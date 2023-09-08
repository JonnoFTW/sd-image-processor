import gc
import math
import queue
import time
import asyncio
import base64
import os
import logging
from io import BytesIO
import atexit
import threading

import PIL.Image
from PIL.PngImagePlugin import PngInfo

import kombu
from kombu import Connection, Producer
from kombu.mixins import ConsumerMixin

import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, EulerAncestralDiscreteScheduler, \
    EulerDiscreteScheduler, KarrasVeScheduler, DPMSolverSinglestepScheduler, HeunDiscreteScheduler, \
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, ScoreSdeVeScheduler, \
    IPNDMScheduler, VQDiffusionScheduler, StableDiffusionUpscalePipeline
from diffusers.models import AutoencoderKL

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import humanize

from sd.attention import StableDiffusionLongPromptWeightingPipeline
from sd.handler import Handler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
for log_name in (
        'urllib3.connectionpool',
        'amqp.connection.Connection.heartbeat_tick',
        'amqp',
):
    logging.getLogger(log_name).setLevel('INFO')

nvmlInit()

# height, width
SIZE_768 = (768, 768)
SIZE_512 = (512, 512)

models = {
    '1.5': {
        'name': "runwayml/stable-diffusion-v1-5",
        'revision': "fp16",
        'scheduler': None,
        'default_size': SIZE_512
    },
    '2.1': {
        "name": "stabilityai/stable-diffusion-2-1",
        "revision": "fp16",
        "scheduler": None,
        'default_size': SIZE_768
    },
    'analog-1.0': {
        "name": "wavymulder/Analog-Diffusion",
        "revision": None,
        "scheduler": lambda: DPMSolverMultistepScheduler.from_pretrained(models['analog-1.0']['name'],
                                                                         subfolder="scheduler"),
        'default_size': SIZE_512
    },
    'anime-3.0': {
        "name": "Linaqruf/anything-v3.0",
        "revision": None,
        "scheduler": None,
        'default_size': SIZE_512
    },
    'hassanblend-1.5': {
        'name': 'hassanblend/hassanblend1.5.1.2',
        'revision': None,
        'scheduler': lambda: DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            trained_betas=None,
            prediction_type='epsilon',
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
        ),
        'vae': 'stabilityai/sd-vae-ft-mse',
        'default_size': SIZE_512
    },
    'rickroll': {
        "name": "TheLastBen/rick-roll-style",
        "revision": None,
        "scheduler": lambda: DPMSolverMultistepScheduler.from_pretrained(models['rickroll']['name'],
                                                                         subfolder="scheduler"),
        'default_size': SIZE_512
    },
    'freedom': {
        'name': 'artificialguybr/freedom',
        'revision': None,
        'scheduler': None,
        'default_size': SIZE_768,
        'vae': 'stabilityai/sd-vae-ft-mse',
    },
    'xl': {
        "name": "stabilityai/stable-diffusion-xl-base-1.0",
        "revision": None,
        'default_size': (1024, 1024),
        'default_steps': 16,
        'vae': "madebyollin/sdxl-vae-fp16-fix",
        'scheduler': None,
        'variant': 'fp16',
        'pipeline': 'sd/xl_pipeline.py'
    }
}

schedulers = {
    'euler-a': EulerAncestralDiscreteScheduler,
    'euler-d': EulerDiscreteScheduler,
    'dpmsolver-multi': DPMSolverMultistepScheduler,
    'dpmsolver-single': DPMSolverSinglestepScheduler,
    'karras': KarrasVeScheduler,
    'huen': HeunDiscreteScheduler,
    'kdpm2-d': KDPM2DiscreteScheduler,
    'kdpm2-a': KDPM2AncestralDiscreteScheduler,
    'lms-d': LMSDiscreteScheduler,
    'pndm': PNDMScheduler,
    'score-sde-ve': ScoreSdeVeScheduler,
    'ipndm': IPNDMScheduler,
    'vqdiffusion': VQDiffusionScheduler
}

DEFAULT_MODEL = '2.1'
DEFAULT_SCHEDULER = 'euler-a'


class Worker(Handler, ConsumerMixin):
    def __init__(self, connection: Connection):
        super().__init__()
        self.connection = connection
        self.queue = queue.Queue()
        self.running = True
        self._pipe = threading.local()
        self._model_name = threading.local()
        self.worker = threading.Thread(target=self.process_queue)
        self.worker.start()

    def get_consumers(self, Consumer, channel):
        print("Making consumer for queue", self.image_requests_queue.name)
        return [
            Consumer(
                queues=[self.image_requests_queue],
                on_message=self.enqueue_image_request,
                accept=['json'],
                auto_declare=True
            )
        ]

    def process_queue(self):
        while self.running:
            try:
                message = self.queue.get(timeout=2)
                self.generate_image(message)
            except queue.Empty:
                pass
        print("Exited process_queue()")

    def enqueue_image_request(self, message: kombu.Message):
        self.queue.put_nowait(message)

    @property
    def pipe(self):
        return getattr(self._pipe, 'val', None)

    @pipe.setter
    def pipe(self, pipe):
        self._pipe.val = pipe

    @pipe.deleter
    def pipe(self):
        del self._pipe.val

    @property
    def model_name(self):
        return getattr(self._model_name, 'val', None)

    @model_name.setter
    def model_name(self, model_name):
        self._model_name.val = model_name

    def generate_image(self, message: kombu.Message):
        payload = message.payload
        payload_to_show = {k: v for k, v in payload.items()}
        if 'img' in payload_to_show:
            payload_to_show['img'] = "<IMAGE>"
        print("[→] Got message:", payload_to_show, message.headers)
        if not payload['prompt'].strip() and 'img' not in payload:
            print(" [✘] Received message with no prompt or image")
            message.ack()
            return
        start = time.time()
        seed = payload.get('seed', int(start))

        try:
            model_name = payload.get('model', DEFAULT_MODEL)
            if model_name not in models:
                raise ValueError(f'model must be one of {", ".join([m for m in models.keys()])}')
            if model_name != self.model_name or self.pipe is None:
                if self.pipe is not None:
                    del self.pipe
                self.pipe = get_pipe(model_name)
                self.model_name = model_name
            if payload.get('scheduler'):
                # load user specified scheduler
                scheduler_name = payload['scheduler']
                if scheduler_name not in schedulers:
                    raise ValueError(f'scheduler must be one of {", ".join([m for m in schedulers.keys()])}')
                print("using user specified scheduler:", scheduler_name)
                scheduler = schedulers[scheduler_name].from_config(self.pipe.scheduler.config)
            elif models[self.model_name].get('scheduler') is not None:
                # load default scheduler for this model
                print("using default scheduler for this model")
                scheduler = models[self.model_name]['scheduler']()
                scheduler_name = scheduler.__class__.__name__
            else:
                # load the global default scheduler
                scheduler_name = DEFAULT_SCHEDULER
                # print("using default scheduler", DEFAULT_SCHEDULER)
                scheduler = schedulers[DEFAULT_SCHEDULER].from_config(self.pipe.scheduler.config)
            logger.info(f"Using scheduler {scheduler_name}")
            self.pipe.scheduler = scheduler
            scheduler_cls = self.pipe.scheduler.__class__.__name__
            with torch.inference_mode():
                default_h, default_w = models[model_name]['default_size']
                height = payload.get('height', default_h)
                width = payload.get('width', default_w)
                generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
                args = dict(
                    prompt=payload['prompt'],
                    height=height,
                    width=width,
                    num_inference_steps=payload.get('steps', models[model_name].get('default_steps', 50)),
                    guidance_scale=payload.get('scale', 7.5),
                    strength=payload.get('strength', 0.8),
                    # scaled_guidance_scale=payload.get('sscale', None),
                    negative_prompt=payload.get('negative_prompt'),
                    generator=generator,
                    # max_embeddings_multiples=3,
                )
                if payload.get('img'):
                    func = self.pipe.img2img
                    input_image = Image.open(BytesIO(base64.b64decode(payload['img']))).convert('RGB')
                    # resize to the lowest multiple of 8
                    new_width = input_image.width
                    new_height = input_image.height

                    if new_width % 8 != 0:
                        new_width -= new_width % 8
                    if new_height % 8 != 0:
                        new_height -= new_height % 8
                    input_image.resize((new_width, new_height))

                    args['height'] = input_image.height
                    args['width'] = input_image.width
                    args['num_inference_steps'] = payload.get('steps', 25)
                    args['image'] = input_image
                else:
                    func = self.pipe.text2img
                image = func(**args).images[0] # type: Image
            took = time.time() - start
            # TODO: implement hi res fix
            #upscale = 'upscale' in payload
            #if upscale:
            #    logger.info("Upscaling")
            #    image = self.upscale(image, args, generator)

            buff = BytesIO()
            meta_data = PngInfo()
            meta_data.add_text('Generated-With', model_name)
            meta_data.add_text('Generated-With-Scheduler', scheduler_cls)
            for field in ('prompt', 'steps', 'scale', 'negative_prompt', 'seed'):
                if payload.get(field):
                    meta_data.add_text(field, str(payload[field]))
            meta_data.add_text('parameters', f'Prompt:{args["prompt"]}\nNegative prompt: {args["negative_prompt"]}\nSteps: {args["num_inference_steps"]}, '
                                             f'Sampler: {scheduler_name}, CFG scale: {args["guidance_scale"]}, Seed: {seed}, Size: {args["height"]}x{args["width"]},\n'
                                             f'Model: {model_name}\n'
                                             f'Generated with: https://github.com/jonnoftw/sd-image-processor'
                               )
            
            image.save(buff, format='png', pnginfo=meta_data)
            del args['generator']
            if 'image' in args:
                args['image'] = payload['img']
            result = {
                'result': {
                    'img_blob': base64.b64encode(buff.getvalue()).decode('ascii'),
                    'ext': 'png',
                    'prompt': payload['prompt'],
                    'negative_prompt': payload.get('negative_prompt'),
                    'duration': round(took, 2),
                    'height': payload.get('height'),
                    'width': payload.get('width'),
                    'steps': payload.get('steps'),
                    'scale': payload.get('scale'),
                    'seed': payload.get('seed', seed),
                    'model': payload.get('model'),
                    'scheduler': payload.get('scheduler'),
             #       'upscale': upscale,
                },
                'args': args,
                'headers': payload.get('headers')
            }
        except Exception as exc:
            result = {'error': str(exc), 'headers': payload.get('headers')}
            logging.critical(exc, exc_info=True)

        with get_connection() as connection:
            with connection.channel() as channel:
                producer = Producer(channel)
                # print("publishing result to", self.exchange, 'routing_key=', message.headers['return-key'])
                producer.publish(
                    result,
                    exchange=self.exchange,
                    routing_key=message.headers['return-key'],
                    serializer='json',
                    retry=True
                )
        message.ack()

    def upscale(self, image, args, generator) -> Image:
        """
        Scale an image up by scale, if <= 1, do nothing
        :param image: the image to be scaled
        :return:
        """
        del self.pipe
        time.sleep(0.25)
        clear_cache()
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_attention_slicing()
        pipeline.enable_sequential_cpu_offload()
        #pipeline.enable_vae_slicing()
        pipeline = pipeline.to("cuda")
        scaled_image = pipeline(
            prompt=args['prompt'],
            image=image,
            negative_prompt=args.get('negative_prompt'),
            generator=generator
        ).images[0]
        del pipeline
        clear_cache()
        return scaled_image


def get_connection():
    broker_url = os.getenv('BROKER_URL', 'amqp://localhost')
    # print("connecting to", broker_url)
    return Connection(broker_url, heartbeat=10)


def run_worker():
    exit_called = False
    try:
        with get_connection() as conn:
            print(' [✓] Awaiting image requests')
            worker = Worker(conn)

            def on_exit():
                if not exit_called:
                    print("calling on_exit()")
                    worker.running = False
                    worker.worker.join()
                    conn.release()

            atexit.register(on_exit)
            asyncio.ensure_future(worker.run())
    except KeyboardInterrupt:
        print("Quitting")
        on_exit()
        exit_called = True


def clear_cache():
    logger.debug("CLEARING CUDA CACHE")
    show_gpu_mem_stats()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.debug("AFTER CACHE CLEAR")
    show_gpu_mem_stats()


def show_gpu_mem_stats():
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    logger.debug(f'total : {info.total}b\t ({humanize.naturalsize(info.total)})')
    logger.debug(f'free  : {info.free}b\t ({humanize.naturalsize(info.free)})')
    logger.debug(f'used  : {info.used}b\t ({humanize.naturalsize(info.used)})')


def get_pipe(name):
    config = models[name]
    model_name = config['name']
    revision = config['revision']
    load_scheduler = config['scheduler']
    clear_cache()
    logger.info(" [✓] Loading model: %s", model_name)

    kwargs = dict(
        # custom_pipeline="lpw_stable_diffusion",
        custom_pipeline="sd/attention.py",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision=revision,
        safety_checker=None,
        requires_safety_checker=False
    )

    vae = config.get('vae')
    if vae:
        logger.info("loading custom VAE: %s", vae)
        vae = AutoencoderKL.from_pretrained(vae, torch_dtype=kwargs['torch_dtype'])
        kwargs['vae'] = vae

    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        **kwargs
    )  # type: StableDiffusionLongPromptWeightingPipeline
    if load_scheduler:
        pipe.scheduler = load_scheduler()
    else:
        scheduler = schedulers[DEFAULT_SCHEDULER]
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    # pipe.enable_sequential_cpu_offload()
    pipe.to("cuda")
    logger.info(" [✓] Finished loading model: %s", model_name)
    return pipe


if __name__ == "__main__":
    run_worker()
