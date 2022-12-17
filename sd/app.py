import queue
import time
import asyncio
import base64
import os
from io import BytesIO
import atexit
import threading

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import kombu
import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

from kombu import Connection, Producer
from kombu.mixins import ConsumerMixin

from sd.attention import StableDiffusionLongPromptWeightingPipeline
from sd.handler import Handler

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


models = {
    '1.5': ("runwayml/stable-diffusion-v1-5", "fp16", None),
    '2.1': ("stabilityai/stable-diffusion-2-1", "fp16", None),
    'analog-1.0': ("wavymulder/Analog-Diffusion", None, True),
    'anime-3.0': ("Linaqruf/anything-v3.0", None, None),
}
DEFAULT_MODEL = '2.1'


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
        print("Exited")

    # @torchdynamo.optimize("ofi")
    # @torchdynamo.optimize("fx2trt")
    def enqueue_image_request(self, message: kombu.Message):
        # print("Enqueuing request")
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
        if not payload['prompt'].strip():
            print(" [✘] Received message with no prompt")
            message.ack()
            return
        start = time.time()
        seed = int(start)

        try:
            model_name = payload.get('model', DEFAULT_MODEL)
            if model_name not in models:
                raise ValueError(f'model must be one of {", ".join([m for m in models.keys()])}')
            if model_name != self.model_name:
                if self.pipe is not None:
                    del self.pipe
                    torch.cuda.empty_cache()
                self.pipe = get_pipe(model_name)
                self.model_name = model_name
            with torch.inference_mode():
                height = payload.get('height', 768)
                width = payload.get('width', 768)
                generator = torch.Generator(device=self.pipe.device).manual_seed(payload.get('seed', seed))
                args = dict(
                    prompt=payload['prompt'],
                    height=height,
                    width=width,
                    num_inference_steps=payload.get('steps', 50),
                    guidance_scale=payload.get('scale', 7.5),
                    # scaled_guidance_scale=payload.get('sscale', None),
                    negative_prompt=payload.get('negative_prompt'),
                    generator=generator
                )
                if payload.get('img'):
                    func = self.pipe.img2img
                    input_image = Image.open(BytesIO(base64.b64decode(payload['img']))).convert('RGB')
                    args['height'] = input_image.height
                    args['width'] = input_image.width
                    args['num_inference_steps'] = 32
                    args['image'] = input_image
                else:
                    func = self.pipe.text2img
                image = func(**args).images[0]
            took = time.time() - start

            buff = BytesIO()
            meta_data = PngInfo()
            meta_data.add_text('Generated-With', model_name)
            for field in ('prompt', 'steps', 'scale', 'negative_prompt', 'seed'):
                if payload.get(field):
                    meta_data.add_text(field, str(payload[field]))

            image.save(buff, format='png', pnginfo=meta_data)

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
                },
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


def get_connection():
    broker_url = os.getenv('BROKER_URL', 'amqp://localhost')
    # print("connecting to", broker_url)
    return Connection(broker_url, heartbeat=10)


def run_worker():
    with get_connection() as conn:
        print(' [✓] Awaiting image requests')
        worker = Worker(conn)

        def on_exit():
            worker.running = False
            worker.worker.join()
            conn.release()

        atexit.register(on_exit)
        asyncio.ensure_future(worker.run())


def get_pipe(name):
    model_name, revision, load_scheduler = models[name]

    logger.info(" [✓] Loading model: %s", model_name)
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="sd/attention.py",
        torch_dtype=torch.float16,
        revision=revision,
        safety_checker=None,
        requires_safety_checker=False
    )  # type: StableDiffusionLongPromptWeightingPipeline
    if load_scheduler:
        pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    pipe.to("cuda")
    return pipe


if __name__ == "__main__":
    try:
        run_worker()
    except KeyboardInterrupt:
        exit("Quitting")
