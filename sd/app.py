import queue
import time
import asyncio
import base64
import os
from io import BytesIO
import atexit
import threading

from PIL.PngImagePlugin import PngInfo

import kombu
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# import torchdynamo

from kombu import Connection, Producer
from kombu.mixins import ConsumerMixin

# from sd.pipeline import StableDiffusionPipeline
from sd.handler import Handler

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# "CompVis/stable-diffusion-v1-4"
# MODEL_NAME = "runwayml/stable-diffusion-v1-5"
MODEL_NAME = "stabilityai/stable-diffusion-2"


class Worker(Handler, ConsumerMixin):
    def __init__(self, connection: Connection, pipe):
        super().__init__()
        self.connection = connection
        self.pipe = pipe
        self.queue = queue.Queue()
        self.running = True
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

    def generate_image(self, message: kombu.Message):
        payload = message.payload
        print("[→] Got message:", payload, message.headers)
        if not payload['prompt'].strip():
            print(" [✘] Received message with no prompt")
            message.ack()
            return
        start = time.time()
        seed = int(start)
        try:
            with torch.inference_mode():
                height = payload.get('height', 768)
                width = payload.get('width', 768)
                generator = torch.Generator(device=_pipe.device).manual_seed(payload.get('seed', seed))
                image = self.pipe(
                    prompt=payload['prompt'],
                    height=height,
                    width=width,
                    num_inference_steps=payload.get('steps', 50),
                    guidance_scale=payload.get('scale', 7.5),
                    negative_prompt=payload.get('negative_prompt'),
                    generator=generator
                ).images[0]
            took = time.time() - start

            buff = BytesIO()
            meta_data = PngInfo()
            meta_data.add_text('Generated-With', MODEL_NAME)
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
                    'seed': payload.get('seed', seed)
                },
                'headers': payload.get('headers')
            }
        except Exception as exc:
            result = {'error': str(exc), 'headers': payload.get('headers')}

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


def run_worker(pipe):
    with get_connection() as conn:
        print(' [✓] Awaiting image requests')
        worker = Worker(conn, pipe)

        def on_exit():
            worker.running = False
            worker.worker.join()
            conn.release()

        atexit.register(on_exit)
        asyncio.ensure_future(worker.run())


if __name__ == "__main__":
    print(" [✓] Loading pipeline")
    _pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        requires_safety_checker=False
    ) # type: StableDiffusionPipeline
    _pipe.scheduler = DPMSolverMultistepScheduler.from_config(_pipe.scheduler.config)
    _pipe.enable_attention_slicing()
    _pipe.enable_xformers_memory_efficient_attention()
    _pipe.to("cuda")

    try:
        run_worker(_pipe)
    except KeyboardInterrupt:
        exit("Quitting")
