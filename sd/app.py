import asyncio
import base64
import os
from io import BytesIO

from PIL.PngImagePlugin import PngInfo

import kombu
import torch
# import torchdynamo

from kombu import Connection, Producer
from kombu.mixins import ConsumerMixin


from sd.pipeline import StableDiffusionPipeline
from sd.handler import Handler


class Worker(Handler, ConsumerMixin):
    def __init__(self, connection: Connection, pipe):
        super().__init__()
        self.connection = connection
        self.pipe = pipe

    def get_consumers(self, Consumer, channel):
        return [
            Consumer(
                queues=[self.image_requests_queue],
                on_message=self.generate_image,
                accept=['json'],
                auto_declare=True,
            )
        ]

    # @torchdynamo.optimize("ofi")
    # @torchdynamo.optimize("fx2trt")
    def generate_image(self, message: kombu.Message):
        payload = message.payload
        print("Got message:", message, payload)
        if not payload['prompt'].strip():
            print("Received message with no prompt")
            message.ack()
            return

        image = self.pipe(
            prompt=payload['prompt'],
            height=payload.get('height', 512),
            width=payload.get('width', 512),
            num_inference_steps=payload.get('steps', 50),
            guidance_scale=payload.get('scale', 7.5),
            negative_prompt=payload.get('negative_prompt'),
            seed=payload.get('seed')
        ).images[0]

        buff = BytesIO()
        meta_data = PngInfo()
        meta_data.add_text('Generated-With', 'StableDiffusion-v-1-4')
        for field in ('prompt', 'steps', 'scale', 'negative_prompt', 'seed'):
            if payload.get(field):
                meta_data.add_text(field, str(payload[field]))

        image.save(buff, format='png', pnginfo=meta_data)

        result = {
            'result': {
                'img_blob': base64.b64encode(buff.getvalue()).decode('ascii'),
                'ext': 'png',
                'prompt': payload['prompt'],
                'negative_prompt': payload.get('negative_prompt')
            },
            'headers': {
                payload.get('headers')
            }
        }
        with self.connection.channel() as channel:
            producer = Producer(channel)
            producer.publish(
                result,
                exchange=self.exchange,
                routing_key='res',
                serializer='json',
                retry=True
            )

        message.ack()


def run_worker(broker_url, pipe):
    with Connection(broker_url) as conn:
        print(' [x] Awaiting image requests')
        worker = Worker(conn, pipe)
        asyncio.ensure_future(worker.run())


if __name__ == "__main__":
    print(" [x] Loading pipeline")
    _pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_type=torch.float16,
        revision="fp16"
    )
    # disable safety check
    _pipe.safety_checker = lambda images, **kwargs: (images,)
    _pipe.enable_attention_slicing()
    _pipe = _pipe.to("cuda")

    try:
        run_worker(os.getenv('BROKER_URL', 'amqp://localhost'), _pipe)
    except KeyboardInterrupt:
        exit("Quitting")
