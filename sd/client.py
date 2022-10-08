#!/usr/bin/env python3
import threading
import base64
from io import BytesIO

from PIL import Image

from kombu import Connection, Producer, Message
from kombu.mixins import ConsumerMixin

from sd.handler import Handler


class ImageConsumer(Handler, ConsumerMixin):

    def __init__(self, connection: Connection):
        super().__init__()
        self.connection = connection

    def get_consumers(self, Consumer, channel):
        return [
            Consumer(
                queues=[self.image_results_queue],
                callbacks=[self.on_response]
            )
        ]

    def on_response(self, body, message: Message):
        print("got message", message)
        result = message.payload['result']
        fname = f"{result['prompt']}.{result['ext']}"
        print("Writing", fname)

        im_bytes = base64.b64decode(result['img_blob'])
        im = Image.open(BytesIO(im_bytes))
        im.save(fname)
        message.ack()


class ImageRequester(Handler):
    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def __call__(self, prompt):
        with self.connection.channel() as channel:
            producer = Producer(channel)
            producer.publish(
                {'prompt': prompt, 'steps': 24},
                queue=self.image_requests_queue,
                exchange=self.exchange,
                routing_key='req',
            )


def run_consumer(connection):
    print("running consumer")
    image_consumer = ImageConsumer(connection)
    image_consumer.run()


def main(broker_url):
    connection = Connection(broker_url)
    thread = threading.Thread(target=run_consumer, args=(connection,))
    thread.start()
    image_requester = ImageRequester(connection)
    while True:
        try:
            prompt = input("Prompt >>> ")
            image_requester(prompt)
        except KeyboardInterrupt:
            thread.join()
            break


if __name__ == '__main__':
    main('amqp://localhost')
