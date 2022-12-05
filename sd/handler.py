from kombu import Exchange, Queue


class Handler:
    def __init__(self):
        self.exchange = Exchange(
            'image_gen',
            durable=True,
            auto_delete=False,
            type='direct'
        )
        self.image_requests_queue = Queue(
            'image_requests',
            exchange=self.exchange,
            durable=True,
            auto_delete=False,
            routing_key='req'
        )
        # self.image_results_queue = Queue(
        #     'image_results',
        #     exchange=self.exchange,
        #     durable=True,
        #     auto_delete=False,
        #     routing_key='res'
        # )
