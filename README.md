# ImageGen Producer

Service to:

1. read image generation requests from rabbitmq
2. generate the requested image
3. put the result back on a queue

Features:

* Doesn't use the safety checker in stable diffusion
* Supports `seed` values
* Uses torchdynamo to speed up generation. Doesn't work well on my 8GB card though.

Running:

```shell
./sd/client.py
```

Will save images to the current folder and start a loop to accept prompts.