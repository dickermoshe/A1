import os
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL.Image import Image

def make():
    os.mkdir('~/.huggingface')
    with open('~/.huggingface/token', 'w') as f:
        f.write(os.environ['HUGGINGFACE_TOKEN'])

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)

    pipe = pipe.to("cuda")

    prompt = "a photograph of an astronaut riding a horse"
    with autocast("cuda"):
        image: Image = pipe(prompt)["sample"][0]

    image.save('image.png')

    # Get file path of image
    image_path = os.path.join(os.getcwd(), 'image.png')
    return {'image_path': image_path}
