import os
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL.Image import Image

def make():
    

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)

    pipe = pipe.to("cuda")

    prompt = "a photograph of an astronaut riding a horse"
    with autocast("cuda"):
        image: Image = pipe(prompt)["sample"][0]
    return {'Done': 'Done'}


if __name__ == '__main__':
    os.mkdir('~/.huggingface')
    with open('~/.huggingface/token', 'w') as f:
        f.write(os.environ['HUGGINGFACE_TOKEN'])
    make()
    
