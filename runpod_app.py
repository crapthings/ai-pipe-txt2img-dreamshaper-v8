import math
import requests

import numpy as np
import torch
import runpod

from utils import extract_origin_pathname, upload_image, rounded_size
from txt2img import txt2img, img2img

def run (job, _generator = None):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')
        debug = _input.get('debug')
        upload_url = _input.get('upload_url')

        prompt = _input.get('prompt', 'a dog')
        negative_prompt = _input.get('negative_prompt', '')
        width = int(np.clip(_input.get('width', 768), 256, 1024))
        height = int(np.clip(_input.get('height', 768), 256, 1024))
        # width = _input.get('width', 768)
        # height = _input.get('height', 768)
        num_inference_steps = _input.get('num_inference_steps', 50)
        guidance_scale = _input.get('guidance_scale', 7.0)
        seed = _input.get('seed')

        upscale = _input.get('upscale')
        strength = _input.get('strength')

        renderWidth, renderHeight = rounded_size(width, height)

        if seed is not None:
            _generator = torch.Generator(device = 'cuda').manual_seed(seed)

        output_image = txt2img(
            prompt = prompt,
            negative_prompt = negative_prompt,
            width = renderWidth,
            height = renderHeight,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            generator = _generator
        ).images[0]

        output_image = output_image.resize([width, height])

        if upscale is not None:
            output_image = output_image.resize([width * upscale, height * upscale])

        if strength is not None:
            output_image = img2img(
                image = output_image,
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = math.ceil(num_inference_steps / strength),
                guidance_scale = guidance_scale,
                strength = strength,
                generator = _generator
            ).images[0]

        # # output
        output_url = extract_origin_pathname(upload_url)
        output = { 'output_url': output_url }

        if debug:
            output_image.save('sample.png')
        else:
            upload_image(upload_url, output_image)

        return output
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
