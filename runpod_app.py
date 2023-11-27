import requests

import torch
import runpod

from utils import extract_origin_pathname, upload_image
from txt2img import txt2img

def run (job, _generator = None):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')
        debug = _input.get('debug')
        upload_url = _input.get('upload_url')

        prompt = _input.get('prompt', 'a dog')
        negative_prompt = _input.get('negative_prompt', '')
        width = _input.get('width', 1024)
        height = _input.get('height', 1024)
        num_inference_steps = _input.get('num_inference_steps', 50)
        guidance_scale = _input.get('guidance_scale', 7.0)
        seed = _input.get('seed')

        if seed is not None:
            _generator = torch.Generator(device = 'cuda').manual_seed(seed)

        output_image = txt2img(
            prompt = prompt,
            negative_prompt = negative_prompt,
            width = width,
            height = height,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            generator = _generator
        ).images[0]

        if debug:
            output_image.save('sample.png')

        # # output
        output_url = extract_origin_pathname(upload_url)
        output = { 'output_url': output_url }

        upload_image(upload_url, output_image)

        return output
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
