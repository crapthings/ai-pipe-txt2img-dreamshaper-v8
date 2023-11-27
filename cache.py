import torch
from diffusers import DiffusionPipeline
from config import model_name, model_dir

print('cache model')

pipe = DiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

print('done')
