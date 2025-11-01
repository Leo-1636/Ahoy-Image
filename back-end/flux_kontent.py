from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers.utils import load_image

import torch
from torchao.quantization import quantize_, int8_weight_only

def load_model():
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    transformer = FluxTransformer2DModel.from_pretrained(
        model = model_id, 
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
    ).to("cuda")
    quantize_(transformer, int8_weight_only())

    pipeline = FluxKontextPipeline.from_pretrained(
        model = model_id,
        transformer = transformer,
        torch_dtype = torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    return pipeline

def generate_image(pipeline, image: str, prompt: str, width: int, height: int):
    image = load_image(image)
    result = pipeline(
        image = image, 
        prompt = prompt, 
        width = width if width is not None else image.size[0],
        height = height if height is not None else image.size[1], 
        guidance_scale = 2.5, 
    ).images[0]
    return result

