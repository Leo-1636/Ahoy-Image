from diffusers import FluxPipeline, FluxTransformer2DModel

import torch
from torchao.quantization import quantize_, int8_weight_only

def load_model():
    model_id = "black-forest-labs/FLUX.1-dev"
    transformer = FluxTransformer2DModel.from_pretrained(
        model = model_id, 
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
    ).to("cuda")
    quantize_(transformer, int8_weight_only())

    pipeline = FluxPipeline.from_pretrained(
        model = model_id,
        transformer = transformer,
        torch_dtype = torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    return pipeline

def generate_image(pipeline, prompt: str, width: int = 1024, height: int = 1024):
    result = pipeline(
        prompt = prompt, 
        width = width, 
        height = height, 
        guidance_scale = 3.5, 
        num_inference_steps = 50
    ).images[0]
    return result