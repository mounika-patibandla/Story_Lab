import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_pipeline(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    return pipe
