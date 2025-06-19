import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import os

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError(
        "HF_TOKEN environment variable is required.\n"
        "Get it at: https://huggingface.co/settings/tokens\n"
        "Request access to: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
    )

print("🔐 Logging into HuggingFace...")
login(token=hf_token)

print("📥 Downloading SDXL 3.5 Large model...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
)

print("💾 Saving model to /workspace/models")
pipe.save_pretrained("/workspace/models")