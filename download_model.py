# download_model.py

import os
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import torch

hf_token = os.environ.get("HF")
if not hf_token:
    raise ValueError(
        "HF_TOKEN environment variable is required.\n"
        "Set it in RunPod's Public Environment Variables.\n"
        "Get your token: https://huggingface.co/settings/tokens\n"
        "Request access: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
    )

login(token=hf_token)
print("✅ Logged in to HuggingFace Hub")

print("⬇️ Downloading Stable Diffusion 3.5 Large model...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
)
pipe.save_pretrained("/workspace/models/")
print("✅ Model saved to /workspace/models")