# download_model.py

import os
from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline
import torch

# Set HuggingFace cache directory to root filesystem
os.environ['HF_HOME'] = '/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/hf_cache/transformers'
os.environ['HF_HUB_CACHE'] = '/hf_cache/hub'

MODEL_DIR = "/models"
MODEL_INDEX = os.path.join(MODEL_DIR, "model_index.json")

if os.path.exists(MODEL_INDEX):
    print(f"Model already exists at {MODEL_DIR}, skipping download.")
else:
    print("Model not found, downloading...")
    
    # Create cache directories
    os.makedirs('/hf_cache', exist_ok=True)
    os.makedirs('/hf_cache/transformers', exist_ok=True)
    os.makedirs('/hf_cache/hub', exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required.\n"
            "Set it in RunPod's Public Environment Variables.\n"
            "Get your token: https://huggingface.co/settings/tokens\n"
            "Request access: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
        )
    
    print("Logging into Hugging Face...")
    login(token=hf_token)
    
    print("Downloading Stable Diffusion 3.5 Large model...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16
    )
    
    print("Saving model to local directory...")
    pipe.save_pretrained(MODEL_DIR)
    print("Model downloaded and saved successfully.")