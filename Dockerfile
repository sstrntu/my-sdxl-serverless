FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python packages
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
    diffusers transformers accelerate safetensors flask runpod

# Accept HuggingFace token as build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download and save SD 3.5 Large model to /workspace/models/ (required for RunPod Serverless)
RUN mkdir -p /workspace/models && \
    python -c "\
import torch; \
from diffusers import StableDiffusion3Pipeline; \
from huggingface_hub import login; \
import os; \
hf_token = os.environ.get('HF_TOKEN'); \
if hf_token: \
    print('Logging in to HuggingFace...'); \
    login(token=hf_token); \
else: \
    print('Warning: No HF_TOKEN provided, attempting without authentication...'); \
print('Downloading SD 3.5 Large model...'); \
pipe = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-large', torch_dtype=torch.bfloat16); \
print('Saving model to /workspace/models/...'); \
pipe.save_pretrained('/workspace/models/'); \
print('Model saved successfully'); \
del pipe; \
"

WORKDIR /app
COPY server.py .

CMD ["python", "server.py"]