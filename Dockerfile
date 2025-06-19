FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
    diffusers transformers accelerate safetensors flask huggingface_hub

# Create workspace for model and app
WORKDIR /workspace

# Pre-download SDXL 3.5 model at build time using build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "\
import os; \
from huggingface_hub import login; \
from diffusers import StableDiffusion3Pipeline; \
login(token=os.environ['HF_TOKEN']); \
pipe = StableDiffusion3Pipeline.from_pretrained( \
    'stabilityai/stable-diffusion-3.5-large', torch_dtype='torch.float16' \
); \
pipe.save_pretrained('/workspace/models')"

# Copy your server script
COPY server.py /app/server.py
WORKDIR /app

# Run the app
CMD ["python", "server.py"]
