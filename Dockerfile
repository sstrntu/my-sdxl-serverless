FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System and Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
    diffusers transformers accelerate safetensors flask huggingface_hub

# Inject HF token
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Preload the model
RUN huggingface-cli login --token $HF_TOKEN && \
    python -c "\
from diffusers import StableDiffusion3Pipeline; \
pipe = StableDiffusion3Pipeline.from_pretrained( \
    'stabilityai/stable-diffusion-3.5-large', torch_dtype='torch.float16' \
); \
pipe.save_pretrained('/workspace/models')"

# App
COPY server.py /app/server.py
WORKDIR /app

CMD ["python", "server.py"]
