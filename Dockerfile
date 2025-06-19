FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System and Python installation with cleanup
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch first (largest packages)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML and AI libraries
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface_hub \
    protobuf>=3.20.0 \
    sentencepiece

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Set working directory under runpod-volume
WORKDIR /runpod-volume/app

# Copy scripts to the runpod-volume location
COPY download_model.py /runpod-volume/app/download_model.py
COPY server.py /runpod-volume/app/server.py

# Set environment variables for cache and model directories under runpod-volume
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache/transformers
ENV HF_HUB_CACHE=/runpod-volume/hf_cache/hub
ENV MODEL_DIR=/runpod-volume/models

# Create necessary directories
RUN mkdir -p /runpod-volume/hf_cache /runpod-volume/models /runpod-volume/app

# Note: Model and cache directories are now all under /runpod-volume for persistence

CMD ["python", "server.py"]
