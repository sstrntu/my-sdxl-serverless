FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HF_DATASETS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip && \
    pip install \
        torch torchvision torchaudio \
        diffusers transformers accelerate safetensors \
        huggingface_hub \
        flask

# Create app directory and copy files
WORKDIR /app
COPY server.py .

# Expose port
EXPOSE 8000

# Launch server
CMD ["python", "server.py"]