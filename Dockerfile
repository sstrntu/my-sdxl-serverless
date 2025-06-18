FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up Python and system libraries
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Python packages
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
    diffusers transformers accelerate safetensors flask huggingface_hub

# Pre-download SDXL 3.5 weights without loading the model
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', local_dir='/models/sdxl', local_dir_use_symlinks=False)"

WORKDIR /app
COPY server.py .

CMD ["python", "server.py"]