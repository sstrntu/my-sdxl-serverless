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

# Create workspace for model and app
WORKDIR /workspace

# Copy the HuggingFace token file into the image
COPY hf_token.txt /workspace/hf_token.txt

# Set HF_TOKEN environment variable from the file
RUN export HF_TOKEN=$(cat /workspace/hf_token.txt) && \
    HF_TOKEN=$HF_TOKEN python download_model.py

# App
COPY server.py /app/server.py
WORKDIR /app

CMD ["python", "server.py"]
