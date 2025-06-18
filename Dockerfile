FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up Python and basic libraries
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
    diffusers transformers accelerate safetensors flask

WORKDIR /app
COPY server.py .

CMD ["python", "server.py"]