FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System and Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio diffusers transformers accelerate safetensors huggingface_hub runpod

WORKDIR /workspace

COPY download_model.py /workspace/download_model.py

# Accept HF_TOKEN as a build argument and set it as environment variable
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download the model using the provided token
RUN python /workspace/download_model.py

COPY server.py /app/server.py
WORKDIR /app
CMD ["python", "server.py"]
