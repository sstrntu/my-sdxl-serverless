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
RUN pip install --upgrade pip

# Install PyTorch packages first with CUDA index
RUN pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining packages from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace

# Copy application files
COPY download_model.py /workspace/download_model.py
COPY server.py /app/server.py

WORKDIR /app
CMD ["python", "server.py"]
