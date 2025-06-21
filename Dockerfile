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

# Copy requirements first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install all Python dependencies from requirements.txt
RUN pip install -r /tmp/requirements.txt --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace

# Copy application files
COPY download_model.py /workspace/download_model.py
COPY server.py /app/server.py

WORKDIR /app
CMD ["python", "server.py"]
