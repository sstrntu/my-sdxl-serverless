FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python packages
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
    diffusers transformers accelerate safetensors flask runpod

# Create workspace dir
RUN mkdir -p /workspace/models

WORKDIR /app
COPY server.py .
COPY download_model.py .

# Run download_model.py at container start if model not found
CMD bash -c '\
if [ ! -f "/workspace/models/model_index.json" ]; then \
    echo "⚙️ Model not found, running download_model.py"; \
    python download_model.py; \
else \
    echo "✅ Model already exists. Skipping download."; \
fi && \
echo "🚀 Starting server..." && python server.py'