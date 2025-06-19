FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System and Python
# RUN apt-get update && apt-get install -y \
#     python3.10 python3-pip git libgl1 libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*

# RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
#     ln -sf /usr/bin/pip3 /usr/bin/pip

# # Install dependencies
# RUN pip install --upgrade pip && \
#     pip install torch torchvision torchaudio \
#     diffusers transformers accelerate safetensors flask huggingface_hub

# Create workspace for model and app
# WORKDIR /workspace

# # Copy your download script
# COPY download_model.py /workspace/download_model.py

# Accept HF_TOKEN as a build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Debug: print first 8 characters of HF_TOKEN only
RUN echo "HF_TOKEN is ${HF_TOKEN}"

# (Comment out or remove model download and server copy for now)
# RUN python /workspace/download_model.py
# COPY server.py /app/server.py
# WORKDIR /app
# CMD ["python", "server.py"]
