from diffusers import DiffusionPipeline
import torch
from flask import Flask, request, jsonify
import os

# Use mounted volume for model cache
MODEL_CACHE_DIR = "/models/sdxl-3.5"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Load the SDXL pipeline from HuggingFace into GPU
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE_DIR
).to("cuda")

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "a fantasy landscape")
    image = pipe(prompt).images[0]
    image_path = "/tmp/output.png"
    image.save(image_path)
    return jsonify({"message": "Image generated", "path": image_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)