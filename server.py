import runpod
from diffusers import DiffusionPipeline
import torch
import os
import base64
import io

# Load model from /workspace/models/ (required for RunPod Serverless)
MODEL_PATH = "/workspace/models"

print(f"🚀 Loading model from: {MODEL_PATH}")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("cuda")

def generate(job):
    """RunPod serverless handler function"""
    try:
        job_input = job["input"]

        # Prompt is required
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' parameter in input."}

        negative_prompt = job_input.get("negative_prompt", "")
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        num_inference_steps = job_input.get("num_inference_steps", 30)

        print(f"Generating image for prompt: {prompt[:50]}...")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps
        ).images[0]

        # Convert image to base64 for RunPod response
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        print("✅ Image generated successfully")

        return {
            "image": image_base64,
            "message": "Image generated successfully"
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}

print("🚀 Starting RunPod Serverless handler...")
runpod.serverless.start({"handler": generate})