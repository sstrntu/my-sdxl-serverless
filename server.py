import runpod
from diffusers import StableDiffusion3Pipeline
import torch
import os
import base64
from io import BytesIO
import subprocess
import sys

# Set HuggingFace cache directory to RunPod network volume
os.environ['HF_HOME'] = '/runpod-volume/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume/hf_cache/transformers'
os.environ['HF_HUB_CACHE'] = '/runpod-volume/hf_cache/hub'

# Additional environment variables to ensure everything saves to runpod-volume
os.environ['TORCH_HOME'] = '/runpod-volume/torch_cache'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/runpod-volume/torch_cache'
os.environ['XDG_CACHE_HOME'] = '/runpod-volume/cache'

# Create additional cache directories
os.makedirs('/runpod-volume/torch_cache', exist_ok=True)
os.makedirs('/runpod-volume/cache', exist_ok=True)

# Model configuration
MODEL_PATH = "/runpod-volume/models"
MODEL_INDEX = os.path.join(MODEL_PATH, "model_index.json")

def download_model_if_needed():
    """Download the model if it doesn't exist locally"""
    if os.path.exists(MODEL_INDEX):
        print(f"✅ Model already exists at {MODEL_PATH}")
        return True
    
    print("📦 Model not found, downloading...")
    try:
        # Run the download script
        result = subprocess.run([sys.executable, "/workspace/download_model.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Model download completed successfully")
        print("📋 Download script output:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Download script warnings:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model download failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def load_model():
    """Load the model into memory"""
    print(f"🚀 Loading model from: {MODEL_PATH}")
    
    # Ensure we're loading from the correct path
    if not os.path.exists(MODEL_INDEX):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Download may have failed.")
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        cache_dir='/runpod-volume/hf_cache'  # Explicitly set cache dir
    ).to("cuda")
    print("✅ Model loaded successfully")
    return pipe

# Initialize model at startup
print("🔄 Initializing model...")
print(f"📍 Model will be stored at: {MODEL_PATH}")
print(f"📍 Cache directory: {os.environ.get('HF_HOME')}")

if download_model_if_needed():
    pipe = load_model()
else:
    print("❌ Failed to download model. Exiting.")
    sys.exit(1)

def handler(job):
    """RunPod serverless handler function"""
    try:
        job_input = job.get("input", {})
        
        # Required parameters
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt")

        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        if not negative_prompt:
            return {"error": "Missing 'negative_prompt' in input"}

        # Optional parameters with tuned defaults for SD 3 Medium
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        num_inference_steps = job_input.get("num_inference_steps", 50)
        guidance_scale = job_input.get("guidance_scale", 7.0)  # Optimal for SD3 Medium
        seed = job_input.get("seed", None)

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        print(f"🎨 Generating image with prompt: {prompt[:50]}...")

        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image": image_base64,
            "message": "✅ Image generation complete",
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        }
    
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})