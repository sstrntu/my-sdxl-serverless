# download_model.py

import os
import shutil
import json
from diffusers import StableDiffusion3Pipeline
import torch

# Constants
MODEL_DIR = "/runpod-volume/models"
MODEL_INDEX = os.path.join(MODEL_DIR, "model_index.json")
MODEL_NAME = "stabilityai/stable-diffusion-3.5-large"
MIN_REQUIRED_SPACE_GB = 15
PATHS_TO_CHECK = ['/runpod-volume', '/', '/tmp']

def get_disk_usage(path):
    """Get disk usage statistics for a given path."""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            'total_gb': total // (1024**3),
            'used_gb': used // (1024**3),
            'free_gb': free // (1024**3),
            'used_percent': (used / total) * 100
        }
    except Exception as e:
        print(f"❌ Error getting disk usage for {path}: {e}")
        return None

def check_available_space(stage=""):
    """Check available space on key mount points with optional stage description."""
    stage_prefix = f"{stage}: " if stage else ""
    print(f"📊 {stage_prefix}Checking disk space...")
    
    for path in PATHS_TO_CHECK:
        if os.path.exists(path):
            usage = get_disk_usage(path)
            if usage:
                print(f"  {path}: {usage['free_gb']}GB free / {usage['total_gb']}GB total ({usage['used_percent']:.1f}% used)")

def check_runpod_space_requirements():
    """Check if runpod-volume has enough space for model download."""
    runpod_usage = get_disk_usage('/runpod-volume')
    if runpod_usage and runpod_usage['free_gb'] < MIN_REQUIRED_SPACE_GB:
        print(f"⚠️  Warning: Only {runpod_usage['free_gb']}GB free space available.")
        print(f"   SD3.5 Large requires ~{MIN_REQUIRED_SPACE_GB}GB. Download may fail.")
        return False
    return True

def setup_model_directory():
    """Create model directory and test write permissions."""
    print(f"📁 Creating model directory: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True, mode=0o777)
    
    # Test write permissions
    test_file = os.path.join(MODEL_DIR, "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Write permissions confirmed for {MODEL_DIR}")
    except Exception as e:
        print(f"❌ Write permission test failed for {MODEL_DIR}: {e}")
        raise

def validate_hf_token():
    """Validate that HuggingFace token is available."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required.\n"
            "Set it in RunPod's Public Environment Variables.\n"
            "Get your token: https://huggingface.co/settings/tokens\n"
            f"Request access: https://huggingface.co/{MODEL_NAME}"
        )
    return hf_token

def download_and_save_model(hf_token):
    """Download and save the model with proper error handling."""
    print("Downloading Stable Diffusion 3.5 Large model...")
    print("⏳ This may take 10-20 minutes depending on connection speed...")
    print("🔑 Using token directly to avoid login disk issues...")
    
    # Download model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        token=hf_token,  # Pass token directly - no login needed
    )
    
    # Check space before saving
    check_available_space("Before saving model")
    
    print("Saving model to local directory...")
    print(f"📍 Saving to: {MODEL_DIR}")
    
    # Show available space
    usage = get_disk_usage('/runpod-volume')
    if usage:
        print(f"💾 Available space: {usage['free_gb']}GB")
    
    # Save the model
    try:
        pipe.save_pretrained(MODEL_DIR)
        print("✅ Model downloaded and saved successfully.")
    except OSError as e:
        print(f"❌ OSError during save: {e}")
        print(f"Error code: {e.errno}")
        print(f"Available space: {usage['free_gb'] if usage else 'unknown'}GB")
        raise

def main():
    """Main download process."""
    # Check initial disk space
    check_available_space("Initial")
    
    # Check if model already exists
    if os.path.exists(MODEL_INDEX):
        print(f"Model already exists at {MODEL_DIR}, skipping download.")
        return
    
    print("Model not found, downloading...")
    
    try:
        # Pre-download checks
        check_available_space("Before download")
        check_runpod_space_requirements()
        setup_model_directory()
        hf_token = validate_hf_token()
        
        # Download and save
        download_and_save_model(hf_token)
        
        # Final space check
        check_available_space("Final")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        check_available_space("At failure")
        raise

if __name__ == "__main__":
    main()