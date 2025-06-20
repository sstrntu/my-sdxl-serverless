# download_model.py

import os
import shutil
from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline
import torch

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



def check_available_space():
    """Check available space on key mount points."""
    print("📊 Checking disk space...")
    
    paths_to_check = ['/runpod-volume', '/', '/tmp']
    for path in paths_to_check:
        if os.path.exists(path):
            usage = get_disk_usage(path)
            if usage:
                print(f"  {path}: {usage['free_gb']}GB free / {usage['total_gb']}GB total ({usage['used_percent']:.1f}% used)")



MODEL_DIR = "/runpod-volume/models"
MODEL_INDEX = os.path.join(MODEL_DIR, "model_index.json")

# Check initial disk space
check_available_space()

if os.path.exists(MODEL_INDEX):
    print(f"Model already exists at {MODEL_DIR}, skipping download.")
else:
    print("Model not found, downloading...")
    
    # Check space before download
    print("\n📊 Disk space before download:")
    check_available_space()
    
    # Verify we have enough space (SD3.5 Large is ~12GB)
    runpod_usage = get_disk_usage('/runpod-volume')
    if runpod_usage and runpod_usage['free_gb'] < 15:
        print(f"⚠️  Warning: Only {runpod_usage['free_gb']}GB free space available.")
        print("   SD3.5 Large requires ~12-15GB. Download may fail.")
    

    
    # Ensure MODEL_DIR has proper permissions and exists
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
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required.\n"
            "Set it in RunPod's Public Environment Variables.\n"
            "Get your token: https://huggingface.co/settings/tokens\n"
            "Request access: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
        )
    
    print("Downloading Stable Diffusion 3.5 Large model...")
    print("⏳ This may take 10-20 minutes depending on connection speed...")
    print("🔑 Using token directly to avoid login disk issues...")
    
    try:
        # Skip login entirely - use token directly in from_pretrained
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.bfloat16,
            token=hf_token,  # Pass token directly - no login needed
        )
        
        # Check space before saving
        print("\n📊 Disk space before saving model:")
        check_available_space()
        
        print("Saving model to local directory...")
        print(f"📍 Saving to: {MODEL_DIR}")
        
        # Check disk space right before save
        usage = get_disk_usage('/runpod-volume')
        if usage:
            print(f"💾 Available space: {usage['free_gb']}GB")
        
        # Save with error handling
        try:
            # Try to save individual components to isolate the issue
            print("🔍 Attempting to save pipeline components individually...")
            
            # First, try saving just the config manually
            test_config_path = os.path.join(MODEL_DIR, "test_config.json")
            try:
                import json
                test_config = {"test": "config"}
                with open(test_config_path, "w") as f:
                    json.dump(test_config, f)
                os.remove(test_config_path)
                print("✅ Manual JSON write test successful")
            except Exception as json_err:
                print(f"❌ Manual JSON write failed: {json_err}")
                raise json_err
            
            # If manual test passes, try the actual save
            pipe.save_pretrained(MODEL_DIR)
            print("✅ Model downloaded and saved successfully.")
            
        except OSError as e:
            print(f"❌ OSError during save: {e}")
            print(f"Error code: {e.errno}")
            print(f"Available space: {usage['free_gb'] if usage else 'unknown'}GB")
            raise
        
        # Check final space
        print("\n📊 Final disk space:")
        check_available_space()
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("\n📊 Disk space at failure:")
        check_available_space()
        raise