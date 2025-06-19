# download_model.py

import os
import shutil
import glob
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

def clear_cache_directories():
    """Clear cache directories to free up space."""
    print("🧹 Clearing cache directories...")
    
    cache_dirs = [
        '/tmp/*',
        '/var/tmp/*',
        '/root/.cache/*',
        '~/.cache/*'
    ]
    
    freed_space = 0
    for cache_pattern in cache_dirs:
        try:
            files = glob.glob(cache_pattern)
            for file_path in files:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_space += size
                    elif os.path.isdir(file_path):
                        size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                 for dirpath, dirnames, filenames in os.walk(file_path)
                                 for filename in filenames)
                        shutil.rmtree(file_path)
                        freed_space += size
        except Exception as e:
            print(f"Warning: Could not clear {cache_pattern}: {e}")
    
    print(f"✅ Freed up {freed_space // (1024**2)} MB of cache space")
    return freed_space

def check_available_space():
    """Check available space on key mount points."""
    print("📊 Checking disk space...")
    
    paths_to_check = ['/runpod-volume', '/', '/tmp']
    for path in paths_to_check:
        if os.path.exists(path):
            usage = get_disk_usage(path)
            if usage:
                print(f"  {path}: {usage['free_gb']}GB free / {usage['total_gb']}GB total ({usage['used_percent']:.1f}% used)")

# Set HuggingFace cache directory to RunPod network volume
os.environ['HF_HOME'] = '/runpod-volume/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume/hf_cache/transformers'
os.environ['HF_HUB_CACHE'] = '/runpod-volume/hf_cache/hub'

# Additional environment variables to ensure everything saves to runpod-volume
os.environ['TORCH_HOME'] = '/runpod-volume/torch_cache'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/runpod-volume/torch_cache'
os.environ['XDG_CACHE_HOME'] = '/runpod-volume/cache'
os.environ['TMPDIR'] = '/runpod-volume/tmp'

# Force HuggingFace config and token storage to runpod-volume
os.environ['HF_TOKEN_PATH'] = '/runpod-volume/hf_cache/token'
os.environ['HOME'] = '/runpod-volume'  # This forces ~/.huggingface to be in runpod-volume

# Create all necessary directories
os.makedirs('/runpod-volume/torch_cache', exist_ok=True)
os.makedirs('/runpod-volume/cache', exist_ok=True)
os.makedirs('/runpod-volume/tmp', exist_ok=True)
os.makedirs('/runpod-volume/.huggingface', exist_ok=True)

MODEL_DIR = "/runpod-volume/models"
MODEL_INDEX = os.path.join(MODEL_DIR, "model_index.json")

# Check initial disk space
check_available_space()

if os.path.exists(MODEL_INDEX):
    print(f"Model already exists at {MODEL_DIR}, skipping download.")
else:
    print("Model not found, downloading...")
    
    # Clear cache to free up space
    clear_cache_directories()
    
    # Check space after cleanup
    print("\n📊 Disk space after cleanup:")
    check_available_space()
    
    # Verify we have enough space (SD3.5 Large is ~12GB)
    runpod_usage = get_disk_usage('/runpod-volume')
    if runpod_usage and runpod_usage['free_gb'] < 15:
        print(f"⚠️  Warning: Only {runpod_usage['free_gb']}GB free space available.")
        print("   SD3.5 Large requires ~12-15GB. Download may fail.")
    
    # Create cache directories on RunPod network volume
    os.makedirs('/runpod-volume/hf_cache', exist_ok=True)
    os.makedirs('/runpod-volume/hf_cache/transformers', exist_ok=True)
    os.makedirs('/runpod-volume/hf_cache/hub', exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required.\n"
            "Set it in RunPod's Public Environment Variables.\n"
            "Get your token: https://huggingface.co/settings/tokens\n"
            "Request access: https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
        )
    
    print("Logging into Hugging Face...")
    login(token=hf_token)
    
    print("Downloading Stable Diffusion 3.5 Large model...")
    print("⏳ This may take 10-20 minutes depending on connection speed...")
    
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.bfloat16
        )
        
        # Check space before saving
        print("\n📊 Disk space before saving model:")
        check_available_space()
        
        print("Saving model to local directory...")
        pipe.save_pretrained(MODEL_DIR)
        print("✅ Model downloaded and saved successfully.")
        
        # Check final space
        print("\n📊 Final disk space:")
        check_available_space()
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("\n📊 Disk space at failure:")
        check_available_space()
        raise