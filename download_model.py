# download_model.py

import os
import shutil
import glob

# 🚨 CRITICAL: Override ALL cache/temp locations BEFORE any imports
import tempfile

# Create the directory first
os.makedirs('/runpod-volume/tmp', exist_ok=True)
os.makedirs('/runpod-volume/cache', exist_ok=True)
os.makedirs('/runpod-volume/torch_cache', exist_ok=True)

# Force ALL temporary operations to use runpod-volume
os.environ['TMPDIR'] = '/runpod-volume/tmp'
os.environ['TEMP'] = '/runpod-volume/tmp'
os.environ['TMP'] = '/runpod-volume/tmp'
tempfile.tempdir = '/runpod-volume/tmp'

# PyTorch specific cache directories
os.environ['TORCH_HOME'] = '/runpod-volume/torch_cache'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/runpod-volume/torch_cache'
os.environ['PYTORCH_JIT_CACHE_DIR'] = '/runpod-volume/torch_cache/jit'

# System cache directories
os.environ['XDG_CACHE_HOME'] = '/runpod-volume/cache'
os.environ['HOME'] = '/runpod-volume'  # This redirects ~/.cache as well

# Create PyTorch JIT cache directory
os.makedirs('/runpod-volume/torch_cache/jit', exist_ok=True)

# 🚨 AGGRESSIVE FIX: Create symlinks for ALL possible temp/cache locations BEFORE imports
import subprocess
try:
    # Create comprehensive symlink redirects
    symlink_redirects = [
        ('/tmp', '/runpod-volume/tmp'),
        ('/var/tmp', '/runpod-volume/tmp'),
        ('/root', '/runpod-volume'),  # This redirects /root/.cache, /root/.torch, etc.
        ('/home/root', '/runpod-volume'),
        ('/.cache', '/runpod-volume/cache'),
        ('/usr/local/lib/python3.10/dist-packages/torch/.cache', '/runpod-volume/torch_cache'),
    ]
    
    for original, target in symlink_redirects:
        try:
            # Skip if original doesn't exist or is already correct
            if os.path.exists(original):
                if os.path.islink(original):
                    current_target = os.readlink(original)
                    if current_target == target:
                        continue  # Already correctly linked
                    os.unlink(original)
                elif os.path.isdir(original):
                    # Move existing contents to target if possible
                    if os.listdir(original):  # Directory not empty
                        subprocess.run(['cp', '-r', f'{original}/*', target], shell=True)
                    subprocess.run(['rm', '-rf', original])
                else:
                    os.remove(original)
            
            # Create parent directory if needed
            parent_dir = os.path.dirname(original)
            if parent_dir and parent_dir != '/' and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Create symlink
            os.symlink(target, original)
            print(f"🔗 Symlinked: {original} -> {target}")
            
        except Exception as e:
            print(f"⚠️ Could not create symlink {original} -> {target}: {e}")
            
except Exception as e:
    print(f"⚠️ Error during symlink creation: {e}")

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

# ✅ Redirect all Hugging Face cache, auth, and token writes to /runpod-volume
os.environ["HF_HOME"] = "/runpod-volume/hf_home"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/runpod-volume/hf_cache"
os.environ["XDG_CONFIG_HOME"] = "/runpod-volume/hf_home"

# Additional environment variables to ensure everything saves to runpod-volume
os.environ['TORCH_HOME'] = '/runpod-volume/torch_cache'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/runpod-volume/torch_cache'
# TMPDIR variables already set at top of file

# Create all necessary directories
os.makedirs('/runpod-volume/hf_home', exist_ok=True)
os.makedirs('/runpod-volume/hf_cache', exist_ok=True)
os.makedirs('/runpod-volume/torch_cache', exist_ok=True)
os.makedirs('/runpod-volume/tmp', exist_ok=True)

# 🚨 AGGRESSIVE FIX: Create symlinks for common cache locations that might be hardcoded
try:
    # Remove and recreate common cache directories as symlinks to runpod-volume
    import subprocess
    
    # List of potential hardcoded cache paths that might be causing issues
    cache_paths_to_redirect = [
        '/root/.cache',
        '/home/.cache', 
        '/.cache',
        '/workspace/.cache'
    ]
    
    for cache_path in cache_paths_to_redirect:
        try:
            # Remove if exists
            if os.path.exists(cache_path):
                if os.path.islink(cache_path):
                    os.unlink(cache_path)
                elif os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                else:
                    os.remove(cache_path)
            
            # Create parent directory if needed
            parent_dir = os.path.dirname(cache_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Create symlink to runpod-volume
            os.symlink('/runpod-volume/hf_cache', cache_path)
            print(f"🔗 Created symlink: {cache_path} -> /runpod-volume/hf_cache")
            
        except Exception as e:
            print(f"⚠️ Could not create symlink for {cache_path}: {e}")
    
except Exception as e:
    print(f"⚠️ Error creating symlinks: {e}")

# Verify tempfile override is working
print(f"🎯 tempfile.tempdir confirmed: {tempfile.tempdir}")
print(f"🎯 TMPDIR environment: {os.environ.get('TMPDIR')}")

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
    os.makedirs('/runpod-volume/hf_cache/models', exist_ok=True)
    os.makedirs('/runpod-volume/hf_cache/transformers', exist_ok=True)
    os.makedirs('/runpod-volume/hf_cache/hub', exist_ok=True)
    
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
            cache_dir='/runpod-volume/hf_cache/models'
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
                
                # Check filesystem stats
                try:
                    import subprocess
                    result = subprocess.run(['df', '-i', '/runpod-volume'], 
                                          capture_output=True, text=True)
                    print("📊 Filesystem inode usage:")
                    print(result.stdout)
                    
                    result2 = subprocess.run(['stat', '-f', '/runpod-volume'], 
                                           capture_output=True, text=True)
                    print("📊 Filesystem stats:")
                    print(result2.stdout)
                except:
                    pass
                raise json_err
            
            # If manual test passes, try the actual save
            pipe.save_pretrained(MODEL_DIR)
            print("✅ Model downloaded and saved successfully.")
            
        except OSError as e:
            print(f"❌ OSError during save: {e}")
            print(f"Error code: {e.errno}")
            print(f"Available space: {usage['free_gb'] if usage else 'unknown'}GB")
            
            # More detailed filesystem analysis
            try:
                import subprocess
                commands = [
                    ('df -h /runpod-volume', 'Disk space'),
                    ('df -i /runpod-volume', 'Inode usage'),
                    ('mount | grep runpod', 'Mount info'),
                    ('ls -la /runpod-volume/', 'Directory contents')
                ]
                
                for cmd, desc in commands:
                    try:
                        result = subprocess.run(cmd.split(), capture_output=True, text=True)
                        print(f"📊 {desc}:")
                        print(result.stdout)
                    except:
                        pass
            except:
                pass
            raise
        
        # Check final space
        print("\n📊 Final disk space:")
        check_available_space()
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("\n📊 Disk space at failure:")
        check_available_space()
        raise