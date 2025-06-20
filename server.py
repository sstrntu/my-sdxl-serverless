import runpod
from diffusers import StableDiffusion3Pipeline
import torch
import os
import base64
from io import BytesIO
import subprocess
import sys
import gc

# ✅ Redirect all Hugging Face cache, auth, and token writes to /runpod-volume
os.environ["HF_HOME"] = "/runpod-volume/hf_home"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/runpod-volume/hf_cache"

# Additional environment variables to ensure everything saves to runpod-volume
os.environ['TORCH_HOME'] = '/runpod-volume/torch_cache'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/runpod-volume/torch_cache'

# 🚀 GPU Memory optimization environment variables for 24GB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better memory debugging

# Create additional cache directories
os.makedirs('/runpod-volume/hf_home', exist_ok=True)
os.makedirs('/runpod-volume/hf_cache', exist_ok=True)
os.makedirs('/runpod-volume/torch_cache', exist_ok=True)

# Model configuration
MODEL_PATH = "/runpod-volume/models"
MODEL_INDEX = os.path.join(MODEL_PATH, "model_index.json")

def clear_gpu_memory():
    """Clear GPU memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - allocated
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total"
    return "CUDA not available"

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
    """Load the model with aggressive 24GB optimizations"""
    print(f"🚀 Loading model optimized for 24GB GPU from: {MODEL_PATH}")
    print(f"🔍 Initial: {get_gpu_memory_info()}")
    
    # Ensure we're loading from the correct path
    if not os.path.exists(MODEL_INDEX):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Download may have failed.")
    
    # Clear any existing GPU memory
    clear_gpu_memory()
    print(f"🧹 After cleanup: {get_gpu_memory_info()}")
    
    try:
        print("💾 Loading model with aggressive 24GB optimizations...")
        
        # Load with balanced device mapping (supported strategy)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            cache_dir='/runpod-volume/hf_cache',
            low_cpu_mem_usage=True,
            device_map="balanced",  # Use balanced instead of cpu
            # variant="fp16",    # Removed - not available for this model
        )
        
        print(f"📊 After balanced loading: {get_gpu_memory_info()}")
        
        # Enable all aggressive memory optimizations AFTER loading
        print("🔧 Enabling maximum memory optimizations...")
        
        # Enable CPU offloading (most important for 24GB)
        pipe.enable_model_cpu_offload()
        print("✅ CPU offloading enabled - model components stay on CPU when not in use")
        
        # Enable maximum attention slicing
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing("max")
            print("✅ Maximum attention slicing enabled")
        
        # Enable VAE slicing to reduce memory during decoding
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        
        # Enable sequential CPU offload for even more aggressive memory management
        if hasattr(pipe, 'enable_sequential_cpu_offload'):
            try:
                pipe.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled - maximum memory efficiency")
            except Exception as e:
                print(f"⚠️ Sequential CPU offload not available: {e}")
        
        # Enable xFormers if available for memory efficient attention
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xFormers memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️ xFormers not available (not critical): {e}")
        
        # Try to enable Torch 2.0 optimizations
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("✅ Torch 2.0 compile optimization enabled")
        except Exception as e:
            print(f"⚠️ Torch compile not available: {e}")
        
        print(f"🎯 Final optimized memory: {get_gpu_memory_info()}")
        print("✅ Model loaded with aggressive 24GB optimizations")
        return pipe
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        print(f"🔍 Error memory state: {get_gpu_memory_info()}")
        
        # Fallback: try loading without device_map
        print("💡 Trying fallback loading without device_map...")
        clear_gpu_memory()
        
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                cache_dir='/runpod-volume/hf_cache',
                low_cpu_mem_usage=True,
                # No device_map - let it load normally
            )
            
            print(f"📊 After fallback loading: {get_gpu_memory_info()}")
            
            # Enable aggressive CPU offloading immediately
            pipe.enable_model_cpu_offload()
            print("✅ CPU offloading enabled (fallback)")
            
            # Enable all memory optimizations
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing("max")
                print("✅ Maximum attention slicing enabled")
            
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
                print("✅ VAE slicing enabled")
            
            print(f"🎯 Fallback memory usage: {get_gpu_memory_info()}")
            print("✅ Model loaded with fallback 24GB optimizations")
            return pipe
            
        except Exception as fallback_error:
            print(f"❌ Fallback loading also failed: {fallback_error}")
            raise

# Initialize model at startup
print("🔄 Initializing model for 24GB GPU...")
print(f"📍 Model will be stored at: {MODEL_PATH}")
print(f"📍 Cache directory: {os.environ.get('HF_HOME')}")

if download_model_if_needed():
    pipe = load_model()
else:
    print("❌ Failed to download model. Exiting.")
    sys.exit(1)

def handler(job):
    """RunPod serverless handler function optimized for 24GB"""
    try:
        job_input = job.get("input", {})
        
        # Required parameters
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt")

        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        if not negative_prompt:
            return {"error": "Missing 'negative_prompt' in input"}

        # Optimized parameters for 24GB GPU - reduced defaults for memory efficiency
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        num_inference_steps = job_input.get("num_inference_steps", 20)  # Reduced for 24GB
        guidance_scale = job_input.get("guidance_scale", 4.5)
        seed = job_input.get("seed", None)
        
        # Limit maximum resolution for 24GB
        max_pixels = 1024 * 1024  # 1MP max for safety
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            width = int(width * scale / 64) * 64  # Round to nearest 64
            height = int(height * scale / 64) * 64
            print(f"⚠️ Resolution reduced for 24GB GPU: {width}x{height}")

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        print(f"🎨 Generating image with prompt: {prompt[:50]}...")
        print(f"🔍 Pre-generation: {get_gpu_memory_info()}")

        # Aggressive memory cleanup before generation
        clear_gpu_memory()
        print(f"🧹 After pre-cleanup: {get_gpu_memory_info()}")

        # Generate image with memory-efficient settings
        with torch.inference_mode():  # Disable gradient computation
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                # Memory efficient generation settings
                output_type="pil",
                return_dict=False,
            )
            
            # Handle different return types
            if isinstance(result, list):
                image = result[0]  # Take first image from list
            else:
                image = result

        # Immediate cleanup after generation
        clear_gpu_memory()
        print(f"🔍 Post-generation: {get_gpu_memory_info()}")

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Clean up image object
        del image
        del buffered
        clear_gpu_memory()

        return {
            "image": image_base64,
            "message": "✅ Image generation complete (24GB optimized)",
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        }
    
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA Out of Memory (24GB): {e}")
        print(f"🔍 Memory state: {get_gpu_memory_info()}")
        clear_gpu_memory()
        return {
            "error": f"GPU memory exceeded. Try smaller image size or fewer steps. Current memory: {get_gpu_memory_info()}"
        }
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        print(f"🔍 Error memory state: {get_gpu_memory_info()}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler (24GB optimized)...")
    runpod.serverless.start({"handler": handler})