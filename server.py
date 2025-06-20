import runpod
from diffusers import StableDiffusion3Pipeline
import torch
import os
import base64
from io import BytesIO
import subprocess
import sys
import gc
import traceback
import numpy as np
from PIL import Image

# 🚀 GPU Memory optimization environment variables for 24GB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better memory debugging


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

def extract_image_from_result(result):
    """
    Enhanced function to extract PIL Image from various result types
    Handles: tuples, lists, PIL Images, tensors, numpy arrays
    """
    try:
        print(f"🔍 Extracting image from result type: {type(result)}")
        
        # Case 1: Direct PIL Image
        if isinstance(result, Image.Image):
            print("✅ Direct PIL Image detected")
            return validate_image_mode(result)
        
        # Case 2: Tuple (common with return_dict=False)
        elif isinstance(result, tuple):
            print(f"📦 Tuple detected with {len(result)} elements")
            for i, item in enumerate(result):
                print(f"  - Element {i}: {type(item)}")
                if isinstance(item, Image.Image):
                    print(f"✅ Found PIL Image at index {i}")
                    return validate_image_mode(item)
                elif isinstance(item, list) and len(item) > 0:
                    return extract_image_from_result(item[0])
            # If no PIL Image found, try first element
            if len(result) > 0:
                return extract_image_from_result(result[0])
        
        # Case 3: List
        elif isinstance(result, list):
            print(f"📋 List detected with {len(result)} elements")
            if len(result) == 0:
                raise ValueError("Empty list returned from pipeline")
            
            for i, item in enumerate(result):
                print(f"  - Element {i}: {type(item)}")
                if isinstance(item, Image.Image):
                    print(f"✅ Found PIL Image at index {i}")
                    return validate_image_mode(item)
            
            # If no PIL Image found, try converting first element
            return extract_image_from_result(result[0])
        
        # Case 4: PyTorch Tensor
        elif isinstance(result, torch.Tensor):
            print(f"🔥 PyTorch Tensor detected: {result.shape}")
            return tensor_to_pil(result)
        
        # Case 5: NumPy Array
        elif isinstance(result, np.ndarray):
            print(f"📊 NumPy Array detected: {result.shape}")
            return numpy_to_pil(result)
        
        # Case 6: Dictionary (if return_dict=True was used)
        elif isinstance(result, dict):
            print(f"📚 Dictionary detected with keys: {list(result.keys())}")
            # Common keys for diffusion pipelines
            for key in ['images', 'image', 'sample', 'samples']:
                if key in result:
                    print(f"🔑 Found key '{key}', extracting...")
                    return extract_image_from_result(result[key])
            raise ValueError(f"No image data found in dictionary keys: {list(result.keys())}")
        
        # Case 7: Unknown type
        else:
            print(f"❓ Unknown result type: {type(result)}")
            # Try to access .images attribute (common in diffusion pipelines)
            if hasattr(result, 'images'):
                print("🔍 Found .images attribute")
                return extract_image_from_result(result.images)
            elif hasattr(result, 'image'):
                print("🔍 Found .image attribute")
                return extract_image_from_result(result.image)
            else:
                raise TypeError(f"Unsupported result type: {type(result)}")
    
    except Exception as e:
        print(f"❌ Error in extract_image_from_result: {e}")
        print(f"📋 Full traceback: {traceback.format_exc()}")
        raise

def validate_image_mode(image):
    """Ensure image is in RGB or RGBA mode for PNG saving"""
    try:
        print(f"🎨 Validating image mode: {image.mode}, size: {image.size}")
        
        if image.mode in ['RGB', 'RGBA']:
            print("✅ Image mode is valid for PNG")
            return image
        elif image.mode == 'L':  # Grayscale
            print("🔄 Converting grayscale to RGB")
            return image.convert('RGB')
        elif image.mode == 'CMYK':
            print("🔄 Converting CMYK to RGB")
            return image.convert('RGB')
        elif image.mode == 'P':  # Palette
            print("🔄 Converting palette to RGB")
            return image.convert('RGB')
        else:
            print(f"🔄 Converting {image.mode} to RGB")
            return image.convert('RGB')
    
    except Exception as e:
        print(f"❌ Error validating image mode: {e}")
        raise

def tensor_to_pil(tensor):
    """Convert PyTorch tensor to PIL Image"""
    try:
        print(f"🔥 Converting tensor: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Move to CPU if on GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert to numpy
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        array = tensor.numpy()
        
        # Handle different tensor formats
        if len(array.shape) == 4:  # Batch dimension
            print("📦 Removing batch dimension")
            array = array[0]
        
        if len(array.shape) == 3:
            if array.shape[0] in [1, 3, 4]:  # Channels first
                print("🔄 Converting from channels-first to channels-last")
                array = np.transpose(array, (1, 2, 0))
        
        # Normalize to 0-255 if needed
        if array.max() <= 1.0:
            print("📊 Normalizing from [0,1] to [0,255]")
            array = (array * 255).astype(np.uint8)
        
        return numpy_to_pil(array)
    
    except Exception as e:
        print(f"❌ Error converting tensor to PIL: {e}")
        raise

def numpy_to_pil(array):
    """Convert NumPy array to PIL Image"""
    try:
        print(f"📊 Converting numpy array: shape={array.shape}, dtype={array.dtype}")
        
        # Ensure uint8
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        
        # Handle different array shapes
        if len(array.shape) == 2:  # Grayscale
            return Image.fromarray(array, mode='L')
        elif len(array.shape) == 3:
            if array.shape[2] == 1:  # Single channel
                return Image.fromarray(array.squeeze(), mode='L')
            elif array.shape[2] == 3:  # RGB
                return Image.fromarray(array, mode='RGB')
            elif array.shape[2] == 4:  # RGBA
                return Image.fromarray(array, mode='RGBA')
        
        raise ValueError(f"Unsupported array shape: {array.shape}")
    
    except Exception as e:
        print(f"❌ Error converting numpy to PIL: {e}")
        raise

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
            low_cpu_mem_usage=True,
            device_map="balanced",  # Use balanced instead of cpu
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
print("🔄 Initializing model...")
print(f"📍 Model will be stored at: {MODEL_PATH}")

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
        if not prompt:
            return {"error": "Missing 'prompt' in input"}

        # Optional parameters with sensible defaults
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted, bad anatomy")
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        num_inference_steps = job_input.get("num_inference_steps", 20)  # Reduced for 24GB
        guidance_scale = job_input.get("guidance_scale", 4.5)
        seed = job_input.get("seed", None)
        
        # Log the parameters being used
        print(f"📝 Parameters:")
        print(f"  - Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        print(f"  - Negative prompt: {negative_prompt}")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Steps: {num_inference_steps}")
        print(f"  - Guidance: {guidance_scale}")
        print(f"  - Seed: {seed}")
        
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
            
            # Use enhanced image extraction
            image = extract_image_from_result(result)

        # Immediate cleanup after generation
        clear_gpu_memory()
        print(f"🔍 Post-generation: {get_gpu_memory_info()}")

        # Convert image to base64 with robust error handling
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            print(f"✅ Image successfully converted to base64 ({len(image_base64)} characters)")
        except Exception as e:
            print(f"❌ Error converting image to base64: {e}")
            print(f"📋 Full traceback: {traceback.format_exc()}")
            raise

        # Clean up image object
        del image
        del buffered
        clear_gpu_memory()

        return {
            "image": image_base64,
            "message": "✅ Image generation complete (24GB optimized)",
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
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
        print(f"📋 Full traceback: {traceback.format_exc()}")
        clear_gpu_memory()
        return {
            "error": f"GPU memory exceeded. Try smaller image size or fewer steps. Current memory: {get_gpu_memory_info()}"
        }
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        print(f"🔍 Error memory state: {get_gpu_memory_info()}")
        print(f"📋 Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler (24GB optimized)...")
    runpod.serverless.start({"handler": handler})