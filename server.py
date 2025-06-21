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

from PIL import Image

# 🚀 GPU Memory optimization environment variables for 24GB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better memory debugging


# Model configuration
MODEL_PATH = "/runpod-volume/models"
MODEL_INDEX = os.path.join(MODEL_PATH, "model_index.json")

# Constants
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, bad anatomy"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 20  # Reduced for 24GB
DEFAULT_GUIDANCE_SCALE = 4.5
MAX_PIXELS_24GB = 1024 * 1024  # 1MP max for safety

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
    """Extract PIL Image from SD3 pipeline result"""
    # SD3 with output_type="pil" and return_dict=False returns a tuple
    # where the first element is a list of PIL Images
    if isinstance(result, tuple) and len(result) > 0:
        images = result[0]  # First element contains the images
        if isinstance(images, list) and len(images) > 0:
            return images[0]  # Return first image
    
    # Fallback: if result is directly a list of images
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    
    # Fallback: if result is directly a PIL Image
    if isinstance(result, Image.Image):
        return result
    
    raise ValueError(f"Unexpected result format: {type(result)}")

def create_pipeline(use_device_map=True):
    """Create StableDiffusion3Pipeline with common parameters"""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    
    if use_device_map:
        kwargs["device_map"] = "balanced"
    
    return StableDiffusion3Pipeline.from_pretrained(MODEL_PATH, **kwargs)

def apply_memory_optimizations(pipe):
    """Apply all memory optimizations to the pipeline"""
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
        pipe = create_pipeline(use_device_map=True)
        print(f"📊 After balanced loading: {get_gpu_memory_info()}")
        
        # Enable all aggressive memory optimizations AFTER loading
        print("🔧 Enabling maximum memory optimizations...")
        apply_memory_optimizations(pipe)
        
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
            pipe = create_pipeline(use_device_map=False)
            print(f"📊 After fallback loading: {get_gpu_memory_info()}")
            
            # Apply optimizations (reusing the same function)
            apply_memory_optimizations(pipe)
            
            print(f"🎯 Fallback memory usage: {get_gpu_memory_info()}")
            print("✅ Model loaded with fallback 24GB optimizations")
            return pipe
            
        except Exception as fallback_error:
            print(f"❌ Fallback loading also failed: {fallback_error}")
            raise

# Initialize model at startup
print("🔄 Initializing model...")
print(f"📍 Model will be stored at: {MODEL_PATH}")

# Check if model exists, if not run download script
if not os.path.exists(MODEL_INDEX):
    print("📦 Model not found, running download script...")
    try:
        result = subprocess.run([sys.executable, "/workspace/download_model.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Model download completed successfully")
        if result.stdout:
            print("📋 Download output:", result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"❌ Model download failed: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
else:
    print(f"✅ Model already exists at {MODEL_PATH}")

# Load the model
pipe = load_model()

def log_memory_state(stage):
    """Log memory state with a descriptive stage name"""
    memory_info = get_gpu_memory_info()
    print(f"🔍 {stage}: {memory_info}")
    return memory_info

def cleanup_and_log(stage):
    """Clear GPU memory and log the result"""
    clear_gpu_memory()
    return log_memory_state(f"After {stage}")

def validate_and_adjust_resolution(width, height):
    """Validate and adjust resolution for 24GB GPU limits"""
    if width * height > MAX_PIXELS_24GB:
        scale = (MAX_PIXELS_24GB / (width * height)) ** 0.5
        new_width = int(width * scale / 64) * 64  # Round to nearest 64
        new_height = int(height * scale / 64) * 64
        print(f"⚠️ Resolution reduced for 24GB GPU: {new_width}x{new_height}")
        return new_width, new_height
    return width, height

def parse_job_parameters(job_input):
    """Parse and validate job input parameters"""
    prompt = job_input.get("prompt")
    if not prompt:
        raise ValueError("Missing 'prompt' in input")
    
    # Extract parameters with defaults
    params = {
        "prompt": prompt,
        "negative_prompt": job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "width": job_input.get("width", DEFAULT_WIDTH),
        "height": job_input.get("height", DEFAULT_HEIGHT),
        "num_inference_steps": job_input.get("num_inference_steps", DEFAULT_STEPS),
        "guidance_scale": job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
        "seed": job_input.get("seed", None)
    }
    
    # Validate and adjust resolution
    params["width"], params["height"] = validate_and_adjust_resolution(params["width"], params["height"])
    
    return params

def log_generation_parameters(params):
    """Log the generation parameters"""
    print(f"📝 Parameters:")
    print(f"  - Prompt: {params['prompt'][:50]}{'...' if len(params['prompt']) > 50 else ''}")
    print(f"  - Negative prompt: {params['negative_prompt']}")
    print(f"  - Resolution: {params['width']}x{params['height']}")
    print(f"  - Steps: {params['num_inference_steps']}")
    print(f"  - Guidance: {params['guidance_scale']}")
    print(f"  - Seed: {params['seed']}")

def create_generator(seed):
    """Create a torch generator with optional seed"""
    if seed is not None:
        return torch.Generator("cuda").manual_seed(seed)
    return None

def generate_image(pipe, params):
    """Generate image using the pipeline with given parameters"""
    generator = create_generator(params["seed"])
    
    print(f"🎨 Generating image with prompt: {params['prompt'][:50]}...")
    log_memory_state("Pre-generation")
    
    # Cleanup before generation
    cleanup_and_log("pre-cleanup")
    
    # Generate image with memory-efficient settings
    with torch.inference_mode():  # Disable gradient computation
        result = pipe(
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            width=params["width"],
            height=params["height"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            generator=generator,
            output_type="pil",
            return_dict=False,
        )
        
        image = extract_image_from_result(result)
    
    # Cleanup after generation
    cleanup_and_log("generation")
    return image

def convert_image_to_base64(image):
    """Convert PIL image to base64 string"""
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(f"✅ Image successfully converted to base64 ({len(image_base64)} characters)")
        return image_base64
    except Exception as e:
        print(f"❌ Error converting image to base64: {e}")
        print(f"📋 Full traceback: {traceback.format_exc()}")
        raise
    finally:
        # Clean up
        if 'buffered' in locals():
            del buffered

def create_success_response(image_base64, params):
    """Create successful response dictionary"""
    return {
        "image": image_base64,
        "message": "✅ Image generation complete (24GB optimized)",
        "parameters": params
    }

def handle_cuda_oom_error(e):
    """Handle CUDA Out of Memory errors"""
    print(f"❌ CUDA Out of Memory (24GB): {e}")
    log_memory_state("OOM Error")
    print(f"📋 Full traceback: {traceback.format_exc()}")
    clear_gpu_memory()
    return {
        "error": f"GPU memory exceeded. Try smaller image size or fewer steps. Current memory: {get_gpu_memory_info()}"
    }

def handle_general_error(e):
    """Handle general errors"""
    print(f"❌ Error in handler: {str(e)}")
    log_memory_state("General Error")
    print(f"📋 Full traceback: {traceback.format_exc()}")
    return {"error": str(e)}

def handler(job):
    """RunPod serverless handler function optimized for 24GB"""
    try:
        job_input = job.get("input", {})
        
        # Parse and validate parameters
        params = parse_job_parameters(job_input)
        
        # Log parameters
        log_generation_parameters(params)
        
        # Generate image
        image = generate_image(pipe, params)
        
        # Convert to base64
        image_base64 = convert_image_to_base64(image)
        
        # Clean up image object
        del image
        clear_gpu_memory()
        
        # Return success response
        return create_success_response(image_base64, params)
    
    except torch.cuda.OutOfMemoryError as e:
        return handle_cuda_oom_error(e)
    except Exception as e:
        return handle_general_error(e)

if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler (24GB optimized)...")
    runpod.serverless.start({"handler": handler})