# Stable Diffusion 3.5 Large Serverless

A RunPod serverless endpoint for generating high-quality images using Stable Diffusion 3.5 Large model with GPU memory optimizations.

## Features

- 🚀 **Stable Diffusion 3.5 Large**: Latest and most powerful SD model
- 💾 **GPU Memory Optimized**: CPU offloading and memory-efficient attention
- 🔧 **Automatic Disk Management**: Handles large model downloads with symlink redirects
- ⚡ **Fast Inference**: Optimized parameters for SD 3.5 Large
- 🛡️ **Error Handling**: Robust error handling with fallback strategies
- 📊 **Memory Monitoring**: Real-time GPU memory usage tracking

## Requirements

- RunPod account with GPU instance (24GB+ required, optimized for 24GB)
- Network volume for model storage (15GB+ required)
- CUDA-compatible GPU

## Deployment

### 1. Clone and Push to Repository

```bash
git clone <your-repo>
cd my-sdxl-serverless
git add .
git commit -m "Updated with memory optimizations"
git push origin main
```

### 2. RunPod Setup

1. Create a new serverless endpoint in RunPod
2. Select GPU with 24GB+ VRAM (optimized for 24GB)
3. Set container image to build from your repository
4. Configure network volume (15GB+ storage)
5. Set environment variables if needed

### 3. Model Download

The model will be automatically downloaded on first startup:
- **Model**: stabilityai/stable-diffusion-3.5-large
- **Size**: ~15GB
- **Location**: `/runpod-volume/models`

## API Usage

### Request Format

```json
{
  "input": {
    "prompt": "A beautiful landscape with mountains and lakes",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 4.5,
    "seed": 42
  }
}
```

**Minimal request (only prompt required):**
```json
{
  "input": {
    "prompt": "A beautiful landscape with mountains and lakes"
  }
}
```

### Response Format

```json
{
  "image": "base64_encoded_image_data",
  "message": "✅ Image generation complete",
  "parameters": {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 4.5,
    "seed": 42
  }
}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of desired image |
| `negative_prompt` | string | `"blurry, low quality, distorted, bad anatomy"` | What to avoid in the image |
| `width` | integer | 1024 | Image width (multiples of 64) |
| `height` | integer | 1024 | Image height (multiples of 64) |
| `num_inference_steps` | integer | 20 | Number of denoising steps (reduced for 24GB) |
| `guidance_scale` | float | 4.5 | How closely to follow the prompt |
| `seed` | integer | random | Seed for reproducible results |

## Memory Optimizations

The project includes several memory optimizations for handling the large SD 3.5 model:

- **CPU Offloading**: Keeps model components on CPU when not in use
- **Attention Slicing**: Reduces memory during attention computation
- **VAE Slicing**: Reduces memory during image decoding
- **Memory Monitoring**: Tracks GPU usage throughout the process
- **Fallback Loading**: Alternative loading strategy if OOM occurs

## File Structure

```
├── Dockerfile              # Container configuration
├── server.py               # Main serverless handler
├── download_model.py       # Model download with optimizations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### CUDA Out of Memory
- Optimized for 24GB GPU with aggressive memory management
- Automatic resolution limiting (max 1024x1024 for safety)
- Sequential CPU offloading keeps components on CPU when not in use
- Check logs for memory usage information

### Model Download Issues
- Ensure network volume has 15GB+ free space
- Check HuggingFace token permissions
- Review disk space monitoring in logs

### Slow Generation
- Reduce `num_inference_steps` (minimum 20)
- Use smaller image dimensions
- CPU offloading trades memory for speed

## Environment Variables

The following are automatically set for optimal performance:

```bash
HF_HOME=/runpod-volume/hf_home
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_HOME=/runpod-volume/torch_cache
```

## Performance Tips

1. **First Run**: Initial startup takes 5-10 minutes for model download
2. **Subsequent Runs**: ~30-60 seconds for model loading
3. **Generation Time**: 30-60 seconds per image depending on parameters
4. **Batch Processing**: Single image generation optimized for memory

## License

This project is open source. Please check individual model licenses:
- Stable Diffusion 3.5 Large: [Stability AI License](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review RunPod logs for detailed error information
3. Ensure GPU memory requirements are met
