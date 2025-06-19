from flask import Flask, request, jsonify
from diffusers import StableDiffusion3Pipeline
import torch
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load model from local path
MODEL_PATH = "/workspace/models"
print(f"🚀 Loading model from: {MODEL_PATH}")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
).to("cuda")

@app.route("/", methods=["POST"])
def generate():
    try:
        input_data = request.json.get("input", {})
        
        # Required parameters (no defaults)
        prompt = input_data.get("prompt")
        negative_prompt = input_data.get("negative_prompt")

        if not prompt:
            return jsonify({"error": "Missing 'prompt' in input"}), 400
        if not negative_prompt:
            return jsonify({"error": "Missing 'negative_prompt' in input"}), 400

        # Optional parameters with tuned defaults
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        guidance_scale = input_data.get("guidance_scale", 3.5)  # Recommended for SD3.5 Large
        strength = input_data.get("strength", 0.9)
        seed = input_data.get("seed", None)

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

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

        return jsonify({
            "image": image_base64,
            "message": "✅ Image generation complete"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)