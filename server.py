from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
import torch
import os

app = Flask(__name__)

# Load model from /workspace/models/ (required for RunPod Serverless)
MODEL_PATH = "/workspace/models"

print(f"🚀 Loading model from: {MODEL_PATH}")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

@app.route("/", methods=["POST"])
def generate():
    try:
        input_data = request.json.get("input", {})

        # Prompt is required
        prompt = input_data.get("prompt")
        if not prompt:
            return jsonify({"error": "Missing 'prompt' parameter in input."}), 400

        negative_prompt = input_data.get("negative_prompt", "")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 30)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps
        ).images[0]

        output_path = "/workspace/output.png"
        image.save(output_path)

        return jsonify({"output": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)