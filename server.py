from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)

# Load model from pre-downloaded path
model_path = "/models/sdxl"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    image = pipe(prompt).images[0]
    image.save("output.png")
    return jsonify({"message": "Image generated and saved to output.png"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)