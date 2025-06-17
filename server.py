from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    image = pipe(prompt).images[0]
    image.save("output.png")
    return jsonify({"message": "Image generated and saved."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)