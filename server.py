from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# Load the pre-downloaded SDXL 3.5 model
print("🔧 Loading SDXL 3.5...")
pipe = DiffusionPipeline.from_pretrained(
    "/app/models/sdxl",
    torch_dtype=torch.float16
).to("cuda")
print("✅ SDXL 3.5 Loaded")

@app.route("/run", methods=["POST"])
def run():
    try:
        # Extract prompt
        input_data = request.get_json(force=True)
        prompt = input_data["input"]["prompt"]

        # Generate image
        image = pipe(prompt).images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return base64-encoded image
        return jsonify({"output": img_str})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)