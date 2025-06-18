import os
import subprocess
from flask import Flask, request, jsonify
from huggingface_hub import snapshot_download

app = Flask(__name__)

# Get Hugging Face token from env variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_DIR = "/models/sdxl"

# Auto-download model if missing
def ensure_model():
    if not os.path.exists(MODEL_DIR):
        print("🔄 Downloading model...")
        snapshot_download(
            repo_id=MODEL_REPO,
            cache_dir=MODEL_DIR,
            token=HF_TOKEN,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ Model downloaded to", MODEL_DIR)
    else:
        print("✅ Model already exists at", MODEL_DIR)

# Initialize once at startup
ensure_model()

@app.route("/", methods=["POST"])
def run():
    data = request.get_json()
    input_data = data.get("input", {})

    # Option 1: Process "prompt" field for inference (replace with real logic)
    if "prompt" in input_data:
        prompt = input_data["prompt"]
        return jsonify({"response": f"You sent: {prompt}"})

    # Option 2: Run shell command via "command" (use with caution)
    if "command" in input_data:
        try:
            result = subprocess.check_output(
                input_data["command"],
                shell=True,
                stderr=subprocess.STDOUT,
                timeout=10
            )
            return jsonify({"output": result.decode()})
        except subprocess.CalledProcessError as e:
            return jsonify({"error": e.output.decode()}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No prompt or command provided"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)