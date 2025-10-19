from huggingface_hub import hf_hub_download
import os

# 🔑 Set your Hugging Face token here (get it from https://huggingface.co/settings/tokens)
HF_TOKEN = "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"

# Repository and filename
repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
filename = "mistral-7b-instruct-v0.2.Q4_0.gguf"

# Where to store the model locally
out_dir = os.path.join(os.getcwd(), "models")
os.makedirs(out_dir, exist_ok=True)

print("⏳ Downloading model (this may take several minutes)...")
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="model",
    token=HF_TOKEN,
    cache_dir=out_dir
)

print("✅ Download complete! Model saved to:", path)
