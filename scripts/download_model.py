from huggingface_hub import hf_hub_download
import os

# 🔑 Set your Hugging Face token here (get it from https://huggingface.co/settings/tokens)
HF_TOKEN = "dfghjk"

# Repository and filename
repo_id = "Choosen model repo id"
filename = "Your guff file name"

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
