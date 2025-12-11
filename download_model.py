from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2-1.5B-Instruct-GGUF"
snapshot_download(
    repo_id=model_id,
    repo_type="model",
    local_dir="./local_models",
    local_dir_use_symlinks=False,
    allow_patterns=["qwen2-1_5b-instruct-q4_0.gguf"]
)