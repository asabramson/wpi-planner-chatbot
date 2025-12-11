from llama_cpp import Llama

llm = Llama(
    model_path="local_models/qwen2-1_5b-instruct-q4_0.gguf",
    n_gpu_layers=-1,
    verbose=True
)

print(llm("Hello, what are the three most common colors on country flags?", max_tokens=256))