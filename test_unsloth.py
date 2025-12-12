from unsloth import FastLanguageModel

max_seq_length = 16384
dtype = None

model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = model_name,
    max_seq_length  = max_seq_length,
    dtype           = dtype,
    load_in_4bit    = True,
)

FastLanguageModel.for_inference(model)

prompt = "Hello, could you briefly explain photosynthesis?"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))