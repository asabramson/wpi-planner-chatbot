from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt
import json
import os

MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # unsloth has over 1000 models to choose from on their huggingface page
NEW_MODEL_NAME = "wpi-advisor-70b"
MAX_SEQ_LENGTH = 4096  # adjust per computing hardware, 4096 fits comfortably on an NVIDIA H100
DTYPE = None # auto-detect (float16 or bfloat16)
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# Alpaca format is used to map prompts into a consistent format (no messy JSON)
# This same exact format is used during inference to maximize performance on the fine-tuned data
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # end-of-sequence token is necessary, otherwise model can generate forever

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="json_data/fine_tuning_transformed.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # rank 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none", 
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # change based on dataset size, 60 steps covers the ~200 prompts with batch 8
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

loss_history = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
steps = [x['step'] for x in trainer.state.log_history if 'loss' in x]

plt.figure(figsize=(10, 6))
plt.plot(steps, loss_history, label='Training Loss', color='blue')
plt.title(f'Fine-Tuning Loss Curve ({NEW_MODEL_NAME})')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss_documentation.png')
print("********Loss documentation saved to 'training_loss_documentation.png'")

# NOTE: If this fails, install llama.cpp (NOT llama-cpp-python), compile if needed, and run:
    # './llama-quantize ../Llama-3.3-70B-Instruct.BF16-00001-of-00003.gguf ../wpi-advisor-final.gguf q4_k_m'
# The resulting .gguf file should be ~40 GB (given the pre-trained foundation model was 70B)
model.save_pretrained_gguf(NEW_MODEL_NAME, tokenizer, quantization_method = "q4_k_m") 
print(f"Model saved in GGUF format to {NEW_MODEL_NAME}")