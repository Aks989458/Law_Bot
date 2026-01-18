import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH = "/content/law-hybrid-rag/data/chunks/law_chunks.json"
LORA_CONFIG_PATH = "/content/law-hybrid-rag/llm/lora_config.json"
OUTPUT_DIR = "/content/law-hybrid-rag/llm/lora_output"

# --------------------------------------------------
# LOAD LORA CONFIG
# --------------------------------------------------
with open(LORA_CONFIG_PATH) as f:
    lora_cfg = json.load(f)

# --------------------------------------------------
# LOAD DATASET (LAW CHUNKS)
# --------------------------------------------------
dataset = load_dataset("json", data_files=DATA_PATH)

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------
# FORMAT DATA (INSTRUCTION STYLE)
# --------------------------------------------------
def format_example(example):
    prompt = f"""
### Instruction:
Explain the following Indian legal provision clearly and accurately.

### Context:
{example['text']}

### Answer:
"""

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# --------------------------------------------------
# QLoRA (4-BIT) CONFIG
# --------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# --------------------------------------------------
# LOAD BASE MODEL (4-BIT)
# --------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# MEMORY SAVERS
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# --------------------------------------------------
# APPLY LORA
# --------------------------------------------------
valid_keys = LoraConfig.__init__.__code__.co_varnames
filtered_cfg = {k: v for k, v in lora_cfg.items() if k in valid_keys}

lora_config = LoraConfig(**filtered_cfg)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# --------------------------------------------------
# TRAINING ARGUMENTS (SAFE FOR COLAB)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # 🔥 IMPORTANT
    gradient_accumulation_steps=8,     # effective batch = 8
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none"                   # disable wandb
)

# --------------------------------------------------
# TRAINER
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# SAVE LORA ADAPTER
# --------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ LoRA training completed successfully!")
print(f"📁 Adapter saved to: {OUTPUT_DIR}")
