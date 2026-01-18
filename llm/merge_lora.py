import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "llm/lora_output"
OUT_PATH = "llm/merged_model"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=None
)

model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
    assign=True
)

model = model.merge_and_unload()
model.save_pretrained(OUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUT_PATH)

print("✅ LoRA merged successfully")
