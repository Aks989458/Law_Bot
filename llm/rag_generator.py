import torch
from llm.model_loader import load_model
from llm.prompt import build_legal_prompt

# Load model ONCE
model, tokenizer = load_model()


def generate_answer(query, top_chunks, max_tokens=300):
    """
    Generate legal answer using RAG + LoRA
    """

    if not top_chunks:
        return "Answer not found in the provided law text."

    prompt = build_legal_prompt(top_chunks, query)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    # Remove prompt echo safely
    answer = decoded.split("### Final Answer:")[-1].strip()
    return answer
