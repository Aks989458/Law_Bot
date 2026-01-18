def build_legal_prompt(chunks, query):
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""
You are a legal assistant trained in Indian law.
Answer the question strictly using the context provided.
Do not repeat the prompt or instructions.

### Question:
{query}

### Context:
{context}

### Final Answer:
"""
    return prompt.strip()
