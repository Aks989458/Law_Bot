from sentence_transformers import CrossEncoder

# Light + accurate model (perfect for law RAG)
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, chunks, top_k=3):
    if not chunks:
        return []

    pairs = [[query, c["text"]] for c in chunks]
    scores = rerank_model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [c for c, s in ranked[:top_k]]
