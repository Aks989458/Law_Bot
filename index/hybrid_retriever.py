import numpy as np

def hybrid_search(query, bm25, vector_index, model, chunks, top_k=5):
    # BM25
    bm25_scores = bm25.get_scores(query.split())
    bm25_ids = np.argsort(bm25_scores)[-top_k:]

    # Vector
    q_emb = model.encode([query])
    _, vec_ids = vector_index.search(q_emb, top_k)

    # Merge
    ids = set(bm25_ids).union(set(vec_ids[0]))
    return [chunks[i] for i in ids]
