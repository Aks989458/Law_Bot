from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "/content/hf_cache"

def build_vector_index():
    os.makedirs(CACHE_DIR, exist_ok=True)

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=CACHE_DIR
    )

    with open("data/chunks/law_chunks.json", "r") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks, model
