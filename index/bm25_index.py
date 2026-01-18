import json
from rank_bm25 import BM25Okapi

def build_bm25():
    with open("/content/law-hybrid-rag/data/chunks/law_chunks.json", "r") as f:
        chunks = json.load(f)

    tokenized = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks
