"""
RAG Retrieval Test Script
Checks ONLY retrieval augmentation (no LLM)
"""

from index.vector_index import build_vector_index
from index.bm25_index import build_bm25
from index.hybrid_retriever import hybrid_search
from index.contextual_filter import contextual_filter
from index.reranker import rerank


def test_retrieval(query: str):
    print("\n" + "=" * 100)
    print(f"🧪 RETRIEVAL TEST QUERY: {query}")
    print("=" * 100 + "\n")

    # -------------------------------
    # Load indexes
    # -------------------------------
    print("📦 Loading vector index...")
    vector_index, chunks, embed_model = build_vector_index()

    print("📦 Loading BM25 index...")
    bm25, _ = build_bm25()

    # -------------------------------
    # Hybrid retrieval
    # -------------------------------
    retrieved_chunks = hybrid_search(
        query=query,
        bm25=bm25,
        vector_index=vector_index,
        model=embed_model,
        chunks=chunks,
        top_k=20
    )

    print(f"\n🔗 Retrieved (BM25 + Vector): {len(retrieved_chunks)} chunks")

    for i, c in enumerate(retrieved_chunks[:5]):
        print("\n" + "-" * 80)
        print(f"[RAW {i+1}] {c['text'][:500]}")

    # -------------------------------
    # Contextual filtering
    # -------------------------------
    filtered_chunks = contextual_filter(
        chunks=retrieved_chunks,
        query=query,
        min_matches=1
    )

    print(f"\n🧪 After contextual filter: {len(filtered_chunks)} chunks")

    for i, c in enumerate(filtered_chunks[:5]):
        print("\n" + "-" * 80)
        print(f"[FILTERED {i+1}] {c['text'][:500]}")

    # -------------------------------
    # Reranking
    # -------------------------------
    final_chunks = rerank(
        query=query,
        chunks=filtered_chunks,
        top_k=3
    )

    print(f"\n🏆 Final reranked chunks sent to LLM: {len(final_chunks)}")

    for i, c in enumerate(final_chunks):
        print("\n" + "=" * 80)
        print(f"[FINAL {i+1}]")
        print(c["text"])

    # -------------------------------
    # PASS / FAIL
    # -------------------------------
    if final_chunks:
        print("\n✅ RETRIEVAL AUGMENTATION WORKING")
    else:
        print("\n❌ RETRIEVAL FAILED (check indexes / filters)")


if __name__ == "__main__":
    test_retrieval("punishment for murder")
