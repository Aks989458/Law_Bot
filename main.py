from ingest.pdf_to_text import convert_all
from ingest.text_cleaner import clean_all
from ingest.chunker import chunk_all

from index.vector_index import build_vector_index
from index.bm25_index import build_bm25
from index.hybrid_retriever import hybrid_search
from index.contextual_filter import contextual_filter
from index.reranker import rerank

from llm.rag_generator import generate_answer

def run_pipeline():
    """
    Continuous interactive Hybrid RAG chatbot
    """

    print("\n" + "=" * 80)
    print("🧑‍⚖️  INDIAN LAW CHATBOT (RAG + LoRA)")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 80 + "\n")

    # 🔥 Build index ONCE
    vector_index, chunks, embed_model = build_vector_index()
    bm25, _ = build_bm25()

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\n👋 Chat ended.")
            break

        if not query:
            print("🤖 LawBot: Please enter a valid legal question.\n")
            continue

        retrieved_chunks = hybrid_search(
            query=query,
            bm25=bm25,
            vector_index=vector_index,
            model=embed_model,
            chunks=chunks,
            top_k=20
        )

        filtered_chunks = contextual_filter(retrieved_chunks, query)
        final_chunks = rerank(query, filtered_chunks, top_k=3)

        if not final_chunks:
            print("🤖 LawBot: No relevant legal context found.\n")
            continue

        answer = generate_answer(query, final_chunks)

        print("\n🤖 LawBot:")
        print(answer)
        print()

if __name__ == "__main__":
    run_pipeline()
