from ingest.pdf_to_text import convert_all
from ingest.text_cleaner import clean_all
from ingest.chunker import chunk_all

from index.vector_index import build_vector_index
from index.bm25_index import build_bm25
from index.hybrid_retriever import hybrid_search
from index.contextual_filter import contextual_filter
from index.reranker import rerank

from llm.rag_generator import generate_answer


# =========================
# Sample Questions
# =========================
SAMPLE_QUESTIONS = {
    "Constitution of India": [
        "What are the fundamental rights guaranteed under Article 21?",
        "What is the procedure to amend the Constitution of India?",
        "Explain the concept of judicial review in India"
    ],
    "Indian Penal Code (IPC)": [
        "What is the punishment for murder under IPC Section 302?",
        "What is the difference between culpable homicide and murder?",
        "What is criminal conspiracy under IPC?"
    ],
    "Criminal Procedure Code (CrPC)": [
        "What is the procedure for filing an FIR under CrPC?",
        "What are the powers of police to arrest without warrant?",
        "What is the difference between summons case and warrant case?"
    ],
    "Indian Evidence Act": [
        "What is the meaning of relevant facts under the Evidence Act?",
        "When is a confession admissible in court?",
        "What is the difference between primary and secondary evidence?"
    ],
    "Out of Scope (Should fail)": [
        "How to cook biryani?",
        "Who won the FIFA World Cup 2018?",
        "What is the capital of Australia?"
    ]
}


def show_sample_questions():
    print("\n📌 SAMPLE QUESTIONS (copy & paste):\n")
    for category, questions in SAMPLE_QUESTIONS.items():
        print(f"--- {category} ---")
        for q in questions:
            print(f"• {q}")
        print()


# =========================
# Main Pipeline
# =========================
def run_pipeline():
    print("\n" + "=" * 80)
    print("⚖️  LAWBOT – INDIAN LAW Q&A SYSTEM (RAG + LoRA)")
    print("Running in Google Colab (Terminal Mode)")
    print("Type 'samples' to see example questions")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 80 + "\n")

    show_sample_questions()

    # Build index once
    print("⏳ Building legal index (first run takes time)...\n")
    vector_index, chunks, embed_model = build_vector_index()
    bm25, _ = build_bm25()
    print("✅ Index ready! Start asking questions.\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            print("\nInput closed.")
            break

        if query.lower() in {"exit", "quit"}:
            print("\n👋 Chat ended.")
            break

        if query.lower() == "samples":
            show_sample_questions()
            continue

        if not query:
            print("🤖 LawBot: Please enter a valid legal question.\n")
            continue

        print("\n⏳ LawBot is thinking...\n")

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
            print("🤖 LawBot: ❌ Question out of scope of Indian law documents.\n")
            continue

        answer = generate_answer(query, final_chunks)

        print("🤖 LawBot:")
        print(answer)
        print()


if __name__ == "__main__":
    run_pipeline()
