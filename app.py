%%writefile app.py
import streamlit as st

from ingest.pdf_to_text import convert_all
from ingest.text_cleaner import clean_all
from ingest.chunker import chunk_all

from index.vector_index import build_vector_index
from index.bm25_index import build_bm25
from index.hybrid_retriever import hybrid_search
from index.contextual_filter import contextual_filter
from index.reranker import rerank

from llm.rag_generator import generate_answer


st.set_page_config(
    page_title="LawBot",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LawBot")
st.subheader(
    "Indian Law Q&A System (Constitution, IPC, CrPC, Evidence Act)"
)
st.markdown(
    "Ask any legal question related to Indian law. Powered by Hybrid RAG + LoRA."
)
st.divider()


@st.cache_resource
def load_index():
    vector_index, chunks, embed_model = build_vector_index()
    bm25, _ = build_bm25()
    return vector_index, chunks, embed_model, bm25


with st.spinner("📚 Building legal index..."):
    vector_index, chunks, embed_model, bm25 = load_index()


query = st.chat_input("Ask your law question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ LawBot is searching law books..."):
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
                answer = "❌ This question is outside Indian law documents."
            else:
                answer = generate_answer(query, final_chunks)

        st.write(answer)
