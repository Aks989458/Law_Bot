# ⚖️ Law Hybrid RAG Chatbot (Indian Law)

A **Hybrid Retrieval-Augmented Generation (RAG) chatbot for Indian Law** built using  
**BM25 + FAISS + Cross-Encoder + LoRA fine-tuned Mistral-7B**.

This system is **Colab-friendly**, accurate, and designed to avoid hallucinations by  
answering strictly from legal texts (IPC, CrPC, Constitution, etc.).

---

## 🚀 Features

- 📄 PDF ingestion (Indian Law Acts)
- 🧹 Cleaning & normalization
- ✂️ Smart legal chunking (section-wise)
- 🔍 Hybrid retrieval (BM25 + Vector Search)
- 🧠 Contextual filtering
- 🎯 Cross-encoder reranking
- 🤖 LoRA fine-tuned Mistral-7B
- ⚡ Runs on Google Colab (low memory)

---

## 🏗️ Architecture Flow

PDFs
↓
Text Extraction
↓
Text Cleaning
↓
Smart Chunking
↓
FAISS Vector Index + BM25 Index
↓
Hybrid Retrieval
↓
Contextual Filter
↓
Reranker
↓
LLM (Mistral-7B + LoRA)
↓
Final Answer


---

## 📁 Project Structure


```bash
law-hybrid-rag/
│
├── data/
│ ├── pdfs/ # Raw PDFs (IPC, CrPC, etc.)
│ ├── raw_text/ # Extracted text
│ ├── cleaned_text/ # Cleaned text
│ └── chunks/ # Chunked JSON
│
├── ingest/
│ ├── pdf_to_text.py
│ ├── text_cleaner.py
│ └── chunker.py
│
├── index/
│ ├── vector_index.py
│ ├── bm25_index.py
│ ├── hybrid_retriever.py
│ ├── contextual_filter.py
│ └── reranker.py
│
├── llm/
│ ├── lora_train.py
│ ├── lora_config.json
│ ├── model_loader.py
│ ├── prompt.py
│ └── rag_generator.py
│
└── main.py

```
---

# ⚙️ Setup Guide (Step-by-Step)

## 1️⃣ Install Requirements (Google Colab)

```bash
pip install -q \
  torch transformers accelerate peft bitsandbytes \
  sentence-transformers faiss-cpu rank-bm25 datasets pymupdf
```

## 2️⃣ Upload Law PDFs


Place all law PDFs inside:

```bash
data/pdfs/
```

Examples:

- ipc.pdf

- crpc.pdf

- constitution.pdf

## 3️⃣ Run Ingestion Pipeline (ONLY ONCE)

```bash 
python ingest/pdf_to_text.py
python ingest/text_cleaner.py
python ingest/chunker.py
```

Output:
data/chunks/law_chunks.json

## 4️⃣ Build Retrieval Indexes

Automatically handled inside main.py:

- FAISS → semantic search

- BM25 → keyword search

## 5️⃣ (Optional) Train QLoRA for Legal Domain

```bash
python llm/lora_train.py
```

Why LoRA?

- Trains only ~1% parameters

- Works on Colab GPU

- Improves legal answer accuracy

Output:
```bash
llm/lora_output/
```


## 6️⃣ Run Full RAG Pipeline

```bash
python main.py
```

Example query:

```bash
query = "punishment for cheating under ipc"
```

---

## 🔍 How Retrieval Works
Hybrid Search

Combines:

- BM25 → exact legal terms

- FAISS → semantic similarity

This ensures no legal section is missed.

- Contextual Filtering

Removes irrelevant chunks using:

- keyword overlap

- legal signals (punishment, fine, imprisonment, offence)

- Reranking

Uses a cross-encoder for precision:

```bash
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 🤖 Answer Generation (RAG + LoRA)

The prompt includes:

the user question

retrieved legal sections

strict instruction to avoid hallucination

The model answers only from provided context.


---


## 🧪 Example Output
Punishment for cheating under IPC (Section 417) includes
imprisonment up to 1 year, or fine, or both...

---

## 🚀 Why This System is Strong

✅ Colab compatible
✅ No hallucinations
✅ Domain-specific LoRA
✅ Hybrid retrieval
✅ Modular design
✅ Easy to extend

---

## 🔮 Future Work

- Streamlit / Web UI

- Multi-act citations

- Judgment ingestion

- Evaluation pipeline

- API deployment

---

## 👨‍💻 Author

Aniket Singh
MCA | AI/ML | RAG Systems | Legal AI
