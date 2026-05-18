---
title: Attention RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---

# 🤖 Attention Is All You Need — RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot built over the seminal Transformer paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).

Built as Week 5 capstone of a structured ML Engineering roadmap.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/Ved2001/attention-rag-chatbot)

---

## What It Does

Ask any question about the Transformer architecture and get a grounded, cited answer retrieved directly from the paper.

**Examples:**
- "What is multi-head attention?"
- "How does scaled dot-product attention work?"
- "Why does the Transformer use positional encoding?"
- "What BLEU score did the Transformer achieve on WMT 2014?"

---

## Pipeline Architecture

```
PDF → Ingest → BM25 Index + ChromaDB Index
                      ↓
              Hybrid Search (BM25 + Semantic)
                      ↓
              RRF Fusion (k=60)
                      ↓
         Cross-Encoder Reranking (top-3)
                      ↓
         HyDE (for queries ≥ 4 words)
                      ↓
         Confidence Threshold Check
                      ↓
         Groq LLM Generation (LCEL)
                      ↓
              Cited Answer + Sources
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | `llama-3.3-70b-versatile` via Groq |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | ChromaDB (persistent) |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Orchestration | LangChain LCEL |
| Evaluation | RAGAS 0.4.3 |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

---

## RAGAS Evaluation Results

Evaluated on a 10-question golden test set built from the paper.

| Metric | Score |
|---|---|
| Faithfulness | 0.86 |
| Answer Relevancy | ~0.78 |
| Context Precision | 0.87 |
| Context Recall | 0.70 |

---

## Project Structure

```
attention-rag-chatbot/
├── src/
│   ├── logger.py        # centralized logging
│   ├── ingest.py        # PDF loading, cleaning, chunking
│   ├── indexing.py      # BM25 + ChromaDB index builder
│   ├── retrieval.py     # Hybrid search, RRF, reranking, HyDE
│   ├── generation.py    # LangChain LCEL chain, fallback, logging
│   └── evaluate.py      # RAGAS evaluation pipeline
├── tests/               # 63 tests — smoke, unit, integration
├── data/
│   ├── raw/             # source PDF
│   └── processed/       # chunked text (JSON)
├── models/              # BM25 index + ChromaDB store
├── logs/                # app.log + query_log.json
├── app.py               # Gradio UI
├── run_eval.py          # RAGAS evaluation runner
├── config.py            # all constants and paths
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Vedant-Nagarkar/attention-rag-chatbot.git
cd attention-rag-chatbot
```

**2. Create conda environment**
```bash
conda create -n attention-rag python=3.11 -y
conda activate attention-rag
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**5. Download the PDF**

Download [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) and save as: data/raw/attention_is_all_you_need.pdf

**6. Run ingestion and indexing**
```bash
python -c "from src.ingest import run_ingest_pipeline; run_ingest_pipeline()"
python -c "from src.indexing import run_indexing_pipeline; run_indexing_pipeline()"
```

---

## How To Run

**Start the chatbot locally:**
```bash
python app.py
```
Open `http://localhost:7860`

**Run RAGAS evaluation:**
```bash
python run_eval.py
```

---

## How To Test

```bash
pytest tests/ -v
```

63 tests across 5 modules — smoke, unit, and integration levels.

---

## Key Design Decisions

- **BM25 always uses original query** — HyDE paragraph hurts keyword matching
- **HyDE skipped for < 4 word queries** — fixes single-word query failure mode
- **Confidence threshold = 0.3** — prevents hallucination on out-of-scope queries
- **RAGAS judge uses 8b model** — saves daily token quota on free Groq tier
- **Logging failure never re-raises** — chatbot never crashes due to observability layer

---

## Author

Vedant Nagarkar — [GitHub](https://github.com/Vedant-Nagarkar) | [HuggingFace](https://huggingface.co/Ved2001)