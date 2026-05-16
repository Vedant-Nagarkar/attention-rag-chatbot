import os
from dotenv import load_dotenv
import gradio as gr

from src.ingest import load_chunks
from src.indexing import run_indexing_pipeline
from src.retrieval import run_retrieval_pipeline, get_groq_llm
from src.generation import run_generation_pipeline
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Initialize pipeline once at startup ─────────────────────
logger.info("Initializing RAG pipeline...")
chunks = load_chunks()
bm25, collection = run_indexing_pipeline()
llm = get_groq_llm()
logger.info("Pipeline ready")


# ── Core chat function ───────────────────────────────────────
def chat(query: str, history: list) -> tuple:
    if not query.strip():
        return "", history, ""

    try:
        retrieved, confident = run_retrieval_pipeline(
            query, bm25, chunks, collection, llm
        )
        result = run_generation_pipeline(query, retrieved, confident)

        answer      = result["answer"]
        is_fallback = result["is_fallback"]
        latency     = result["latency_ms"]

        if is_fallback:
            sources_md = "⚠️ **Fallback triggered** — query outside paper scope."
        else:
            sources_lines = [
                f"**Chunk {i+1}** (ID: {c['chunk_id']} | Score: {c['rerank_score']:.4f})\n"
                f"> {c['text'][:200]}...\n"
                for i, c in enumerate(retrieved)
            ]
            sources_md = (
                f"**Retrieved Sources** | Latency: {latency}ms\n\n" +
                "\n---\n".join(sources_lines)
            )

        history = history + [{"role": "user", "content": query},
                             {"role": "assistant", "content": answer}]

        logger.info(f"Response sent — latency={latency}ms, fallback={is_fallback}")
        return "", history, sources_md

    except Exception as e:
        logger.error(f"Chat function failed: {e}", exc_info=True)
        history = history + [{"role": "user", "content": query},
                             {"role": "assistant", "content": "Something went wrong. Please try again."}]
        return "", history, ""


# ── Gradio UI ────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Attention Is All You Need — RAG Chatbot") as demo:

        gr.Markdown("""
        # 🤖 Attention Is All You Need — RAG Chatbot
        Ask questions about the Transformer architecture paper by Vaswani et al. (2017).

        **Pipeline:** Hybrid Search (BM25 + ChromaDB) → RRF Fusion → Cross-Encoder Reranking → HyDE → Groq LLM
        """)

        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    value=[],
                    elem_id="chatbot"
                )
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask something about the Transformer paper...",
                        label="",
                        scale=8,
                        container=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "What is multi-head attention?",
                        "How does scaled dot-product attention work?",
                        "Why does the Transformer use positional encoding?",
                        "What BLEU score did the Transformer achieve?",
                        "How does the Transformer achieve parallelization?",
                    ],
                    inputs=query_input,
                    label="Example Questions"
                )

            with gr.Column(scale=4):
                sources_output = gr.Markdown(
                    value="*Retrieved sources will appear here after each query.*"
                )

        gr.Markdown("""
        ---
        **Model:** `llama-3.3-70b-versatile` via Groq |
        **Embeddings:** `all-MiniLM-L6-v2` |
        **Reranker:** `ms-marco-MiniLM-L-6-v2` |
        **RAGAS:** Faithfulness 0.625 | Context Precision 0.917 | Context Recall 1.0
        """)

        submit_btn.click(
            fn=chat,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot, sources_output]
        )
        query_input.submit(
            fn=chat,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot, sources_output]
        )

    return demo


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )