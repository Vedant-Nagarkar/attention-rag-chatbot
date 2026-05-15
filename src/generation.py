import json
import os
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LOGS_DIR
from src.logger import get_logger
from src.retrieval import get_groq_llm

logger = get_logger(__name__)

# Fallback message when confidence is too low
FALLBACK_MESSAGE = (
    "I don't have enough information in the paper to answer this question confidently. "
    "Please ask something related to the Transformer architecture, attention mechanisms, "
    "or the contents of 'Attention Is All You Need'."
)

# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are an expert assistant on the Transformer architecture paper 
'Attention Is All You Need' by Vaswani et al. (2017).

Answer the question using ONLY the context provided below.
Be precise, technical, and cite which part of the context supports your answer.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a single context string for the prompt.
    Each chunk is numbered and separated clearly.

    Args:
        chunks: reranked list of chunk dicts

    Returns:
        str: formatted context string
    """
    logger.debug(f"format_context called — {len(chunks)} chunks")

    try:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Chunk {i+1} | ID: {chunk['chunk_id']} | "
                f"Score: {chunk['rerank_score']:.4f}]\n{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Context formatted — {len(context)} total characters")
        return context

    except Exception as e:
        logger.error(f"Context formatting failed: {e}", exc_info=True)
        raise


def generate_answer(
    query: str,
    chunks: list[dict],
    is_confident: bool
) -> dict:
    """
    Generates a final answer from retrieved chunks using LangChain LCEL.
    Returns fallback message if confidence threshold was not met.

    Args:
        query: original user query
        chunks: reranked chunks from retrieval pipeline
        is_confident: confidence flag from retrieval pipeline

    Returns:
        dict with keys: answer, sources, is_fallback, latency_ms
    """
    logger.debug(f"generate_answer called — query='{query}', confident={is_confident}")

    start_time = datetime.now()

    try:
        # Return fallback immediately if not confident
        if not is_confident:
            logger.warning(f"Fallback triggered for query: '{query}'")
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "is_fallback": True,
                "latency_ms": 0
            }

        # Format context from chunks
        context = format_context(chunks)

        # Build LCEL chain
        llm = get_groq_llm()
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        # Run chain
        answer = chain.invoke({
            "context": context,
            "question": query
        })

        # Calculate latency
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Extract source chunk IDs for citation
        sources = [
            {
                "chunk_id": c["chunk_id"],
                "rerank_score": round(c["rerank_score"], 4),
                "text_preview": c["text"][:150] + "..."
            }
            for c in chunks
        ]

        logger.info(
            f"Answer generated — latency: {latency_ms}ms, "
            f"answer length: {len(answer)} chars"
        )

        return {
            "answer": answer,
            "sources": sources,
            "is_fallback": False,
            "latency_ms": latency_ms
        }

    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        raise


def log_query(
    query: str,
    result: dict,
    chunks: list[dict]
) -> None:
    """
    Logs the full query-retrieval-generation cycle to a JSON log file.
    This is your observability layer — every query is recorded.

    Log format per entry:
    - timestamp
    - query
    - retrieved chunk IDs and scores
    - answer (or fallback)
    - latency
    - is_fallback flag

    Args:
        query: original user query
        result: output dict from generate_answer()
        chunks: retrieved chunks
    """
    logger.debug("log_query called")

    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "is_fallback": result["is_fallback"],
            "latency_ms": result["latency_ms"],
            "answer_length": len(result["answer"]),
            "retrieved_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "rerank_score": round(c.get("rerank_score", 0), 4),
                    "source": c.get("source", "unknown")
                }
                for c in chunks
            ],
            "answer_preview": result["answer"][:200]
        }

        # Append to query_log.json
        log_path = os.path.join(LOGS_DIR, "query_log.json")
        os.makedirs(LOGS_DIR, exist_ok=True)

        # Load existing logs or start fresh
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        logger.debug(f"Query logged to {log_path}")

    except Exception as e:
        # Logging failure should never crash the app
        logger.error(f"Failed to log query: {e}", exc_info=True)


def run_generation_pipeline(
    query: str,
    chunks: list[dict],
    is_confident: bool
) -> dict:
    """
    Master function — generates answer and logs the full cycle.
    This is what app.py calls.

    Args:
        query: user query
        chunks: reranked chunks from retrieval pipeline
        is_confident: confidence flag from retrieval pipeline

    Returns:
        dict with answer, sources, is_fallback, latency_ms
    """
    logger.info(f"=== Generation Pipeline Started — query='{query}' ===")

    result = generate_answer(query, chunks, is_confident)
    log_query(query, result, chunks)

    logger.info(
        f"=== Generation Pipeline Complete — "
        f"fallback={result['is_fallback']}, latency={result['latency_ms']}ms ==="
    )
    return result