# src/retrieval.py
import os
from dotenv import load_dotenv

import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    BM25_TOP_K,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CONFIDENCE_THRESHOLD,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    HYDE_MIN_QUERY_WORDS,
    RERANK_TOP_N,
    RRF_K,
    SEMANTIC_TOP_K,
    TEMPERATURE,
)
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Cross-encoder model for reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_groq_llm() -> ChatGroq:
    """
    Initializes and returns the Groq LLM client.
    Reads GROQ_API_KEY from .env file.

    Returns:
        ChatGroq instance
    """
    logger.debug("get_groq_llm called")

    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.critical("GROQ_API_KEY not found in .env file")
            raise ValueError("GROQ_API_KEY missing from .env")

        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            api_key=api_key
        )
        logger.info(f"Groq LLM initialized — model: {GROQ_MODEL}")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}", exc_info=True)
        raise


def bm25_search(
    bm25: BM25Okapi,
    chunks: list[dict],
    query: str,
    top_k: int = BM25_TOP_K
) -> list[dict]:
    """
    Runs BM25 keyword search on the chunk corpus.

    Args:
        bm25: BM25Okapi index
        chunks: original list of chunk dicts (for text lookup)
        query: user query string
        top_k: number of results to return

    Returns:
        list of dicts with keys: chunk_id, text, score, source
    """
    logger.debug(f"bm25_search called — query='{query}', top_k={top_k}")

    try:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = [
            {
                "chunk_id": chunks[i]["chunk_id"],
                "text": chunks[i]["text"],
                "score": float(scores[i]),
                "source": "bm25"
            }
            for i in top_indices
            if scores[i] > 0  # ignore zero-score results
        ]

        logger.debug(f"BM25 returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"BM25 search failed: {e}", exc_info=True)
        raise


def semantic_search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = SEMANTIC_TOP_K
) -> list[dict]:
    """
    Runs semantic similarity search on ChromaDB collection.

    Args:
        collection: ChromaDB collection object
        query: user query string
        top_k: number of results to return

    Returns:
        list of dicts with keys: chunk_id, text, score, source
    """
    logger.debug(f"semantic_search called — query='{query}', top_k={top_k}")

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        formatted = [
            {
                "chunk_id": int(results["ids"][0][i]),
                "text": results["documents"][0][i],
                "score": float(1 - results["distances"][0][i]),  # cosine: 1-distance = similarity
                "source": "semantic"
            }
            for i in range(len(results["ids"][0]))
        ]

        logger.debug(f"Semantic search returned {len(formatted)} results")
        return formatted

    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        raise


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    semantic_results: list[dict],
    k: int = RRF_K
) -> list[dict]:
    """
    Merges BM25 and semantic results using Reciprocal Rank Fusion.
    RRF score = 1/(k + rank) summed across both result lists.
    Higher RRF score = better combined ranking.

    Args:
        bm25_results: ranked list from BM25
        semantic_results: ranked list from semantic search
        k: RRF constant (default 60, standard value)

    Returns:
        merged list sorted by RRF score descending
    """
    logger.debug("reciprocal_rank_fusion called")

    try:
        rrf_scores = {}
        chunk_texts = {}

        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_texts[cid] = result["text"]

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_texts[cid] = result["text"]

        # Sort by RRF score descending
        merged = [
            {
                "chunk_id": cid,
                "text": chunk_texts[cid],
                "score": score,
                "source": "rrf"
            }
            for cid, score in sorted(
                rrf_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        logger.debug(f"RRF fusion produced {len(merged)} unique chunks")
        return merged

    except Exception as e:
        logger.error(f"RRF fusion failed: {e}", exc_info=True)
        raise


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = RERANK_TOP_N
) -> list[dict]:
    """
    Reranks chunks using a Cross-Encoder model.
    Cross-encoder scores query+chunk pairs jointly — more accurate
    than bi-encoder similarity but slower (only run on top RRF results).

    Args:
        query: user query string
        chunks: merged chunks from RRF
        top_n: number of top chunks to keep after reranking

    Returns:
        top_n chunks sorted by cross-encoder score descending
    """
    logger.debug(f"rerank_chunks called — {len(chunks)} chunks, top_n={top_n}")

    try:
        cross_encoder = CrossEncoder(RERANKER_MODEL)

        # Create (query, chunk_text) pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = cross_encoder.predict(pairs)

        # Attach scores and sort
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

        logger.info(
            f"Reranking complete — top score: {reranked[0]['rerank_score']:.4f}, "
            f"kept {len(reranked)} of {len(chunks)} chunks"
        )
        return reranked

    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        raise


def generate_hypothetical_document(query: str, llm: ChatGroq) -> str:
    """
    HyDE: generates a hypothetical answer to the query using the LLM.
    This hypothetical answer is then used as the retrieval query
    instead of the original short query — improves semantic matching.

    Only used when query has >= HYDE_MIN_QUERY_WORDS words.

    Args:
        query: original user query
        llm: ChatGroq instance

    Returns:
        str: hypothetical document text
    """
    logger.debug(f"generate_hypothetical_document called — query='{query}'")

    try:
        hyde_prompt = PromptTemplate.from_template(
            "Write a short technical passage (3-4 sentences) that would "
            "directly answer this question about the Transformer architecture paper:\n\n"
            "Question: {query}\n\n"
            "Passage:"
        )

        chain = hyde_prompt | llm | StrOutputParser()
        hypothetical_doc = chain.invoke({"query": query})

        logger.info(f"HyDE generated hypothetical document ({len(hypothetical_doc)} chars)")
        logger.debug(f"HyDE document: {hypothetical_doc[:100]}...")
        return hypothetical_doc

    except Exception as e:
        logger.error(f"HyDE generation failed: {e}", exc_info=True)
        raise


def run_retrieval_pipeline(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    collection: chromadb.Collection,
    llm: ChatGroq
) -> tuple[list[dict], bool]:
    """
    Master retrieval function — runs the full retrieval pipeline:
    1. HyDE (if query is long enough)
    2. BM25 search
    3. Semantic search
    4. RRF fusion
    5. Cross-encoder reranking
    6. Confidence threshold check

    Args:
        query: user query string
        bm25: BM25Okapi index
        chunks: original chunk list
        collection: ChromaDB collection
        llm: ChatGroq instance

    Returns:
        tuple of (top reranked chunks, is_confident bool)
    """
    logger.info(f"=== Retrieval Pipeline Started — query='{query}' ===")

    # Step 1 — HyDE (skip for short queries)
    retrieval_query = query
    word_count = len(query.split())

    if word_count >= HYDE_MIN_QUERY_WORDS:
        logger.info("Query long enough — applying HyDE")
        retrieval_query = generate_hypothetical_document(query, llm)
    else:
        logger.warning(
            f"Query too short for HyDE ({word_count} words < {HYDE_MIN_QUERY_WORDS}) — using original query"
        )

    # Step 2 — BM25 search (always uses original query)
    bm25_results   = bm25_search(bm25, chunks, query)

    # Step 3 — Semantic search (uses HyDE query if applicable)
    semantic_results = semantic_search(collection, retrieval_query)

    # Step 4 — RRF fusion
    merged = reciprocal_rank_fusion(bm25_results, semantic_results)

    # Step 5 — Rerank
    reranked = rerank_chunks(query, merged)

    # Step 6 — Confidence check
    top_score = reranked[0]["rerank_score"] if reranked else 0
    is_confident = top_score >= CONFIDENCE_THRESHOLD

    if not is_confident:
        logger.warning(
            f"Low confidence — top rerank score {top_score:.4f} < threshold {CONFIDENCE_THRESHOLD}"
        )
    else:
        logger.info(f"Confident retrieval — top score: {top_score:.4f}")

    logger.info(f"=== Retrieval Pipeline Complete — {len(reranked)} chunks returned ===")
    return reranked, is_confident