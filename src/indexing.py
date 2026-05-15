# src/indexing.py
import json
import os
import pickle

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    MODELS_DIR,
    PROCESSED_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Path where BM25 index is saved
BM25_INDEX_PATH = os.path.join(MODELS_DIR, "bm25_index.pkl")


def load_chunks_from_disk() -> list[dict]:
    """
    Loads chunks from data/processed/chunks.json.
    Indexing always reads from saved chunks, never re-ingests.

    Returns:
        list of chunk dicts
    """
    chunks_path = os.path.join(PROCESSED_DIR, "chunks.json")
    logger.debug(f"load_chunks_from_disk called — path: {chunks_path}")

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from disk")
        return chunks

    except FileNotFoundError:
        logger.critical(f"chunks.json not found at {chunks_path}. Run ingest pipeline first.")
        raise

    except Exception as e:
        logger.error(f"Failed to load chunks: {e}", exc_info=True)
        raise


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds a BM25 keyword index from chunks.
    Tokenizes each chunk by whitespace — simple but effective for
    technical vocabulary like 'multi-head attention', 'softmax', etc.

    Args:
        chunks: list of chunk dicts from ingest pipeline

    Returns:
        BM25Okapi index object
    """
    logger.debug(f"build_bm25_index called — {len(chunks)} chunks")

    try:
        # Tokenize: lowercase + split on whitespace
        tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_chunks)
        logger.info(f"BM25 index built — {len(tokenized_chunks)} documents indexed")
        return bm25

    except Exception as e:
        logger.error(f"BM25 index build failed: {e}", exc_info=True)
        raise


def save_bm25_index(bm25: BM25Okapi) -> str:
    """
    Saves BM25 index to models/bm25_index.pkl using pickle.
    BM25 has no native save method — pickle is the standard approach.

    Args:
        bm25: BM25Okapi index object

    Returns:
        str: path where index was saved
    """
    logger.debug(f"save_bm25_index called — saving to {BM25_INDEX_PATH}")

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)

        logger.info(f"BM25 index saved to {BM25_INDEX_PATH}")
        return BM25_INDEX_PATH

    except Exception as e:
        logger.error(f"Failed to save BM25 index: {e}", exc_info=True)
        raise


def load_bm25_index() -> BM25Okapi:
    """
    Loads saved BM25 index from models/bm25_index.pkl.
    Use this on subsequent runs to skip rebuilding.

    Returns:
        BM25Okapi index object
    """
    logger.debug(f"load_bm25_index called — path: {BM25_INDEX_PATH}")

    try:
        if not os.path.exists(BM25_INDEX_PATH):
            logger.warning("BM25 index not found — will rebuild")
            raise FileNotFoundError(f"No BM25 index at {BM25_INDEX_PATH}")

        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)

        logger.info("BM25 index loaded from disk")
        return bm25

    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}", exc_info=True)
        raise


def build_chromadb_index(chunks: list[dict]) -> chromadb.Collection:
    """
    Builds a persistent ChromaDB semantic index from chunks.
    Uses sentence-transformers all-MiniLM-L6-v2 for embeddings.
    Skips rebuilding if collection already exists with same chunk count.

    Args:
        chunks: list of chunk dicts from ingest pipeline

    Returns:
        ChromaDB collection object
    """
    logger.debug(f"build_chromadb_index called — {len(chunks)} chunks")

    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)

        # Persistent client — survives across runs
        client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Embedding function using sentence-transformers
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Get or create collection
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        # Skip if already populated with same number of chunks
        existing_count = collection.count()
        if existing_count == len(chunks):
            logger.info(
                f"ChromaDB collection already has {existing_count} chunks — skipping rebuild"
            )
            return collection

        # Clear and rebuild if count mismatch
        if existing_count > 0:
            logger.warning(
                f"ChromaDB count mismatch ({existing_count} vs {len(chunks)}) — rebuilding"
            )
            client.delete_collection(CHROMA_COLLECTION)
            collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"}
            )

        # Add chunks in batches of 50
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.add(
                ids=[str(c["chunk_id"]) for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"chunk_id": c["chunk_id"], "char_count": c["char_count"]} for c in batch]
            )
            logger.debug(f"Added batch {i // batch_size + 1} — chunks {i} to {i + len(batch)}")

        logger.info(f"ChromaDB index built — {collection.count()} chunks indexed")
        return collection

    except Exception as e:
        logger.error(f"ChromaDB index build failed: {e}", exc_info=True)
        raise


def load_chromadb_collection() -> chromadb.Collection:
    """
    Loads existing ChromaDB collection from models/chroma_store/.
    Use this on subsequent runs — no rebuilding needed.

    Returns:
        ChromaDB collection object
    """
    logger.debug("load_chromadb_collection called")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef
        )

        logger.info(f"ChromaDB collection loaded — {collection.count()} chunks")
        return collection

    except Exception as e:
        logger.error(f"Failed to load ChromaDB collection: {e}", exc_info=True)
        raise


def run_indexing_pipeline() -> tuple[BM25Okapi, chromadb.Collection]:
    """
    Master function — builds both BM25 and ChromaDB indexes.
    Loads chunks from disk, builds both indexes, saves BM25.
    ChromaDB persists automatically via PersistentClient.

    Returns:
        tuple of (BM25Okapi index, ChromaDB collection)
    """
    logger.info("=== Indexing Pipeline Started ===")

    chunks = load_chunks_from_disk()

    # BM25
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    # ChromaDB
    collection = build_chromadb_index(chunks)

    logger.info("=== Indexing Pipeline Complete — BM25 + ChromaDB ready ===")
    return bm25, collection