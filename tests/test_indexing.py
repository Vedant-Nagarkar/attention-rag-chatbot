import os
import pickle
import pytest

from rank_bm25 import BM25Okapi

from src.indexing import (
    load_chunks_from_disk,
    build_bm25_index,
    save_bm25_index,
    load_bm25_index,
    build_chromadb_index,
    load_chromadb_collection,
    run_indexing_pipeline,
    BM25_INDEX_PATH,
)
from config import CHROMA_DIR, MODELS_DIR


# ── Smoke Tests ──────────────────────────────────────────────
class TestSmoke:
    def test_indexing_module_imports(self):
        """Smoke: indexing module imports without error."""
        from src import indexing
        assert indexing is not None

    def test_bm25_index_file_exists(self):
        """Smoke: BM25 index file exists after pipeline has run."""
        assert os.path.exists(BM25_INDEX_PATH), (
            "BM25 index not found. Run run_indexing_pipeline() first."
        )

    def test_chroma_dir_exists(self):
        """Smoke: ChromaDB directory exists after pipeline has run."""
        assert os.path.exists(CHROMA_DIR), (
            "ChromaDB store not found. Run run_indexing_pipeline() first."
        )


# ── Unit Tests ───────────────────────────────────────────────
class TestUnit:
    def test_build_bm25_index_returns_bm25okapi(self):
        """Unit: build_bm25_index returns BM25Okapi object."""
        sample_chunks = [
            {"chunk_id": 0, "text": "attention is all you need", "char_count": 25},
            {"chunk_id": 1, "text": "transformer architecture encoder decoder", "char_count": 40},
            {"chunk_id": 2, "text": "multi head attention mechanism", "char_count": 30},
        ]
        bm25 = build_bm25_index(sample_chunks)
        assert isinstance(bm25, BM25Okapi)

    def test_build_bm25_index_corpus_size(self):
        """Unit: BM25 index corpus size matches input chunks."""
        sample_chunks = [
            {"chunk_id": i, "text": f"sample text chunk number {i}", "char_count": 25}
            for i in range(10)
        ]
        bm25 = build_bm25_index(sample_chunks)
        assert bm25.corpus_size == 10

    def test_bm25_search_returns_scores(self):
        """Unit: BM25 scores relevant document higher than irrelevant."""
        sample_chunks = [
            {"chunk_id": 0, "text": "attention mechanism transformer model", "char_count": 38},
            {"chunk_id": 1, "text": "cooking recipes pasta carbonara", "char_count": 31},
            {"chunk_id": 2, "text": "neural network deep learning", "char_count": 28},
            {"chunk_id": 3, "text": "encoder decoder architecture", "char_count": 28},
            {"chunk_id": 4, "text": "softmax function probability", "char_count": 28},
        ]
        bm25 = build_bm25_index(sample_chunks)
        scores = bm25.get_scores(["attention", "transformer"])
        assert scores[0] > scores[1]

    def test_save_bm25_index_creates_file(self, tmp_path, monkeypatch):
        """Unit: save_bm25_index creates a pkl file."""
        monkeypatch.setattr("src.indexing.BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
        sample_chunks = [
            {"chunk_id": 0, "text": "attention is all you need", "char_count": 25},
        ]
        bm25 = build_bm25_index(sample_chunks)
        path = save_bm25_index(bm25)
        assert os.path.exists(path)


# ── Integration Tests ────────────────────────────────────────
class TestIntegration:
    def test_load_chunks_from_disk_returns_86(self):
        """Integration: loads exactly 86 chunks from disk."""
        chunks = load_chunks_from_disk()
        assert len(chunks) == 86

    def test_load_bm25_index_is_valid(self):
        """Integration: loaded BM25 index is usable."""
        bm25 = load_bm25_index()
        assert isinstance(bm25, BM25Okapi)
        assert bm25.corpus_size == 86

    def test_load_chromadb_collection_count(self):
        """Integration: ChromaDB collection has 86 chunks."""
        collection = load_chromadb_collection()
        assert collection.count() == 86

    def test_run_indexing_pipeline_returns_tuple(self):
        """Integration: pipeline returns (BM25Okapi, Collection) tuple."""
        bm25, collection = run_indexing_pipeline()
        assert isinstance(bm25, BM25Okapi)
        assert collection.count() == 86

    def test_bm25_retrieval_on_real_query(self):
        """Integration: BM25 returns relevant chunks for attention query."""
        from src.indexing import load_chunks_from_disk
        chunks = load_chunks_from_disk()
        bm25 = load_bm25_index()
        scores = bm25.get_scores("multi-head attention".split())
        top_idx = scores.argmax()
        assert scores[top_idx] > 0
        assert "attention" in chunks[top_idx]["text"].lower()