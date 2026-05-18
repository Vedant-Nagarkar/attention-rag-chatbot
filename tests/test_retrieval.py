import os
import pytest
from unittest.mock import MagicMock, patch

from src.retrieval import (
    bm25_search,
    semantic_search,
    reciprocal_rank_fusion,
    rerank_chunks,
    run_retrieval_pipeline,
    get_groq_llm,
)
from src.indexing import run_indexing_pipeline, load_chunks_from_disk
from config import CONFIDENCE_THRESHOLD


# ── Smoke Tests ──────────────────────────────────────────────
class TestSmoke:
    def test_retrieval_module_imports(self):
        """Smoke: retrieval module imports without error."""
        from src import retrieval
        assert retrieval is not None

    def test_get_groq_llm_initializes(self):
        """Smoke: Groq LLM initializes with API key from .env."""
        llm = get_groq_llm()
        assert llm is not None


# ── Unit Tests ───────────────────────────────────────────────
class TestUnit:
    @pytest.fixture
    def sample_chunks(self):
        return [
            {"chunk_id": 0, "text": "attention mechanism transformer model encoder", "char_count": 45},
            {"chunk_id": 1, "text": "cooking recipes pasta carbonara italian food", "char_count": 44},
            {"chunk_id": 2, "text": "multi head attention parallel layers heads", "char_count": 42},
            {"chunk_id": 3, "text": "neural network deep learning backpropagation", "char_count": 44},
            {"chunk_id": 4, "text": "softmax function probability distribution", "char_count": 41},
        ]

    @pytest.fixture
    def sample_bm25(self, sample_chunks):
        from src.indexing import build_bm25_index
        return build_bm25_index(sample_chunks)

    def test_bm25_search_returns_list(self, sample_bm25, sample_chunks):
        """Unit: bm25_search returns a list."""
        results = bm25_search(sample_bm25, sample_chunks, "attention transformer")
        assert isinstance(results, list)

    def test_bm25_search_result_keys(self, sample_bm25, sample_chunks):
        """Unit: bm25_search results have required keys."""
        results = bm25_search(sample_bm25, sample_chunks, "attention transformer")
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "score" in r
            assert "source" in r

    def test_bm25_search_source_label(self, sample_bm25, sample_chunks):
        """Unit: bm25_search results are labeled with source=bm25."""
        results = bm25_search(sample_bm25, sample_chunks, "attention transformer")
        for r in results:
            assert r["source"] == "bm25"

    def test_rrf_fusion_merges_results(self):
        """Unit: RRF fusion combines two result lists correctly."""
        bm25_results = [
            {"chunk_id": 0, "text": "chunk 0", "score": 1.0, "source": "bm25"},
            {"chunk_id": 1, "text": "chunk 1", "score": 0.8, "source": "bm25"},
        ]
        semantic_results = [
            {"chunk_id": 1, "text": "chunk 1", "score": 0.9, "source": "semantic"},
            {"chunk_id": 2, "text": "chunk 2", "score": 0.7, "source": "semantic"},
        ]
        merged = reciprocal_rank_fusion(bm25_results, semantic_results)
        assert isinstance(merged, list)
        assert len(merged) == 3  # 3 unique chunks

    def test_rrf_fusion_boosts_overlap(self):
        """Unit: chunk appearing in both lists gets higher RRF score."""
        bm25_results = [
            {"chunk_id": 0, "text": "chunk 0", "score": 1.0, "source": "bm25"},
            {"chunk_id": 1, "text": "chunk 1", "score": 0.8, "source": "bm25"},
        ]
        semantic_results = [
            {"chunk_id": 0, "text": "chunk 0", "score": 0.9, "source": "semantic"},
            {"chunk_id": 2, "text": "chunk 2", "score": 0.7, "source": "semantic"},
        ]
        merged = reciprocal_rank_fusion(bm25_results, semantic_results)
        # chunk_id 0 appears in both — should be ranked first
        assert merged[0]["chunk_id"] == 0

    def test_rrf_fusion_source_label(self):
        """Unit: RRF results are labeled with source=rrf."""
        bm25_results = [
            {"chunk_id": 0, "text": "chunk 0", "score": 1.0, "source": "bm25"},
        ]
        semantic_results = [
            {"chunk_id": 1, "text": "chunk 1", "score": 0.9, "source": "semantic"},
        ]
        merged = reciprocal_rank_fusion(bm25_results, semantic_results)
        for r in merged:
            assert r["source"] == "rrf"

    def test_rerank_chunks_returns_top_n(self):
        """Unit: rerank_chunks returns exactly top_n results."""
        chunks = [
            {"chunk_id": i, "text": f"attention transformer chunk {i} " * 5, "score": 0.5, "source": "rrf"}
            for i in range(5)
        ]
        reranked = rerank_chunks("what is attention", chunks, top_n=2)
        assert len(reranked) == 2

    def test_rerank_chunks_have_score(self):
        """Unit: reranked chunks have rerank_score key."""
        chunks = [
            {"chunk_id": i, "text": f"attention transformer chunk {i} " * 5, "score": 0.5, "source": "rrf"}
            for i in range(3)
        ]
        reranked = rerank_chunks("what is attention", chunks)
        for r in reranked:
            assert "rerank_score" in r


# ── Integration Tests ────────────────────────────────────────
class TestIntegration:
    @pytest.fixture(scope="class")
    def pipeline_components(self):
        """Loads real indexes once for all integration tests."""
        chunks = load_chunks_from_disk()
        bm25, collection = run_indexing_pipeline()
        llm = get_groq_llm()
        return chunks, bm25, collection, llm

    def test_full_retrieval_pipeline_returns_tuple(self, pipeline_components):
        """Integration: pipeline returns (list, bool) tuple."""
        chunks, bm25, collection, llm = pipeline_components
        result, confident = run_retrieval_pipeline(
            "What is multi-head attention?",
            bm25, chunks, collection, llm
        )
        assert isinstance(result, list)
        assert isinstance(confident, bool)

    def test_retrieval_returns_top_3_chunks(self, pipeline_components):
        """Integration: pipeline returns exactly 3 reranked chunks."""
        chunks, bm25, collection, llm = pipeline_components
        result, _ = run_retrieval_pipeline(
            "How does scaled dot-product attention work?",
            bm25, chunks, collection, llm
        )
        assert len(result) == 3

    def test_retrieval_confident_on_relevant_query(self, pipeline_components):
        """Integration: confident=True for clear in-scope query."""
        chunks, bm25, collection, llm = pipeline_components
        _, confident = run_retrieval_pipeline(
            "What is the Transformer architecture?",
            bm25, chunks, collection, llm
        )
        assert confident is True

    def test_retrieval_fallback_on_irrelevant_query(self, pipeline_components):
        """Integration: confident=False for out-of-scope query."""
        chunks, bm25, collection, llm = pipeline_components
        _, confident = run_retrieval_pipeline(
            "What is the recipe for pasta carbonara?",
            bm25, chunks, collection, llm
        )
        assert confident is False