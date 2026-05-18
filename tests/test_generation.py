import pytest
from unittest.mock import MagicMock

from src.generation import (
    format_context,
    generate_answer,
    log_query,
    run_generation_pipeline,
    FALLBACK_MESSAGE,
)


# ── Smoke Tests ──────────────────────────────────────────────
class TestSmoke:
    def test_generation_module_imports(self):
        """Smoke: generation module imports without error."""
        from src import generation
        assert generation is not None

    def test_fallback_message_is_string(self):
        """Smoke: FALLBACK_MESSAGE is a non-empty string."""
        assert isinstance(FALLBACK_MESSAGE, str)
        assert len(FALLBACK_MESSAGE) > 0


# ── Unit Tests ───────────────────────────────────────────────
class TestUnit:
    @pytest.fixture
    def sample_chunks(self):
        return [
            {"chunk_id": 0, "text": "Attention is all you need.", "rerank_score": 5.5, "source": "rrf"},
            {"chunk_id": 1, "text": "Multi-head attention runs in parallel.", "rerank_score": 4.2, "source": "rrf"},
            {"chunk_id": 2, "text": "Positional encoding injects order.", "rerank_score": 3.1, "source": "rrf"},
        ]

    def test_format_context_returns_string(self, sample_chunks):
        """Unit: format_context returns a string."""
        context = format_context(sample_chunks)
        assert isinstance(context, str)

    def test_format_context_contains_all_chunks(self, sample_chunks):
        """Unit: formatted context contains text from all chunks."""
        context = format_context(sample_chunks)
        for chunk in sample_chunks:
            assert chunk["text"] in context

    def test_format_context_contains_chunk_ids(self, sample_chunks):
        """Unit: formatted context labels each chunk."""
        context = format_context(sample_chunks)
        assert "Chunk 1" in context
        assert "Chunk 2" in context
        assert "Chunk 3" in context

    def test_generate_answer_returns_fallback(self, sample_chunks):
        """Unit: generate_answer returns fallback when not confident."""
        result = generate_answer("some query", sample_chunks, is_confident=False)
        assert result["is_fallback"] is True
        assert result["answer"] == FALLBACK_MESSAGE
        assert result["sources"] == []

    def test_generate_answer_fallback_no_latency(self, sample_chunks):
        """Unit: fallback response has zero latency."""
        result = generate_answer("some query", sample_chunks, is_confident=False)
        assert result["latency_ms"] == 0

    def test_log_query_does_not_raise(self, sample_chunks, tmp_path, monkeypatch):
        """Unit: log_query runs without raising even with tmp path."""
        monkeypatch.setattr("src.generation.LOGS_DIR", str(tmp_path))
        result = {
            "answer": "Test answer",
            "is_fallback": False,
            "latency_ms": 100
        }
        log_query("test query", result, sample_chunks)


# ── Integration Tests ────────────────────────────────────────
class TestIntegration:
    @pytest.fixture(scope="class")
    def pipeline_output(self):
        """Runs real retrieval once for all integration tests."""
        from src.ingest import load_chunks
        from src.indexing import run_indexing_pipeline
        from src.retrieval import run_retrieval_pipeline, get_groq_llm

        chunks = load_chunks()
        bm25, collection = run_indexing_pipeline()
        llm = get_groq_llm()
        retrieved, confident = run_retrieval_pipeline(
            "What is multi-head attention?",
            bm25, chunks, collection, llm
        )
        return retrieved, confident

    def test_run_generation_pipeline_returns_dict(self, pipeline_output):
        """Integration: pipeline returns dict with required keys."""
        retrieved, confident = pipeline_output
        result = run_generation_pipeline(
            "What is multi-head attention?", retrieved, confident
        )
        assert "answer" in result
        assert "sources" in result
        assert "is_fallback" in result
        assert "latency_ms" in result

    def test_generation_answer_is_string(self, pipeline_output):
        """Integration: generated answer is a non-empty string."""
        retrieved, confident = pipeline_output
        result = run_generation_pipeline(
            "What is multi-head attention?", retrieved, confident
        )
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_generation_not_fallback_on_relevant_query(self, pipeline_output):
        """Integration: relevant query does not trigger fallback."""
        retrieved, confident = pipeline_output
        result = run_generation_pipeline(
            "What is multi-head attention?", retrieved, confident
        )
        assert result["is_fallback"] is False