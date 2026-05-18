# tests/test_ingest.py
import os
import json
import pytest

from src.ingest import (
    check_pdf_exists,
    extract_text_from_pdf,
    clean_text,
    chunk_text,
    save_chunks,
    load_chunks,
    run_ingest_pipeline,
)
from config import PDF_PATH, PROCESSED_DIR


# ── Smoke Tests ──────────────────────────────────────────────
class TestSmoke:
    def test_ingest_module_imports(self):
        """Smoke: ingest module imports without error."""
        from src import ingest
        assert ingest is not None

    def test_pdf_exists(self):
        """Smoke: PDF file is present in data/raw/."""
        assert os.path.exists(PDF_PATH), (
            f"PDF not found at {PDF_PATH}. "
            f"Download from https://arxiv.org/pdf/1706.03762"
        )

    def test_chunks_json_exists(self):
        """Smoke: chunks.json exists after ingest pipeline has run."""
        chunks_path = os.path.join(PROCESSED_DIR, "chunks.json")
        assert os.path.exists(chunks_path), (
            "chunks.json not found. Run run_ingest_pipeline() first."
        )


# ── Unit Tests ───────────────────────────────────────────────
class TestUnit:
    def test_clean_text_removes_extra_newlines(self):
        """Unit: clean_text collapses 3+ newlines into 2."""
        raw = "hello\n\n\n\nworld"
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_clean_text_removes_double_spaces(self):
        """Unit: clean_text collapses multiple spaces."""
        raw = "hello   world"
        result = clean_text(raw)
        assert "  " not in result

    def test_clean_text_fixes_hyphenation(self):
        """Unit: clean_text fixes hyphenated line breaks."""
        raw = "trans-\nformer"
        result = clean_text(raw)
        assert "transformer" in result

    def test_chunk_text_returns_list(self):
        """Unit: chunk_text returns a non-empty list of dicts."""
        sample_text = "The Transformer model uses attention mechanisms. " * 50
        chunks = chunk_text(sample_text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_text_dict_keys(self):
        """Unit: each chunk has required keys."""
        sample_text = "The Transformer model uses attention mechanisms. " * 50
        chunks = chunk_text(sample_text)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "char_count" in chunk

    def test_chunk_text_ids_are_sequential(self):
        """Unit: chunk IDs are sequential starting from 0."""
        sample_text = "The Transformer model uses attention mechanisms. " * 50
        chunks = chunk_text(sample_text)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_text_char_count_matches(self):
        """Unit: char_count matches actual text length."""
        sample_text = "The Transformer model uses attention mechanisms. " * 50
        chunks = chunk_text(sample_text)
        for chunk in chunks:
            assert chunk["char_count"] == len(chunk["text"])


# ── Integration Tests ────────────────────────────────────────
class TestIntegration:
    def test_load_chunks_returns_86(self):
        """Integration: load_chunks returns 86 chunks from saved JSON."""
        chunks = load_chunks()
        assert len(chunks) == 86

    def test_load_chunks_content_is_valid(self):
        """Integration: loaded chunks have non-empty text."""
        chunks = load_chunks()
        for chunk in chunks:
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_extract_text_from_pdf(self):
        """Integration: PDF extraction returns substantial text."""
        text = extract_text_from_pdf(PDF_PATH)
        assert isinstance(text, str)
        assert len(text) > 10000  # paper should have at least 10k chars
        assert "Transformer" in text  # sanity check

    def test_save_and_load_chunks_roundtrip(self, tmp_path, monkeypatch):
        """Integration: save then load chunks returns same data."""
        sample_chunks = [
            {"chunk_id": 0, "text": "hello world", "char_count": 11},
            {"chunk_id": 1, "text": "attention is all you need", "char_count": 25},
        ]
        # Monkeypatch PROCESSED_DIR to tmp_path
        monkeypatch.setattr("src.ingest.PROCESSED_DIR", str(tmp_path))

        save_chunks(sample_chunks)
        loaded = load_chunks()
        assert loaded == sample_chunks