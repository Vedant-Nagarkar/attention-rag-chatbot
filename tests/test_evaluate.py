import os
import json
import pytest

from src.evaluate import (
    GOLDEN_TEST_SET,
    save_golden_set,
)
from config import GOLDEN_SET_PATH, RAGAS_RESULTS_PATH


# ── Smoke Tests ──────────────────────────────────────────────
class TestSmoke:
    def test_evaluate_module_imports(self):
        """Smoke: evaluate module imports without error."""
        from src import evaluate
        assert evaluate is not None

    def test_golden_test_set_exists(self):
        """Smoke: GOLDEN_TEST_SET is defined and non-empty."""
        assert isinstance(GOLDEN_TEST_SET, list)
        assert len(GOLDEN_TEST_SET) == 10

    def test_ragas_results_file_exists(self):
        """Smoke: ragas_results.json exists after eval has run."""
        assert os.path.exists(RAGAS_RESULTS_PATH), (
            "ragas_results.json not found. Run run_eval.py first."
        )


# ── Unit Tests ───────────────────────────────────────────────
class TestUnit:
    def test_golden_set_has_required_keys(self):
        """Unit: every golden set item has question and ground_truth."""
        for item in GOLDEN_TEST_SET:
            assert "question" in item
            assert "ground_truth" in item

    def test_golden_set_no_empty_fields(self):
        """Unit: no question or ground_truth is empty."""
        for item in GOLDEN_TEST_SET:
            assert len(item["question"].strip()) > 0
            assert len(item["ground_truth"].strip()) > 0

    def test_golden_set_questions_are_unique(self):
        """Unit: all 10 questions are unique."""
        questions = [item["question"] for item in GOLDEN_TEST_SET]
        assert len(questions) == len(set(questions))

    def test_save_golden_set_creates_file(self, tmp_path, monkeypatch):
        """Unit: save_golden_set writes JSON to disk."""
        monkeypatch.setattr("src.evaluate.GOLDEN_SET_PATH", str(tmp_path / "golden.json"))
        monkeypatch.setattr("config.GOLDEN_SET_PATH", str(tmp_path / "golden.json"))
        path = save_golden_set()
        assert os.path.exists(path)

    def test_save_golden_set_valid_json(self, tmp_path, monkeypatch):
        """Unit: saved golden set is valid parseable JSON."""
        output_path = str(tmp_path / "golden.json")
        monkeypatch.setattr("src.evaluate.GOLDEN_SET_PATH", output_path)
        monkeypatch.setattr("config.GOLDEN_SET_PATH", output_path)
        save_golden_set()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 10


# ── Integration Tests ────────────────────────────────────────
class TestIntegration:
    def test_ragas_results_has_required_keys(self):
        """Integration: ragas_results.json has all four metric keys."""
        with open(RAGAS_RESULTS_PATH, "r") as f:
            results = json.load(f)
        assert "faithfulness" in results
        assert "answer_relevancy" in results
        assert "context_precision" in results
        assert "context_recall" in results

    def test_faithfulness_score_above_threshold(self):
        """Integration: faithfulness score is above 0.60."""
        with open(RAGAS_RESULTS_PATH, "r") as f:
            results = json.load(f)
        score = results["faithfulness"]
        assert isinstance(score, float)
        assert score >= 0.60, f"Faithfulness {score} below threshold 0.60"

    def test_context_precision_above_threshold(self):
        """Integration: context precision is above 0.70."""
        with open(RAGAS_RESULTS_PATH, "r") as f:
            results = json.load(f)
        score = results["context_precision"]
        assert isinstance(score, float)
        assert score >= 0.70, f"Context precision {score} below threshold 0.70"

    def test_context_recall_above_threshold(self):
        """Integration: context recall is above 0.60."""
        with open(RAGAS_RESULTS_PATH, "r") as f:
            results = json.load(f)
        score = results["context_recall"]
        assert isinstance(score, float)
        assert score >= 0.60, f"Context recall {score} below threshold 0.60"