"""Tests for KVL/GLMM vocabulary difficulty metrics."""

from pathlib import Path

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.run_store import compute_summary_stats
from slm_experiments.evaluation.kvl import (
    KVL_HARD_THRESHOLD,
    KvlLookup,
    compute_kvl_metrics,
    empty_kvl_metrics,
)
from slm_experiments.evaluation.metrics import TextEvaluator

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_lookup(tmp_path):
    """Copy test fixture into an isolated lookup directory."""
    lookup_dir = tmp_path / "kvl"
    lookup_dir.mkdir()
    source = FIXTURE_DIR / "kvl_lookup_es.json"
    (lookup_dir / "kvl_lookup_es.json").write_text(source.read_text(), encoding="utf-8")
    return KvlLookup(data_dir=str(lookup_dir))


class TestComputeKvlMetrics:
    def test_easy_and_hard_words(self, fixture_lookup):
        content_words = {"friend", "like", "establishment", "fundamentally"}
        metrics = compute_kvl_metrics(
            "ignored",
            "es",
            content_words=content_words,
            kvl_lookup=fixture_lookup,
        )

        assert metrics["kvl_l1"] == "es"
        assert metrics["kvl_content_word_count"] == 4
        assert metrics["kvl_lookup_count"] == 4
        assert metrics["kvl_oov_count"] == 0
        assert metrics["kvl_lookup_coverage"] == 1.0
        assert metrics["kvl_mean_score"] == pytest.approx(
            round((2.1 + 0.5 + (-2.3) + (-3.1)) / 4, 4)
        )
        assert metrics["kvl_min_score"] == -3.1
        assert metrics["kvl_pct_hard_words"] == 0.5

    def test_empty_response(self):
        metrics = empty_kvl_metrics("es")

        assert metrics["kvl_content_word_count"] == 0
        assert metrics["kvl_lookup_count"] == 0
        assert metrics["kvl_oov_count"] == 0
        assert metrics["kvl_lookup_coverage"] == 0.0
        assert metrics["kvl_mean_score"] is None
        assert metrics["kvl_min_score"] is None
        assert metrics["kvl_pct_hard_words"] is None

    def test_word_not_in_lookup_excluded_from_mean(self, fixture_lookup):
        content_words = {"friend", "notinlookup"}
        metrics = compute_kvl_metrics(
            "ignored",
            "es",
            content_words=content_words,
            kvl_lookup=fixture_lookup,
        )

        assert metrics["kvl_content_word_count"] == 2
        assert metrics["kvl_lookup_count"] == 1
        assert metrics["kvl_oov_count"] == 1
        assert metrics["kvl_lookup_coverage"] == 0.5
        assert metrics["kvl_mean_score"] == 2.1
        assert metrics["kvl_min_score"] == 2.1
        assert metrics["kvl_pct_hard_words"] == 0.0

    def test_hard_threshold_constant(self):
        assert KVL_HARD_THRESHOLD == -1.0

    def test_no_lookup_hits_returns_none_aggregates(self, fixture_lookup):
        content_words = {"notinlookup", "alsoabsent"}
        metrics = compute_kvl_metrics(
            "ignored",
            "es",
            content_words=content_words,
            kvl_lookup=fixture_lookup,
        )

        assert metrics["kvl_lookup_count"] == 0
        assert metrics["kvl_oov_count"] == 2
        assert metrics["kvl_lookup_coverage"] == 0.0
        assert metrics["kvl_mean_score"] is None


class TestKvlLookup:
    def test_loads_fixture(self, fixture_lookup):
        lookup = fixture_lookup.load("es")
        assert lookup["friend"] == 2.1

    def test_unsupported_l1_raises(self, fixture_lookup):
        with pytest.raises(ValueError, match="Unsupported L1"):
            fixture_lookup.load("fr")

    def test_missing_file_raises(self, tmp_path):
        lookup = KvlLookup(data_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            lookup.load("es")


class TestPipelineKvlIntegration:
    class MockSuccessModel:
        def generate(self, prompt: str, config: ExperimentConfig) -> dict:
            return {
                "response": "A friend is a person you like.",
                "response_time_seconds": 1.0,
                "generation_successful": True,
            }

    class MockFailureModel:
        def generate(self, prompt: str, config: ExperimentConfig) -> dict:
            return {
                "response": "",
                "response_time_seconds": 0.5,
                "generation_successful": False,
            }

    def test_success_path_populates_kvl_columns(self, fixture_lookup):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p01", kvl_l1="es")
        pipeline = ExperimentPipeline(kvl_lookup=fixture_lookup)
        result = pipeline.run("What is a friend?", config, self.MockSuccessModel())

        assert result.generation_successful is True
        assert result.kvl_l1 == "es"
        assert result.kvl_content_word_count > 0
        assert result.kvl_lookup_count > 0
        assert result.kvl_lookup_coverage > 0.0
        assert result.kvl_mean_score is not None

    def test_failure_path_sets_kvl_defaults(self, fixture_lookup):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p02", kvl_l1="es")
        pipeline = ExperimentPipeline(kvl_lookup=fixture_lookup)
        result = pipeline.run("Hello?", config, self.MockFailureModel())

        assert result.generation_successful is False
        assert result.kvl_l1 == "es"
        assert result.kvl_content_word_count == 0
        assert result.kvl_lookup_count == 0
        assert result.kvl_oov_count == 0
        assert result.kvl_mean_score is None

    def test_summary_includes_kvl_stats(self, fixture_lookup):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p01", kvl_l1="es")
        pipeline = ExperimentPipeline(kvl_lookup=fixture_lookup)
        result = pipeline.run("What is a friend?", config, self.MockSuccessModel())

        summary = compute_summary_stats([result])
        assert "kvl_mean_score" in summary["overall"]
        assert "kvl_lookup_coverage" in summary["overall"]
        assert "kvl_oov_count" in summary["overall"]


class TestRealLookupSmoke:
    """Smoke test against committed production lookup files."""

    @pytest.fixture
    def repo_lookup(self):
        repo_root = Path(__file__).resolve().parent.parent
        return KvlLookup(data_dir=str(repo_root / "data" / "kvl"))

    def test_production_lookup_has_words(self, repo_lookup):
        lookup = repo_lookup.load("es")
        assert len(lookup) > 6000
        assert "friend" in lookup

    def test_content_words_from_evaluator(self, repo_lookup):
        evaluator = TextEvaluator()
        text = "The establishment is fundamentally important."
        content_words = evaluator.extract_content_words(text)
        metrics = compute_kvl_metrics(
            text,
            "es",
            content_words=content_words,
            kvl_lookup=repo_lookup,
        )

        assert metrics["kvl_content_word_count"] == len(content_words)
        if metrics["kvl_lookup_count"] > 0:
            assert metrics["kvl_mean_score"] is not None
            assert metrics["kvl_mean_score"] < metrics["kvl_mean_score"] + 1
