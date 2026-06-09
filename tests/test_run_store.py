"""Tests for RunStore run bundle output."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import RunStore, compute_summary_stats, make_run_id


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockSuccessModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.0,
            "generation_successful": True,
        }


class MockFailureModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.5,
            "generation_successful": False,
        }


def _make_results():
    pipeline = ExperimentPipeline()
    config_ok = ExperimentConfig(
        model_name="Qwen3",
        config_weighting=False,
        config_prompting=True,
        prompt_id="p01",
    )
    config_fail = ExperimentConfig(
        model_name="Qwen3",
        config_weighting=True,
        config_prompting=False,
        prompt_id="p02",
    )
    ok = pipeline.run("What is a friend?", config_ok, MockSuccessModel())
    fail = pipeline.run("What is a friend?", config_fail, MockFailureModel())
    return [ok, fail]


class TestRunStore:
    def test_make_run_id(self):
        ts = datetime(2026, 6, 6, 14, 30, 22)
        run_id = make_run_id(1, "factorial", started_at=ts)
        assert run_id == "20260606_143022_phase1_factorial"

    def test_write_bundle_creates_artifacts(self, tmp_path: Path):
        results = _make_results()
        store = RunStore(tmp_path)
        started = datetime(2026, 6, 6, 14, 30, 22, tzinfo=timezone.utc)
        run_id = make_run_id(1, "factorial", started_at=started)

        out_dir = store.write_bundle(
            run_id,
            results,
            phase=1,
            experiment="factorial",
            cli_args=["--prompts", "2"],
            models=["Qwen3"],
            prompt_count=2,
            started_at=started,
            completed_at=datetime(2026, 6, 6, 14, 31, 0, tzinfo=timezone.utc),
        )

        assert out_dir.exists()
        for name in ("manifest.json", "specification.csv", "full.csv", "summary.json"):
            assert (out_dir / name).exists()

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 1
        assert manifest["experiment"] == "factorial"
        assert manifest["observations"]["total"] == 2
        assert manifest["observations"]["successful"] == 1
        assert manifest["observations"]["failed"] == 1

    def test_specification_csv_columns(self, tmp_path: Path):
        results = _make_results()
        store = RunStore(tmp_path)
        run_id = make_run_id(1, "factorial")
        out_dir = store.write_bundle(run_id, results, phase=1, experiment="factorial")

        spec = pd.read_csv(out_dir / "specification.csv")
        expected_cols = [
            "model",
            "config_weighting",
            "config_prompting",
            "prompt_id",
            "answer",
            "time_spent",
            "generation_successful",
            "flesch_kincaid_grade",
            "gunning_fog",
            "spache_readability",
            "word_count",
            "difficult_words",
        ]
        assert list(spec.columns) == expected_cols
        assert "weight_factor" not in spec.columns

    def test_summary_excludes_failed_from_metric_means(self, tmp_path: Path):
        results = _make_results()
        summary = compute_summary_stats(results)

        assert summary["metadata"]["successful_experiments"] == 1
        assert summary["metadata"]["failed_experiments"] == 1

        fk_mean = summary["overall"]["flesch_kincaid_grade"]["mean"]
        wc_mean = summary["overall"]["word_count"]["mean"]
        assert wc_mean > 0

        # If failed obs were included, word_count mean would be diluted toward 0
        successful_only = [r for r in results if r.generation_successful]
        expected_wc = successful_only[0].word_count
        assert abs(wc_mean - expected_wc) < 0.01
        assert abs(fk_mean - successful_only[0].flesch_kincaid_grade) < 0.01

    def test_read_manifest_and_summary(self, tmp_path: Path):
        results = _make_results()
        store = RunStore(tmp_path)
        run_id = make_run_id(1, "factorial")
        store.write_bundle(run_id, results, phase=1, experiment="factorial")

        manifest = store.read_manifest(run_id)
        summary = store.read_summary(run_id)
        assert manifest["run_id"] == run_id
        assert "overall" in summary

    def test_list_runs(self, tmp_path: Path):
        results = _make_results()
        store = RunStore(tmp_path)
        run_id = make_run_id(1, "factorial")
        store.write_bundle(run_id, results, phase=1, experiment="factorial")

        manifests = store.list_runs()
        assert len(manifests) == 1
        assert manifests[0]["run_id"] == run_id
