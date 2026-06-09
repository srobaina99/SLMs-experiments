"""Tests for human evaluation export/import."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.human.export import (
    EXPORT_COLUMNS,
    HumanExporter,
    config_label,
    stratified_sample,
)
from importlib import import_module

HumanImporter = import_module("slm_experiments.human.import").HumanImporter

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


def _build_factorial_bundle(tmp_path: Path, num_prompts: int = 3) -> tuple[str, RunStore]:
    """Create a factorial-like run bundle with 4 configs × num_prompts rows."""
    pipeline = ExperimentPipeline()
    configs = [
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=False,
            config_prompting=False,
            prompt_id=f"P{i + 1}",
        )
        for i in range(num_prompts)
    ] + [
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=True,
            config_prompting=False,
            prompt_id=f"P{i + 1}",
        )
        for i in range(num_prompts)
    ] + [
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=False,
            config_prompting=True,
            prompt_id=f"P{i + 1}",
        )
        for i in range(num_prompts)
    ] + [
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=True,
            config_prompting=True,
            prompt_id=f"P{i + 1}",
        )
        for i in range(num_prompts)
    ]

    results = [
        pipeline.run("What is a friend?", config, MockSuccessModel())
        for config in configs
    ]

    store = RunStore(tmp_path)
    started = datetime(2026, 6, 6, 14, 30, 22, tzinfo=timezone.utc)
    run_id = make_run_id(1, "factorial", started_at=started.replace(tzinfo=None))
    store.write_bundle(
        run_id,
        results,
        phase=1,
        experiment="factorial",
        cli_args=["--prompts", str(num_prompts)],
        models=["Qwen3"],
        prompt_count=num_prompts,
        started_at=started,
        completed_at=datetime(2026, 6, 6, 14, 31, 0, tzinfo=timezone.utc),
    )
    return run_id, store


class TestConfigLabel:
    def test_config_labels(self):
        assert config_label(False, False) == "control"
        assert config_label(True, False) == "weighting_only"
        assert config_label(False, True) == "prompting_only"
        assert config_label(True, True) == "both"


class TestStratifiedSample:
    def test_returns_all_when_sample_exceeds_rows(self):
        df = pd.DataFrame(
            {
                "config": ["control", "both", "control", "both"],
                "value": [1, 2, 3, 4],
            }
        )
        sampled = stratified_sample(df, n=10, seed=42)
        assert len(sampled) == 4

    def test_respects_sample_size(self):
        df = pd.DataFrame(
            {
                "config": ["control"] * 20 + ["both"] * 20,
                "value": list(range(40)),
            }
        )
        sampled = stratified_sample(df, n=8, seed=42)
        assert len(sampled) == 8
        assert set(sampled["config"]) == {"control", "both"}


class TestHumanExport:
    def test_export_sample_size_and_columns(self, tmp_path: Path):
        run_id, store = _build_factorial_bundle(tmp_path, num_prompts=5)
        exporter = HumanExporter(results_root=tmp_path)

        out_path, row_count = exporter.export(run_id, sample=6, seed=42)

        assert out_path == store.run_dir(run_id) / "human_review.csv"
        assert row_count == 6

        review = pd.read_csv(out_path)
        assert list(review.columns) == EXPORT_COLUMNS
        assert review["response_appropriateness"].isna().all()
        assert review["vocabulary_level"].isna().all()
        assert review["notes"].isna().all()

        manifest = json.loads((store.run_dir(run_id) / "manifest.json").read_text())
        assert manifest["artifacts"]["human_review_csv"] == "human_review.csv"
        assert manifest["human_eval"]["exported_rows"] == 6

    def test_export_all_when_sample_larger_than_run(self, tmp_path: Path):
        run_id, _store = _build_factorial_bundle(tmp_path, num_prompts=2)
        exporter = HumanExporter(results_root=tmp_path)

        _out_path, row_count = exporter.export(run_id, sample=60, seed=42)
        assert row_count == 8


class TestHumanImport:
    def test_import_merges_tags(self, tmp_path: Path):
        run_id, store = _build_factorial_bundle(tmp_path, num_prompts=3)
        exporter = HumanExporter(results_root=tmp_path)
        out_path, _ = exporter.export(run_id, sample=4, seed=42)

        review = pd.read_csv(out_path)
        review.loc[0, "response_appropriateness"] = 4.0
        review.loc[0, "vocabulary_level"] = "beginner"
        review.loc[0, "notes"] = "clear and simple"
        review.loc[1, "response_appropriateness"] = 2.0
        review.loc[1, "vocabulary_level"] = "intermediate"
        tagged_path = store.run_dir(run_id) / "tagged.csv"
        review.to_csv(tagged_path, index=False)

        importer = HumanImporter(results_root=tmp_path)
        updated = importer.import_tags(run_id, tagged_path)
        assert updated == 2

        full_df = store.read_full_csv(run_id)
        first_id = review.loc[0, "experiment_id"]
        row = full_df[full_df["experiment_id"] == first_id].iloc[0]
        assert row["response_appropriateness"] == 4.0
        assert row["vocabulary_level"] == "beginner"
        assert row["notes"] == "clear and simple"

    def test_import_rejects_unknown_experiment_id(self, tmp_path: Path):
        run_id, store = _build_factorial_bundle(tmp_path, num_prompts=2)
        exporter = HumanExporter(results_root=tmp_path)
        out_path, _ = exporter.export(run_id, sample=2, seed=42)

        review = pd.read_csv(out_path)
        review.loc[0, "experiment_id"] = "missing-id"
        bad_path = store.run_dir(run_id) / "bad_tags.csv"
        review.to_csv(bad_path, index=False)

        importer = HumanImporter(results_root=tmp_path)
        with pytest.raises(ValueError, match="unknown experiment_id"):
            importer.import_tags(run_id, bad_path)


class TestHumanRoundTrip:
    def test_export_import_round_trip(self, tmp_path: Path):
        run_id, store = _build_factorial_bundle(tmp_path, num_prompts=4)
        exporter = HumanExporter(results_root=tmp_path)
        importer = HumanImporter(results_root=tmp_path)

        out_path, exported_count = exporter.export(run_id, sample=8, seed=7)
        review = pd.read_csv(out_path)
        review["response_appropriateness"] = [5, 4, 3, 2, 1, 5, 4, 3][:exported_count]
        review["vocabulary_level"] = ["beginner"] * exported_count
        review["notes"] = [f"note-{i}" for i in range(exported_count)]
        review.to_csv(out_path, index=False)

        updated = importer.import_tags(run_id, out_path)
        assert updated == exported_count

        full_df = store.read_full_csv(run_id)
        tagged_ids = set(review["experiment_id"])
        tagged_rows = full_df[full_df["experiment_id"].isin(tagged_ids)]
        assert len(tagged_rows) == exported_count
        assert tagged_rows["response_appropriateness"].notna().all()
        assert tagged_rows["vocabulary_level"].eq("beginner").all()

        manifest = json.loads((store.run_dir(run_id) / "manifest.json").read_text())
        assert manifest["human_eval"]["updated_rows"] == exported_count
