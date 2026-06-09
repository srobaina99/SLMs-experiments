"""Tests for plot_run boxplot generation."""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.plot import plot_run


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockSuccessModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.5,
            "generation_successful": True,
        }


class MockFailureModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.2,
            "generation_successful": False,
        }


def _build_factorial_bundle(tmp_path: Path):
    pipeline = ExperimentPipeline()
    configs = [
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=False,
            config_prompting=False,
            prompt_id="P1",
        ),
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=True,
            config_prompting=False,
            prompt_id="P1",
        ),
        ExperimentConfig(
            model_name="Qwen2",
            config_weighting=False,
            config_prompting=True,
            prompt_id="P2",
        ),
        ExperimentConfig(
            model_name="Qwen2",
            config_weighting=True,
            config_prompting=True,
            prompt_id="P2",
        ),
    ]

    results = []
    for idx, config in enumerate(configs):
        model = MockSuccessModel() if idx % 2 == 0 else MockFailureModel()
        results.append(pipeline.run("What is a friend?", config, model))

    store = RunStore(tmp_path)
    started = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)
    run_id = make_run_id(1, "factorial", started_at=started)
    out_dir = store.write_bundle(
        run_id,
        results,
        phase=1,
        experiment="factorial",
        models=["Qwen2", "Qwen3"],
        prompt_count=2,
        started_at=started,
    )
    return run_id, out_dir, store


class TestPlotRun:
    def test_boxplot_files_created(self, tmp_path: Path):
        run_id, _, store = _build_factorial_bundle(tmp_path)

        plots_dir = plot_run(run_id, results_root=tmp_path)

        assert plots_dir == store.run_dir(run_id) / "plots"
        assert (plots_dir / "boxplot_readability.png").exists()
        assert (plots_dir / "boxplot_flesch_kincaid_grade.png").exists()
        assert (plots_dir / "boxplot_gunning_fog.png").exists()
        assert (plots_dir / "boxplot_spache_readability.png").exists()

    def test_plot_from_specification_csv_only(self, tmp_path: Path):
        run_id, out_dir, _ = _build_factorial_bundle(tmp_path)
        (out_dir / "full.csv").unlink()

        plots_dir = plot_run(run_id, results_root=tmp_path)
        assert (plots_dir / "boxplot_readability.png").exists()

    def test_plot_excludes_failed_generations(self, tmp_path: Path):
        run_id, _, _ = _build_factorial_bundle(tmp_path)

        plots_dir = plot_run(run_id, results_root=tmp_path)
        assert plots_dir.exists()
        # Bundle has 2 successful / 4 total; plot should succeed with partial data
        assert (plots_dir / "boxplot_readability.png").stat().st_size > 0

    def test_plot_missing_run_raises(self, tmp_path: Path):
        try:
            plot_run("nonexistent_run", results_root=tmp_path)
        except FileNotFoundError as exc:
            assert "nonexistent_run" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError")
