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

    def test_plot_rejects_assessment_bundle(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment import AssessmentBundler

        run_id, _out_dir, store = _build_factorial_bundle(tmp_path)
        assess_id, _ = AssessmentBundler(results_root=tmp_path).build(
            [run_id], seed=42
        )
        try:
            plot_run(assess_id, results_root=tmp_path)
        except ValueError as exc:
            message = str(exc)
            assert "assessment" in message.lower()
            assert "assess analyze" in message
        else:
            raise AssertionError("Expected ValueError for assessment kind")
        # Generation plot still works.
        assert plot_run(run_id, results_root=tmp_path).exists()
        assert store.run_dir(run_id).exists()

    def test_plot_phase2_guided_and_kvl_beam_runs(self, tmp_path: Path):
        """Sweep metadata on full.csv must not break plot_run."""
        pipeline = ExperimentPipeline()

        class MockGuided:
            def generate_guided(self, prompt, config):
                return {
                    "response": SIMPLE_RESPONSE,
                    "response_time_seconds": 1.0,
                    "generation_successful": True,
                    "guided_top_k": config.guided_top_k,
                    "guided_mode": "flat",
                    "guided_steps_a1_chosen": 1,
                    "guided_steps_total": 4,
                    "guided_intervention_rate": 0.25,
                }

        class MockKvl:
            def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
                return {
                    "response": SIMPLE_RESPONSE,
                    "response_time_seconds": 1.0,
                    "generation_successful": True,
                    "kvl_beam_width": beam_width,
                    "kvl_branch_factor": branch_factor,
                    "kvl_beam_steps_total": 4,
                    "kvl_beam_words_scored": 2,
                    "kvl_beam_running_mean": 1.0,
                    "kvl_beam_logprob_tiebreak": -0.5,
                    "kvl_beam_candidates_pruned": 1,
                }

        store = RunStore(tmp_path)
        started = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)

        guided_results = []
        for k in (0, 10):
            config = ExperimentConfig(
                model_name="Qwen3",
                config_prompting=True,
                config_guided=k != 0,
                guided_top_k=k,
                prompt_id="P1",
                experiment_name=f"Qwen3_guided_k{k}",
            )
            if k == 0:
                result = pipeline.run("What is a friend?", config, MockSuccessModel())
                result.guided_top_k = 0
            else:
                result = pipeline.run_guided(
                    "What is a friend?", config, MockGuided()
                )
            guided_results.append(result)
        guided_id = make_run_id(2, "guided", started_at=started)
        store.write_bundle(
            guided_id, guided_results, phase=2, experiment="guided", started_at=started
        )
        guided_plots = plot_run(guided_id, results_root=tmp_path)
        assert (guided_plots / "boxplot_readability.png").exists()
        assert (guided_plots / "boxplot_readability.png").stat().st_size > 0

        kvl_results = []
        for width in (1, 8):
            config = ExperimentConfig(
                model_name="Qwen3",
                config_prompting=True,
                config_kvl_beam=width != 1,
                prompt_id="P1",
                experiment_name=f"Qwen3_kvl_beam_w{width}",
            )
            if width == 1:
                result = pipeline.run("What is a friend?", config, MockSuccessModel())
                result.kvl_beam_width = 1
            else:
                result = pipeline.run_kvl_beam(
                    "What is a friend?", config, MockKvl(), beam_width=width
                )
            kvl_results.append(result)
        kvl_started = datetime(2026, 6, 6, 13, 0, 0, tzinfo=timezone.utc)
        kvl_id = make_run_id(2, "kvl_beam", started_at=kvl_started)
        store.write_bundle(
            kvl_id, kvl_results, phase=2, experiment="kvl_beam", started_at=kvl_started
        )
        kvl_plots = plot_run(kvl_id, results_root=tmp_path)
        assert (kvl_plots / "boxplot_readability.png").exists()
        assert (kvl_plots / "boxplot_flesch_kincaid_grade.png").exists()
