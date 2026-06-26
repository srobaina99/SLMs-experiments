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

COMPLEX_RESPONSE = (
    "The multifaceted ramifications of epistemological paradigms necessitate "
    "comprehensive interdisciplinary investigation. Furthermore, methodological "
    "inconsistencies undermine the validity of longitudinal analyses."
)


class MockSuccessModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.0,
            "generation_successful": True,
        }


class MockBeamModelWrapper:
    def generate_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        selection_method: str = "a1_ratio",
    ) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.0,
            "generation_successful": True,
            "beam_selection_method": selection_method,
            "beam_a1_ratio": 0.75,
            "beam_a1_count": 3,
            "beam_content_word_count": 6,
            "beam_cumulative_logprob": -1.2,
            "beam_width": beam_width,
        }


class MockGuidedModelWrapper:
    def generate_guided(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.0,
            "generation_successful": True,
            "guided_top_k": config.guided_top_k,
            "guided_mode": config.guided_mode,
            "guided_steps_a1_chosen": 5,
            "guided_steps_total": 20,
            "guided_intervention_rate": 0.25,
        }


class MockKvlBeamModelWrapper:
    def generate_kvl_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        branch_factor: int = 10,
    ) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.0,
            "generation_successful": True,
            "kvl_beam_width": beam_width,
            "kvl_branch_factor": branch_factor,
            "kvl_beam_steps_total": 30,
            "kvl_beam_words_scored": 5,
            "kvl_beam_running_mean": 2.0,
            "kvl_beam_logprob_tiebreak": -1.0,
            "kvl_beam_candidates_pruned": 80,
        }


class MockComplexModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": COMPLEX_RESPONSE,
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


def _make_a1_pipeline_results():
    """One A1-passing and one valid-but-failing result from the pipeline."""
    pipeline = ExperimentPipeline()
    config_simple = ExperimentConfig(
        model_name="Qwen3",
        config_weighting=False,
        config_prompting=True,
        prompt_id="p01",
    )
    config_complex = ExperimentConfig(
        model_name="Qwen3",
        config_weighting=True,
        config_prompting=False,
        prompt_id="p02",
    )
    simple = pipeline.run("What is a friend?", config_simple, MockSuccessModel())
    complex = pipeline.run("What is a friend?", config_complex, MockComplexModel())
    return [simple, complex]


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
            "meets_a1_criteria",
            "flesch_kincaid_grade",
            "gunning_fog",
            "spache_readability",
            "word_count",
            "difficult_words",
        ]
        assert list(spec.columns) == expected_cols
        assert "weight_factor" not in spec.columns

    def test_summary_includes_a1_pass_rate(self, tmp_path: Path):
        results = _make_results()
        success = next(r for r in results if r.generation_successful)
        success.meets_a1_criteria = True
        fail = next(r for r in results if not r.generation_successful)
        fail.meets_a1_criteria = False

        summary = compute_summary_stats(results)

        assert summary["metadata"]["a1_pass_experiments"] == 1
        assert summary["metadata"]["a1_pass_rate"] == 0.5
        assert summary["by_config"]["prompting_only"]["a1_pass_count"] == 1
        assert summary["by_config"]["prompting_only"]["a1_pass_rate"] == 1.0
        assert summary["by_config"]["weighting_only"]["a1_pass_count"] == 0
        assert summary["by_config"]["weighting_only"]["a1_pass_rate"] == 0.0

    def test_summary_a1_pass_rate_from_pipeline_results(self):
        results = _make_a1_pipeline_results()
        summary = compute_summary_stats(results)

        assert summary["metadata"]["a1_pass_experiments"] == 1
        assert summary["metadata"]["a1_pass_rate"] == 0.5
        assert summary["by_config"]["prompting_only"]["a1_pass_count"] == 1
        assert summary["by_config"]["weighting_only"]["a1_pass_count"] == 0
        assert summary["by_config"]["prompting_only"]["generation_successful_count"] == 1
        assert summary["by_config"]["prompting_only"]["generation_failure_rate"] == 0.0
        assert summary["by_config"]["weighting_only"]["generation_successful_count"] == 1
        assert summary["by_config"]["weighting_only"]["generation_failure_rate"] == 0.0
        assert summary["by_config"]["prompting_only"]["a1_pass_rate_given_valid"] == 1.0
        assert summary["by_config"]["weighting_only"]["a1_pass_rate_given_valid"] == 0.0

        valid_count = sum(
            group["generation_successful_count"]
            for group in summary["by_config"].values()
        )
        a1_pass_among_valid = sum(
            group["a1_pass_rate_given_valid"] * group["generation_successful_count"]
            for group in summary["by_config"].values()
        ) / valid_count
        assert a1_pass_among_valid == 0.5

    def test_specification_csv_meets_a1_criteria_values(self, tmp_path: Path):
        results = _make_a1_pipeline_results()
        store = RunStore(tmp_path)
        run_id = make_run_id(1, "factorial")
        out_dir = store.write_bundle(run_id, results, phase=1, experiment="factorial")

        spec = pd.read_csv(out_dir / "specification.csv")
        by_prompt = spec.set_index("prompt_id")["meets_a1_criteria"]
        assert by_prompt["p01"] == True  # noqa: E712
        assert by_prompt["p02"] == False  # noqa: E712

    def test_summary_a1_secondary_fields(self):
        results = _make_results()
        summary = compute_summary_stats(results)

        assert summary["by_config"]["prompting_only"]["generation_successful_count"] == 1
        assert summary["by_config"]["prompting_only"]["generation_failure_rate"] == 0.0
        assert summary["by_config"]["weighting_only"]["generation_successful_count"] == 0
        assert summary["by_config"]["weighting_only"]["generation_failure_rate"] == 1.0

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

    def test_summary_includes_sweep_buckets_for_phase2(self, tmp_path: Path):
        pipeline = ExperimentPipeline()

        def _result(weight: float, fk: float) -> ExperimentResult:
            config = ExperimentConfig(
                model_name="Qwen3",
                config_weighting=True,
                config_prompting=True,
                weight_factor=weight,
                prompt_id="p01",
            )
            result = pipeline.run("What is a friend?", config, MockSuccessModel())
            result.flesch_kincaid_grade = fk
            return result

        results = [_result(1.0, 4.0), _result(1.5, 3.0), _result(1.5, 2.5)]
        summary = compute_summary_stats(results, experiment="weights")

        assert summary["metadata"]["sweep_dimension"] == "weight_factor"
        assert summary["metadata"]["sweep_values"] == ["1", "1.5"]
        assert summary["by_weight_factor"]["1"]["count"] == 1
        assert summary["by_weight_factor"]["1.5"]["count"] == 2
        assert summary["by_weight_factor"]["1.5"]["flesch_kincaid_grade"]["mean"] == 2.75
        assert list(summary["by_config"].keys()) == ["both"]

    def test_summary_includes_prompting_shot_buckets(self, tmp_path: Path):
        pipeline = ExperimentPipeline()

        def _result(num_shots: int, fk: float) -> ExperimentResult:
            config = ExperimentConfig(
                model_name="Qwen3",
                config_weighting=False,
                config_prompting=True,
                num_shots=num_shots,
                prompt_id="p01",
            )
            result = pipeline.run("What is a friend?", config, MockSuccessModel())
            result.flesch_kincaid_grade = fk
            return result

        results = [_result(0, 4.0), _result(1, 3.5), _result(3, 3.0)]
        summary = compute_summary_stats(results, experiment="prompting")

        assert summary["metadata"]["sweep_dimension"] == "num_shots"
        assert summary["metadata"]["sweep_values"] == ["0", "1", "3"]
        assert summary["by_num_shots"]["0"]["flesch_kincaid_grade"]["mean"] == 4.0
        assert summary["by_num_shots"]["3"]["flesch_kincaid_grade"]["mean"] == 3.0
        assert list(summary["by_config"].keys()) == ["prompting_only"]

    def test_summary_includes_beam_width_buckets(self, tmp_path: Path):
        pipeline = ExperimentPipeline()
        model = MockBeamModelWrapper()

        def _result(beam_width: int, fk: float) -> ExperimentResult:
            config = ExperimentConfig(
                model_name="Qwen3",
                config_weighting=False,
                config_prompting=True,
                prompt_id="p01",
                experiment_name=f"Qwen3_beam_w{beam_width}",
            )
            result = pipeline.run_beam(
                "What is a friend?", config, model, beam_width=beam_width
            )
            result.flesch_kincaid_grade = fk
            return result

        results = [_result(4, 4.0), _result(8, 3.5), _result(4, 3.0)]
        summary = compute_summary_stats(results, experiment="beam")

        assert summary["metadata"]["sweep_dimension"] == "beam_width"
        assert summary["metadata"]["sweep_values"] == ["4", "8"]
        assert summary["by_beam_width"]["4"]["count"] == 2
        assert summary["by_beam_width"]["8"]["count"] == 1
        assert summary["by_beam_width"]["4"]["flesch_kincaid_grade"]["mean"] == 3.5
        assert list(summary["by_config"].keys()) == ["prompting_only"]

    def test_summary_includes_guided_top_k_buckets(self, tmp_path: Path):
        pipeline = ExperimentPipeline()
        model = MockGuidedModelWrapper()

        def _result(guided_top_k: int, fk: float) -> ExperimentResult:
            config = ExperimentConfig(
                model_name="Qwen3",
                config_weighting=False,
                config_prompting=True,
                config_guided=True,
                guided_top_k=guided_top_k,
                guided_mode="flat",
                prompt_id="p01",
                experiment_name=f"Qwen3_guided_k{guided_top_k}",
            )
            result = pipeline.run_guided(
                "What is a friend?", config, model
            )
            result.flesch_kincaid_grade = fk
            return result

        results = [_result(5, 4.0), _result(10, 3.5), _result(5, 3.0)]
        summary = compute_summary_stats(results, experiment="guided")

        assert summary["metadata"]["sweep_dimension"] == "guided_top_k"
        assert summary["metadata"]["sweep_values"] == ["5", "10"]
        assert summary["by_guided_top_k"]["5"]["count"] == 2
        assert summary["by_guided_top_k"]["10"]["count"] == 1
        assert summary["by_guided_top_k"]["5"]["flesch_kincaid_grade"]["mean"] == 3.5
        assert list(summary["by_config"].keys()) == ["prompting_only"]

    def test_summary_includes_kvl_beam_width_buckets(self, tmp_path: Path):
        pipeline = ExperimentPipeline()
        model = MockKvlBeamModelWrapper()

        def _result(kvl_beam_width: int, fk: float) -> ExperimentResult:
            config = ExperimentConfig(
                model_name="Qwen3",
                config_weighting=False,
                config_prompting=True,
                config_kvl_beam=True,
                prompt_id="p01",
                experiment_name=f"Qwen3_kvl_beam_w{kvl_beam_width}",
            )
            result = pipeline.run_kvl_beam(
                "What is a friend?",
                config,
                model,
                beam_width=kvl_beam_width,
            )
            result.flesch_kincaid_grade = fk
            result.meets_a1_criteria = True
            return result

        results = [_result(4, 4.0), _result(8, 3.5), _result(4, 3.0)]
        store = RunStore(tmp_path)
        run_id = make_run_id(2, "kvl_beam")
        out_dir = store.write_bundle(
            run_id, results, phase=2, experiment="kvl_beam"
        )
        summary = json.loads((out_dir / "summary.json").read_text())

        assert summary["metadata"]["sweep_dimension"] == "kvl_beam_width"
        assert summary["metadata"]["sweep_values"] == ["4", "8"]
        assert summary["by_kvl_beam_width"]["4"]["count"] == 2
        assert summary["by_kvl_beam_width"]["8"]["count"] == 1
        assert summary["by_kvl_beam_width"]["4"]["a1_pass_rate"] == 1.0
        assert summary["by_kvl_beam_width"]["4"]["flesch_kincaid_grade"]["mean"] == 3.5

        full = pd.read_csv(out_dir / "full.csv")
        assert "kvl_beam_width" in full.columns
        assert set(full["kvl_beam_width"].tolist()) == {4, 8}

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
