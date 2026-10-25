"""Tests for Phase 2 guided decoding top-K pool sweep configs and runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS, STANDARD_PROMPTS
from slm_experiments.phase2.guided import (
    DEFAULT_TOP_K_POOL_GRID,
    GuidedSweepRunner,
    create_guided_configs,
    guided_top_k_from_config,
    parse_top_k_pools,
)

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockGuidedModelWrapper:
    def __init__(self, model_name: str, seed: int = 42, **kwargs):
        self.model_name = model_name
        self.seed = seed
        self.generate_calls = 0
        self.generate_guided_calls = 0
        self.cleanup_calls = 0

    def generate(self, prompt, config):
        self.generate_calls += 1
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
        }

    def generate_guided(self, prompt, config):
        self.generate_guided_calls += 1
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.5,
            "generation_successful": True,
            "guided_top_k": config.guided_top_k,
            "guided_mode": config.guided_mode,
            "guided_steps_a1_chosen": 12,
            "guided_steps_total": 20,
            "guided_intervention_rate": 0.6,
        }

    def cleanup(self):
        self.cleanup_calls += 1


class TestGuidedConfigs:
    def test_create_guided_configs_count(self):
        configs = create_guided_configs(DEFAULT_TOP_K_POOL_GRID)
        assert DEFAULT_TOP_K_POOL_GRID == [0, 5, 10, 20]
        assert len(configs) == len(MODEL_CONFIGS) * len(DEFAULT_TOP_K_POOL_GRID)

    def test_guided_configs_prompting_on_weighting_off(self):
        configs = create_guided_configs([5, 10])
        for config in configs:
            assert config.config_weighting is False
            assert config.config_prompting is True
            assert config.config_guided is True
            assert config.config_kvl_beam is False
            assert config.num_shots == 0
            assert config.temperature == 0.0
            assert config.top_k == 50
            assert config.guided_mode == "flat"

    def test_guided_baseline_pool_zero(self):
        configs = create_guided_configs([0, 5])
        by_pool = {c.guided_top_k: c for c in configs if c.model_name == "Qwen3"}
        assert by_pool[0].config_guided is False
        assert by_pool[0].config_prompting is True
        assert by_pool[0].config_weighting is False
        assert by_pool[0].experiment_name.endswith("_guided_k0")
        assert by_pool[5].config_guided is True

    def test_guided_top_k_encoded_in_experiment_name(self):
        configs = create_guided_configs([5, 20])
        for config in configs:
            pool_size = guided_top_k_from_config(config)
            assert pool_size in (5, 20)
            assert config.experiment_name.endswith(f"_guided_k{pool_size}")

    def test_guided_mode_passed_to_configs(self):
        configs = create_guided_configs([10], guided_mode="trie")
        for config in configs:
            assert config.guided_mode == "trie"

    def test_parse_top_k_pools(self):
        assert parse_top_k_pools("0,5,10,20") == [0, 5, 10, 20]
        assert parse_top_k_pools("5,10,20") == [5, 10, 20]

    def test_parse_top_k_pools_invalid(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            parse_top_k_pools("-1,5")

    def test_parse_top_k_pools_empty_or_whitespace(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_top_k_pools("")
        with pytest.raises(ValueError, match="must not be empty"):
            parse_top_k_pools("   ")
        with pytest.raises(ValueError, match="at least one value"):
            parse_top_k_pools(",")
        with pytest.raises(ValueError, match="at least one value"):
            parse_top_k_pools(" , , ")

    def test_parse_top_k_pools_trailing_comma(self):
        assert parse_top_k_pools("0,5,10,") == [0, 5, 10]
        assert parse_top_k_pools(" 5 , 10 , ") == [5, 10]

    def test_default_grid_includes_unconstrained_baseline(self):
        assert 0 in DEFAULT_TOP_K_POOL_GRID
        assert DEFAULT_TOP_K_POOL_GRID[0] == 0


class TestGuidedSweepRunner:
    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_default_run_produces_48_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockGuidedModelWrapper(
            name, seed=seed
        )

        runner = GuidedSweepRunner(results_root=tmp_path)
        run_id, out_dir = runner.run(
            prompts="3",
            models="all",
            seed=42,
            no_plot=True,
            cli_args=["--prompts", "3"],
        )

        # 4 models × 4 pools (incl. baseline) × 3 prompts = 48
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 2
        assert manifest["experiment"] == "guided"
        assert manifest["prompt_count"] == 3
        assert manifest["observations"]["total"] == 48
        assert manifest["models"] == list(MODEL_CONFIGS.keys())

        full = (out_dir / "full.csv").read_text()
        assert full.count("\n") == 49  # header + 48 rows
        assert "guided_intervention_rate" in full
        assert "guided_top_k" in full

        assert mock_get_wrapper.call_count == len(MODEL_CONFIGS)

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_baseline_routes_to_generate(self, mock_get_wrapper, tmp_path: Path):
        wrapper = MockGuidedModelWrapper("Qwen3")
        mock_get_wrapper.return_value = wrapper

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1",
            models="Qwen3",
            top_k_pools="0,5",
            no_plot=True,
        )

        assert wrapper.generate_calls == 1
        assert wrapper.generate_guided_calls == 1

        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["by_guided_top_k"]["0"]["count"] == 1
        assert summary["by_guided_top_k"]["5"]["count"] == 1

        full = (out_dir / "full.csv").read_text()
        assert "Qwen3_guided_k0" in full
        assert "Qwen3_guided_k5" in full

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_single_model_subset(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Qwen3",
            top_k_pools="5,10",
            no_plot=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 4  # 2 pools × 2 prompts
        mock_get_wrapper.assert_called_once_with("Qwen3", seed=42)

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_guided_metadata_in_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1", models="Qwen3", top_k_pools="10", no_plot=True
        )

        full = (out_dir / "full.csv").read_text()
        assert ",0.6," in full or ",0,6," in full  # guided_intervention_rate column
        assert ",10," in full  # guided_top_k

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_configs_passed_to_pipeline(self, mock_get_wrapper, tmp_path: Path):
        captured = []

        class CapturingWrapper(MockGuidedModelWrapper):
            def generate_guided(self, prompt, config):
                captured.append(config)
                return super().generate_guided(prompt, config)

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1", models="Qwen3", top_k_pools="20", mode="trie", no_plot=True
        )

        assert len(captured) == 1
        config = captured[0]
        assert config.config_weighting is False
        assert config.config_prompting is True
        assert config.config_guided is True
        assert config.prompt_id == "P1"
        assert config.experiment_name == "Qwen3_guided_k20"
        assert config.guided_top_k == 20
        assert config.guided_mode == "trie"
        assert config.temperature == 0.0
        assert config.top_k == 50
        assert config.config_kvl_beam is False
        assert config.config_weighting is False

    def test_guided_top_k_from_config_invalid(self):
        config = ExperimentConfig(experiment_name="invalid_name")
        with pytest.raises(ValueError, match="Cannot parse guided top-k"):
            guided_top_k_from_config(config)

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_prompts_all_tiny_grid_count(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="all",
            models="Qwen3",
            top_k_pools="0,5",
            no_plot=True,
        )

        n_prompts = len(STANDARD_PROMPTS)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["prompt_count"] == n_prompts
        assert manifest["observations"]["total"] == n_prompts * 2

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_unknown_model_raises(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")
        runner = GuidedSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            runner.run(prompts="1", models="NotAModel", top_k_pools="5", no_plot=True)
        mock_get_wrapper.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_no_plot_true_skips_plot_run(
        self, mock_get_wrapper, mock_plot_run, tmp_path: Path
    ):
        mock_plot_run.return_value = tmp_path / "plots"
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", top_k_pools="5", no_plot=True)

        mock_plot_run.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_no_plot_false_calls_plot_run(
        self, mock_get_wrapper, mock_plot_run, tmp_path: Path
    ):
        mock_plot_run.return_value = tmp_path / "plots"
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", top_k_pools="5", no_plot=False)

        mock_plot_run.assert_called_once()
        assert mock_plot_run.call_args.kwargs["results_root"] == tmp_path

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_baseline_stamped_guided_metadata(self, mock_get_wrapper, tmp_path: Path):
        wrapper = MockGuidedModelWrapper("Qwen3")
        mock_get_wrapper.return_value = wrapper

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1", models="Qwen3", top_k_pools="0", no_plot=True
        )

        assert wrapper.generate_calls == 1
        assert wrapper.generate_guided_calls == 0

        full = pd.read_csv(out_dir / "full.csv")
        assert len(full) == 1
        row = full.iloc[0]
        assert int(row["guided_top_k"]) == 0
        assert int(row["guided_steps_a1_chosen"]) == 0
        assert int(row["guided_steps_total"]) == 0
        assert float(row["guided_intervention_rate"]) == 0.0
        assert row["guided_mode"] == "flat"

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_summary_sweep_buckets_and_rates(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockGuidedModelWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2", models="Qwen3", top_k_pools="0,5", no_plot=True
        )

        summary = json.loads((out_dir / "summary.json").read_text())
        by_pool = summary["by_guided_top_k"]
        assert by_pool["0"]["count"] == 2
        assert by_pool["5"]["count"] == 2
        for pool_stats in by_pool.values():
            assert pool_stats["a1_pass_rate"] == pytest.approx(1.0)
            assert pool_stats["generation_failure_rate"] == pytest.approx(0.0)
            assert pool_stats["hit_max_tokens_rate"] == pytest.approx(0.0)

        assert "Qwen3" in summary["by_model"]
        qwen = summary["by_model"]["Qwen3"]
        assert qwen["count"] == 4
        assert qwen["a1_pass_rate"] == pytest.approx(1.0)
        assert qwen["generation_failure_rate"] == pytest.approx(0.0)
        assert qwen["hit_max_tokens_rate"] == pytest.approx(0.0)
        assert qwen["by_guided_top_k"]["0"]["count"] == 2
        assert qwen["by_guided_top_k"]["5"]["count"] == 2

        overall = summary["metadata"]
        assert overall["a1_pass_rate"] == pytest.approx(1.0)
        assert overall["generation_failure_rate"] == pytest.approx(0.0)
        assert overall["hit_max_tokens_rate"] == pytest.approx(0.0)

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_generation_failure_recorded(self, mock_get_wrapper, tmp_path: Path):
        class FailingWrapper(MockGuidedModelWrapper):
            def generate_guided(self, prompt, config):
                self.generate_guided_calls += 1
                return {
                    "response": "",
                    "response_time_seconds": 0.4,
                    "generation_successful": False,
                    "guided_top_k": config.guided_top_k,
                    "guided_mode": config.guided_mode,
                    "guided_steps_a1_chosen": 0,
                    "guided_steps_total": 0,
                    "guided_intervention_rate": 0.0,
                }

        mock_get_wrapper.return_value = FailingWrapper("Qwen3")

        runner = GuidedSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1", models="Qwen3", top_k_pools="10", no_plot=True
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 1

        full = pd.read_csv(out_dir / "full.csv")
        assert len(full) == 1
        assert bool(full.iloc[0]["generation_successful"]) is False

        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["metadata"]["failed_experiments"] == 1
        assert summary["metadata"]["generation_failure_rate"] == pytest.approx(1.0)
        assert summary["metadata"]["a1_pass_rate"] == pytest.approx(0.0)

    @patch("slm_experiments.phase2.guided.get_model_wrapper")
    def test_wrapper_cleanup_called(self, mock_get_wrapper, tmp_path: Path):
        wrappers = []

        def factory(name, seed=42, **kwargs):
            wrapper = MockGuidedModelWrapper(name, seed=seed)
            wrappers.append(wrapper)
            return wrapper

        mock_get_wrapper.side_effect = factory

        runner = GuidedSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="all", top_k_pools="5", no_plot=True)

        assert len(wrappers) == len(MODEL_CONFIGS)
        assert all(w.cleanup_calls == 1 for w in wrappers)
