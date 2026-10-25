"""Tests for Phase 2 weight sweep configs and runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from slm_experiments.core.prompts import MODEL_CONFIGS, STANDARD_PROMPTS
from slm_experiments.phase2.weights import (
    DEFAULT_WEIGHT_GRID,
    WeightSweepRunner,
    create_weight_configs,
    parse_weights,
)

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockModelWrapper:
    def __init__(self, model_name: str, seed: int = 42, **kwargs):
        self.model_name = model_name
        self.seed = seed

    def generate(self, prompt, config):
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
        }

    def cleanup(self):
        pass


class TestWeightConfigs:
    def test_create_weight_configs_count(self):
        configs = create_weight_configs(DEFAULT_WEIGHT_GRID)
        assert len(configs) == len(MODEL_CONFIGS) * len(DEFAULT_WEIGHT_GRID)

    def test_weight_configs_have_both_interventions_on(self):
        configs = create_weight_configs([1.5, 2.0])
        for config in configs:
            assert config.config_weighting is True
            assert config.config_prompting is True
            assert config.num_shots == 0

    def test_weight_factors_applied(self):
        configs = create_weight_configs([1.5, 4.0])
        factors = sorted({c.weight_factor for c in configs})
        assert factors == [1.5, 4.0]

    def test_parse_weights(self):
        assert parse_weights("1.0,1.5,2.0") == [1.0, 1.5, 2.0]

    def test_parse_weights_invalid(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_weights("0,1.5")


class TestWeightSweepRunner:
    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_default_run_produces_84_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockModelWrapper(
            name, seed=seed
        )

        runner = WeightSweepRunner(results_root=tmp_path)
        run_id, out_dir = runner.run(
            prompts="3",
            models="all",
            seed=42,
            cli_args=["--prompts", "3"],
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 2
        assert manifest["experiment"] == "weights"
        assert manifest["prompt_count"] == 3
        assert manifest["observations"]["total"] == 84
        assert manifest["models"] == list(MODEL_CONFIGS.keys())

        full = (out_dir / "full.csv").read_text()
        assert full.count("\n") == 85  # header + 84 rows

        assert mock_get_wrapper.call_count == len(MODEL_CONFIGS)

    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_single_model_subset(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockModelWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Qwen3",
            weights="1.5,2.0",
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 4  # 2 weights × 2 prompts
        mock_get_wrapper.assert_called_once_with("Qwen3", seed=42)

    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_configs_passed_to_pipeline(self, mock_get_wrapper, tmp_path: Path):
        captured_configs = []

        class CapturingWrapper(MockModelWrapper):
            def generate(self, prompt, config):
                captured_configs.append(config)
                return super().generate(prompt, config)

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", weights="1.5")

        assert len(captured_configs) == 1
        assert captured_configs[0].config_weighting is True
        assert captured_configs[0].config_prompting is True
        assert captured_configs[0].weight_factor == 1.5
        assert captured_configs[0].prompt_id == "P1"
        assert captured_configs[0].experiment_name == "Qwen3_weighted_prompted_1_5"


class TestParseWeightsEdgeCases:
    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    def test_parse_weights_empty_or_whitespace_raises(self, raw):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_weights(raw)

    def test_parse_weights_trailing_comma(self):
        assert parse_weights("1.0,1.5,") == [1.0, 1.5]

    def test_parse_weights_skips_blank_entries(self):
        assert parse_weights("1.0,,2.0") == [1.0, 2.0]

    def test_parse_weights_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_weights("1.0,abc")

    def test_parse_weights_identity_baseline(self):
        assert parse_weights("1.0") == [1.0]
        assert 1.0 in DEFAULT_WEIGHT_GRID


class TestWeightConfigDetails:
    def test_create_weight_configs_identity_weighting_on(self):
        configs = create_weight_configs([1.0])
        assert configs
        for config in configs:
            assert config.weight_factor == 1.0
            assert config.config_weighting is True
            assert config.config_prompting is True
            assert config.temperature == 0.0
            assert config.top_k == 50
            assert config.max_new_tokens == 200

    def test_weight_experiment_name_uses_underscore_for_decimal(self):
        configs = create_weight_configs([1.5, 2.0])
        names = {c.experiment_name for c in configs if c.model_name == "Qwen3"}
        assert "Qwen3_weighted_prompted_1_5" in names
        assert "Qwen3_weighted_prompted_2_0" in names

    def test_weight_configs_kvl_guided_beam_flags_off(self):
        configs = create_weight_configs(DEFAULT_WEIGHT_GRID)
        for config in configs:
            assert config.config_guided is False
            assert config.config_kvl_beam is False


class TestWeightSweepRunnerGaps:
    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_prompts_all_tiny_grid_observation_count(
        self, mock_get_wrapper, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="all",
            models="Qwen3",
            weights="1.0",
            no_plot=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert len(STANDARD_PROMPTS) == 25
        assert manifest["prompt_count"] == 25
        assert manifest["observations"]["total"] == 25  # 1 model × 1 weight × 25 prompts

    def test_unknown_model_raises(self, tmp_path: Path):
        runner = WeightSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            runner.run(prompts="1", models="NotAModel", weights="1.0", no_plot=True)

    def test_prompt_count_zero_raises(self, tmp_path: Path):
        runner = WeightSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="prompt count must be >= 1"):
            runner.run(prompts="0", models="Qwen3", weights="1.0", no_plot=True)

    def test_prompt_count_invalid_raises(self, tmp_path: Path):
        runner = WeightSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError):
            runner.run(prompts="nope", models="Qwen3", weights="1.0", no_plot=True)

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_no_plot_true_skips_plot(
        self, mock_get_wrapper, mock_plot, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1",
            models="Qwen3",
            weights="1.0",
            no_plot=True,
        )

        mock_plot.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_no_plot_false_calls_plot_run(
        self, mock_get_wrapper, mock_plot, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Qwen3")
        mock_plot.return_value = Path("/tmp/plots")

        runner = WeightSweepRunner(results_root=tmp_path)
        run_id, _ = runner.run(
            prompts="1",
            models="Qwen3",
            weights="1.0",
            no_plot=False,
        )

        mock_plot.assert_called_once_with(run_id, results_root=tmp_path)

    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_enable_cefr_sp_false_stamped_on_configs(
        self, mock_get_wrapper, tmp_path: Path
    ):
        captured = []

        class CapturingWrapper(MockModelWrapper):
            def generate(self, prompt, config):
                captured.append(config)
                return super().generate(prompt, config)

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1",
            models="Qwen3",
            weights="1.0",
            no_plot=True,
            enable_cefr_sp=False,
        )

        assert captured
        for config in captured:
            assert config.enable_cefr_sp is False

    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_generation_failure_recorded_rates_use_all_rows(
        self, mock_get_wrapper, tmp_path: Path
    ):
        class FailingWrapper(MockModelWrapper):
            def generate(self, prompt, config):
                if prompt == STANDARD_PROMPTS[0]:
                    return {
                        "response": "",
                        "response_time_seconds": 0.2,
                        "generation_successful": False,
                        "hit_max_tokens": False,
                    }
                return {
                    "response": SIMPLE_RESPONSE,
                    "response_time_seconds": 1.0,
                    "generation_successful": True,
                    "hit_max_tokens": True,
                }

        mock_get_wrapper.return_value = FailingWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Qwen3",
            weights="1.0,1.5",
            no_plot=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 4
        assert manifest["observations"]["failed"] == 2
        assert manifest["observations"]["successful"] == 2

        full = pd.read_csv(out_dir / "full.csv")
        spec = pd.read_csv(out_dir / "specification.csv")
        assert len(full) == 4
        assert int((~full["generation_successful"]).sum()) == 2
        assert int((~spec["generation_successful"]).sum()) == 2

        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["metadata"]["generation_failure_rate"] == pytest.approx(0.5)
        assert summary["metadata"]["hit_max_tokens_rate"] == pytest.approx(0.5)
        assert summary["metadata"]["a1_pass_rate"] == pytest.approx(0.5)

        successful = full[full["generation_successful"]]
        assert summary["overall"]["word_count"]["mean"] == pytest.approx(
            float(successful["word_count"].mean())
        )
        assert summary["overall"]["word_count"]["mean"] != pytest.approx(
            float(full["word_count"].mean())
        )

    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_summary_has_rates_and_sweep_buckets(
        self, mock_get_wrapper, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Qwen3")

        runner = WeightSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Qwen3",
            weights="1.0,1.5",
            no_plot=True,
        )

        summary = json.loads((out_dir / "summary.json").read_text())
        for key in (
            "a1_pass_rate",
            "generation_failure_rate",
            "hit_max_tokens_rate",
        ):
            assert key in summary["metadata"]

        assert "by_weight_factor" in summary
        assert set(summary["by_weight_factor"]) == {"1", "1.5"}
        for bucket in summary["by_weight_factor"].values():
            assert "a1_pass_rate" in bucket
            assert "generation_failure_rate" in bucket
            assert "hit_max_tokens_rate" in bucket
            assert bucket["count"] == 2

        assert "by_model" in summary
        assert "Qwen3" in summary["by_model"]
        model_stats = summary["by_model"]["Qwen3"]
        assert "a1_pass_rate" in model_stats
        assert "generation_failure_rate" in model_stats
        assert "hit_max_tokens_rate" in model_stats
        assert "by_weight_factor" in model_stats

    @patch("slm_experiments.phase2.weights.get_model_wrapper")
    def test_wrappers_cleaned_up(self, mock_get_wrapper, tmp_path: Path):
        wrappers = []

        def factory(name, seed=42, **kwargs):
            wrapper = MockModelWrapper(name, seed=seed)
            wrapper.cleanup = MagicMock()
            wrappers.append(wrapper)
            return wrapper

        mock_get_wrapper.side_effect = factory

        runner = WeightSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1",
            models="Qwen3,Phi3",
            weights="1.0",
            no_plot=True,
        )

        assert len(wrappers) == 2
        for wrapper in wrappers:
            wrapper.cleanup.assert_called_once()
