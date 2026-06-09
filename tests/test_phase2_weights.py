"""Tests for Phase 2 weight sweep configs and runner."""

import json
from pathlib import Path
from unittest.mock import patch

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
