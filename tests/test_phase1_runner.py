"""Tests for Phase 1 factorial configs and runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS, STANDARD_PROMPTS
from slm_experiments.phase1.configs import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_WEIGHT_FACTOR,
    create_factorial_configs,
    get_config_by_name,
    get_configs_for_model,
)
from slm_experiments.phase1.runner import FactorialRunner, parse_models, parse_prompts

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockModelWrapper:
    """Returns success for most prompts; fails on the third prompt."""

    def __init__(self, model_name: str, seed: int = 42, **kwargs):
        self.model_name = model_name
        self.seed = seed
        self.generate_calls = 0

    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        self.generate_calls += 1
        if prompt == STANDARD_PROMPTS[2]:
            return {
                "response": "",
                "response_time_seconds": 0.4,
                "generation_successful": False,
            }
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
        }

    def cleanup(self):
        pass


class TestFactorialConfigs:
    def test_create_factorial_configs_count(self):
        configs = create_factorial_configs()
        assert len(configs) == len(MODEL_CONFIGS) * 4

    def test_intervention_variants_per_model(self):
        for model_name in MODEL_CONFIGS:
            model_configs = get_configs_for_model(model_name)
            assert len(model_configs) == 4
            combos = {(c.config_weighting, c.config_prompting) for c in model_configs}
            assert combos == {(False, False), (True, False), (False, True), (True, True)}

    def test_default_system_prompt_and_weight(self):
        configs = create_factorial_configs()
        for config in configs:
            assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
            assert config.weight_factor == DEFAULT_WEIGHT_FACTOR

    def test_get_config_by_name(self):
        config = get_config_by_name("Qwen3_control")
        assert config.model_name == "Qwen3"
        assert config.config_weighting is False
        assert config.config_prompting is False

    def test_get_config_by_name_unknown(self):
        with pytest.raises(ValueError, match="not found"):
            get_config_by_name("Missing_config")


class TestFactorialRunner:
    @patch("slm_experiments.phase1.runner.get_model_wrapper")
    def test_default_run_produces_48_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockModelWrapper(
            name, seed=seed
        )

        runner = FactorialRunner(results_root=tmp_path)
        run_id, out_dir = runner.run(
            prompts="3",
            models="all",
            seed=42,
            cli_args=["--prompts", "3"],
        )

        assert len(list(out_dir.glob("*"))) >= 4
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 1
        assert manifest["experiment"] == "factorial"
        assert manifest["prompt_count"] == 3
        assert manifest["observations"]["total"] == 48
        assert manifest["models"] == list(MODEL_CONFIGS.keys())
        assert manifest["cli_args"] == ["--prompts", "3"]
        assert set(manifest["artifacts"]) == {
            "specification_csv",
            "full_csv",
            "summary_json",
        }

        full = (out_dir / "full.csv").read_text()
        assert full.count("\n") == 49  # header + 48 rows

        assert mock_get_wrapper.call_count == len(MODEL_CONFIGS)
        for call in mock_get_wrapper.call_args_list:
            assert call.kwargs.get("seed") == 42

    @patch("slm_experiments.phase1.runner.get_model_wrapper")
    def test_failed_generations_excluded_from_summary_means(
        self, mock_get_wrapper, tmp_path: Path
    ):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockModelWrapper(
            name, seed=seed
        )

        runner = FactorialRunner(results_root=tmp_path)
        _, out_dir = runner.run(prompts="3", models="Qwen3")

        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["metadata"]["total_experiments"] == 12
        assert summary["metadata"]["failed_experiments"] == 4
        assert summary["metadata"]["successful_experiments"] == 8

        wc_mean = summary["overall"]["word_count"]["mean"]
        assert wc_mean > 0

        full_rows = (out_dir / "full.csv").read_text().splitlines()
        failed_rows = [line for line in full_rows if ",False," in line or line.endswith(",False")]
        assert len(failed_rows) >= 4

        successful_wc = wc_mean
        assert successful_wc > 10

    @patch("slm_experiments.phase1.runner.get_model_wrapper")
    def test_single_model_subset(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockModelWrapper("Qwen3")

        runner = FactorialRunner(results_root=tmp_path)
        _, out_dir = runner.run(prompts="2", models="Qwen3")

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 8
        mock_get_wrapper.assert_called_once_with("Qwen3", seed=42)

    def test_parse_prompts_all(self):
        assert parse_prompts("all") == STANDARD_PROMPTS

    def test_parse_prompts_count(self):
        assert parse_prompts("3") == STANDARD_PROMPTS[:3]

    def test_parse_models_all(self):
        assert parse_models("all") == list(MODEL_CONFIGS.keys())

    def test_parse_models_subset(self):
        assert parse_models("Qwen3,Phi3") == ["Qwen3", "Phi3"]

    def test_parse_models_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            parse_models("NotAModel")
