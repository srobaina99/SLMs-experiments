"""Tests for prompting shot templates and Phase 2 prompting sweep."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from slm_experiments.core.prompts import (
    MODEL_CONFIGS,
    SHOT_EXAMPLES,
    STANDARD_PROMPTS,
    build_contextual_prompt,
)
from slm_experiments.phase2.prompting import (
    DEFAULT_SHOT_GRID,
    PromptingSweepRunner,
    create_prompting_configs,
    parse_shots,
)

SIMPLE_RESPONSE = "A cat is a small animal. It is soft."


class MockModelWrapper:
    def __init__(self, model_name: str, seed: int = 42, **kwargs):
        self.model_name = model_name
        self.seed = seed

    def generate(self, prompt, config):
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 0.8,
            "generation_successful": True,
        }

    def cleanup(self):
        pass


class TestBuildContextualPrompt:
    def test_zero_shot_has_context_only(self):
        prompt = build_contextual_prompt("What is a dog?", num_shots=0)
        assert "# Context" in prompt
        assert "What is a dog?" in prompt
        assert "# Example" not in prompt
        assert "# Examples" not in prompt

    def test_one_shot_includes_single_example(self):
        prompt = build_contextual_prompt("What is a dog?", num_shots=1)
        assert "# Example" in prompt
        assert SHOT_EXAMPLES[0] in prompt
        assert SHOT_EXAMPLES[1] not in prompt
        assert prompt.endswith("What is a dog?")

    def test_few_shot_includes_three_examples(self):
        prompt = build_contextual_prompt("What is a dog?", num_shots=3)
        assert "# Examples" in prompt
        for example in SHOT_EXAMPLES:
            assert example in prompt
        assert prompt.endswith("What is a dog?")

    def test_invalid_shot_count(self):
        with pytest.raises(ValueError, match="exceeds available"):
            build_contextual_prompt("Hi", num_shots=99)

    def test_shot_examples_do_not_overlap_evaluation_prompts(self):
        shot_questions = {
            example.split("\n", maxsplit=1)[0].removeprefix("Question: ")
            for example in SHOT_EXAMPLES
        }
        overlap = shot_questions & set(STANDARD_PROMPTS)
        assert not overlap, f"Few-shot examples must not match evaluation prompts: {overlap}"


class TestPromptingConfigs:
    def test_create_prompting_configs_count(self):
        configs = create_prompting_configs(DEFAULT_SHOT_GRID)
        assert len(configs) == len(MODEL_CONFIGS) * len(DEFAULT_SHOT_GRID)

    def test_prompting_configs_weighting_off(self):
        configs = create_prompting_configs([0, 1, 3])
        for config in configs:
            assert config.config_weighting is False
            assert config.config_prompting is True

    def test_shot_levels_applied(self):
        configs = create_prompting_configs([0, 3])
        shot_levels = sorted({c.num_shots for c in configs})
        assert shot_levels == [0, 3]

    def test_parse_shots(self):
        assert parse_shots("0,1,3") == [0, 1, 3]


class TestPromptingSweepRunner:
    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_default_run_produces_36_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockModelWrapper(
            name, seed=seed
        )

        runner = PromptingSweepRunner(results_root=tmp_path)
        run_id, out_dir = runner.run(
            prompts="3",
            models="all",
            seed=42,
            cli_args=["--prompts", "3"],
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 2
        assert manifest["experiment"] == "prompting"
        assert manifest["prompt_count"] == 3
        assert manifest["observations"]["total"] == 36
        assert manifest["models"] == list(MODEL_CONFIGS.keys())

        full = (out_dir / "full.csv").read_text()
        assert full.count("\n") == 37  # header + 36 rows

    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_single_model_subset(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockModelWrapper("Phi3")

        runner = PromptingSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(prompts="2", models="Phi3", shots="0,1")

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 4  # 2 shots × 2 prompts
        mock_get_wrapper.assert_called_once_with("Phi3", seed=42)
