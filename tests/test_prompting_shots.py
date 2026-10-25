"""Tests for prompting shot templates and Phase 2 prompting sweep."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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


class TestParseShotsEdgeCases:
    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    def test_parse_shots_empty_or_whitespace_raises(self, raw):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_shots(raw)

    def test_parse_shots_trailing_comma(self):
        assert parse_shots("0,1,") == [0, 1]

    def test_parse_shots_skips_blank_entries(self):
        assert parse_shots("0,,3") == [0, 3]

    def test_parse_shots_negative_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            parse_shots("-1,3")

    def test_parse_shots_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_shots("0,abc")

    def test_parse_shots_zero_shot_baseline(self):
        assert parse_shots("0") == [0]
        assert 0 in DEFAULT_SHOT_GRID


class TestPromptingConfigDetails:
    def test_create_prompting_configs_defaults_and_names(self):
        assert DEFAULT_SHOT_GRID == [0, 1, 3]
        configs = create_prompting_configs([0, 1, 3])
        by_shots = {
            c.num_shots: c for c in configs if c.model_name == "Qwen3"
        }
        assert set(by_shots) == {0, 1, 3}
        for num_shots, config in by_shots.items():
            assert config.experiment_name == f"Qwen3_prompting_{num_shots}shot"
            assert config.temperature == 0.0
            assert config.top_k == 50
            assert config.max_new_tokens == 200
            assert config.weight_factor == 1.0
            assert config.config_prompting is True
            assert config.config_weighting is False
            assert config.config_guided is False
            assert config.config_kvl_beam is False

    def test_prompting_configs_kvl_guided_beam_flags_off(self):
        configs = create_prompting_configs(DEFAULT_SHOT_GRID)
        for config in configs:
            assert config.config_guided is False
            assert config.config_kvl_beam is False


class TestPromptingSweepRunnerGaps:
    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_prompts_all_tiny_grid_observation_count(
        self, mock_get_wrapper, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Phi3")

        runner = PromptingSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="all",
            models="Phi3",
            shots="0",
            no_plot=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert len(STANDARD_PROMPTS) == 25
        assert manifest["prompt_count"] == 25
        assert manifest["observations"]["total"] == 25  # 1 model × 1 shot × 25 prompts

    def test_unknown_model_raises(self, tmp_path: Path):
        runner = PromptingSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            runner.run(prompts="1", models="NotAModel", shots="0", no_plot=True)

    def test_prompt_count_zero_raises(self, tmp_path: Path):
        runner = PromptingSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="prompt count must be >= 1"):
            runner.run(prompts="0", models="Phi3", shots="0", no_plot=True)

    def test_prompt_count_invalid_raises(self, tmp_path: Path):
        runner = PromptingSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError):
            runner.run(prompts="nope", models="Phi3", shots="0", no_plot=True)

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_no_plot_true_skips_plot(
        self, mock_get_wrapper, mock_plot, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Phi3")

        runner = PromptingSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Phi3", shots="0", no_plot=True)

        mock_plot.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_no_plot_false_calls_plot_run(
        self, mock_get_wrapper, mock_plot, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Phi3")
        mock_plot.return_value = Path("/tmp/plots")

        runner = PromptingSweepRunner(results_root=tmp_path)
        run_id, _ = runner.run(
            prompts="1", models="Phi3", shots="0", no_plot=False
        )

        mock_plot.assert_called_once_with(run_id, results_root=tmp_path)

    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_enable_cefr_sp_false_stamped_on_configs(
        self, mock_get_wrapper, tmp_path: Path
    ):
        captured = []

        class CapturingWrapper(MockModelWrapper):
            def generate(self, prompt, config):
                captured.append(config)
                return super().generate(prompt, config)

        mock_get_wrapper.return_value = CapturingWrapper("Phi3")

        runner = PromptingSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1",
            models="Phi3",
            shots="0",
            no_plot=True,
            enable_cefr_sp=False,
        )

        assert captured
        for config in captured:
            assert config.enable_cefr_sp is False

    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
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
                    "response_time_seconds": 0.8,
                    "generation_successful": True,
                    "hit_max_tokens": True,
                }

        mock_get_wrapper.return_value = FailingWrapper("Phi3")

        runner = PromptingSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Phi3",
            shots="0,1",
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

    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_summary_has_rates_and_sweep_buckets(
        self, mock_get_wrapper, tmp_path: Path
    ):
        mock_get_wrapper.return_value = MockModelWrapper("Phi3")

        runner = PromptingSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Phi3",
            shots="0,1",
            no_plot=True,
        )

        summary = json.loads((out_dir / "summary.json").read_text())
        for key in (
            "a1_pass_rate",
            "generation_failure_rate",
            "hit_max_tokens_rate",
        ):
            assert key in summary["metadata"]

        assert "by_num_shots" in summary
        assert set(summary["by_num_shots"]) == {"0", "1"}
        for bucket in summary["by_num_shots"].values():
            assert "a1_pass_rate" in bucket
            assert "generation_failure_rate" in bucket
            assert "hit_max_tokens_rate" in bucket
            assert bucket["count"] == 2

        assert "by_model" in summary
        assert "Phi3" in summary["by_model"]
        model_stats = summary["by_model"]["Phi3"]
        assert "a1_pass_rate" in model_stats
        assert "generation_failure_rate" in model_stats
        assert "hit_max_tokens_rate" in model_stats
        assert "by_num_shots" in model_stats

    @patch("slm_experiments.phase2.prompting.get_model_wrapper")
    def test_wrappers_cleaned_up(self, mock_get_wrapper, tmp_path: Path):
        wrappers = []

        def factory(name, seed=42, **kwargs):
            wrapper = MockModelWrapper(name, seed=seed)
            wrapper.cleanup = MagicMock()
            wrappers.append(wrapper)
            return wrapper

        mock_get_wrapper.side_effect = factory

        runner = PromptingSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1",
            models="Qwen3,Phi3",
            shots="0",
            no_plot=True,
        )

        assert len(wrappers) == 2
        for wrapper in wrappers:
            wrapper.cleanup.assert_called_once()
