"""Tests for Phase 2 beam width sweep configs and runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS
from slm_experiments.phase2.beam import (
    DEFAULT_BEAM_WIDTH_GRID,
    BeamSweepRunner,
    beam_width_from_config,
    create_beam_configs,
    parse_widths,
)

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockBeamModelWrapper:
    def __init__(self, model_name: str, seed: int = 42, **kwargs):
        self.model_name = model_name
        self.seed = seed
        self.cleanup_calls = 0
        self.generate_beam_calls = 0

    def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
        self.generate_beam_calls += 1
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.5,
            "generation_successful": True,
            "beam_selection_method": selection_method,
            "beam_a1_ratio": 0.75,
            "beam_a1_count": 3,
            "beam_content_word_count": 6,
            "beam_cumulative_logprob": -1.2,
            "beam_width": beam_width,
        }

    def cleanup(self):
        self.cleanup_calls += 1


class TestBeamConfigs:
    def test_create_beam_configs_count(self):
        configs = create_beam_configs(DEFAULT_BEAM_WIDTH_GRID)
        assert len(configs) == len(MODEL_CONFIGS) * len(DEFAULT_BEAM_WIDTH_GRID)

    def test_beam_configs_prompting_on_weighting_off(self):
        configs = create_beam_configs([4, 8])
        for config in configs:
            assert config.config_weighting is False
            assert config.config_prompting is True
            assert config.num_shots == 0
            assert config.temperature == 0.0
            assert config.top_k == 50
            assert config.config_guided is False
            assert config.config_kvl_beam is False

    def test_beam_width_encoded_in_experiment_name(self):
        configs = create_beam_configs([4, 10])
        for config in configs:
            width = beam_width_from_config(config)
            assert width in (4, 10)
            assert config.experiment_name.endswith(f"_beam_w{width}")

    def test_parse_widths(self):
        assert parse_widths("4,8,10") == [4, 8, 10]

    def test_parse_widths_invalid(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_widths("0,4")

    def test_parse_widths_empty_or_whitespace(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_widths("")
        with pytest.raises(ValueError, match="must not be empty"):
            parse_widths("   ")
        with pytest.raises(ValueError, match="at least one value"):
            parse_widths(",")

    def test_parse_widths_trailing_comma(self):
        assert parse_widths("4,8,") == [4, 8]
        assert parse_widths(" 4 , 10 , ") == [4, 10]


class TestBeamSweepRunner:
    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_default_run_produces_36_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockBeamModelWrapper(
            name, seed=seed
        )

        runner = BeamSweepRunner(results_root=tmp_path)
        run_id, out_dir = runner.run(
            prompts="3",
            models="all",
            seed=42,
            cli_args=["--prompts", "3"],
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 2
        assert manifest["experiment"] == "beam"
        assert manifest["prompt_count"] == 3
        assert manifest["observations"]["total"] == 36
        assert manifest["models"] == list(MODEL_CONFIGS.keys())

        full = (out_dir / "full.csv").read_text()
        assert full.count("\n") == 37  # header + 36 rows
        assert "beam_a1_ratio" in full
        assert "beam_width" in full

        assert mock_get_wrapper.call_count == len(MODEL_CONFIGS)

    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_single_model_subset(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockBeamModelWrapper("Qwen3")

        runner = BeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Qwen3",
            widths="4,8",
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 4  # 2 widths × 2 prompts
        mock_get_wrapper.assert_called_once_with("Qwen3", seed=42)

    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_beam_metadata_in_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockBeamModelWrapper("Qwen3")

        runner = BeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(prompts="1", models="Qwen3", widths="8")

        full = (out_dir / "full.csv").read_text()
        assert ",0.75," in full or ",0,75," in full  # beam_a1_ratio column
        assert ",8," in full  # beam_width

    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_configs_passed_to_pipeline(self, mock_get_wrapper, tmp_path: Path):
        captured = []

        class CapturingWrapper(MockBeamModelWrapper):
            def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
                captured.append((config, beam_width))
                return super().generate_beam(
                    prompt, config, beam_width=beam_width, selection_method=selection_method
                )

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = BeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="10")

        assert len(captured) == 1
        config, width = captured[0]
        assert config.config_weighting is False
        assert config.config_prompting is True
        assert config.prompt_id == "P1"
        assert config.experiment_name == "Qwen3_beam_w10"
        assert width == 10

    def test_beam_width_from_config_invalid(self):
        config = ExperimentConfig(experiment_name="invalid_name")
        with pytest.raises(ValueError, match="Cannot parse beam width"):
            beam_width_from_config(config)

    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_unknown_model_raises(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockBeamModelWrapper("Qwen3")
        runner = BeamSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            runner.run(prompts="1", models="NotAModel", widths="4", no_plot=True)
        mock_get_wrapper.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_no_plot_true_skips_plot_run(
        self, mock_get_wrapper, mock_plot_run, tmp_path: Path
    ):
        mock_plot_run.return_value = tmp_path / "plots"
        mock_get_wrapper.return_value = MockBeamModelWrapper("Qwen3")

        runner = BeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="4", no_plot=True)

        mock_plot_run.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_no_plot_false_calls_plot_run(
        self, mock_get_wrapper, mock_plot_run, tmp_path: Path
    ):
        mock_plot_run.return_value = tmp_path / "plots"
        mock_get_wrapper.return_value = MockBeamModelWrapper("Qwen3")

        runner = BeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="4", no_plot=False)

        mock_plot_run.assert_called_once()
        assert mock_plot_run.call_args.kwargs["results_root"] == tmp_path

    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_wrapper_cleanup_called(self, mock_get_wrapper, tmp_path: Path):
        wrappers = []

        def factory(name, seed=42, **kwargs):
            wrapper = MockBeamModelWrapper(name, seed=seed)
            wrappers.append(wrapper)
            return wrapper

        mock_get_wrapper.side_effect = factory

        runner = BeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="all", widths="4", no_plot=True)

        assert len(wrappers) == len(MODEL_CONFIGS)
        assert all(w.cleanup_calls == 1 for w in wrappers)

    @patch("slm_experiments.phase2.beam.get_model_wrapper")
    def test_failed_generation_still_in_bundle(self, mock_get_wrapper, tmp_path: Path):
        class FailingWrapper(MockBeamModelWrapper):
            def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
                self.generate_beam_calls += 1
                return {
                    "response": "",
                    "response_time_seconds": 0.4,
                    "generation_successful": False,
                    "beam_selection_method": selection_method,
                    "beam_a1_ratio": 0.0,
                    "beam_a1_count": 0,
                    "beam_content_word_count": 0,
                    "beam_cumulative_logprob": 0.0,
                    "beam_width": beam_width,
                }

        mock_get_wrapper.return_value = FailingWrapper("Qwen3")

        runner = BeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1", models="Qwen3", widths="4", no_plot=True
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 1

        full = pd.read_csv(out_dir / "full.csv")
        assert len(full) == 1
        assert bool(full.iloc[0]["generation_successful"]) is False

        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["metadata"]["failed_experiments"] == 1
        assert summary["metadata"]["generation_failure_rate"] == pytest.approx(1.0)
