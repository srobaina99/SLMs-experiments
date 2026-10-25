"""Tests for Phase 2 KVL beam width sweep configs and runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS, STANDARD_PROMPTS
from slm_experiments.phase2.kvl_beam import (
    DEFAULT_KVL_BEAM_WIDTH_GRID,
    KvlBeamSweepRunner,
    create_kvl_beam_configs,
    kvl_beam_width_from_config,
    parse_widths,
)

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockKvlBeamModelWrapper:
    def __init__(self, model_name: str, seed: int = 42, **kwargs):
        self.model_name = model_name
        self.seed = seed
        self.kwargs = kwargs
        self.generate_calls = 0
        self.generate_kvl_beam_calls = 0
        self.cleanup_calls = 0

    def generate(self, prompt, config):
        self.generate_calls += 1
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
        }

    def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
        self.generate_kvl_beam_calls += 1
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 2.5,
            "generation_successful": True,
            "kvl_beam_width": beam_width,
            "kvl_branch_factor": branch_factor,
            "kvl_beam_steps_total": 10,
            "kvl_beam_words_scored": 5,
            "kvl_beam_running_mean": 2.0,
            "kvl_beam_logprob_tiebreak": -1.0,
            "kvl_beam_candidates_pruned": 20,
        }

    def cleanup(self):
        self.cleanup_calls += 1


class TestKvlBeamConfigs:
    def test_create_kvl_beam_configs_count(self):
        configs = create_kvl_beam_configs(DEFAULT_KVL_BEAM_WIDTH_GRID)
        assert DEFAULT_KVL_BEAM_WIDTH_GRID == [1, 4, 6, 8]
        assert len(configs) == len(MODEL_CONFIGS) * len(DEFAULT_KVL_BEAM_WIDTH_GRID)

    def test_kvl_beam_configs_isolation_flags(self):
        configs = create_kvl_beam_configs([4, 8], branch_factor=10, kvl_l1="es")
        for config in configs:
            assert config.config_kvl_beam is True
            assert config.config_weighting is False
            assert config.config_prompting is True
            assert config.config_guided is False
            assert config.num_shots == 0
            assert config.temperature == 0.0
            assert config.top_k == 50
            assert config.kvl_branch_factor == 10
            assert config.kvl_l1 == "es"

    def test_kvl_beam_baseline_width_one(self):
        configs = create_kvl_beam_configs([1, 4])
        by_width = {c.kvl_beam_width: c for c in configs if c.model_name == "Qwen3"}
        assert by_width[1].config_kvl_beam is False
        assert by_width[1].config_prompting is True
        assert by_width[1].config_weighting is False
        assert by_width[1].experiment_name.endswith("_kvl_beam_w1")
        assert by_width[4].config_kvl_beam is True

    def test_kvl_beam_width_encoded_in_experiment_name(self):
        configs = create_kvl_beam_configs([4, 8])
        for config in configs:
            width = kvl_beam_width_from_config(config)
            assert width in (4, 8)
            assert config.experiment_name.endswith(f"_kvl_beam_w{width}")
            assert config.kvl_beam_width == width

    def test_create_kvl_beam_configs_custom_l1(self):
        configs = create_kvl_beam_configs([4], branch_factor=5, kvl_l1="de")
        assert len(configs) == len(MODEL_CONFIGS)
        for config in configs:
            assert config.kvl_l1 == "de"
            assert config.kvl_branch_factor == 5

    def test_parse_widths(self):
        assert parse_widths("1,4,6,8") == [1, 4, 6, 8]
        assert parse_widths("4,8") == [4, 8]

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
        with pytest.raises(ValueError, match="at least one value"):
            parse_widths(" , , ")

    def test_parse_widths_trailing_comma(self):
        assert parse_widths("1,4,8,") == [1, 4, 8]
        assert parse_widths(" 4 , 8 , ") == [4, 8]

    def test_kvl_beam_width_from_config_invalid(self):
        config = ExperimentConfig(experiment_name="invalid_name")
        with pytest.raises(ValueError, match="Cannot parse KVL beam width"):
            kvl_beam_width_from_config(config)


class TestKvlBeamSweepRunner:
    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_default_run_produces_48_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.side_effect = lambda name, seed=42, **kwargs: MockKvlBeamModelWrapper(
            name, seed=seed
        )

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        run_id, out_dir = runner.run(
            prompts="3",
            models="all",
            seed=42,
            no_plot=True,
            cli_args=["--prompts", "3"],
        )

        # 4 models × 4 widths (incl. greedy baseline) × 3 prompts = 48
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["run_id"] == run_id
        assert manifest["phase"] == 2
        assert manifest["experiment"] == "kvl_beam"
        assert manifest["prompt_count"] == 3
        assert manifest["observations"]["total"] == 48
        assert manifest["models"] == list(MODEL_CONFIGS.keys())

        full = (out_dir / "full.csv").read_text()
        assert full.count("\n") == 49  # header + 48 rows
        assert "kvl_beam_width" in full
        assert "kvl_beam_running_mean" in full

        assert mock_get_wrapper.call_count == len(MODEL_CONFIGS)
        for call in mock_get_wrapper.call_args_list:
            assert call.kwargs.get("seed") == 42
            assert call.kwargs.get("timeout_seconds") == 7200

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_baseline_routes_to_generate(self, mock_get_wrapper, tmp_path: Path):
        wrapper = MockKvlBeamModelWrapper("Qwen3")
        mock_get_wrapper.return_value = wrapper

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1",
            models="Qwen3",
            widths="1,4",
            no_plot=True,
        )

        assert wrapper.generate_calls == 1
        assert wrapper.generate_kvl_beam_calls == 1

        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["by_kvl_beam_width"]["1"]["count"] == 1
        assert summary["by_kvl_beam_width"]["4"]["count"] == 1

        full = (out_dir / "full.csv").read_text()
        assert "Qwen3_kvl_beam_w1" in full
        assert "Qwen3_kvl_beam_w4" in full

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_width_one_does_not_call_generate_kvl_beam(
        self, mock_get_wrapper, tmp_path: Path
    ):
        wrapper = MockKvlBeamModelWrapper("Qwen3")
        mock_get_wrapper.return_value = wrapper

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="1", no_plot=True)

        assert wrapper.generate_calls == 1
        assert wrapper.generate_kvl_beam_calls == 0

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_single_model_subset(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2",
            models="Qwen3",
            widths="4,8",
            no_plot=True,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["observations"]["total"] == 4  # 2 widths × 2 prompts
        mock_get_wrapper.assert_called_once_with(
            "Qwen3", seed=42, timeout_seconds=7200
        )

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_kvl_metadata_in_results(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1", models="Qwen3", widths="4", no_plot=True
        )

        full = pd.read_csv(out_dir / "full.csv")
        row = full.iloc[0]
        assert int(row["kvl_beam_width"]) == 4
        assert int(row["kvl_branch_factor"]) == 10
        assert int(row["kvl_beam_steps_total"]) == 10
        assert int(row["kvl_beam_words_scored"]) == 5
        assert float(row["kvl_beam_running_mean"]) == pytest.approx(2.0)
        assert int(row["kvl_beam_candidates_pruned"]) == 20

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_baseline_stamped_kvl_metadata(self, mock_get_wrapper, tmp_path: Path):
        wrapper = MockKvlBeamModelWrapper("Qwen3")
        mock_get_wrapper.return_value = wrapper

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1", models="Qwen3", widths="1", no_plot=True
        )

        assert wrapper.generate_kvl_beam_calls == 0
        full = pd.read_csv(out_dir / "full.csv")
        row = full.iloc[0]
        assert int(row["kvl_beam_width"]) == 1
        assert int(row["kvl_beam_steps_total"]) == 0
        assert int(row["kvl_beam_words_scored"]) == 0
        assert pd.isna(row["kvl_beam_running_mean"])
        assert int(row["kvl_beam_candidates_pruned"]) == 0

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_summary_by_kvl_beam_width(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="2", models="Qwen3", widths="1,4", no_plot=True
        )

        summary = json.loads((out_dir / "summary.json").read_text())
        by_width = summary["by_kvl_beam_width"]
        assert by_width["1"]["count"] == 2
        assert by_width["4"]["count"] == 2
        for width_stats in by_width.values():
            assert width_stats["a1_pass_rate"] == pytest.approx(1.0)
            assert width_stats["generation_failure_rate"] == pytest.approx(0.0)
            assert width_stats["hit_max_tokens_rate"] == pytest.approx(0.0)

        assert "Qwen3" in summary["by_model"]
        qwen = summary["by_model"]["Qwen3"]
        assert qwen["by_kvl_beam_width"]["1"]["count"] == 2
        assert qwen["by_kvl_beam_width"]["4"]["count"] == 2

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_configs_passed_to_pipeline(self, mock_get_wrapper, tmp_path: Path):
        captured = []

        class CapturingWrapper(MockKvlBeamModelWrapper):
            def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
                captured.append((config, beam_width, branch_factor))
                return super().generate_kvl_beam(
                    prompt, config, beam_width=beam_width, branch_factor=branch_factor
                )

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="4", no_plot=True)

        assert len(captured) == 1
        config, width, branch_factor = captured[0]
        assert config.config_weighting is False
        assert config.config_prompting is True
        assert config.config_kvl_beam is True
        assert config.config_guided is False
        assert config.prompt_id == "P1"
        assert config.experiment_name == "Qwen3_kvl_beam_w4"
        assert config.kvl_beam_width == 4
        assert config.kvl_l1 == "es"
        assert config.kvl_branch_factor == 10
        assert config.temperature == 0.0
        assert config.top_k == 50
        assert width == 4
        assert branch_factor == 10

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_custom_branch_factor_and_kvl_l1(self, mock_get_wrapper, tmp_path: Path):
        captured = []

        class CapturingWrapper(MockKvlBeamModelWrapper):
            def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
                captured.append((config, beam_width, branch_factor))
                return super().generate_kvl_beam(
                    prompt, config, beam_width=beam_width, branch_factor=branch_factor
                )

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        runner.run(
            prompts="1",
            models="Qwen3",
            widths="8",
            branch_factor=5,
            kvl_l1="de",
            no_plot=True,
        )

        assert len(captured) == 1
        config, width, branch_factor = captured[0]
        assert width == 8
        assert branch_factor == 5
        assert config.kvl_branch_factor == 5
        assert config.kvl_l1 == "de"

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_no_plot_true_skips_plot_run(
        self, mock_get_wrapper, mock_plot_run, tmp_path: Path
    ):
        mock_plot_run.return_value = tmp_path / "plots"
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="4", no_plot=True)

        mock_plot_run.assert_not_called()

    @patch("slm_experiments.plot.plot_run")
    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_no_plot_false_calls_plot_run(
        self, mock_get_wrapper, mock_plot_run, tmp_path: Path
    ):
        mock_plot_run.return_value = tmp_path / "plots"
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="Qwen3", widths="4", no_plot=False)

        mock_plot_run.assert_called_once()
        assert mock_plot_run.call_args.kwargs["results_root"] == tmp_path

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_cefr_sp_disable_stamped(self, mock_get_wrapper, tmp_path: Path):
        captured = []

        class CapturingWrapper(MockKvlBeamModelWrapper):
            def generate(self, prompt, config):
                captured.append(config)
                return super().generate(prompt, config)

            def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
                captured.append(config)
                return super().generate_kvl_beam(
                    prompt, config, beam_width=beam_width, branch_factor=branch_factor
                )

        mock_get_wrapper.return_value = CapturingWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="1",
            models="Qwen3",
            widths="1,4",
            enable_cefr_sp=False,
            no_plot=True,
        )

        assert len(captured) == 2
        assert all(c.enable_cefr_sp is False for c in captured)

        full = pd.read_csv(out_dir / "full.csv")
        assert (full["cefr_sp_enabled"] == False).all()  # noqa: E712
        assert full["cefr_sp_level"].isna().all()

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_failed_generation_still_in_bundle(self, mock_get_wrapper, tmp_path: Path):
        class FailingWrapper(MockKvlBeamModelWrapper):
            def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
                self.generate_kvl_beam_calls += 1
                return {
                    "response": "",
                    "response_time_seconds": 0.4,
                    "generation_successful": False,
                    "kvl_beam_width": beam_width,
                    "kvl_branch_factor": branch_factor,
                    "kvl_beam_steps_total": 0,
                    "kvl_beam_words_scored": 0,
                    "kvl_beam_running_mean": None,
                    "kvl_beam_logprob_tiebreak": 0.0,
                    "kvl_beam_candidates_pruned": 0,
                }

        mock_get_wrapper.return_value = FailingWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
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

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_wrapper_cleanup_called(self, mock_get_wrapper, tmp_path: Path):
        wrappers = []

        def factory(name, seed=42, **kwargs):
            wrapper = MockKvlBeamModelWrapper(name, seed=seed)
            wrappers.append(wrapper)
            return wrapper

        mock_get_wrapper.side_effect = factory

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        runner.run(prompts="1", models="all", widths="4", no_plot=True)

        assert len(wrappers) == len(MODEL_CONFIGS)
        assert all(w.cleanup_calls == 1 for w in wrappers)

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_prompts_all_tiny_grid_count(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")

        runner = KvlBeamSweepRunner(results_root=tmp_path)
        _, out_dir = runner.run(
            prompts="all",
            models="Qwen3",
            widths="1,4",
            no_plot=True,
        )

        n_prompts = len(STANDARD_PROMPTS)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["prompt_count"] == n_prompts
        assert manifest["observations"]["total"] == n_prompts * 2

    @patch("slm_experiments.phase2.kvl_beam.get_model_wrapper")
    def test_unknown_model_raises(self, mock_get_wrapper, tmp_path: Path):
        mock_get_wrapper.return_value = MockKvlBeamModelWrapper("Qwen3")
        runner = KvlBeamSweepRunner(results_root=tmp_path)
        with pytest.raises(ValueError, match="Unknown model"):
            runner.run(prompts="1", models="NotAModel", widths="4", no_plot=True)
        mock_get_wrapper.assert_not_called()
