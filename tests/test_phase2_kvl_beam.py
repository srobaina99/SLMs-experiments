"""Tests for Phase 2 KVL beam width sweep configs and runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS
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
        self.generate_calls = 0
        self.generate_kvl_beam_calls = 0

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
        pass


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
            assert config.num_shots == 0
            assert config.temperature == 0.0
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

    def test_kvl_beam_width_from_config_invalid(self):
        config = ExperimentConfig(experiment_name="invalid_name")
        with pytest.raises(ValueError, match="Cannot parse KVL beam width"):
            kvl_beam_width_from_config(config)


class TestKvlBeamSweepRunner:
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
