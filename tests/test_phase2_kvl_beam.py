"""Tests for Phase 2 KVL beam width sweep configs."""

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS
from slm_experiments.phase2.kvl_beam import (
    DEFAULT_KVL_BEAM_WIDTH_GRID,
    create_kvl_beam_configs,
    kvl_beam_width_from_config,
    parse_widths,
)


class TestKvlBeamConfigs:
    def test_create_kvl_beam_configs_count(self):
        configs = create_kvl_beam_configs(DEFAULT_KVL_BEAM_WIDTH_GRID)
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
        assert parse_widths("4,8") == [4, 8]

    def test_parse_widths_invalid(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_widths("0,4")

    def test_kvl_beam_width_from_config_invalid(self):
        config = ExperimentConfig(experiment_name="invalid_name")
        with pytest.raises(ValueError, match="Cannot parse KVL beam width"):
            kvl_beam_width_from_config(config)
