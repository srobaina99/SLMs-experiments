"""Tests for ExperimentConfig."""

from dataclasses import fields

from slm_experiments.core.config import ExperimentConfig


LEGACY_FIELDS = {"weighted_words_enabled", "enable_thinking", "verbose"}


class TestExperimentConfig:
    def test_defaults(self):
        config = ExperimentConfig()
        assert config.model_name == "Qwen3"
        assert config.config_weighting is False
        assert config.config_prompting is False
        assert config.weight_factor == 1.0
        assert config.num_shots == 0
        assert config.temperature == 0.0
        assert config.enable_cefr_sp is True

    def test_to_dict(self):
        config = ExperimentConfig(
            model_name="Phi3",
            config_weighting=True,
            prompt_id="p01",
        )
        d = config.to_dict()
        assert d["model_name"] == "Phi3"
        assert d["config_weighting"] is True
        assert d["prompt_id"] == "p01"
        assert isinstance(d, dict)

    def test_no_legacy_fields(self):
        field_names = {f.name for f in fields(ExperimentConfig)}
        assert LEGACY_FIELDS.isdisjoint(field_names)

    def test_kept_fields(self):
        field_names = {f.name for f in fields(ExperimentConfig)}
        expected = {
            "model_name",
            "model_id",
            "system_prompt",
            "config_weighting",
            "config_prompting",
            "weight_factor",
            "num_shots",
            "prompt_id",
            "temperature",
            "top_k",
            "max_new_tokens",
            "experiment_name",
            "description",
        }
        assert "top_p" not in field_names
