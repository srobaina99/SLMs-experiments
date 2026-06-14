"""Tests for ExperimentPipeline with mocked model wrapper."""

from dataclasses import fields

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockSuccessModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.23,
            "generation_successful": True,
        }


class MockFailureModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.5,
            "generation_successful": False,
        }


class MockEmptySuccessModel:
    """Model reports success but returns empty text."""

    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "   ",
            "response_time_seconds": 0.1,
            "generation_successful": True,
        }


class TestExperimentPipeline:
    def test_success_path(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p01")
        pipeline = ExperimentPipeline()
        result = pipeline.run("What is a friend?", config, MockSuccessModel())

        assert isinstance(result, ExperimentResult)
        assert result.generation_successful is True
        assert result.word_count > 0
        assert result.gunning_fog != 0.0 or result.spache_readability != 0.0
        assert result.cleaned_response

    def test_failure_path(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p02")
        pipeline = ExperimentPipeline()
        result = pipeline.run("What is a friend?", config, MockFailureModel())

        assert result.generation_successful is False
        assert result.flesch_kincaid_grade == 0.0
        assert result.gunning_fog == 0.0
        assert result.spache_readability == 0.0
        assert result.word_count == 0

    def test_empty_response_treated_as_failure(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p03")
        pipeline = ExperimentPipeline()
        result = pipeline.run("Hello?", config, MockEmptySuccessModel())

        assert result.generation_successful is False
        assert result.flesch_kincaid_grade == 0.0

    def test_result_no_legacy_fields(self):
        field_names = {f.name for f in fields(ExperimentResult)}
        assert "weighted_words_enabled" not in field_names
        assert "enable_thinking" not in field_names
        assert "weight_factor" in field_names
        assert "num_shots" in field_names
        assert "temperature" in field_names
        assert "meets_a1_criteria" in field_names

    def test_meets_a1_criteria_set_on_success(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p01")
        pipeline = ExperimentPipeline()
        result = pipeline.run("What is a friend?", config, MockSuccessModel())

        assert result.generation_successful is True
        assert result.meets_a1_criteria is True

    def test_meets_a1_criteria_false_on_failure(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p02")
        pipeline = ExperimentPipeline()
        result = pipeline.run("What is a friend?", config, MockFailureModel())

        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
