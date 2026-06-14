"""Tests for ExperimentPipeline with mocked model wrapper."""

from dataclasses import fields

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)

COMPLEX_RESPONSE = (
    "The multifaceted ramifications of epistemological paradigms necessitate "
    "comprehensive interdisciplinary investigation. Furthermore, methodological "
    "inconsistencies undermine the validity of longitudinal analyses."
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


class MockComplexModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": COMPLEX_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
        }


class MockBeamSuccessModel:
    def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
            "beam_selection_method": selection_method,
            "beam_a1_ratio": 0.75,
            "beam_a1_count": 3,
            "beam_content_word_count": 6,
            "beam_cumulative_logprob": -1.2,
            "beam_width": beam_width,
        }


class MockBeamComplexModel:
    def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
        return {
            "response": COMPLEX_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
            "beam_selection_method": selection_method,
            "beam_a1_ratio": 0.1,
            "beam_a1_count": 0,
            "beam_content_word_count": 10,
            "beam_cumulative_logprob": -2.0,
            "beam_width": beam_width,
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

    def test_meets_a1_criteria_false_when_text_too_complex(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p04")
        pipeline = ExperimentPipeline()
        result = pipeline.run("Explain epistemology.", config, MockComplexModel())

        assert result.generation_successful is True
        assert result.meets_a1_criteria is False
        assert (
            result.flesch_kincaid_grade > 5
            or result.gunning_fog > 6
            or result.spache_readability > 4
        )

    def test_run_beam_sets_meets_a1_true_on_simple_text(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p05")
        pipeline = ExperimentPipeline()
        result = pipeline.run_beam(
            "What is a friend?", config, MockBeamSuccessModel(), beam_width=4
        )

        assert result.generation_successful is True
        assert result.meets_a1_criteria is True

    def test_run_beam_sets_meets_a1_false_on_complex_text(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p06")
        pipeline = ExperimentPipeline()
        result = pipeline.run_beam(
            "Explain epistemology.", config, MockBeamComplexModel(), beam_width=4
        )

        assert result.generation_successful is True
        assert result.meets_a1_criteria is False
