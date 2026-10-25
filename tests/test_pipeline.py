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


class MockBeamFailureModel:
    def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
        return {
            "response": "",
            "response_time_seconds": 0.2,
            "generation_successful": False,
            "beam_selection_method": selection_method,
            "beam_a1_ratio": 0.0,
            "beam_a1_count": 0,
            "beam_content_word_count": 0,
            "beam_cumulative_logprob": 0.0,
            "beam_width": beam_width,
        }


class MockHitMaxModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
            "hit_max_tokens": True,
        }


class MockThinkingOnlyModel:
    """Qwen3-style thinking block with no leftover answer text."""

    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "<think>internal chain of thought about friends</think>",
            "response_time_seconds": 0.4,
            "generation_successful": True,
        }


class MockThinkingThenAnswerModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": (
                "<think>internal chain of thought</think>\n" + SIMPLE_RESPONSE
            ),
            "response_time_seconds": 0.4,
            "generation_successful": True,
        }


class RecordingGenerateModel:
    """Captures the config forwarded by pipeline.run (weights / prompting carriers)."""

    def __init__(self):
        self.seen_config = None

    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        self.seen_config = config
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
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

    def test_meets_a1_criteria_false_when_cefr_sp_not_a1(self):
        """A1 gate follows CEFR-SP level, not FK/Fog/Spache."""

        class HardCefrSpScorer:
            def score_sentences(self, sentences):
                return [
                    {"label": 5, "probs": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
                    for _ in sentences
                ]

        config = ExperimentConfig(model_name="Qwen3", prompt_id="p04")
        pipeline = ExperimentPipeline(cefr_sp_scorer=HardCefrSpScorer())
        # Simple readability text, but CEFR-SP labels are C2 → gate fails
        result = pipeline.run("What is a friend?", config, MockSuccessModel())

        assert result.generation_successful is True
        assert result.cefr_sp_level == "C2"
        assert result.meets_a1_criteria is False

    def test_run_beam_sets_meets_a1_true_on_simple_text(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p05")
        pipeline = ExperimentPipeline()
        result = pipeline.run_beam(
            "What is a friend?", config, MockBeamSuccessModel(), beam_width=4
        )

        assert result.generation_successful is True
        assert result.meets_a1_criteria is True

    def test_run_beam_sets_meets_a1_false_when_cefr_sp_not_a1(self):
        class HardCefrSpScorer:
            def score_sentences(self, sentences):
                return [
                    {"label": 4, "probs": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
                    for _ in sentences
                ]

        config = ExperimentConfig(model_name="Qwen3", prompt_id="p06")
        pipeline = ExperimentPipeline(cefr_sp_scorer=HardCefrSpScorer())
        result = pipeline.run_beam(
            "What is a friend?", config, MockBeamSuccessModel(), beam_width=4
        )

        assert result.generation_successful is True
        assert result.cefr_sp_level == "C1"
        assert result.meets_a1_criteria is False

    def test_run_forwards_weighting_and_prompting_config(self):
        """Phase 2 weights sweep carrier: weighting ON + prompting ON."""
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p07",
            config_weighting=True,
            config_prompting=True,
            weight_factor=2.0,
            num_shots=0,
        )
        model = RecordingGenerateModel()
        result = ExperimentPipeline().run("What is a friend?", config, model)

        assert model.seen_config is config
        assert model.seen_config.config_weighting is True
        assert model.seen_config.config_prompting is True
        assert model.seen_config.weight_factor == 2.0
        assert result.config_weighting is True
        assert result.config_prompting is True
        assert result.weight_factor == 2.0
        assert result.generation_successful is True

    def test_run_forwards_prompting_only_config(self):
        """Phase 2 prompting / guided-baseline / kvl-baseline carrier."""
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p08",
            config_weighting=False,
            config_prompting=True,
            weight_factor=1.0,
            num_shots=3,
        )
        model = RecordingGenerateModel()
        result = ExperimentPipeline().run("What is a friend?", config, model)

        assert model.seen_config.config_weighting is False
        assert model.seen_config.config_prompting is True
        assert model.seen_config.num_shots == 3
        assert result.config_weighting is False
        assert result.config_prompting is True
        assert result.num_shots == 3
        assert result.generation_successful is True

    def test_failure_still_has_result_fields(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p09")
        result = ExperimentPipeline().run(
            "What is a friend?", config, MockFailureModel()
        )

        assert isinstance(result, ExperimentResult)
        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
        assert result.prompt == "What is a friend?"
        assert result.response == ""
        assert result.word_count == 0
        assert result.kvl_mean_score is None
        assert result.kvl_content_word_count == 0
        assert result.cefr_sp_level is None
        assert result.flesch_kincaid_grade == 0.0

    def test_hit_max_tokens_recorded_on_success(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p10")
        result = ExperimentPipeline().run(
            "What is a friend?", config, MockHitMaxModel()
        )

        assert result.generation_successful is True
        assert result.hit_max_tokens is True

    def test_hit_max_tokens_defaults_false(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p11")
        result = ExperimentPipeline().run(
            "What is a friend?", config, MockSuccessModel()
        )

        assert result.hit_max_tokens is False

    def test_thinking_tags_only_treated_as_failure(self):
        """Formatter strips Qwen3 <think> blocks; leftover empty text is a failure."""
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p12")
        result = ExperimentPipeline().run(
            "What is a friend?", config, MockThinkingOnlyModel()
        )

        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
        assert result.cleaned_response == ""
        assert result.word_count == 0

    def test_thinking_tags_stripped_before_evaluation(self):
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p13")
        result = ExperimentPipeline().run(
            "What is a friend?", config, MockThinkingThenAnswerModel()
        )

        assert result.generation_successful is True
        assert "<think>" not in result.cleaned_response
        assert "</think>" not in result.cleaned_response
        assert "internal chain of thought" not in result.cleaned_response
        assert "friend" in result.cleaned_response.lower()

    def test_meets_a1_criteria_false_when_cefr_sp_disabled(self):
        config = ExperimentConfig(
            model_name="Qwen3", prompt_id="p14", enable_cefr_sp=False
        )
        result = ExperimentPipeline().run(
            "What is a friend?", config, MockSuccessModel()
        )

        assert result.generation_successful is True
        assert result.cefr_sp_enabled is False
        assert result.cefr_sp_level is None
        assert result.meets_a1_criteria is False

    def test_run_beam_failure_records_metadata(self):
        """Deprecated beam path still evaluates internally (CLI hard-fails separately)."""
        config = ExperimentConfig(model_name="Qwen3", prompt_id="p15")
        result = ExperimentPipeline().run_beam(
            "What is a friend?", config, MockBeamFailureModel(), beam_width=8
        )

        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
        assert result.beam_width == 8
        assert result.beam_selection_method == "a1_ratio"
        assert result.word_count == 0
        assert result.kvl_mean_score is None

    def test_run_beam_records_hit_max_tokens(self):
        class HitMaxBeam:
            def generate_beam(
                self, prompt, config, beam_width=4, selection_method="a1_ratio"
            ):
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
                    "hit_max_tokens": True,
                }

        config = ExperimentConfig(model_name="Qwen3", prompt_id="p16")
        result = ExperimentPipeline().run_beam(
            "What is a friend?", config, HitMaxBeam(), beam_width=4
        )

        assert result.generation_successful is True
        assert result.hit_max_tokens is True
