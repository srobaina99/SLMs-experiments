"""Tests for ExperimentPipeline.run_guided with mocked model wrapper."""

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockGuidedSuccessModel:
    def generate_guided(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.5,
            "generation_successful": True,
            "guided_top_k": config.guided_top_k,
            "guided_mode": config.guided_mode,
            "guided_steps_a1_chosen": 8,
            "guided_steps_total": 20,
            "guided_intervention_rate": 0.4,
        }


class TestRunGuided:
    def test_success_populates_guided_and_kvl_metrics(self):
        config = ExperimentConfig(
            model_name="Qwen3",
            config_guided=True,
            guided_top_k=10,
            guided_mode="flat",
            kvl_l1="es",
            prompt_id="p01",
        )
        pipeline = ExperimentPipeline()
        result = pipeline.run_guided(
            "What is a friend?",
            config,
            MockGuidedSuccessModel(),
        )

        assert isinstance(result, ExperimentResult)
        assert result.generation_successful is True
        assert result.meets_a1_criteria is True
        assert result.guided_top_k == 10
        assert result.guided_mode == "flat"
        assert result.guided_steps_a1_chosen == 8
        assert result.guided_steps_total == 20
        assert result.guided_intervention_rate == 0.4
        assert result.kvl_mean_score is not None
        assert result.kvl_content_word_count > 0
        assert result.kvl_l1 == "es"
