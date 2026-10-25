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


class MockGuidedFailureModel:
    def generate_guided(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.2,
            "generation_successful": False,
            "guided_top_k": config.guided_top_k,
            "guided_mode": config.guided_mode,
            "guided_steps_a1_chosen": 0,
            "guided_steps_total": 0,
            "guided_intervention_rate": 0.0,
        }


class RecordingGuidedModel:
    def __init__(self):
        self.called = False
        self.seen_config = None

    def generate_guided(self, prompt: str, config: ExperimentConfig) -> dict:
        self.called = True
        self.seen_config = config
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.0,
            "generation_successful": True,
            "guided_top_k": config.guided_top_k,
            "guided_mode": config.guided_mode,
            "guided_steps_a1_chosen": 1,
            "guided_steps_total": 5,
            "guided_intervention_rate": 0.2,
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

    def test_failure_preserves_guided_metadata(self):
        config = ExperimentConfig(
            model_name="Qwen3",
            config_guided=True,
            guided_top_k=10,
            guided_mode="flat",
            prompt_id="p02",
        )
        result = ExperimentPipeline().run_guided(
            "What is a friend?",
            config,
            MockGuidedFailureModel(),
        )

        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
        assert result.guided_top_k == 10
        assert result.guided_mode == "flat"
        assert result.guided_steps_a1_chosen == 0
        assert result.guided_steps_total == 0
        assert result.guided_intervention_rate == 0.0
        assert result.word_count == 0
        assert result.kvl_mean_score is None

    def test_records_hit_max_tokens(self):
        class HitMaxGuided:
            def generate_guided(self, prompt, config):
                return {
                    "response": SIMPLE_RESPONSE,
                    "response_time_seconds": 1.0,
                    "generation_successful": True,
                    "guided_top_k": config.guided_top_k,
                    "guided_mode": config.guided_mode,
                    "guided_steps_a1_chosen": 8,
                    "guided_steps_total": 20,
                    "guided_intervention_rate": 0.4,
                    "hit_max_tokens": True,
                }

        config = ExperimentConfig(
            model_name="Qwen3",
            config_guided=True,
            guided_top_k=10,
            prompt_id="p03",
        )
        result = ExperimentPipeline().run_guided(
            "What is a friend?", config, HitMaxGuided()
        )

        assert result.generation_successful is True
        assert result.hit_max_tokens is True

    def test_meets_a1_false_when_cefr_sp_not_a1(self):
        class HardCefrSpScorer:
            def score_sentences(self, sentences):
                return [
                    {"label": 5, "probs": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
                    for _ in sentences
                ]

        config = ExperimentConfig(
            model_name="Qwen3",
            config_guided=True,
            guided_top_k=10,
            prompt_id="p04",
        )
        result = ExperimentPipeline(cefr_sp_scorer=HardCefrSpScorer()).run_guided(
            "What is a friend?", config, MockGuidedSuccessModel()
        )

        assert result.generation_successful is True
        assert result.cefr_sp_level == "C2"
        assert result.meets_a1_criteria is False
        assert result.guided_intervention_rate == 0.4

    def test_pool_zero_does_not_reject_run_guided(self):
        """Pipeline.run_guided does not no-op or reject guided_top_k=0.

        Phase 2 runner routes pool 0 through pipeline.run (unconstrained greedy)
        and stamps guided_* zeros. If run_guided is called with pool 0 anyway,
        it still invokes generate_guided and records whatever metadata the model
        returns.
        """
        config = ExperimentConfig(
            model_name="Qwen3",
            config_guided=True,
            config_prompting=True,
            guided_top_k=0,
            guided_mode="flat",
            prompt_id="p05",
        )
        model = RecordingGuidedModel()
        result = ExperimentPipeline().run_guided(
            "What is a friend?", config, model
        )

        assert model.called is True
        assert model.seen_config.guided_top_k == 0
        assert result.guided_top_k == 0
        assert result.generation_successful is True
        assert result.guided_steps_total == 5

    def test_thinking_tags_only_treated_as_failure(self):
        class ThinkingOnlyGuided:
            def generate_guided(self, prompt, config):
                return {
                    "response": "<think>hidden reasoning</think>",
                    "response_time_seconds": 0.3,
                    "generation_successful": True,
                    "guided_top_k": config.guided_top_k,
                    "guided_mode": config.guided_mode,
                    "guided_steps_a1_chosen": 2,
                    "guided_steps_total": 4,
                    "guided_intervention_rate": 0.5,
                }

        config = ExperimentConfig(
            model_name="Qwen3",
            config_guided=True,
            guided_top_k=10,
            prompt_id="p06",
        )
        result = ExperimentPipeline().run_guided(
            "What is a friend?", config, ThinkingOnlyGuided()
        )

        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
        assert result.guided_steps_total == 4
        assert result.guided_intervention_rate == 0.5
