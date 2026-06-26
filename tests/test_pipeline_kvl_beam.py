"""Tests for ExperimentPipeline.run_kvl_beam with mocked model wrapper."""

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult

SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)


class MockKvlBeamSuccessModel:
    def generate_kvl_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        branch_factor: int = 10,
    ) -> dict:
        return {
            "response": SIMPLE_RESPONSE,
            "response_time_seconds": 1.5,
            "generation_successful": True,
            "kvl_beam_width": beam_width,
            "kvl_branch_factor": branch_factor,
            "kvl_beam_steps_total": 42,
            "kvl_beam_words_scored": 8,
            "kvl_beam_running_mean": 2.5,
            "kvl_beam_logprob_tiebreak": -1.8,
            "kvl_beam_candidates_pruned": 120,
        }


class MockKvlBeamFailureModel:
    def generate_kvl_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        branch_factor: int = 10,
    ) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.3,
            "generation_successful": False,
            "kvl_beam_width": beam_width,
            "kvl_branch_factor": branch_factor,
            "kvl_beam_steps_total": 0,
            "kvl_beam_words_scored": 0,
            "kvl_beam_running_mean": None,
            "kvl_beam_logprob_tiebreak": 0.0,
            "kvl_beam_candidates_pruned": 0,
        }


class TestRunKvlBeam:
    def test_success_populates_kvl_beam_metadata(self):
        config = ExperimentConfig(
            model_name="Qwen3",
            config_kvl_beam=True,
            kvl_l1="es",
            prompt_id="p01",
        )
        pipeline = ExperimentPipeline()
        result = pipeline.run_kvl_beam(
            "What is a friend?",
            config,
            MockKvlBeamSuccessModel(),
            beam_width=8,
            branch_factor=10,
        )

        assert isinstance(result, ExperimentResult)
        assert result.generation_successful is True
        assert result.meets_a1_criteria is True
        assert result.kvl_beam_width == 8
        assert result.kvl_branch_factor == 10
        assert result.kvl_beam_steps_total == 42
        assert result.kvl_beam_words_scored == 8
        assert result.kvl_beam_running_mean == 2.5
        assert result.kvl_beam_logprob_tiebreak == -1.8
        assert result.kvl_beam_candidates_pruned == 120
        assert result.kvl_mean_score is not None
        assert result.kvl_content_word_count > 0

    def test_failure_gets_empty_defaults(self):
        config = ExperimentConfig(
            model_name="Qwen3",
            config_kvl_beam=True,
            prompt_id="p02",
        )
        pipeline = ExperimentPipeline()
        result = pipeline.run_kvl_beam(
            "What is a friend?",
            config,
            MockKvlBeamFailureModel(),
            beam_width=4,
            branch_factor=10,
        )

        assert result.generation_successful is False
        assert result.meets_a1_criteria is False
        assert result.kvl_beam_width == 4
        assert result.kvl_branch_factor == 10
        assert result.kvl_beam_steps_total == 0
        assert result.kvl_beam_words_scored == 0
        assert result.kvl_beam_running_mean is None
        assert result.kvl_beam_logprob_tiebreak == 0.0
        assert result.kvl_beam_candidates_pruned == 0
        assert result.flesch_kincaid_grade == 0.0
        assert result.word_count == 0
