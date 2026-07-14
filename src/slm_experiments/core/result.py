"""Experiment result dataclass and factory methods."""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from slm_experiments.core.config import ExperimentConfig


@dataclass
class ExperimentResult:
    """Results from a single prompt-response experiment."""

    experiment_id: str
    timestamp: datetime
    config_name: str

    prompt: str
    system_prompt: str
    response: str

    model: str
    model_id: str
    config_weighting: bool
    config_prompting: bool
    prompt_id: str

    weight_factor: float
    temperature: float

    response_time_seconds: float
    generation_successful: bool = True
    meets_a1_criteria: bool = False
    num_shots: int = 0

    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    spache_readability: float = 0.0

    word_count: int = 0
    token_count: Optional[int] = None
    difficult_words: int = 0

    kvl_l1: str = "es"
    kvl_content_word_count: int = 0
    kvl_lookup_count: int = 0
    kvl_oov_count: int = 0
    kvl_lookup_coverage: float = 0.0
    kvl_mean_score: Optional[float] = None
    kvl_min_score: Optional[float] = None
    kvl_pct_hard_words: Optional[float] = None

    cefr_sp_enabled: bool = False
    cefr_sp_sentence_count: int = 0
    cefr_sp_level: Optional[str] = None
    cefr_sp_level_ordinal: Optional[float] = None
    cefr_sp_max_level_ordinal: Optional[int] = None
    cefr_sp_pct_a1: Optional[float] = None
    cefr_sp_adjacency: Optional[float] = None
    cefr_sp_expected_level: Optional[float] = None

    cleaned_response: str = ""

    beam_selection_method: Optional[str] = None
    beam_a1_ratio: Optional[float] = None
    beam_a1_count: Optional[int] = None
    beam_content_word_count: Optional[int] = None
    beam_cumulative_logprob: Optional[float] = None
    beam_width: Optional[int] = None

    guided_top_k: Optional[int] = None
    guided_mode: Optional[str] = None
    guided_steps_a1_chosen: Optional[int] = None
    guided_steps_total: Optional[int] = None
    guided_intervention_rate: Optional[float] = None

    kvl_beam_width: Optional[int] = None
    kvl_branch_factor: Optional[int] = None
    kvl_beam_steps_total: Optional[int] = None
    kvl_beam_words_scored: Optional[int] = None
    kvl_beam_running_mean: Optional[float] = None
    kvl_beam_logprob_tiebreak: Optional[float] = None
    kvl_beam_candidates_pruned: Optional[int] = None

    response_appropriateness: Optional[float] = None
    vocabulary_level: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def create_from_response(
        cls,
        prompt: str,
        response: str,
        config: ExperimentConfig,
        response_time: float,
        text_metrics: Dict[str, Any],
        experiment_name: str = "default",
        cleaned_response: str = "",
        generation_successful: bool = True,
        meets_a1_criteria: bool = False,
        kvl_metrics: Optional[Dict[str, Any]] = None,
        cefr_sp_metrics: Optional[Dict[str, Any]] = None,
    ) -> "ExperimentResult":
        """Create ExperimentResult from response data and metrics."""
        grade_indices = text_metrics.get("grade_level_indices", {})
        readability_scores = text_metrics.get("readability_scores", {})
        text_stats = text_metrics.get("text_statistics", {})
        kvl = kvl_metrics or {}
        cefr_sp = cefr_sp_metrics or {}

        return cls(
            experiment_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            config_name=experiment_name,
            prompt=prompt,
            system_prompt=config.system_prompt,
            response=response,
            cleaned_response=cleaned_response,
            model=config.model_name,
            model_id=config.model_id,
            config_weighting=config.config_weighting,
            config_prompting=config.config_prompting,
            prompt_id=config.prompt_id,
            weight_factor=config.weight_factor,
            num_shots=config.num_shots,
            temperature=config.temperature,
            response_time_seconds=response_time,
            generation_successful=generation_successful,
            meets_a1_criteria=meets_a1_criteria,
            flesch_kincaid_grade=grade_indices.get("flesch_kincaid_grade", 0.0),
            gunning_fog=grade_indices.get("gunning_fog", 0.0),
            spache_readability=readability_scores.get("spache_readability", 0.0),
            word_count=text_stats.get("word_count", 0),
            token_count=text_stats.get("token_count"),
            difficult_words=text_stats.get("difficult_words", 0),
            kvl_l1=str(kvl.get("kvl_l1", config.kvl_l1)),
            kvl_content_word_count=int(kvl.get("kvl_content_word_count", 0)),
            kvl_lookup_count=int(kvl.get("kvl_lookup_count", 0)),
            kvl_oov_count=int(kvl.get("kvl_oov_count", 0)),
            kvl_lookup_coverage=float(kvl.get("kvl_lookup_coverage", 0.0)),
            kvl_mean_score=kvl.get("kvl_mean_score"),
            kvl_min_score=kvl.get("kvl_min_score"),
            kvl_pct_hard_words=kvl.get("kvl_pct_hard_words"),
            cefr_sp_enabled=bool(cefr_sp.get("cefr_sp_enabled", False)),
            cefr_sp_sentence_count=int(cefr_sp.get("cefr_sp_sentence_count", 0)),
            cefr_sp_level=cefr_sp.get("cefr_sp_level"),
            cefr_sp_level_ordinal=cefr_sp.get("cefr_sp_level_ordinal"),
            cefr_sp_max_level_ordinal=cefr_sp.get("cefr_sp_max_level_ordinal"),
            cefr_sp_pct_a1=cefr_sp.get("cefr_sp_pct_a1"),
            cefr_sp_adjacency=cefr_sp.get("cefr_sp_adjacency"),
            cefr_sp_expected_level=cefr_sp.get("cefr_sp_expected_level"),
        )

    @classmethod
    def create_from_beam_response(
        cls,
        prompt: str,
        response: str,
        config: ExperimentConfig,
        response_time: float,
        text_metrics: Dict[str, Any],
        experiment_name: str = "default",
        cleaned_response: str = "",
        generation_successful: bool = True,
        meets_a1_criteria: bool = False,
        kvl_metrics: Optional[Dict[str, Any]] = None,
        cefr_sp_metrics: Optional[Dict[str, Any]] = None,
        beam_selection_method: str = "a1_ratio",
        beam_a1_ratio: float = 0.0,
        beam_a1_count: int = 0,
        beam_content_word_count: int = 0,
        beam_cumulative_logprob: float = 0.0,
        beam_width: int = 4,
    ) -> "ExperimentResult":
        """Create ExperimentResult from beam search response data."""
        result = cls.create_from_response(
            prompt=prompt,
            response=response,
            config=config,
            response_time=response_time,
            text_metrics=text_metrics,
            experiment_name=experiment_name,
            cleaned_response=cleaned_response,
            generation_successful=generation_successful,
            meets_a1_criteria=meets_a1_criteria,
            kvl_metrics=kvl_metrics,
            cefr_sp_metrics=cefr_sp_metrics,
        )
        result.beam_selection_method = beam_selection_method
        result.beam_a1_ratio = beam_a1_ratio
        result.beam_a1_count = beam_a1_count
        result.beam_content_word_count = beam_content_word_count
        result.beam_cumulative_logprob = beam_cumulative_logprob
        result.beam_width = beam_width
        return result

    @classmethod
    def create_from_guided_response(
        cls,
        prompt: str,
        response: str,
        config: ExperimentConfig,
        response_time: float,
        text_metrics: Dict[str, Any],
        experiment_name: str = "default",
        cleaned_response: str = "",
        generation_successful: bool = True,
        meets_a1_criteria: bool = False,
        kvl_metrics: Optional[Dict[str, Any]] = None,
        cefr_sp_metrics: Optional[Dict[str, Any]] = None,
        guided_top_k: int = 10,
        guided_mode: str = "flat",
        guided_steps_a1_chosen: int = 0,
        guided_steps_total: int = 0,
        guided_intervention_rate: float = 0.0,
    ) -> "ExperimentResult":
        """Create ExperimentResult from guided decoding response data."""
        result = cls.create_from_response(
            prompt=prompt,
            response=response,
            config=config,
            response_time=response_time,
            text_metrics=text_metrics,
            experiment_name=experiment_name,
            cleaned_response=cleaned_response,
            generation_successful=generation_successful,
            meets_a1_criteria=meets_a1_criteria,
            kvl_metrics=kvl_metrics,
            cefr_sp_metrics=cefr_sp_metrics,
        )
        result.guided_top_k = guided_top_k
        result.guided_mode = guided_mode
        result.guided_steps_a1_chosen = guided_steps_a1_chosen
        result.guided_steps_total = guided_steps_total
        result.guided_intervention_rate = guided_intervention_rate
        return result

    @classmethod
    def create_from_kvl_beam_response(
        cls,
        prompt: str,
        response: str,
        config: ExperimentConfig,
        response_time: float,
        text_metrics: Dict[str, Any],
        experiment_name: str = "default",
        cleaned_response: str = "",
        generation_successful: bool = True,
        meets_a1_criteria: bool = False,
        kvl_metrics: Optional[Dict[str, Any]] = None,
        cefr_sp_metrics: Optional[Dict[str, Any]] = None,
        kvl_beam_width: int = 4,
        kvl_branch_factor: int = 10,
        kvl_beam_steps_total: int = 0,
        kvl_beam_words_scored: int = 0,
        kvl_beam_running_mean: Optional[float] = None,
        kvl_beam_logprob_tiebreak: float = 0.0,
        kvl_beam_candidates_pruned: int = 0,
    ) -> "ExperimentResult":
        """Create ExperimentResult from KVL beam search response data."""
        result = cls.create_from_response(
            prompt=prompt,
            response=response,
            config=config,
            response_time=response_time,
            text_metrics=text_metrics,
            experiment_name=experiment_name,
            cleaned_response=cleaned_response,
            generation_successful=generation_successful,
            meets_a1_criteria=meets_a1_criteria,
            kvl_metrics=kvl_metrics,
            cefr_sp_metrics=cefr_sp_metrics,
        )
        result.kvl_beam_width = kvl_beam_width
        result.kvl_branch_factor = kvl_branch_factor
        result.kvl_beam_steps_total = kvl_beam_steps_total
        result.kvl_beam_words_scored = kvl_beam_words_scored
        result.kvl_beam_running_mean = kvl_beam_running_mean
        result.kvl_beam_logprob_tiebreak = kvl_beam_logprob_tiebreak
        result.kvl_beam_candidates_pruned = kvl_beam_candidates_pruned
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result_dict = asdict(self)
        result_dict["timestamp"] = self.timestamp.isoformat()
        return result_dict
