"""Generate → format → evaluate → record pipeline."""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.result import ExperimentResult
from slm_experiments.evaluation.a1_criteria import meets_a1_criteria
from slm_experiments.evaluation.formatter import ResponseFormatter
from slm_experiments.evaluation.kvl import KvlLookup, compute_kvl_metrics, empty_kvl_metrics
from slm_experiments.evaluation.metrics import TextEvaluator


@runtime_checkable
class ModelWrapper(Protocol):
    """Minimal model interface for the experiment pipeline."""

    def generate(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """Return at least response, response_time_seconds, generation_successful."""
        ...


@runtime_checkable
class BeamModelWrapper(Protocol):
    """Model interface for beam search experiments."""

    def generate_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        selection_method: str = "a1_ratio",
    ) -> Dict[str, Any]:
        """Return response, beam metadata, response_time_seconds, generation_successful."""
        ...


@runtime_checkable
class KvlBeamModelWrapper(Protocol):
    """Model interface for KVL-scored beam search experiments."""

    def generate_kvl_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        branch_factor: int = 10,
    ) -> Dict[str, Any]:
        """Return response, kvl_beam metadata, response_time_seconds, generation_successful."""
        ...


class ExperimentPipeline:
    """Single-observation pipeline: generate, format, evaluate, record."""

    def __init__(
        self,
        text_evaluator: Optional[TextEvaluator] = None,
        formatter: Optional[ResponseFormatter] = None,
        kvl_lookup: Optional[KvlLookup] = None,
    ):
        self.text_evaluator = text_evaluator or TextEvaluator()
        self.formatter = formatter or ResponseFormatter()
        self.kvl_lookup = kvl_lookup or KvlLookup()

    def _compute_kvl_metrics(self, cleaned: str, l1: str) -> Dict[str, Any]:
        content_words = self.text_evaluator.extract_content_words(cleaned)
        return compute_kvl_metrics(
            cleaned,
            l1,
            content_words=content_words,
            kvl_lookup=self.kvl_lookup,
        )

    def run(
        self,
        prompt: str,
        config: ExperimentConfig,
        model: ModelWrapper,
        experiment_name: Optional[str] = None,
    ) -> ExperimentResult:
        """Execute one prompt through the full pipeline."""
        name = experiment_name or config.experiment_name
        response_data = model.generate(prompt, config)

        raw_response = response_data.get("response") or ""
        response_time = float(response_data.get("response_time_seconds", 0.0))
        model_success = bool(response_data.get("generation_successful", False))

        cleaned = self.formatter.clean_response_for_evaluation(raw_response)
        is_successful = model_success and bool(cleaned.strip())

        if is_successful:
            text_metrics = self.text_evaluator.evaluate_text_comprehensive(cleaned)
            grade = text_metrics.get("grade_level_indices", {})
            read = text_metrics.get("readability_scores", {})
            meets_a1 = meets_a1_criteria(
                grade.get("flesch_kincaid_grade", 0.0),
                grade.get("gunning_fog", 0.0),
                read.get("spache_readability", 0.0),
                generation_valid=True,
            )
            return ExperimentResult.create_from_response(
                prompt=prompt,
                response=raw_response,
                config=config,
                response_time=response_time,
                text_metrics=text_metrics,
                experiment_name=name,
                cleaned_response=cleaned,
                generation_successful=True,
                meets_a1_criteria=meets_a1,
                kvl_metrics=self._compute_kvl_metrics(cleaned, config.kvl_l1),
            )

        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
        return ExperimentResult.create_from_response(
            prompt=prompt,
            response=raw_response,
            config=config,
            response_time=response_time,
            text_metrics=empty_metrics,
            experiment_name=name,
            cleaned_response=cleaned,
            generation_successful=False,
            meets_a1_criteria=False,
            kvl_metrics=empty_kvl_metrics(config.kvl_l1),
        )

    def run_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        model: BeamModelWrapper,
        beam_width: int,
        experiment_name: Optional[str] = None,
        selection_method: str = "a1_ratio",
    ) -> ExperimentResult:
        """Execute one prompt through beam search, format, evaluate, record."""
        name = experiment_name or config.experiment_name
        response_data = model.generate_beam(
            prompt,
            config,
            beam_width=beam_width,
            selection_method=selection_method,
        )

        raw_response = response_data.get("response") or ""
        response_time = float(response_data.get("response_time_seconds", 0.0))
        model_success = bool(response_data.get("generation_successful", False))

        cleaned = self.formatter.clean_response_for_evaluation(raw_response)
        is_successful = model_success and bool(cleaned.strip())

        beam_kwargs = {
            "beam_selection_method": response_data.get(
                "beam_selection_method", selection_method
            ),
            "beam_a1_ratio": float(response_data.get("beam_a1_ratio", 0.0)),
            "beam_a1_count": int(response_data.get("beam_a1_count", 0)),
            "beam_content_word_count": int(
                response_data.get("beam_content_word_count", 0)
            ),
            "beam_cumulative_logprob": float(
                response_data.get("beam_cumulative_logprob", 0.0)
            ),
            "beam_width": int(response_data.get("beam_width", beam_width)),
        }

        if is_successful:
            text_metrics = self.text_evaluator.evaluate_text_comprehensive(cleaned)
            grade = text_metrics.get("grade_level_indices", {})
            read = text_metrics.get("readability_scores", {})
            meets_a1 = meets_a1_criteria(
                grade.get("flesch_kincaid_grade", 0.0),
                grade.get("gunning_fog", 0.0),
                read.get("spache_readability", 0.0),
                generation_valid=True,
            )
            return ExperimentResult.create_from_beam_response(
                prompt=prompt,
                response=raw_response,
                config=config,
                response_time=response_time,
                text_metrics=text_metrics,
                experiment_name=name,
                cleaned_response=cleaned,
                generation_successful=True,
                meets_a1_criteria=meets_a1,
                kvl_metrics=self._compute_kvl_metrics(cleaned, config.kvl_l1),
                **beam_kwargs,
            )

        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
        return ExperimentResult.create_from_beam_response(
            prompt=prompt,
            response=raw_response,
            config=config,
            response_time=response_time,
            text_metrics=empty_metrics,
            experiment_name=name,
            cleaned_response=cleaned,
            generation_successful=False,
            meets_a1_criteria=False,
            kvl_metrics=empty_kvl_metrics(config.kvl_l1),
            **beam_kwargs,
        )

    def run_kvl_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        model: KvlBeamModelWrapper,
        beam_width: int,
        branch_factor: int = 10,
        experiment_name: Optional[str] = None,
    ) -> ExperimentResult:
        """Execute one prompt through KVL beam search, format, evaluate, record."""
        name = experiment_name or config.experiment_name
        response_data = model.generate_kvl_beam(
            prompt,
            config,
            beam_width=beam_width,
            branch_factor=branch_factor,
        )

        raw_response = response_data.get("response") or ""
        response_time = float(response_data.get("response_time_seconds", 0.0))
        model_success = bool(response_data.get("generation_successful", False))

        cleaned = self.formatter.clean_response_for_evaluation(raw_response)
        is_successful = model_success and bool(cleaned.strip())

        kvl_beam_kwargs = {
            "kvl_beam_width": int(response_data.get("kvl_beam_width", beam_width)),
            "kvl_branch_factor": int(
                response_data.get("kvl_branch_factor", branch_factor)
            ),
            "kvl_beam_steps_total": int(response_data.get("kvl_beam_steps_total", 0)),
            "kvl_beam_words_scored": int(response_data.get("kvl_beam_words_scored", 0)),
            "kvl_beam_running_mean": response_data.get("kvl_beam_running_mean"),
            "kvl_beam_logprob_tiebreak": float(
                response_data.get("kvl_beam_logprob_tiebreak", 0.0)
            ),
            "kvl_beam_candidates_pruned": int(
                response_data.get("kvl_beam_candidates_pruned", 0)
            ),
        }

        if is_successful:
            text_metrics = self.text_evaluator.evaluate_text_comprehensive(cleaned)
            grade = text_metrics.get("grade_level_indices", {})
            read = text_metrics.get("readability_scores", {})
            meets_a1 = meets_a1_criteria(
                grade.get("flesch_kincaid_grade", 0.0),
                grade.get("gunning_fog", 0.0),
                read.get("spache_readability", 0.0),
                generation_valid=True,
            )
            return ExperimentResult.create_from_kvl_beam_response(
                prompt=prompt,
                response=raw_response,
                config=config,
                response_time=response_time,
                text_metrics=text_metrics,
                experiment_name=name,
                cleaned_response=cleaned,
                generation_successful=True,
                meets_a1_criteria=meets_a1,
                kvl_metrics=self._compute_kvl_metrics(cleaned, config.kvl_l1),
                **kvl_beam_kwargs,
            )

        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
        return ExperimentResult.create_from_kvl_beam_response(
            prompt=prompt,
            response=raw_response,
            config=config,
            response_time=response_time,
            text_metrics=empty_metrics,
            experiment_name=name,
            cleaned_response=cleaned,
            generation_successful=False,
            meets_a1_criteria=False,
            kvl_metrics=empty_kvl_metrics(config.kvl_l1),
            **kvl_beam_kwargs,
        )
