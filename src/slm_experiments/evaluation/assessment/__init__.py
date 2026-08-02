"""Immutable assessment bundles and scoring seam."""

from slm_experiments.evaluation.assessment.bundle import AssessmentBundler
from slm_experiments.evaluation.assessment.judge import (
    JudgeExporter,
    JudgeImporter,
    validate_judge_scores,
)
from slm_experiments.evaluation.assessment.scorers import (
    ensure_scorer_registered,
    list_scorers,
    register_scorer,
    score_items,
)

# Import side-effect: register built-in assessment scorers.
from slm_experiments.evaluation.assessment import cefr_tsar as _cefr_tsar  # noqa: F401
from slm_experiments.evaluation.assessment import kvl_v2 as _kvl_v2  # noqa: F401

__all__ = [
    "AssessmentBundler",
    "JudgeExporter",
    "JudgeImporter",
    "ensure_scorer_registered",
    "list_scorers",
    "register_scorer",
    "score_items",
    "validate_judge_scores",
]
