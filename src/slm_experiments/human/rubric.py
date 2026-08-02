"""Versioned beginner-suitability rubric constants (English column keys)."""

from __future__ import annotations

from pathlib import Path

RUBRIC_VERSION = "beginner_suitability_rubric_v0"

# English data keys; Spanish anchors live in the companion markdown.
OVERALL_SUITABILITY = "overall_suitability"
VOCABULARY_ACCESSIBILITY = "vocabulary_accessibility"
SYNTAX_ACCESSIBILITY = "syntax_accessibility"
ANSWER_ADEQUACY = "answer_adequacy"

RATING_DIMENSIONS = (
    OVERALL_SUITABILITY,
    VOCABULARY_ACCESSIBILITY,
    SYNTAX_ACCESSIBILITY,
    ANSWER_ADEQUACY,
)

RATING_MIN = 1
RATING_MAX = 4

# Binary human-suitable gate (vocab/syntax are diagnostic only).
OVERALL_SUITABLE_THRESHOLD = 3
ADEQUACY_SUITABLE_THRESHOLD = 3

NOTES_COLUMN = "notes"

RUBRIC_MARKDOWN = Path(__file__).resolve().parent / "rubrics" / f"{RUBRIC_VERSION}.md"


def is_human_suitable(median_overall: float, median_adequacy: float) -> bool:
    """True iff median overall ≥ 3 AND median adequacy ≥ 3."""
    return (
        median_overall >= OVERALL_SUITABLE_THRESHOLD
        and median_adequacy >= ADEQUACY_SUITABLE_THRESHOLD
    )
