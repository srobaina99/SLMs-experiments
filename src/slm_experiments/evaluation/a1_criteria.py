"""Primary A1 pass/fail gate based on CEFR-SP document level."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def meets_a1_criteria(
    *,
    generation_valid: bool,
    cefr_sp_level: Optional[str] = None,
    cefr_sp_enabled: bool = False,
    cefr_sp_metrics: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True when a valid generation is CEFR-SP document level A1.

    The gate uses Arase et al. CEFR-SP aggregates (``cefr_sp_level == "A1"``),
    not US readability formulas (FK / Fog / Spache). Those formulas remain
    recorded as secondary descriptive metrics.

    When CEFR-SP is disabled or unscored (null level), the gate is False.
    """
    if not generation_valid:
        return False

    if cefr_sp_metrics is not None:
        cefr_sp_enabled = bool(cefr_sp_metrics.get("cefr_sp_enabled", False))
        cefr_sp_level = cefr_sp_metrics.get("cefr_sp_level")  # type: ignore[assignment]

    if not cefr_sp_enabled:
        return False
    return cefr_sp_level == "A1"
