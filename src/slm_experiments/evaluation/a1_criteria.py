from dataclasses import dataclass


@dataclass(frozen=True)
class A1ReadabilityThresholds:
    flesch_kincaid_max: float = 5.0
    gunning_fog_max: float = 6.0
    spache_max: float = 4.0


DEFAULT_A1_THRESHOLDS = A1ReadabilityThresholds()


def meets_a1_criteria(
    flesch_kincaid: float,
    gunning_fog: float,
    spache: float,
    *,
    generation_valid: bool,
    thresholds: A1ReadabilityThresholds = DEFAULT_A1_THRESHOLDS,
) -> bool:
    """Return True when a valid generation passes the automated readability proxy.

    This is a conjunction of US readability formula thresholds (FK / Fog / Spache),
    not a CEFR A1 communicative assessment. The function/field name is historical.
    """
    if not generation_valid:
        return False
    return (
        flesch_kincaid <= thresholds.flesch_kincaid_max
        and gunning_fog <= thresholds.gunning_fog_max
        and spache <= thresholds.spache_max
    )
