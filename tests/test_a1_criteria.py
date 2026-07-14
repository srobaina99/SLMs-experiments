"""Tests for A1 pass criteria (CEFR-SP gate)."""

from slm_experiments.evaluation.a1_criteria import meets_a1_criteria


def test_passes_when_cefr_sp_level_is_a1():
    assert meets_a1_criteria(
        generation_valid=True,
        cefr_sp_enabled=True,
        cefr_sp_level="A1",
    )


def test_fails_when_cefr_sp_level_is_a2_or_higher():
    for level in ("A2", "B1", "B2", "C1", "C2"):
        assert not meets_a1_criteria(
            generation_valid=True,
            cefr_sp_enabled=True,
            cefr_sp_level=level,
        )


def test_fails_when_generation_invalid():
    assert not meets_a1_criteria(
        generation_valid=False,
        cefr_sp_enabled=True,
        cefr_sp_level="A1",
    )


def test_fails_when_cefr_sp_disabled():
    assert not meets_a1_criteria(
        generation_valid=True,
        cefr_sp_enabled=False,
        cefr_sp_level="A1",
    )


def test_fails_when_cefr_sp_level_missing():
    assert not meets_a1_criteria(
        generation_valid=True,
        cefr_sp_enabled=True,
        cefr_sp_level=None,
    )


def test_accepts_cefr_sp_metrics_mapping():
    assert meets_a1_criteria(
        generation_valid=True,
        cefr_sp_metrics={
            "cefr_sp_enabled": True,
            "cefr_sp_level": "A1",
        },
    )
    assert not meets_a1_criteria(
        generation_valid=True,
        cefr_sp_metrics={
            "cefr_sp_enabled": True,
            "cefr_sp_level": "C1",
        },
    )
