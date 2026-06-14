"""Tests for A1 pass criteria."""

from slm_experiments.evaluation.a1_criteria import meets_a1_criteria


def test_passes_when_all_thresholds_met():
    assert meets_a1_criteria(
        4.0, 5.0, 3.0, generation_valid=True
    )


def test_fails_when_fk_exceeds_threshold():
    assert not meets_a1_criteria(
        5.1, 5.0, 3.0, generation_valid=True
    )


def test_fails_when_fog_exceeds_threshold():
    assert not meets_a1_criteria(
        4.0, 6.1, 3.0, generation_valid=True
    )


def test_fails_when_spache_exceeds_threshold():
    assert not meets_a1_criteria(
        4.0, 5.0, 4.1, generation_valid=True
    )


def test_fails_when_generation_invalid_even_if_metrics_zero():
    assert not meets_a1_criteria(
        0.0, 0.0, 0.0, generation_valid=False
    )


def test_boundary_values_pass():
    assert meets_a1_criteria(
        5.0, 6.0, 4.0, generation_valid=True
    )
