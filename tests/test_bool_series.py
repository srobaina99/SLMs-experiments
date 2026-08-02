"""CSV-safe boolean coercion (string \"False\" must not become True)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slm_experiments.core.bool_series import coerce_bool, coerce_bool_series
from slm_experiments.core.run_store import _aggregate_metric_stats
from slm_experiments.evaluation.assessment.analysis_helpers import _bool_series
from slm_experiments.evaluation.assessment.paired_deltas import _stratum_rates


class TestCoerceBool:
    def test_string_false_is_false(self):
        assert coerce_bool("False") is False
        assert coerce_bool("false") is False
        assert coerce_bool("FALSE") is False
        assert coerce_bool("0") is False
        assert coerce_bool("no") is False

    def test_string_true_is_true(self):
        assert coerce_bool("True") is True
        assert coerce_bool("true") is True
        assert coerce_bool("1") is True
        assert coerce_bool("yes") is True

    def test_native_and_missing(self):
        assert coerce_bool(True) is True
        assert coerce_bool(False) is False
        assert coerce_bool(None) is False
        assert coerce_bool(pd.NA) is False
        assert coerce_bool(float("nan")) is False
        assert coerce_bool(0) is False
        assert coerce_bool(1) is True


class TestCoerceBoolSeries:
    def test_string_false_not_truthy_via_astype(self):
        """Regression: Series.astype(bool) makes non-empty 'False' → True."""
        raw = pd.Series(["False", "True", "false", "true", False, True, None, ""])
        buggy = raw.fillna(False).astype(bool)
        assert buggy.tolist()[0] is True  # documents the footgun

        fixed = coerce_bool_series(raw)
        assert fixed.tolist() == [False, True, False, True, False, True, False, False]
        assert fixed.dtype == bool

    def test_assessment_wrappers_delegate(self):
        series = pd.Series(["False", "True"])
        assert _bool_series(series).tolist() == [False, True]

    def test_stratum_rates_with_string_csv_flags(self):
        df = pd.DataFrame(
            {
                "generation_successful": ["True", "False", "true", "false"],
                "hit_max_tokens": ["False", "True", "false", "false"],
                "meets_a1_criteria": ["True", "False", "False", "False"],
            }
        )
        rates = _stratum_rates(df)
        assert rates["count"] == 4
        assert rates["generation_failure_rate"] == pytest.approx(0.5)
        assert rates["hit_max_tokens_rate"] == pytest.approx(0.25)
        assert rates["a1_pass_rate"] == pytest.approx(0.25)

    def test_run_store_aggregate_with_string_csv_flags(self):
        df = pd.DataFrame(
            {
                "generation_successful": ["True", "False"],
                "hit_max_tokens": ["False", "False"],
                "meets_a1_criteria": ["False", "False"],
                "flesch_kincaid_grade": [2.0, np.nan],
            }
        )
        stats = _aggregate_metric_stats(df)
        assert stats["generation_successful_count"] == 1
        assert stats["generation_failure_rate"] == pytest.approx(0.5)
        assert stats["hit_max_tokens_rate"] == pytest.approx(0.0)
        assert stats["a1_pass_rate"] == pytest.approx(0.0)
