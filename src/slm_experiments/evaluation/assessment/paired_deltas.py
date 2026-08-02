"""Paired-delta + percentile bootstrap machinery (issue #19)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from slm_experiments.evaluation.assessment.analysis_helpers import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CI,
    DEFAULT_RESAMPLES,
    METRIC_SPECS,
    Direction,
    EasierOrientation,
    _bool_series,
    arm_key,
    resolve_arm,
)
from slm_experiments.human.study_export import parse_experiment_family

def easier_deltas(
    raw_deltas: np.ndarray,
    *,
    easier_orientation: EasierOrientation,
) -> np.ndarray:
    """Orientation-normalised deltas where positive always means easier."""
    arr = np.asarray(raw_deltas, dtype=float)
    if easier_orientation == "lower":
        return -arr
    return arr.copy()


def ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    """Descriptive flag: True when the two-sided CI does not contain 0."""
    return not (ci_low <= 0.0 <= ci_high)


def direction_from_ci(ci_low: float, ci_high: float) -> Direction:
    """Descriptive direction from an easier_delta CI (positive = easier)."""
    if ci_low > 0.0:
        return "easier"
    if ci_high < 0.0:
        return "harder"
    return "none"

def percentile_bootstrap_ci(
    values: Union[Sequence[float], np.ndarray],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Percentile bootstrap CI for the mean of ``values``.

    Resamples ``values`` with replacement, takes the mean of each replicate,
    then forms the two-sided interval from the empirical quantiles at
    ``(1 - ci) / 2`` and ``1 - (1 - ci) / 2`` via ``numpy.quantile`` with
    ``method="linear"`` (NumPy's default linear interpolation).

    Returns ``point_estimate``, ``ci_low``, ``ci_high``. Reusable by issue #20
    (human adequacy non-inferiority) — do not specialize this helper.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "point_estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    point = float(arr.mean())
    if arr.size == 1 or n_resamples <= 0:
        return {"point_estimate": point, "ci_low": point, "ci_high": point}

    if rng is None:
        if seed is None:
            seed = DEFAULT_BOOTSTRAP_SEED
        rng = np.random.default_rng(seed)

    n = arr.size
    # Resample with replacement; mean per replicate.
    indices = rng.integers(0, n, size=(int(n_resamples), n))
    means = arr[indices].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    low = float(np.quantile(means, alpha, method="linear"))
    high = float(np.quantile(means, 1.0 - alpha, method="linear"))
    return {"point_estimate": point, "ci_low": low, "ci_high": high}

def build_paired_deltas(
    baseline: pd.DataFrame,
    intervention: pd.DataFrame,
    *,
    value_col: str,
    prompt_col: str = "prompt_id",
) -> pd.DataFrame:
    """
    Build per-``prompt_id`` raw deltas (intervention − baseline).

    Incomplete pairs (missing either side or non-finite value) are dropped.
    ``n_pairs_dropped`` is attached to every retained row for convenience.
    """
    base = baseline[[prompt_col, value_col]].copy()
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    base = base.dropna(subset=[value_col]).drop_duplicates(subset=[prompt_col], keep="first")

    interv = intervention[[prompt_col, value_col]].copy()
    interv[value_col] = pd.to_numeric(interv[value_col], errors="coerce")
    interv = interv.dropna(subset=[value_col]).drop_duplicates(
        subset=[prompt_col], keep="first"
    )

    candidate_prompts = sorted(
        set(baseline[prompt_col].astype(str)) | set(intervention[prompt_col].astype(str))
    )
    merged = base.merge(
        interv,
        on=prompt_col,
        how="inner",
        suffixes=("_baseline", "_intervention"),
    )
    merged["delta"] = merged[f"{value_col}_intervention"] - merged[f"{value_col}_baseline"]
    n_kept = int(len(merged))
    n_dropped = int(len(candidate_prompts) - n_kept)
    merged["n_pairs_dropped"] = n_dropped
    merged = merged.sort_values(prompt_col).reset_index(drop=True)
    return merged[[prompt_col, "delta", "n_pairs_dropped"]]

def _stratum_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Rate conventions matching ``RunStore._aggregate_metric_stats``."""
    n = int(len(df))
    if n == 0:
        return {
            "generation_failure_rate": 0.0,
            "hit_max_tokens_rate": 0.0,
            "a1_pass_rate": 0.0,
            "count": 0,
        }
    successful = _bool_series(df["generation_successful"]) if "generation_successful" in df else pd.Series(False, index=df.index)
    maxed = (
        _bool_series(df["hit_max_tokens"])
        if "hit_max_tokens" in df
        else pd.Series(False, index=df.index)
    )
    if "meets_a1_criteria" in df.columns:
        a1 = _bool_series(df["meets_a1_criteria"])
        a1_pass_rate = float(a1.sum() / n)
    else:
        a1_pass_rate = 0.0
    return {
        "generation_failure_rate": float(1.0 - successful.sum() / n),
        "hit_max_tokens_rate": float(maxed.sum() / n),
        "a1_pass_rate": a1_pass_rate,
        "count": n,
    }


def _metric_frame(
    map_rows: pd.DataFrame,
    scores: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    """Per-prompt rows with a finite metric value after status gating."""
    spec = METRIC_SPECS[metric_name]
    col = spec["column"]
    working = map_rows.copy()

    if spec["source"] == "scores":
        if scores.empty or "item_id" not in scores.columns:
            return pd.DataFrame(columns=["prompt_id", col])
        score_cols = ["item_id", col]
        status_col = spec["status_column"]
        if status_col and status_col in scores.columns:
            score_cols.append(status_col)
        scored = scores[score_cols].drop_duplicates("item_id", keep="first")
        working = working.merge(scored, on="item_id", how="inner")
        if status_col and status_col in working.columns:
            working = working[working[status_col] == spec["status_ok"]]
    else:
        # CEFR-SP ordinal lives on item_map after the schema extension.
        if col not in working.columns:
            return pd.DataFrame(columns=["prompt_id", col])

    # Analysis runs over scored (in_sample) items when the flag exists.
    if "in_sample" in working.columns:
        working = working[_bool_series(working["in_sample"])]
    working = working[working["item_id"].astype(str).str.strip() != ""]
    working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=[col])
    return working[["prompt_id", col]].drop_duplicates("prompt_id", keep="first")


def _annotate_map(item_map: pd.DataFrame) -> pd.DataFrame:
    out = item_map.copy()
    out["family"] = out["source_run_id"].map(
        lambda rid: parse_experiment_family(str(rid or ""))
    )
    out["arm"] = [
        resolve_arm(row, family=row["family"]) for _, row in out.iterrows()
    ]
    out["arm_key"] = [
        arm_key(row, family=row["family"]) for _, row in out.iterrows()
    ]
    return out
