"""Paired-delta + percentile bootstrap analysis over assessment bundles.

Exploratory / descriptive framing only: raw 95% CIs, a descriptive
``ci_excludes_zero`` flag, and no confirmatory language.

Also houses automatic-vs-human validation and adequacy non-inferiority
(issue #20): Kendall τ-b / Spearman ρ, binary agreement, CEFR-band
confusion, and the 0.5 adequacy-drop guardrail on the human subset.

Formal analysis should use an unsampled assessment bundle — sampling can
break ``(model, prompt_id)`` pairs, which then drop out of ``n_pairs`` and
appear in ``n_pairs_dropped``. Rates still use all ``item_map`` rows.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from slm_experiments.core.run_store import KIND_ASSESSMENT, RunStore
from slm_experiments.human.rubric import (
    ANSWER_ADEQUACY,
    OVERALL_SUITABILITY,
    is_human_suitable,
)
from slm_experiments.human.study_export import (
    STUDY_DIRNAME,
    parse_experiment_family,
)

DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_RESAMPLES = 10_000
DEFAULT_CI = 0.95

ANALYSIS_DIRNAME = "analysis"
PAIRED_DELTAS_FILENAME = "paired_deltas.csv"
ANALYSIS_JSON_FILENAME = "analysis.json"
VALIDATION_CSV_FILENAME = "validation.csv"
ADEQUACY_CSV_FILENAME = "adequacy_noninferiority.csv"

ADEQUACY_MARGIN = 0.5
# Distinct namespace so adequacy RNG streams cannot collide with #19 keys
# (``{model}|{family}|{arm}|{metric}``).
ADEQUACY_GROUP_KEY_PREFIX = "adequacy"

CEFR_BANDS_FULL = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_BANDS_COLLAPSED = ("A1", "higher", "unknown")
HUMAN_BINARY_LABELS = ("suitable", "not_suitable")

EasierOrientation = Literal["lower", "higher"]
ArmLabel = Literal["baseline", "intervention"]
Direction = Literal["easier", "harder", "none"]

# Per-family neutral (identity) point and sweep column.
FAMILY_BASELINE: Dict[str, Dict[str, Any]] = {
    "weights": {"column": "weight_factor", "value": 1.0},
    "prompting": {"column": "num_shots", "value": 0},
    "guided": {"column": "guided_top_k", "value": 0},
    "kvl_beam": {"column": "kvl_beam_width", "value": 1},
}

# Metric specs: analysis reads ONLY assessment-bundle artifacts.
METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "cefr_sp_level_ordinal": {
        "column": "cefr_sp_level_ordinal",
        "source": "item_map",
        "scale": "0-5",
        "easier_orientation": "lower",
        "role": "primary",
        "status_column": None,
        "status_ok": None,
    },
    "cefr_tsar_ensemble_ordinal": {
        "column": "cefr_tsar_ensemble_ordinal",
        "source": "scores",
        "scale": "1-6",
        "easier_orientation": "lower",
        "role": "secondary",
        "status_column": "cefr_tsar_status",
        "status_ok": "ok",
    },
    "kvl_v2_mean_score": {
        "column": "kvl_v2_mean_score",
        "source": "scores",
        "scale": "approx_-6_to_+5",
        "easier_orientation": "higher",
        "role": "secondary",
        "status_column": "kvl_v2_status",
        "status_ok": "ok",
    },
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "<na>", "none"}


def resolve_family(row: pd.Series, family: Optional[str] = None) -> str:
    """Resolve sweep family from an explicit arg or ``source_run_id``."""
    if family is not None and str(family).strip():
        return str(family).strip()
    source = row.get("source_run_id", "") if hasattr(row, "get") else ""
    return parse_experiment_family(str(source or ""))


def _sweep_value(row: pd.Series, column: str) -> Optional[float]:
    if _is_missing(row.get(column)):
        return None
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return None


def resolve_arm(row: pd.Series, family: Optional[str] = None) -> ArmLabel:
    """
    Family-aware baseline vs intervention label.

    Resolves the sweep family first, then applies **only** that family's
    sweep column (Decision 2). Unknown families fall back to ``config``.
    """
    fam = resolve_family(row, family)
    spec = FAMILY_BASELINE.get(fam)
    if spec is None:
        config = str(row.get("config", "") or "")
        return "baseline" if config == "control" else "intervention"

    value = _sweep_value(row, spec["column"])
    if value is None:
        config = str(row.get("config", "") or "")
        return "baseline" if config == "control" else "intervention"
    return "baseline" if value == float(spec["value"]) else "intervention"


def arm_key(row: pd.Series, family: Optional[str] = None) -> str:
    """Stable arm identifier from the family's sweep column value."""
    fam = resolve_family(row, family)
    spec = FAMILY_BASELINE.get(fam)
    if spec is None:
        return resolve_arm(row, family=fam)
    value = _sweep_value(row, spec["column"])
    if value is None:
        return resolve_arm(row, family=fam)
    col = spec["column"]
    if float(value).is_integer():
        return f"{col}={int(value)}"
    return f"{col}={value}"


def baseline_arm_key(family: str) -> str:
    spec = FAMILY_BASELINE[family]
    col = spec["column"]
    value = float(spec["value"])
    if value.is_integer():
        return f"{col}={int(value)}"
    return f"{col}={value}"


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


def group_bootstrap_rng(bootstrap_seed: int, group_key: str) -> np.random.Generator:
    """
    Deterministic RNG for one ``(group, metric)`` key.

    Seed entropy is derived from ``bootstrap_seed`` and ``group_key`` alone so
    adding an unrelated model / reordering a dict cannot change an existing
    group's CI stream.
    """
    digest = hashlib.sha256(
        f"{int(bootstrap_seed)}\0{group_key}".encode("utf-8")
    ).digest()
    entropy = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


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


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


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



# ---------------------------------------------------------------------------
# Issue #20 — ordinal association, binary agreement, confusion, adequacy
# ---------------------------------------------------------------------------


def orient_ordinal_higher_suitable(
    values: Union[Sequence[float], np.ndarray],
    *,
    easier_orientation: EasierOrientation,
) -> np.ndarray:
    """Flip scorer ordinals so higher always means easier / more suitable."""
    arr = np.asarray(values, dtype=float)
    if easier_orientation == "lower":
        return -arr
    return arr.copy()


def average_ranks(values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """1-based average ranks; ties share the mid-rank."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        mid = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = mid
        i = j + 1
    return ranks


def kendall_tau_b(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Kendall τ-b with tie correction (unweighted).

    Returns ``{"value": float|None, "reason": str|None}``. Undefined when
    either variable is all-tied (denominator zero) or n < 2.
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    n = int(xv.size)
    if n < 2:
        return {"value": None, "reason": "fewer than 2 finite paired observations"}

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        dx = xv[i + 1 :] - xv[i]
        dy = yv[i + 1 :] - yv[i]
        usable = (dx != 0) & (dy != 0)
        prod = dx[usable] * dy[usable]
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))

    n0 = n * (n - 1) / 2.0

    def _tie_pairs(vals: np.ndarray) -> float:
        _, counts = np.unique(vals, return_counts=True)
        return float(np.sum(counts * (counts - 1) / 2.0))

    n1 = _tie_pairs(xv)
    n2 = _tie_pairs(yv)
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0.0:
        return {
            "value": None,
            "reason": "undefined under all-tied ordinals (τ-b denominator zero)",
        }
    return {"value": float((concordant - discordant) / denom), "reason": None}


def weighted_kendall_tau_b(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    weights: Union[Sequence[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Pair-weighted Kendall τ-b.

    Each unordered pair ``(i, j)`` contributes ``w_i * w_j`` to the
    concordant / discordant / tie-in-x / tie-in-y / total-pair sums:

    ``τ_b_w = (P_w − Q_w) / sqrt((n0_w − n1_w)(n0_w − n2_w))``.

    When all weights are equal this reduces exactly to unweighted ``kendall_tau_b``.
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != xv.shape:
        raise ValueError("weights length mismatch")
    mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(w)
    xv, yv, w = xv[mask], yv[mask], w[mask]
    n = int(xv.size)
    if n < 2:
        return {"value": None, "reason": "fewer than 2 finite paired observations"}

    p_w = 0.0
    q_w = 0.0
    n0_w = 0.0
    n1_w = 0.0
    n2_w = 0.0
    for i in range(n - 1):
        wij = w[i] * w[i + 1 :]
        n0_w += float(wij.sum())
        dx = xv[i + 1 :] - xv[i]
        dy = yv[i + 1 :] - yv[i]
        tie_x = dx == 0
        tie_y = dy == 0
        n1_w += float(wij[tie_x].sum())
        n2_w += float(wij[tie_y].sum())
        usable = (~tie_x) & (~tie_y)
        if not np.any(usable):
            continue
        prod = dx[usable] * dy[usable]
        w_u = wij[usable]
        p_w += float(w_u[prod > 0].sum())
        q_w += float(w_u[prod < 0].sum())

    denom = np.sqrt((n0_w - n1_w) * (n0_w - n2_w))
    if denom == 0.0:
        return {
            "value": None,
            "reason": "undefined under all-tied ordinals (τ-b denominator zero)",
        }
    return {"value": float((p_w - q_w) / denom), "reason": None}


def spearman_rho(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Spearman ρ via Pearson correlation of average ranks (secondary).

    Returns ``{"value": float|None, "reason": str|None}``. Undefined when
    either ranked variable has zero variance (all ties) or n < 2.
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    n = int(xv.size)
    if n < 2:
        return {"value": None, "reason": "fewer than 2 finite paired observations"}

    rx = average_ranks(xv)
    ry = average_ranks(yv)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return {
            "value": None,
            "reason": "undefined under all-tied ordinals (Spearman denominator zero)",
        }
    return {"value": float(np.sum(rx * ry) / denom), "reason": None}


def _weight_normalized_mean(
    values: np.ndarray,
    weights: Optional[np.ndarray],
) -> float:
    vals = np.asarray(values, dtype=float)
    if weights is None:
        return float(vals.mean()) if vals.size else float("nan")
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0.0 or vals.size == 0:
        return float("nan")
    return float(np.sum(w * vals) / total)


def binary_suitability_agreement(
    human_suitable: Union[Sequence[bool], np.ndarray],
    scorer_suitable: Union[Sequence[bool], np.ndarray],
    *,
    weights: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Percent agreement, 2×2 counts, sensitivity, specificity.

    Weighted estimators use Horvitz–Thompson-style weight-normalised means
    (``sum(w * indicator) / sum(w)``). No chance-corrected coefficient.
    """
    human = np.asarray(human_suitable, dtype=bool)
    scorer = np.asarray(scorer_suitable, dtype=bool)
    if human.shape != scorer.shape:
        raise ValueError("human_suitable and scorer_suitable length mismatch")
    n = int(human.size)
    w = None if weights is None else np.asarray(weights, dtype=float)
    if w is not None and w.shape != human.shape:
        raise ValueError("weights length mismatch")

    tp = int(np.sum(human & scorer))
    fn = int(np.sum(human & ~scorer))
    fp = int(np.sum(~human & scorer))
    tn = int(np.sum(~human & ~scorer))
    agree = (human == scorer).astype(float)

    def _block(use_weights: bool) -> Dict[str, Any]:
        ww = w if use_weights else None
        pct = _weight_normalized_mean(agree, ww)
        pos = human
        neg = ~human
        if use_weights and ww is not None:
            sens = (
                float(np.sum(ww[pos] * scorer[pos].astype(float)) / np.sum(ww[pos]))
                if np.any(pos) and float(np.sum(ww[pos])) > 0
                else None
            )
            spec = (
                float(np.sum(ww[neg] * (~scorer[neg]).astype(float)) / np.sum(ww[neg]))
                if np.any(neg) and float(np.sum(ww[neg])) > 0
                else None
            )
        else:
            sens = float(scorer[pos].mean()) if np.any(pos) else None
            spec = float((~scorer[neg]).mean()) if np.any(neg) else None
        return {
            "percent_agreement": None if n == 0 or not np.isfinite(pct) else float(pct),
            "sensitivity": sens,
            "specificity": spec,
            "n": n,
        }

    return {
        "weighted": _block(True) if w is not None else _block(False),
        "unweighted": _block(False),
        "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def confusion_by_cefr_band(
    scorer_bands: Union[Sequence[Any], np.ndarray],
    human_suitable: Union[Sequence[bool], np.ndarray],
) -> Dict[str, Any]:
    """
    Confusion matrices: full A1..C2 × human binary, and collapsed A1/higher.

    Counts plus row proportions. Unknown scorer bands map to ``unknown`` in
    the collapsed matrix; unmatched labels do not inflate full-band rows.
    """
    raw_bands = [
        str(b).strip().upper() if not _is_missing(b) else "UNKNOWN" for b in scorer_bands
    ]
    human = np.asarray(human_suitable, dtype=bool)

    def _matrix(row_labels: Sequence[str], mapped: Sequence[str]) -> Dict[str, Any]:
        counts: List[List[int]] = []
        props: List[List[float]] = []
        mapped_arr = np.asarray(mapped, dtype=object)
        for key in row_labels:
            mask = mapped_arr == key
            n_row = int(mask.sum())
            n_suit = int(np.sum(human[mask])) if n_row else 0
            n_not = n_row - n_suit
            counts.append([n_suit, n_not])
            if n_row == 0:
                props.append([0.0, 0.0])
            else:
                props.append([n_suit / n_row, n_not / n_row])
        return {
            "bands": list(row_labels),
            "human_labels": list(HUMAN_BINARY_LABELS),
            "counts": counts,
            "row_proportions": props,
        }

    full_mapped = [b if b in CEFR_BANDS_FULL else "" for b in raw_bands]
    full = _matrix(CEFR_BANDS_FULL, full_mapped)

    collapsed_mapped = []
    for b in raw_bands:
        if b == "A1":
            collapsed_mapped.append("A1")
        elif b in CEFR_BANDS_FULL:
            collapsed_mapped.append("higher")
        else:
            collapsed_mapped.append("unknown")
    collapsed = _matrix(CEFR_BANDS_COLLAPSED, collapsed_mapped)
    return {"full_band": full, "collapsed_band": collapsed}


def adequacy_group_key(model: str, family: str, arm: str) -> str:
    """RNG group key for adequacy bootstrap (``adequacy|...`` namespace)."""
    return f"{ADEQUACY_GROUP_KEY_PREFIX}|{model}|{family}|{arm}"


def adequacy_noninferiority(
    drops: Union[Sequence[float], np.ndarray],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    group_key: str,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    margin: float = ADEQUACY_MARGIN,
    n_pairs_dropped: int = 0,
) -> Dict[str, Any]:
    """
    Adequacy non-inferiority on paired drops (baseline − intervention).

    ``preserved`` iff the percentile-bootstrap CI upper bound is strictly
    below ``margin`` (default 0.5). Descriptive framing only.
    """
    arr = np.asarray(drops, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_pairs = int(arr.size)
    if n_pairs == 0:
        return {
            "drop": None,
            "ci_low": None,
            "ci_high": None,
            "preserved": False,
            "margin": float(margin),
            "n_pairs": 0,
            "n_pairs_dropped": int(n_pairs_dropped),
            "reason": "no paired human-rated adequacy observations",
        }

    rng = group_bootstrap_rng(bootstrap_seed, group_key)
    boot = percentile_bootstrap_ci(arr, n_resamples=n_resamples, ci=ci, rng=rng)
    ci_high = boot["ci_high"]
    preserved = bool(ci_high < float(margin))
    return {
        "drop": boot["point_estimate"],
        "ci_low": boot["ci_low"],
        "ci_high": ci_high,
        "preserved": preserved,
        "margin": float(margin),
        "n_pairs": n_pairs,
        "n_pairs_dropped": int(n_pairs_dropped),
        "reason": None,
    }


def _percentile_from_replicates(
    replicates: np.ndarray,
    *,
    point: float,
    ci: float,
) -> Dict[str, Optional[float]]:
    arr = np.asarray(replicates, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"point_estimate": point, "ci_low": None, "ci_high": None}
    alpha = (1.0 - float(ci)) / 2.0
    return {
        "point_estimate": float(point),
        "ci_low": float(np.quantile(arr, alpha, method="linear")),
        "ci_high": float(np.quantile(arr, 1.0 - alpha, method="linear")),
    }


def _bootstrap_item_statistic(
    n_items: int,
    statistic: Callable[[np.ndarray], Optional[float]],
    *,
    bootstrap_seed: int,
    group_key: str,
    n_resamples: int,
    ci: float,
    point: Optional[float],
) -> Dict[str, Optional[float]]:
    """Item-level percentile bootstrap for an arbitrary scalar statistic."""
    if point is None or n_items < 1:
        return {"point_estimate": point, "ci_low": None, "ci_high": None}
    if n_items == 1 or n_resamples <= 0:
        return {"point_estimate": point, "ci_low": point, "ci_high": point}

    rng = group_bootstrap_rng(bootstrap_seed, group_key)
    replicates = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n_items, size=n_items)
        val = statistic(idx)
        replicates[i] = float(val) if val is not None and np.isfinite(val) else np.nan
    return _percentile_from_replicates(replicates, point=float(point), ci=ci)


SPEARMAN_WEIGHTED_NOT_REPORTED = (
    "not reported — no weighted Spearman ρ definition adopted; "
    "weighted association uses pair-weighted Kendall τ-b only"
)


def _association_block(
    human: np.ndarray,
    scorer_oriented: np.ndarray,
    weights: np.ndarray,
    *,
    bootstrap_seed: int,
    group_key_prefix: str,
    n_resamples: int,
    ci: float,
) -> Dict[str, Any]:
    """
    Association with item-bootstrap CIs.

    Weighted row: pair-weighted Kendall τ-b (primary). Spearman ρ is
    explicitly ``not_reported`` under weighting — no weighted ρ adopted.
    Unweighted row: standard τ-b + Spearman ρ.
    CIs resample items with replacement and recompute the same statistic.
    """
    tau_u = kendall_tau_b(human, scorer_oriented)
    rho_u = spearman_rho(human, scorer_oriented)
    tau_w = weighted_kendall_tau_b(human, scorer_oriented, weights)
    n = int(human.size)

    def _ci_for(
        *,
        name: str,
        weighting: str,
        point: Optional[float],
        reason: Optional[str],
        statistic: Callable[[np.ndarray], Optional[float]],
    ) -> Dict[str, Any]:
        boot = _bootstrap_item_statistic(
            n,
            statistic,
            bootstrap_seed=bootstrap_seed,
            group_key=f"{group_key_prefix}|{weighting}|{name}",
            n_resamples=n_resamples,
            ci=ci,
            point=point,
        )
        return {
            "value": point,
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "reason": reason,
            # "ok" when computed; "undefined" when ties / n<2 make the
            # coefficient undefined (distinct from weighted "not_reported").
            "status": "ok" if point is not None else "undefined",
        }

    unweighted = {
        "kendall_tau_b": _ci_for(
            name="kendall_tau_b",
            weighting="u",
            point=tau_u["value"],
            reason=tau_u["reason"],
            statistic=lambda idx: kendall_tau_b(
                human[idx], scorer_oriented[idx]
            )["value"],
        ),
        "spearman_rho": _ci_for(
            name="spearman_rho",
            weighting="u",
            point=rho_u["value"],
            reason=rho_u["reason"],
            statistic=lambda idx: spearman_rho(
                human[idx], scorer_oriented[idx]
            )["value"],
        ),
        "n": n,
    }

    weighted = {
        "kendall_tau_b": _ci_for(
            name="kendall_tau_b",
            weighting="w",
            point=tau_w["value"],
            reason=tau_w["reason"],
            statistic=lambda idx: weighted_kendall_tau_b(
                human[idx], scorer_oriented[idx], weights[idx]
            )["value"],
        ),
        # Visible absence: no weighted Spearman adopted.
        "spearman_rho": {
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "reason": SPEARMAN_WEIGHTED_NOT_REPORTED,
            "status": "not_reported",
        },
        "n": n,
    }
    return {"weighted": weighted, "unweighted": unweighted}


def _agreement_with_ci(
    human: np.ndarray,
    scorer: np.ndarray,
    weights: np.ndarray,
    *,
    bootstrap_seed: int,
    group_key_prefix: str,
    n_resamples: int,
    ci: float,
) -> Dict[str, Any]:
    base = binary_suitability_agreement(human, scorer, weights=weights)
    n = int(human.size)
    if n == 0:
        return base
    agree = (human == scorer).astype(float)

    rng_u = group_bootstrap_rng(bootstrap_seed, f"{group_key_prefix}|agree|u")
    boot_u = percentile_bootstrap_ci(agree, n_resamples=n_resamples, ci=ci, rng=rng_u)
    base["unweighted"]["ci_low"] = boot_u["ci_low"]
    base["unweighted"]["ci_high"] = boot_u["ci_high"]

    def weighted_agree(idx: np.ndarray) -> Optional[float]:
        return _weight_normalized_mean(agree[idx], weights[idx])

    boot_w = _bootstrap_item_statistic(
        n,
        weighted_agree,
        bootstrap_seed=bootstrap_seed,
        group_key=f"{group_key_prefix}|agree|w",
        n_resamples=n_resamples,
        ci=ci,
        point=base["weighted"]["percent_agreement"],
    )
    base["weighted"]["ci_low"] = boot_w["ci_low"]
    base["weighted"]["ci_high"] = boot_w["ci_high"]
    return base


def _load_human_study(
    run_dir: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """Return (consensus, source_key_analysis, status_reason)."""
    study_dir = run_dir / STUDY_DIRNAME
    consensus_path = study_dir / "consensus.csv"
    if not consensus_path.exists():
        return None, None, "study/consensus.csv not found"
    consensus = pd.read_csv(consensus_path)
    if consensus.empty:
        return consensus, None, "study/consensus.csv is empty"
    source_key_path = study_dir / "source_key.csv"
    if source_key_path.exists():
        source_key = pd.read_csv(source_key_path)
        if "role" in source_key.columns:
            source_key = source_key[source_key["role"].astype(str) == "analysis"]
    else:
        source_key = pd.DataFrame(columns=["item_id", "inclusion_weight"])
    return consensus, source_key, "ok"


def _scorer_suitable_cefr_sp(row: pd.Series) -> Optional[bool]:
    if "meets_a1_criteria" in row.index and not _is_missing(row.get("meets_a1_criteria")):
        val = row["meets_a1_criteria"]
        if isinstance(val, (str,)):
            return val.strip().lower() in {"true", "1", "yes"}
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return bool(val)
    level = row.get("cefr_sp_level")
    if _is_missing(level):
        return None
    return str(level).strip().upper() == "A1"


def _scorer_suitable_tsar(row: pd.Series) -> Optional[bool]:
    status = row.get("cefr_tsar_status")
    if not _is_missing(status) and str(status) != "ok":
        return None
    label = row.get("cefr_tsar_ensemble_label")
    if _is_missing(label):
        return None
    return str(label).strip().upper() == "A1"


def _validation_metadata() -> Dict[str, Any]:
    return {
        "estimators": {
            "binary_agreement_weighted": (
                "weight-normalised percent agreement: "
                "sum(inclusion_weight * 1_agree) / sum(inclusion_weight)"
            ),
            "binary_agreement_unweighted": (
                "mean of per-item agreement indicators"
            ),
            "kendall_tau_b_weighted": (
                "pair-weighted Kendall τ-b: each unordered pair (i, j) "
                "contributes w_i*w_j to concordant / discordant / tie sums; "
                "primary weighted association"
            ),
            "kendall_tau_b_unweighted": (
                "standard Kendall τ-b with tie correction"
            ),
            "spearman_rho_unweighted": (
                "Pearson correlation of average ranks; reported unweighted only"
            ),
            "spearman_rho_weighted": SPEARMAN_WEIGHTED_NOT_REPORTED,
        },
        "inclusion_weight_caveat": (
            "inclusion_weight is P(select | not in calibration), not the "
            "unconditional inclusion probability; known deferred defect from "
            "the human-study exporter — weights are used as stored."
        ),
        "chance_correction": (
            "omitted — protocol dropped Krippendorff's alpha; no Cohen's kappa "
            "or other chance-corrected agreement coefficient"
        ),
        "ordinal_orientation": (
            "scorer ordinals orientation-normalised so higher = easier / more "
            "suitable (CEFR-SP 0–5 and TSAR 1–6 flipped via negation)"
        ),
        "association_primary": "weighted_kendall_tau_b",
        "association_secondary": "unweighted_spearman_rho",
        "human_consensus": "per-item unweighted median across raters",
        "krippendorff_alpha": "not used",
        "tsar_role": "diagnostic_only",
        "tsar_caveat": (
            "TSAR ensemble A1 discrimination is weak (ensemble A1 F1 = 0.00 on "
            "the TSAR test split); reported as diagnostic only — never a gate"
        ),
        "framing": "exploratory_descriptive",
    }


def _build_scorer_validation(
    frame: pd.DataFrame,
    *,
    scorer_name: str,
    role: str,
    ordinal_col: str,
    band_col: str,
    suitable_col: str,
    easier_orientation: EasierOrientation,
    bootstrap_seed: int,
    n_resamples: int,
    ci: float,
    model: str,
) -> Dict[str, Any]:
    if frame.empty or suitable_col not in frame.columns:
        return {
            "role": role,
            "status": "insufficient_data",
            "reason": "no overlapping human consensus and scorer rows",
        }
    working = frame.dropna(subset=[OVERALL_SUITABILITY, suitable_col]).copy()
    if working.empty:
        return {
            "role": role,
            "status": "insufficient_data",
            "reason": "no overlapping human consensus and scorer rows",
        }

    human_overall = working[OVERALL_SUITABILITY].to_numpy(dtype=float)
    human_adeq = pd.to_numeric(working[ANSWER_ADEQUACY], errors="coerce").to_numpy(
        dtype=float
    )
    human_bin = working["human_suitable"].astype(bool).to_numpy()
    scorer_bin = working[suitable_col].astype(bool).to_numpy()
    weights = (
        pd.to_numeric(working["inclusion_weight"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float)
    )
    raw_ord = pd.to_numeric(working[ordinal_col], errors="coerce").to_numpy(dtype=float)
    oriented = orient_ordinal_higher_suitable(
        raw_ord, easier_orientation=easier_orientation
    )
    assoc_mask = np.isfinite(human_overall) & np.isfinite(oriented)
    prefix = f"validation|{model}|{scorer_name}"

    assoc_overall = _association_block(
        human_overall[assoc_mask],
        oriented[assoc_mask],
        weights[assoc_mask],
        bootstrap_seed=bootstrap_seed,
        group_key_prefix=f"{prefix}|overall",
        n_resamples=n_resamples,
        ci=ci,
    )
    adeq_mask = np.isfinite(human_adeq) & np.isfinite(oriented)
    assoc_adeq = _association_block(
        human_adeq[adeq_mask],
        oriented[adeq_mask],
        weights[adeq_mask],
        bootstrap_seed=bootstrap_seed,
        group_key_prefix=f"{prefix}|adequacy_dim",
        n_resamples=n_resamples,
        ci=ci,
    )
    agreement = _agreement_with_ci(
        human_bin,
        scorer_bin,
        weights,
        bootstrap_seed=bootstrap_seed,
        group_key_prefix=prefix,
        n_resamples=n_resamples,
        ci=ci,
    )
    confusion = confusion_by_cefr_band(working[band_col].tolist(), human_bin)
    return {
        "role": role,
        "status": "ok",
        "ordinal_orientation_applied": easier_orientation,
        "association": {
            "overall_suitability": assoc_overall,
            "answer_adequacy": assoc_adeq,
        },
        "binary_agreement": agreement,
        "confusion": confusion,
        "n_items": int(len(working)),
    }


def _build_judge_validation(
    frame: pd.DataFrame,
    judge_scores: pd.DataFrame,
    *,
    bootstrap_seed: int,
    n_resamples: int,
    ci: float,
    model: str,
) -> Dict[str, Any]:
    if judge_scores is None or judge_scores.empty or "item_id" not in judge_scores.columns:
        return {
            "status": "absent",
            "reason": "judge/judge_scores.csv not present or empty",
        }
    needed = {OVERALL_SUITABILITY, ANSWER_ADEQUACY}
    if not needed.issubset(set(judge_scores.columns)):
        return {
            "status": "absent",
            "reason": "judge_scores.csv missing overall_suitability / answer_adequacy",
        }
    judge = judge_scores.copy()
    judge["item_id"] = judge["item_id"].astype(str)
    judge[OVERALL_SUITABILITY] = pd.to_numeric(
        judge[OVERALL_SUITABILITY], errors="coerce"
    )
    judge[ANSWER_ADEQUACY] = pd.to_numeric(judge[ANSWER_ADEQUACY], errors="coerce")
    judge = judge.dropna(subset=[OVERALL_SUITABILITY, ANSWER_ADEQUACY])
    judge["judge_suitable"] = [
        is_human_suitable(float(o), float(a))
        for o, a in zip(judge[OVERALL_SUITABILITY], judge[ANSWER_ADEQUACY])
    ]
    merged = frame.merge(
        judge[["item_id", "judge_suitable", OVERALL_SUITABILITY]].rename(
            columns={OVERALL_SUITABILITY: "judge_overall"}
        ),
        on="item_id",
        how="inner",
    )
    if merged.empty:
        return {
            "status": "insufficient_data",
            "reason": "no overlap between human consensus and judge scores",
        }
    merged["judge_band"] = "unknown"
    merged["judge_ordinal"] = merged["judge_overall"]
    return _build_scorer_validation(
        merged,
        scorer_name="judge",
        role="secondary",
        ordinal_col="judge_ordinal",
        band_col="judge_band",
        suitable_col="judge_suitable",
        easier_orientation="higher",
        bootstrap_seed=bootstrap_seed,
        n_resamples=n_resamples,
        ci=ci,
        model=model,
    )


def build_validation_section(
    annotated: pd.DataFrame,
    scores: pd.DataFrame,
    consensus: pd.DataFrame,
    source_key: pd.DataFrame,
    *,
    judge_scores: Optional[pd.DataFrame] = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Build validation JSON section + flat CSV rows (model-first)."""
    meta = _validation_metadata()
    if consensus is None or consensus.empty:
        return (
            {
                "status": "no_human_study",
                "reason": "empty consensus",
                "metadata": meta,
                "by_model": {},
            },
            pd.DataFrame(),
        )

    cons = consensus.copy()
    cons["item_id"] = cons["item_id"].astype(str)
    for col in (OVERALL_SUITABILITY, ANSWER_ADEQUACY):
        if col in cons.columns:
            cons[col] = pd.to_numeric(cons[col], errors="coerce")
    if "human_suitable" not in cons.columns:
        cons["human_suitable"] = [
            (
                is_human_suitable(float(o), float(a))
                if pd.notna(o) and pd.notna(a)
                else None
            )
            for o, a in zip(cons[OVERALL_SUITABILITY], cons[ANSWER_ADEQUACY])
        ]

    key = source_key.copy() if source_key is not None else pd.DataFrame()
    if not key.empty:
        key["item_id"] = key["item_id"].astype(str)
        weight_cols = ["item_id"]
        if "inclusion_weight" in key.columns:
            weight_cols.append("inclusion_weight")
        key = key[weight_cols].drop_duplicates("item_id", keep="first")
    else:
        key = pd.DataFrame(columns=["item_id", "inclusion_weight"])

    map_cols = [
        "item_id",
        "model",
        "prompt_id",
        "family",
        "arm",
        "arm_key",
        "cefr_sp_level",
        "cefr_sp_level_ordinal",
        "meets_a1_criteria",
        "generation_successful",
        "hit_max_tokens",
    ]
    present = [c for c in map_cols if c in annotated.columns]
    base = annotated[present].copy()
    base["item_id"] = base["item_id"].astype(str)
    base = base[base["item_id"].str.strip() != ""]

    score_cols = [
        "item_id",
        "cefr_tsar_ensemble_label",
        "cefr_tsar_ensemble_ordinal",
        "cefr_tsar_status",
    ]
    if not scores.empty and "item_id" in scores.columns:
        sc = scores[[c for c in score_cols if c in scores.columns]].drop_duplicates(
            "item_id", keep="first"
        )
        sc["item_id"] = sc["item_id"].astype(str)
        base = base.merge(sc, on="item_id", how="left")
    else:
        for c in score_cols[1:]:
            if c not in base.columns:
                base[c] = pd.NA

    base = base.merge(cons, on="item_id", how="inner")
    if base.empty:
        return (
            {
                "status": "insufficient_data",
                "reason": "no overlap between consensus item_ids and item_map",
                "metadata": meta,
                "by_model": {},
            },
            pd.DataFrame(),
        )
    if "inclusion_weight" in key.columns and not key.empty:
        base = base.merge(key, on="item_id", how="left")
    if "inclusion_weight" not in base.columns:
        base["inclusion_weight"] = 1.0
    base["inclusion_weight"] = pd.to_numeric(
        base["inclusion_weight"], errors="coerce"
    ).fillna(1.0)

    base["cefr_sp_suitable"] = [_scorer_suitable_cefr_sp(r) for _, r in base.iterrows()]
    base["cefr_tsar_suitable"] = [_scorer_suitable_tsar(r) for _, r in base.iterrows()]
    base["cefr_sp_band"] = base["cefr_sp_level"] if "cefr_sp_level" in base.columns else pd.NA
    base["cefr_tsar_band"] = (
        base["cefr_tsar_ensemble_label"]
        if "cefr_tsar_ensemble_label" in base.columns
        else pd.NA
    )

    by_model: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    models = sorted({str(m) for m in base["model"].dropna().unique()})
    for model in models:
        model_all = annotated[annotated["model"].astype(str) == model]
        rates = _stratum_rates(model_all)
        model_human = base[base["model"].astype(str) == model]
        scorers: Dict[str, Any] = {
            "cefr_sp": _build_scorer_validation(
                model_human,
                scorer_name="cefr_sp",
                role="primary",
                ordinal_col="cefr_sp_level_ordinal",
                band_col="cefr_sp_band",
                suitable_col="cefr_sp_suitable",
                easier_orientation="lower",
                bootstrap_seed=bootstrap_seed,
                n_resamples=resamples,
                ci=ci,
                model=model,
            ),
            "cefr_tsar": _build_scorer_validation(
                model_human.dropna(subset=["cefr_tsar_suitable"]),
                scorer_name="cefr_tsar",
                role="diagnostic",
                ordinal_col="cefr_tsar_ensemble_ordinal",
                band_col="cefr_tsar_band",
                suitable_col="cefr_tsar_suitable",
                easier_orientation="lower",
                bootstrap_seed=bootstrap_seed,
                n_resamples=resamples,
                ci=ci,
                model=model,
            ),
        }
        if judge_scores is not None:
            scorers["judge"] = _build_judge_validation(
                model_human,
                judge_scores,
                bootstrap_seed=bootstrap_seed,
                n_resamples=resamples,
                ci=ci,
                model=model,
            )
        else:
            scorers["judge"] = {
                "status": "absent",
                "reason": "judge/judge_scores.csv not present",
            }

        by_model[model] = {"rates": rates, "scorers": scorers}

        for sname, sblock in scorers.items():
            if "binary_agreement" not in sblock:
                csv_rows.append(
                    {
                        "model": model,
                        "scorer": sname,
                        "role": sblock.get("role"),
                        "status": sblock.get("status"),
                        "generation_failure_rate": rates["generation_failure_rate"],
                        "hit_max_tokens_rate": rates["hit_max_tokens_rate"],
                        "a1_pass_rate": rates["a1_pass_rate"],
                    }
                )
                continue
            ba = sblock.get("binary_agreement") or {}
            for weighting in ("weighted", "unweighted"):
                block = ba.get(weighting) or {}
                tau = (
                    (sblock.get("association") or {})
                    .get("overall_suitability", {})
                    .get(weighting, {})
                    .get("kendall_tau_b", {})
                )
                rho = (
                    (sblock.get("association") or {})
                    .get("overall_suitability", {})
                    .get(weighting, {})
                    .get("spearman_rho", {})
                )
                csv_rows.append(
                    {
                        "model": model,
                        "scorer": sname,
                        "role": sblock.get("role"),
                        "status": sblock.get("status", "ok"),
                        "weighting": weighting,
                        "percent_agreement": block.get("percent_agreement"),
                        "agreement_ci_low": block.get("ci_low"),
                        "agreement_ci_high": block.get("ci_high"),
                        "sensitivity": block.get("sensitivity"),
                        "specificity": block.get("specificity"),
                        "kendall_tau_b": tau.get("value"),
                        "kendall_tau_b_ci_low": tau.get("ci_low"),
                        "kendall_tau_b_ci_high": tau.get("ci_high"),
                        "kendall_tau_b_status": tau.get("status"),
                        "spearman_rho": rho.get("value"),
                        "spearman_rho_ci_low": rho.get("ci_low"),
                        "spearman_rho_ci_high": rho.get("ci_high"),
                        "spearman_rho_status": rho.get("status"),
                        "n_items": sblock.get("n_items"),
                        "tp": (ba.get("counts") or {}).get("tp"),
                        "fn": (ba.get("counts") or {}).get("fn"),
                        "fp": (ba.get("counts") or {}).get("fp"),
                        "tn": (ba.get("counts") or {}).get("tn"),
                        "generation_failure_rate": rates["generation_failure_rate"],
                        "hit_max_tokens_rate": rates["hit_max_tokens_rate"],
                        "a1_pass_rate": rates["a1_pass_rate"],
                    }
                )

    section = {"status": "ok", "metadata": meta, "by_model": by_model}
    csv_df = pd.DataFrame(csv_rows)
    if not csv_df.empty:
        sort_cols = [c for c in ("model", "scorer", "weighting") if c in csv_df.columns]
        csv_df = csv_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        csv_df = csv_df[["model"] + [c for c in csv_df.columns if c != "model"]]
    return section, csv_df


def build_adequacy_noninferiority_section(
    annotated: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    margin: float = ADEQUACY_MARGIN,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Paired adequacy-drop non-inferiority on human-rated items only."""
    meta = {
        "margin": float(margin),
        "preserved_rule": "ci_high < margin",
        "drop_definition": "baseline_answer_adequacy - intervention_answer_adequacy",
        "pairing_unit": "(model, prompt_id) within family arms",
        "bootstrap": {
            "method": "percentile_bootstrap",
            "resamples": int(resamples),
            "ci": float(ci),
            "seed": int(bootstrap_seed),
            "group_key_namespace": ADEQUACY_GROUP_KEY_PREFIX,
        },
        "framing": "exploratory_descriptive",
        "note": (
            "'preserved' / 'not preserved' are descriptive labels from the "
            "CI-vs-margin rule; they do not prove non-inferiority."
        ),
    }
    if consensus is None or consensus.empty:
        return (
            {
                "status": "no_human_study",
                "reason": "empty consensus",
                "margin": float(margin),
                "metadata": meta,
                "by_model": {},
            },
            pd.DataFrame(),
        )

    cons = consensus.copy()
    cons["item_id"] = cons["item_id"].astype(str)
    if ANSWER_ADEQUACY not in cons.columns:
        return (
            {
                "status": "insufficient_data",
                "reason": "consensus missing answer_adequacy",
                "margin": float(margin),
                "metadata": meta,
                "by_model": {},
            },
            pd.DataFrame(),
        )
    cons[ANSWER_ADEQUACY] = pd.to_numeric(cons[ANSWER_ADEQUACY], errors="coerce")
    cons = cons.dropna(subset=[ANSWER_ADEQUACY])

    # Ensure item_id dtype matches.
    ann = annotated.copy()
    ann["item_id"] = ann["item_id"].astype(str)
    human = ann.merge(cons[["item_id", ANSWER_ADEQUACY]], on="item_id", how="inner")
    if human.empty:
        return (
            {
                "status": "insufficient_data",
                "reason": "no human-rated items overlap item_map",
                "margin": float(margin),
                "metadata": meta,
                "by_model": {},
            },
            pd.DataFrame(),
        )

    by_model: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    models = sorted({str(m) for m in human["model"].dropna().unique()})
    for model in models:
        model_df = human[human["model"].astype(str) == model]
        model_all = ann[ann["model"].astype(str) == model]
        family_blocks: Dict[str, Any] = {}
        families = sorted({str(f) for f in model_df["family"].dropna().unique()})
        for family in families:
            if family not in FAMILY_BASELINE:
                family_blocks[family] = {
                    "status": "no_baseline",
                    "reason": f"Unknown sweep family {family!r}",
                }
                continue
            fam_df = model_df[model_df["family"] == family]
            fam_all = model_all[model_all["family"] == family]
            base_key = baseline_arm_key(family)
            baseline_rows = fam_df[fam_df["arm_key"] == base_key]
            if baseline_rows.empty:
                family_blocks[family] = {
                    "status": "no_baseline",
                    "reason": f"No human-rated baseline ({base_key})",
                    "baseline_arm": base_key,
                }
                continue
            baseline_rates = _stratum_rates(
                fam_all[fam_all["arm_key"] == base_key]
                if not fam_all.empty
                else baseline_rows
            )
            intervention_keys = sorted(
                k
                for k in fam_df["arm_key"].unique()
                if k != base_key
                and resolve_arm(
                    fam_df[fam_df["arm_key"] == k].iloc[0], family=family
                )
                == "intervention"
            )
            by_arm: Dict[str, Any] = {}
            for ikey in intervention_keys:
                int_rows = fam_df[fam_df["arm_key"] == ikey]
                int_rates = _stratum_rates(
                    fam_all[fam_all["arm_key"] == ikey]
                    if not fam_all.empty
                    else int_rows
                )
                base_vals = baseline_rows[["prompt_id", ANSWER_ADEQUACY]].drop_duplicates(
                    "prompt_id", keep="first"
                )
                int_vals = int_rows[["prompt_id", ANSWER_ADEQUACY]].drop_duplicates(
                    "prompt_id", keep="first"
                )
                pairs = build_paired_deltas(
                    base_vals.rename(columns={ANSWER_ADEQUACY: "val"}),
                    int_vals.rename(columns={ANSWER_ADEQUACY: "val"}),
                    value_col="val",
                )
                if pairs.empty:
                    drops = np.array([], dtype=float)
                    n_dropped = len(
                        set(base_vals["prompt_id"].astype(str))
                        | set(int_vals["prompt_id"].astype(str))
                    )
                else:
                    # build_paired_deltas = intervention − baseline; flip → drop.
                    drops = -pairs["delta"].to_numpy(dtype=float)
                    n_dropped = int(pairs["n_pairs_dropped"].iloc[0])

                result = adequacy_noninferiority(
                    drops,
                    bootstrap_seed=bootstrap_seed,
                    group_key=adequacy_group_key(model, family, ikey),
                    n_resamples=resamples,
                    ci=ci,
                    margin=margin,
                    n_pairs_dropped=n_dropped,
                )
                arm_block = {
                    **result,
                    "generation_failure_rate": int_rates["generation_failure_rate"],
                    "hit_max_tokens_rate": int_rates["hit_max_tokens_rate"],
                    "a1_pass_rate": int_rates["a1_pass_rate"],
                    "baseline_generation_failure_rate": baseline_rates[
                        "generation_failure_rate"
                    ],
                    "baseline_hit_max_tokens_rate": baseline_rates[
                        "hit_max_tokens_rate"
                    ],
                    "baseline_a1_pass_rate": baseline_rates["a1_pass_rate"],
                }
                by_arm[ikey] = arm_block
                csv_rows.append(
                    {
                        "model": model,
                        "family": family,
                        "arm": ikey,
                        "drop": arm_block["drop"],
                        "ci_low": arm_block["ci_low"],
                        "ci_high": arm_block["ci_high"],
                        "preserved": arm_block["preserved"],
                        "margin": margin,
                        "n_pairs": arm_block["n_pairs"],
                        "n_pairs_dropped": arm_block["n_pairs_dropped"],
                        "generation_failure_rate": arm_block["generation_failure_rate"],
                        "hit_max_tokens_rate": arm_block["hit_max_tokens_rate"],
                        "a1_pass_rate": arm_block["a1_pass_rate"],
                    }
                )
            family_blocks[family] = {
                "status": "ok",
                "baseline_arm": base_key,
                "by_arm": by_arm,
            }
        by_model[model] = {"by_family": family_blocks}

    section = {
        "status": "ok",
        "margin": float(margin),
        "metadata": meta,
        "by_model": by_model,
    }
    csv_df = pd.DataFrame(csv_rows)
    if not csv_df.empty:
        csv_df = csv_df.sort_values(["model", "family", "arm"]).reset_index(drop=True)
        csv_df = csv_df[["model"] + [c for c in csv_df.columns if c != "model"]]
    return section, csv_df


def analyze_assessment_bundle(
    assessment_run_id: str,
    *,
    results_root: Optional[Union[Path, str]] = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
) -> Path:
    """
    Write paired-delta, validation, and adequacy artifacts under ``analysis/``.

    Updates the assessment manifest ``analysis`` block and ``artifacts``.
    Never mutates source generation runs. Validation / adequacy sections run
    only when ``study/consensus.csv`` exists; otherwise they record
    ``status: no_human_study`` and the paired-delta sections still write.
    """
    if results_root is not None:
        root = Path(results_root)
    else:
        from slm_experiments.models.base import REPO_ROOT as repo_root

        root = Path(repo_root) / "results"

    store = RunStore(root)
    run_dir = store.run_dir(assessment_run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Assessment bundle not found: {assessment_run_id}")

    manifest = store.read_manifest(assessment_run_id)
    if manifest.get("kind") != KIND_ASSESSMENT:
        raise ValueError(
            f"Run {assessment_run_id} is not kind=assessment "
            f"(got {manifest.get('kind')!r})"
        )

    item_map = store.read_item_map_csv(assessment_run_id)
    scores_path = run_dir / "scores.csv"
    scores = (
        pd.read_csv(scores_path)
        if scores_path.exists()
        else pd.DataFrame(columns=["item_id"])
    )

    annotated = _annotate_map(item_map)
    analysis_dir = run_dir / ANALYSIS_DIRNAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: List[str] = []
    by_model: Dict[str, Any] = {}
    delta_rows: List[Dict[str, Any]] = []

    models = sorted({str(m) for m in annotated["model"].dropna().unique()})
    for model in models:
        model_df = annotated[annotated["model"].astype(str) == model]
        families = sorted({str(f) for f in model_df["family"].dropna().unique()})
        family_blocks: Dict[str, Any] = {}

        for family in families:
            fam_df = model_df[model_df["family"] == family]
            if family not in FAMILY_BASELINE:
                reason = f"Unknown sweep family {family!r}; skipped."
                family_blocks[family] = {"status": "no_baseline", "reason": reason}
                warnings_list.append(f"{model}/{family}: {reason}")
                warnings.warn(reason, stacklevel=2)
                continue

            base_key = baseline_arm_key(family)
            baseline_rows = fam_df[fam_df["arm_key"] == base_key]
            if baseline_rows.empty:
                reason = (
                    f"No neutral baseline ({base_key}) for family {family!r} "
                    f"and model {model!r}."
                )
                family_blocks[family] = {
                    "status": "no_baseline",
                    "reason": reason,
                    "baseline_arm": base_key,
                }
                warnings_list.append(f"{model}/{family}: {reason}")
                warnings.warn(reason, stacklevel=2)
                continue

            baseline_rates = _stratum_rates(baseline_rows)
            intervention_keys = sorted(
                k
                for k in fam_df["arm_key"].unique()
                if k != base_key and resolve_arm(
                    fam_df[fam_df["arm_key"] == k].iloc[0], family=family
                )
                == "intervention"
            )

            by_arm: Dict[str, Any] = {}
            for ikey in intervention_keys:
                int_rows = fam_df[fam_df["arm_key"] == ikey]
                int_rates = _stratum_rates(int_rows)
                a1_delta = float(int_rates["a1_pass_rate"] - baseline_rates["a1_pass_rate"])

                metrics_out: Dict[str, Any] = {}
                for metric_name, spec in METRIC_SPECS.items():
                    base_vals = _metric_frame(baseline_rows, scores, metric_name)
                    int_vals = _metric_frame(int_rows, scores, metric_name)
                    pairs = build_paired_deltas(
                        base_vals,
                        int_vals,
                        value_col=spec["column"],
                    )
                    n_pairs = int(len(pairs))
                    n_dropped = (
                        int(pairs["n_pairs_dropped"].iloc[0]) if n_pairs else (
                            len(
                                set(base_vals["prompt_id"].astype(str))
                                | set(int_vals["prompt_id"].astype(str))
                            )
                            if not base_vals.empty or not int_vals.empty
                            else 0
                        )
                    )

                    group_key = f"{model}|{family}|{ikey}|{metric_name}"
                    if n_pairs == 0:
                        metric_block = {
                            "delta": None,
                            "easier_delta": None,
                            "ci_low": None,
                            "ci_high": None,
                            "ci_excludes_zero": False,
                            "direction": "none",
                            "n_pairs": 0,
                            "n_pairs_dropped": n_dropped,
                            "scale": spec["scale"],
                            "easier_orientation": spec["easier_orientation"],
                            "role": spec["role"],
                        }
                    else:
                        raw = pairs["delta"].to_numpy(dtype=float)
                        easy = easier_deltas(
                            raw, easier_orientation=spec["easier_orientation"]
                        )
                        rng = group_bootstrap_rng(bootstrap_seed, group_key)
                        boot = percentile_bootstrap_ci(
                            easy,
                            n_resamples=resamples,
                            ci=ci,
                            rng=rng,
                        )
                        excludes = ci_excludes_zero(boot["ci_low"], boot["ci_high"])
                        direction = direction_from_ci(boot["ci_low"], boot["ci_high"])
                        metric_block = {
                            "delta": float(raw.mean()),
                            "easier_delta": float(easy.mean()),
                            "ci_low": boot["ci_low"],
                            "ci_high": boot["ci_high"],
                            "ci_excludes_zero": excludes,
                            "direction": direction,
                            "n_pairs": n_pairs,
                            "n_pairs_dropped": int(pairs["n_pairs_dropped"].iloc[0]),
                            "scale": spec["scale"],
                            "easier_orientation": spec["easier_orientation"],
                            "role": spec["role"],
                        }

                    metrics_out[metric_name] = metric_block
                    delta_rows.append(
                        {
                            "model": model,
                            "family": family,
                            "arm": ikey,
                            "metric": metric_name,
                            "role": spec["role"],
                            "scale": spec["scale"],
                            "easier_orientation": spec["easier_orientation"],
                            "delta": metric_block["delta"],
                            "easier_delta": metric_block["easier_delta"],
                            "ci_low": metric_block["ci_low"],
                            "ci_high": metric_block["ci_high"],
                            "ci_excludes_zero": metric_block["ci_excludes_zero"],
                            "direction": metric_block["direction"],
                            "n_pairs": metric_block["n_pairs"],
                            "n_pairs_dropped": metric_block["n_pairs_dropped"],
                            "generation_failure_rate": int_rates[
                                "generation_failure_rate"
                            ],
                            "hit_max_tokens_rate": int_rates["hit_max_tokens_rate"],
                            "a1_pass_rate": int_rates["a1_pass_rate"],
                            "baseline_generation_failure_rate": baseline_rates[
                                "generation_failure_rate"
                            ],
                            "baseline_hit_max_tokens_rate": baseline_rates[
                                "hit_max_tokens_rate"
                            ],
                            "baseline_a1_pass_rate": baseline_rates["a1_pass_rate"],
                            "a1_pass_rate_delta": a1_delta,
                            "intervention_count": int_rates["count"],
                            "baseline_count": baseline_rates["count"],
                        }
                    )

                by_arm[ikey] = {
                    "rates": {
                        "generation_failure_rate": int_rates["generation_failure_rate"],
                        "hit_max_tokens_rate": int_rates["hit_max_tokens_rate"],
                        "a1_pass_rate": int_rates["a1_pass_rate"],
                        "baseline_generation_failure_rate": baseline_rates[
                            "generation_failure_rate"
                        ],
                        "baseline_hit_max_tokens_rate": baseline_rates[
                            "hit_max_tokens_rate"
                        ],
                        "baseline_a1_pass_rate": baseline_rates["a1_pass_rate"],
                        "a1_pass_rate_delta": a1_delta,
                        "intervention_count": int_rates["count"],
                        "baseline_count": baseline_rates["count"],
                    },
                    "metrics": metrics_out,
                }

            family_blocks[family] = {
                "status": "ok",
                "baseline_arm": base_key,
                "baseline_rates": baseline_rates,
                "by_arm": by_arm,
            }

        by_model[model] = {"by_family": family_blocks}

    metadata = {
        "bootstrap_seed": int(bootstrap_seed),
        "resamples": int(resamples),
        "ci": float(ci),
        "method": "percentile_bootstrap",
        "framing": "exploratory_descriptive",
        "multiplicity_correction": None,
        "metrics": {
            name: {
                "scale": spec["scale"],
                "easier_orientation": spec["easier_orientation"],
                "role": spec["role"],
                "source": spec["source"],
            }
            for name, spec in METRIC_SPECS.items()
        },
        "note": (
            "Formal analysis should use an unsampled assessment bundle; "
            "sampling can break pairs (counted in n_pairs_dropped). "
            "Rates use all item_map rows. Raw CIs only; ci_excludes_zero is "
            "descriptive."
        ),
    }

    # Issue #20 — validation + adequacy (human study optional).
    consensus, source_key, study_status = _load_human_study(run_dir)
    judge_path = run_dir / "judge" / "judge_scores.csv"
    judge_scores = (
        pd.read_csv(judge_path) if judge_path.exists() else None
    )

    if study_status != "ok" or consensus is None:
        validation_section: Dict[str, Any] = {
            "status": "no_human_study",
            "reason": study_status,
            "metadata": _validation_metadata(),
            "by_model": {},
        }
        adequacy_section: Dict[str, Any] = {
            "status": "no_human_study",
            "reason": study_status,
            "margin": float(ADEQUACY_MARGIN),
            "metadata": {
                "margin": float(ADEQUACY_MARGIN),
                "preserved_rule": "ci_high < margin",
                "framing": "exploratory_descriptive",
            },
            "by_model": {},
        }
        validation_df = pd.DataFrame()
        adequacy_df = pd.DataFrame()
    else:
        validation_section, validation_df = build_validation_section(
            annotated,
            scores,
            consensus,
            source_key if source_key is not None else pd.DataFrame(),
            judge_scores=judge_scores,
            bootstrap_seed=bootstrap_seed,
            resamples=resamples,
            ci=ci,
        )
        adequacy_section, adequacy_df = build_adequacy_noninferiority_section(
            annotated,
            consensus,
            bootstrap_seed=bootstrap_seed,
            resamples=resamples,
            ci=ci,
        )

    analysis_doc: Dict[str, Any] = {
        "metadata": metadata,
        "by_model": by_model,
        "validation": validation_section,
        "adequacy_noninferiority": adequacy_section,
        "warnings": warnings_list,
    }

    deltas_df = pd.DataFrame(delta_rows)
    if not deltas_df.empty:
        # Model-first stratification (AGENTS.md §6).
        deltas_df = deltas_df.sort_values(
            ["model", "family", "arm", "metric"]
        ).reset_index(drop=True)
        # Ensure model is the first column.
        cols = ["model"] + [c for c in deltas_df.columns if c != "model"]
        deltas_df = deltas_df[cols]
    else:
        deltas_df = pd.DataFrame(
            columns=[
                "model",
                "family",
                "arm",
                "metric",
                "delta",
                "easier_delta",
                "ci_low",
                "ci_high",
                "ci_excludes_zero",
                "direction",
                "n_pairs",
                "n_pairs_dropped",
                "generation_failure_rate",
                "hit_max_tokens_rate",
                "a1_pass_rate_delta",
            ]
        )

    deltas_path = analysis_dir / PAIRED_DELTAS_FILENAME
    analysis_path = analysis_dir / ANALYSIS_JSON_FILENAME
    validation_path = analysis_dir / VALIDATION_CSV_FILENAME
    adequacy_path = analysis_dir / ADEQUACY_CSV_FILENAME
    # Match generation/assessment CSV writers: missing numerics → empty cells
    # (not the literal string "NaN"). Explicit na_rep pins the contract.
    deltas_df.to_csv(deltas_path, index=False, na_rep="")
    analysis_path.write_text(json.dumps(analysis_doc, indent=2), encoding="utf-8")

    artifacts = dict(manifest.get("artifacts") or {})
    artifacts["paired_deltas_csv"] = f"{ANALYSIS_DIRNAME}/{PAIRED_DELTAS_FILENAME}"
    artifacts["analysis_json"] = f"{ANALYSIS_DIRNAME}/{ANALYSIS_JSON_FILENAME}"

    if validation_section.get("status") == "ok" and not validation_df.empty:
        validation_df.to_csv(validation_path, index=False, na_rep="")
        artifacts["validation_csv"] = f"{ANALYSIS_DIRNAME}/{VALIDATION_CSV_FILENAME}"
    elif validation_path.exists():
        validation_path.unlink()
        artifacts.pop("validation_csv", None)

    if adequacy_section.get("status") == "ok" and not adequacy_df.empty:
        adequacy_df.to_csv(adequacy_path, index=False, na_rep="")
        artifacts["adequacy_noninferiority_csv"] = (
            f"{ANALYSIS_DIRNAME}/{ADEQUACY_CSV_FILENAME}"
        )
    elif adequacy_path.exists():
        adequacy_path.unlink()
        artifacts.pop("adequacy_noninferiority_csv", None)

    # Update manifest (assessment bundler owns its manifest; additive only).
    manifest["analysis"] = {
        "bootstrap_seed": int(bootstrap_seed),
        "resamples": int(resamples),
        "ci": float(ci),
    }
    manifest["artifacts"] = artifacts
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return analysis_dir
