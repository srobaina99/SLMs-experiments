"""Shared constants and helpers for assessment analysis (#19 / #20)."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
import pandas as pd

from slm_experiments.core.bool_series import coerce_bool_series as _bool_series
from slm_experiments.human.study_export import parse_experiment_family

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
