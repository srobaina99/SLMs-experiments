"""Reliability and consensus for three-rater ordinal ratings.

Reports exact + adjacent (±1) percent agreement per dimension and consensus
medians. Krippendorff's alpha is intentionally not computed.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from slm_experiments.human.rubric import (
    ANSWER_ADEQUACY,
    OVERALL_SUITABILITY,
    RATING_DIMENSIONS,
    RUBRIC_VERSION,
    is_human_suitable,
)


def _pairwise_pairs(rater_ids: Sequence[str]) -> List[tuple[str, str]]:
    return list(combinations(sorted(rater_ids), 2))


def _agreement_rates(
    wide: pd.DataFrame,
    dimension: str,
    pairs: Sequence[tuple[str, str]],
) -> Dict[str, Optional[float]]:
    """Mean pairwise exact and adjacent (±1) agreement for one dimension."""
    if wide.empty or not pairs:
        return {"exact_agreement": None, "adjacent_agreement": None}

    exact_rates: List[float] = []
    adjacent_rates: List[float] = []
    for left, right in pairs:
        left_col = f"{dimension}__{left}"
        right_col = f"{dimension}__{right}"
        if left_col not in wide.columns or right_col not in wide.columns:
            continue
        subset = wide[[left_col, right_col]].dropna()
        if subset.empty:
            continue
        diffs = (subset[left_col] - subset[right_col]).abs()
        exact_rates.append(float((diffs == 0).mean()))
        adjacent_rates.append(float((diffs <= 1).mean()))

    if not exact_rates:
        return {"exact_agreement": None, "adjacent_agreement": None}
    return {
        "exact_agreement": float(sum(exact_rates) / len(exact_rates)),
        "adjacent_agreement": float(sum(adjacent_rates) / len(adjacent_rates)),
    }


def pivot_ratings_long_to_wide(
    ratings: pd.DataFrame,
    dimensions: Iterable[str] = RATING_DIMENSIONS,
) -> pd.DataFrame:
    """Pivot long-format ratings to one row per item_id with dimension__rater columns."""
    if ratings.empty:
        return pd.DataFrame(columns=["item_id"])

    frames: List[pd.DataFrame] = [ratings[["item_id"]].drop_duplicates()]
    for dim in dimensions:
        if dim not in ratings.columns:
            continue
        pivoted = ratings.pivot(index="item_id", columns="rater_id", values=dim)
        pivoted = pivoted.rename(columns={c: f"{dim}__{c}" for c in pivoted.columns})
        pivoted = pivoted.reset_index()
        frames.append(pivoted)

    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on="item_id", how="outer")
    return wide


def consensus_medians(
    ratings: pd.DataFrame,
    dimensions: Sequence[str] = RATING_DIMENSIONS,
) -> pd.DataFrame:
    """
    Per-item median across raters for each dimension, plus binary human_suitable.

    ``human_suitable`` uses median overall ≥ 3 AND median adequacy ≥ 3.
    """
    if ratings.empty:
        cols = ["item_id", *dimensions, "human_suitable", "n_raters"]
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for item_id, group in ratings.groupby("item_id", sort=False):
        record: Dict[str, Any] = {
            "item_id": item_id,
            "n_raters": int(group["rater_id"].nunique()),
        }
        for dim in dimensions:
            values = pd.to_numeric(group[dim], errors="coerce").dropna()
            record[dim] = float(values.median()) if not values.empty else None
        overall = record.get(OVERALL_SUITABILITY)
        adequacy = record.get(ANSWER_ADEQUACY)
        if overall is None or adequacy is None:
            record["human_suitable"] = None
        else:
            record["human_suitable"] = bool(
                is_human_suitable(float(overall), float(adequacy))
            )
        rows.append(record)

    return pd.DataFrame(rows)


def compute_reliability(
    ratings: pd.DataFrame,
    dimensions: Sequence[str] = RATING_DIMENSIONS,
) -> Dict[str, Any]:
    """
    Exact + adjacent (±1) percent agreement per dimension, plus consensus table.

    Returns a JSON-serialisable dict. Does **not** compute Krippendorff's alpha.
    """
    required = {"item_id", "rater_id", *dimensions}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing columns: {sorted(missing)}")

    working = ratings.copy()
    for dim in dimensions:
        working[dim] = pd.to_numeric(working[dim], errors="coerce")

    rater_ids = sorted(working["rater_id"].astype(str).unique().tolist())
    pairs = _pairwise_pairs(rater_ids)
    wide = pivot_ratings_long_to_wide(working, dimensions)

    by_dimension: Dict[str, Dict[str, Optional[float]]] = {}
    for dim in dimensions:
        by_dimension[dim] = _agreement_rates(wide, dim, pairs)

    consensus = consensus_medians(working, dimensions)
    suitable = consensus["human_suitable"].dropna()
    suitable_rate = float(suitable.mean()) if len(suitable) else None

    return {
        "rubric_version": RUBRIC_VERSION,
        "n_items": int(working["item_id"].nunique()),
        "n_raters": len(rater_ids),
        "rater_ids": rater_ids,
        "method": {
            "exact_agreement": "mean pairwise percent exact agreement",
            "adjacent_agreement": "mean pairwise percent agreement within ±1",
            "consensus": "per-item median across raters",
            "krippendorff_alpha": "not computed (dropped by design)",
        },
        "by_dimension": by_dimension,
        "human_suitable_rate": suitable_rate,
        "consensus": consensus.to_dict(orient="records"),
    }
