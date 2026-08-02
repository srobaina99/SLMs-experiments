"""Assessment-only KVL v2 scorer (occurrence-level, lemmatized).

Diagnostic only — never an A1 gate. Does not touch KVL beam decoding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from slm_experiments.core.bool_series import coerce_bool_series as _bool_series
from slm_experiments.evaluation.assessment.scorers import (
    ensure_scorer_registered,
    register_scorer,
)
from slm_experiments.evaluation.kvl import (
    DEFAULT_KVL_L1,
    KVL_V2_LOWER_TAIL_PERCENTILE,
    KvlLookup,
    compute_kvl_v2_metrics,
    empty_kvl_v2_metrics,
)
from slm_experiments.evaluation.metrics import resolve_lemmatizer_backend

KVL_V2_SCORER_NAME = "kvl_v2"

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_ERROR = "error"

SCORE_COLUMNS = (
    "item_id",
    "kvl_v2_l1",
    "kvl_v2_token_count",
    "kvl_v2_lookup_count",
    "kvl_v2_oov_count",
    "kvl_v2_lookup_coverage",
    "kvl_v2_mean_score",
    "kvl_v2_hard_token_share",
    "kvl_v2_lower_tail_score",
    "kvl_v2_status",
    "kvl_v2_error",
)

_SWEEP_COLUMNS = (
    "weight_factor",
    "num_shots",
    "guided_top_k",
    "kvl_beam_width",
    "beam_width",
)

_SWEEP_SECTION = {
    "weight_factor": "by_weight_factor",
    "num_shots": "by_num_shots",
    "guided_top_k": "by_guided_top_k",
    "kvl_beam_width": "by_kvl_beam_width",
    "beam_width": "by_beam_width",
}


def empty_kvl_v2_row(
    item_id: str,
    *,
    status: str,
    error: Optional[str] = None,
    l1: str = DEFAULT_KVL_L1,
) -> Dict[str, Any]:
    """Explicit missing/error row for scores.csv."""
    row = empty_kvl_v2_metrics(l1)
    row["item_id"] = item_id
    row["kvl_v2_status"] = status
    row["kvl_v2_error"] = error
    return row


def _row_from_metrics(
    item_id: str,
    metrics: Dict[str, object],
    *,
    status: str = STATUS_OK,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"item_id": item_id}
    for key in SCORE_COLUMNS:
        if key in ("item_id", "kvl_v2_status", "kvl_v2_error"):
            continue
        row[key] = metrics.get(key)
    row["kvl_v2_status"] = status
    row["kvl_v2_error"] = error
    return row


@register_scorer(KVL_V2_SCORER_NAME)
def score_kvl_v2(items: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
    """Score successful non-empty cleaned_response texts with KVL v2.

    Never mutates generation runs; never sets meets_a1_criteria.
    Extra keyword args are ignored (score_items may pass per-scorer options).
    """
    if items.empty or "item_id" not in items.columns:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))

    working = items.copy()
    if "cleaned_response" not in working.columns:
        working["cleaned_response"] = ""
    working["cleaned_response"] = (
        working["cleaned_response"].fillna("").astype(str).str.strip()
    )

    # Assessment KVL v2 is fixed to Spanish L1. Callers may still pass a
    # ``kvl_l1`` column on items; it is ignored (public API stays stable).
    l1 = DEFAULT_KVL_L1

    lookup = KvlLookup()
    rows: List[Dict[str, Any]] = []
    for _, item in working.iterrows():
        item_id = str(item["item_id"])
        text = str(item["cleaned_response"])
        if not text:
            rows.append(empty_kvl_v2_row(item_id, status=STATUS_MISSING, l1=l1))
            continue
        try:
            metrics = compute_kvl_v2_metrics(
                text,
                l1,
                kvl_lookup=lookup,
                lemmatize=True,
            )
            rows.append(_row_from_metrics(item_id, metrics))
        except Exception as exc:  # noqa: BLE001 — surface as per-row error state
            rows.append(
                empty_kvl_v2_row(
                    item_id,
                    status=STATUS_ERROR,
                    error=str(exc),
                    l1=l1,
                )
            )

    out = pd.DataFrame(rows)
    order = working["item_id"].astype(str).tolist()
    out["item_id"] = out["item_id"].astype(str)
    out = out.set_index("item_id").reindex(order).reset_index()
    return out[list(SCORE_COLUMNS)]


def ensure_registered() -> None:
    """Re-register after ``clear_scorers`` (tests) or ensure import side-effect."""
    ensure_scorer_registered(KVL_V2_SCORER_NAME, score_kvl_v2)


def kvl_v2_scorer_revision() -> Dict[str, Any]:
    """Pinned metadata for assessment manifests."""
    return {
        "registered": True,
        "version": "v2",
        "token_semantics": "occurrence_level_lemmatized_content_words",
        "lower_tail_percentile": KVL_V2_LOWER_TAIL_PERCENTILE,
        "a1_gate": False,
        "default_l1": DEFAULT_KVL_L1,
        "lemmatizer": resolve_lemmatizer_backend(),
        "note": (
            "Never interpret kvl_v2_mean_score without kvl_v2_lookup_coverage. "
            "KVL beam decoding is unchanged."
        ),
    }


def _format_sweep_key(column: str, value: Any) -> str:
    if column in ("beam_width", "kvl_beam_width", "num_shots", "guided_top_k"):
        return str(int(value))
    return f"{float(value):g}"


def _kvl_v2_metric_block(
    map_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Failure/truncation on all map rows; KVL v2 means on ok scored joins."""
    n = len(map_df)
    successful = _bool_series(map_df["generation_successful"]) if n else pd.Series(dtype=bool)
    stats: Dict[str, Any] = {
        "count": int(n),
        "generation_successful_count": int(successful.sum()) if n else 0,
        "generation_failure_rate": float(1 - successful.mean()) if n else 0.0,
    }
    if "hit_max_tokens" in map_df.columns and n:
        maxed = _bool_series(map_df["hit_max_tokens"])
        stats["hit_max_tokens_count"] = int(maxed.sum())
        stats["hit_max_tokens_rate"] = float(maxed.mean())
    else:
        stats["hit_max_tokens_count"] = 0
        stats["hit_max_tokens_rate"] = 0.0

    empty_metrics = {
        "kvl_v2_mean_coverage": None,
        "kvl_v2_mean_score": None,
        "kvl_v2_mean_hard_token_share": None,
        "kvl_v2_mean_lower_tail_score": None,
        "kvl_v2_scored_count": 0,
    }
    if n == 0 or scores_df.empty:
        stats.update(empty_metrics)
        return stats

    in_sample = map_df.copy()
    if "in_sample" in in_sample.columns:
        in_sample = in_sample.loc[_bool_series(in_sample["in_sample"])]
    in_sample = in_sample[in_sample["item_id"].astype(str).str.len() > 0]
    merged = in_sample.merge(scores_df, on="item_id", how="left")
    item_level = merged.drop_duplicates(subset=["item_id"], keep="first")
    if "kvl_v2_status" not in item_level.columns:
        stats.update(
            {
                "kvl_v2_mean_coverage": None,
                "kvl_v2_mean_score": None,
                "kvl_v2_mean_hard_token_share": None,
                "kvl_v2_mean_lower_tail_score": None,
                "kvl_v2_scored_count": 0,
            }
        )
        return stats
    ok = item_level[item_level["kvl_v2_status"] == STATUS_OK]
    stats["kvl_v2_scored_count"] = int(len(ok))
    if ok.empty:
        stats.update(
            {
                "kvl_v2_mean_coverage": None,
                "kvl_v2_mean_score": None,
                "kvl_v2_mean_hard_token_share": None,
                "kvl_v2_mean_lower_tail_score": None,
            }
        )
        return stats

    coverage = pd.to_numeric(ok["kvl_v2_lookup_coverage"], errors="coerce").dropna()
    stats["kvl_v2_mean_coverage"] = (
        float(coverage.mean()) if not coverage.empty else None
    )
    # Mean score only over rows with positive coverage (never interpret bare mean).
    covered = ok[pd.to_numeric(ok["kvl_v2_lookup_coverage"], errors="coerce") > 0]
    means = pd.to_numeric(covered["kvl_v2_mean_score"], errors="coerce").dropna()
    stats["kvl_v2_mean_score"] = float(means.mean()) if not means.empty else None
    hard = pd.to_numeric(covered["kvl_v2_hard_token_share"], errors="coerce").dropna()
    stats["kvl_v2_mean_hard_token_share"] = (
        float(hard.mean()) if not hard.empty else None
    )
    tail = pd.to_numeric(covered["kvl_v2_lower_tail_score"], errors="coerce").dropna()
    stats["kvl_v2_mean_lower_tail_score"] = (
        float(tail.mean()) if not tail.empty else None
    )
    return stats


def compute_kvl_v2_assessment_summary(
    scores_df: pd.DataFrame,
    item_map_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Per-model / per-sweep KVL v2 stats beside failure / truncation rates."""
    summary: Dict[str, Any] = {
        "overall": _kvl_v2_metric_block(item_map_df, scores_df),
        "by_model": {},
        "metadata": {
            "scorer": KVL_V2_SCORER_NAME,
            "version": "v2",
            "a1_gate": False,
            "note": (
                "Never interpret kvl_v2_mean_score without kvl_v2_mean_coverage / "
                "per-item kvl_v2_lookup_coverage."
            ),
        },
    }

    if "model" in item_map_df.columns and not item_map_df.empty:
        by_model: Dict[str, Any] = {}
        for model_name, group in item_map_df.groupby("model", sort=True):
            by_model[str(model_name)] = _kvl_v2_metric_block(group, scores_df)
        summary["by_model"] = by_model

    sweep_dimensions: List[str] = []
    sweep_values: Dict[str, List[str]] = {}
    for col in _SWEEP_COLUMNS:
        if col not in item_map_df.columns:
            continue
        values = item_map_df[col].dropna().unique()
        if len(values) <= 1:
            continue
        section = _SWEEP_SECTION[col]
        grouped: Dict[str, Any] = {}
        for value in sorted(values, key=lambda v: float(v)):
            mask = item_map_df[col] == value
            grouped[_format_sweep_key(col, value)] = _kvl_v2_metric_block(
                item_map_df.loc[mask], scores_df
            )
        summary[section] = grouped
        sweep_dimensions.append(col)
        sweep_values[col] = list(grouped.keys())
    # List (not overwritten string) so mixed-family bundles keep all axes.
    summary["metadata"]["sweep_dimension"] = sweep_dimensions
    summary["metadata"]["sweep_values"] = sweep_values

    return summary


# Import-time registration.
ensure_registered()
