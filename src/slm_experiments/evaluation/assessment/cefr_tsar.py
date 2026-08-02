"""Assessment-only TSAR ModernBERT ensemble CEFR cross-check.

Pinned three-model confidence-max ensemble (TSAR 2025 official evaluator).
Ordinal / disagreement diagnostic only — never an A1 gate (C1).
If an A1-class TSAR signal is ever needed, use TRAIN_DOC_EN alone (C2),
not this ensemble (ensemble A1 F1 = 0.00 on the TSAR test split).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from slm_experiments.evaluation.assessment.scorers import (
    ensure_scorer_registered,
    register_scorer,
)


TSAR_SCORER_NAME = "cefr_tsar"

# Decision 1 / plan SHAs — pin exactly (upstream HF tips drift).
TSAR_MODELS: Tuple[Dict[str, str], ...] = (
    {
        "key": "doc_en",
        "model_id": "AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
        "revision": "3c29f5fbcdc753e99bb437ff9303df983486915b",
        "training_split": "TRAIN_DOC_EN",
    },
    {
        "key": "doc_sent_en",
        "model_id": "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
        "revision": "b00d1d4780e46f6410ea8a8509649044dee18298",
        "training_split": "TRAIN_DOC_SENT_EN",
    },
    {
        "key": "reference_alllang",
        "model_id": "AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
        "revision": "83337437aa82277e96b293665dc3186088a4a839",
        "training_split": "REFERENCE_ALLLANG",
    },
)

# C2: sole A1-signal fallback (A1 F1 ≈ 0.80) — not used by the ensemble path.
A1_SIGNAL_FALLBACK_KEY = "doc_en"
TRAIN_DOC_EN_MODEL = next(m for m in TSAR_MODELS if m["key"] == A1_SIGNAL_FALLBACK_KEY)

# id2label 0..5 = A1..C2; ordinal = index + 1 (plan Decision 1).
CEFR_TSAR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
_LABEL_TO_ORDINAL = {lvl: i + 1 for i, lvl in enumerate(CEFR_TSAR_LEVELS)}

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_ERROR = "error"

DEFAULT_BATCH_SIZE = 8

_MEMBER_KEYS = tuple(m["key"] for m in TSAR_MODELS)

SCORE_COLUMNS = (
    "item_id",
    "cefr_sp_level",
    *(
        col
        for key in _MEMBER_KEYS
        for col in (f"cefr_tsar_{key}_label", f"cefr_tsar_{key}_confidence")
    ),
    "cefr_tsar_ensemble_label",
    "cefr_tsar_ensemble_ordinal",
    "cefr_tsar_ensemble_confidence",
    "cefr_tsar_status",
    "cefr_tsar_error",
    "cefr_tsar_disagrees_with_cefr_sp",
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


def label_to_ordinal(label: Optional[str]) -> Optional[int]:
    """Map A1..C2 → 1..6 (index + 1). Returns None for unknown/missing."""
    if label is None:
        return None
    text = str(label).strip().upper()
    if not text:
        return None
    return _LABEL_TO_ORDINAL.get(text)


def normalize_label(label: Any) -> Optional[str]:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    text = str(label).strip().upper()
    if text in _LABEL_TO_ORDINAL:
        return text
    # HF pipelines sometimes emit LABEL_0 .. LABEL_5
    if text.startswith("LABEL_"):
        try:
            idx = int(text.split("_", 1)[1])
            if 0 <= idx < len(CEFR_TSAR_LEVELS):
                return CEFR_TSAR_LEVELS[idx]
        except ValueError:
            return None
    return None


def aggregate_confidence_max(
    members: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep the member prediction with the highest softmax score (not majority)."""
    if not members:
        raise ValueError("confidence-max requires at least one member prediction")
    best = max(members, key=lambda m: float(m["score"]))
    label = normalize_label(best["label"])
    return {
        "key": best.get("key"),
        "label": label,
        "score": float(best["score"]),
        "ordinal": label_to_ordinal(label),
    }


def disagrees_with_cefr_sp(
    cefr_sp_level: Optional[str],
    tsar_label: Optional[str],
) -> Optional[bool]:
    """True when both discrete labels exist and differ; None if either missing."""
    left = normalize_label(cefr_sp_level)
    right = normalize_label(tsar_label)
    if left is None or right is None:
        return None
    return left != right


def detect_device() -> str:
    """CUDA → MPS → CPU. Never hard-code device=0."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "TSAR scoring requires optional extras. "
            'Install with: pip install -e ".[cefr-tsar]"'
        ) from exc

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_attn_implementation(device: str) -> str:
    """P100-safe attention: never flash_attention_2; eager on sm_60, else sdpa."""
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                major, _minor = torch.cuda.get_device_capability(0)
                if major <= 6:
                    return "eager"
        except Exception:
            return "eager"
    return "sdpa"


def empty_tsar_row(
    item_id: str,
    *,
    status: str,
    error: Optional[str] = None,
    cefr_sp_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Explicit missing/error row for scores.csv."""
    row: Dict[str, Any] = {
        "item_id": item_id,
        "cefr_sp_level": normalize_label(cefr_sp_level),
        "cefr_tsar_ensemble_label": None,
        "cefr_tsar_ensemble_ordinal": None,
        "cefr_tsar_ensemble_confidence": None,
        "cefr_tsar_status": status,
        "cefr_tsar_error": error,
        "cefr_tsar_disagrees_with_cefr_sp": None,
    }
    for key in _MEMBER_KEYS:
        row[f"cefr_tsar_{key}_label"] = None
        row[f"cefr_tsar_{key}_confidence"] = None
    return row


def _pipeline_device_arg(device: str) -> Any:
    """Map logical device string to transformers pipeline device arg."""
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0  # first CUDA device; detect_device chose CUDA, not a hard-coded prefer
    if device == "mps":
        return "mps"
    return device


def predict_member_labels(
    texts: Sequence[str],
    *,
    models: Sequence[Dict[str, str]] = TSAR_MODELS,
    device: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[List[Dict[str, Any]]]:
    """Run the three pinned classifiers; return per-text list of member top-1s.

    Import / load is lazy so unit tests can mock this function without
    downloading weights or importing torch.
    """
    if not texts:
        return []

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "TSAR scoring requires optional extras. "
            'Install with: pip install -e ".[cefr-tsar]" '
            "(transformers>=4.55, torch)."
        ) from exc

    resolved_device = device or detect_device()
    attn = resolve_attn_implementation(resolved_device)
    pipe_device = _pipeline_device_arg(resolved_device)

    pipelines = []
    for spec in models:
        pipelines.append(
            (
                spec["key"],
                pipeline(
                    "text-classification",
                    model=spec["model_id"],
                    revision=spec["revision"],
                    device=pipe_device,
                    dtype="auto",
                    model_kwargs={"attn_implementation": attn},
                ),
            )
        )

    # Batched forward per model, then zip by text index.
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    for key, pipe in pipelines:
        raw = pipe(
            list(texts),
            batch_size=batch_size,
            truncation=True,
            top_k=1,
        )
        # top_k=1 → list of dict OR list of single-element lists depending on version.
        normalized: List[Dict[str, Any]] = []
        for entry in raw:
            if isinstance(entry, list):
                entry = max(entry, key=lambda d: d["score"]) if entry else {}
            label = normalize_label(entry.get("label"))
            normalized.append(
                {
                    "key": key,
                    "label": label,
                    "score": float(entry.get("score", 0.0)),
                }
            )
        per_model[key] = normalized

    rows: List[List[Dict[str, Any]]] = []
    for i in range(len(texts)):
        rows.append([per_model[m["key"]][i] for m in models])
    return rows


def _row_from_members(
    item_id: str,
    members: Sequence[Dict[str, Any]],
    cefr_sp_level: Optional[str],
) -> Dict[str, Any]:
    ensemble = aggregate_confidence_max(members)
    sp_level = normalize_label(cefr_sp_level)
    # Winning label must normalize to A1..C2; never report ok + null label.
    if ensemble["label"] is None:
        row = empty_tsar_row(
            item_id,
            status=STATUS_ERROR,
            error="unnormalizable ensemble member label",
            cefr_sp_level=sp_level,
        )
        by_key = {m["key"]: m for m in members}
        for key in _MEMBER_KEYS:
            pred = by_key.get(key, {})
            row[f"cefr_tsar_{key}_label"] = normalize_label(pred.get("label"))
            score = pred.get("score")
            row[f"cefr_tsar_{key}_confidence"] = (
                float(score) if score is not None else None
            )
        return row

    row = empty_tsar_row(item_id, status=STATUS_OK, cefr_sp_level=sp_level)
    by_key = {m["key"]: m for m in members}
    for key in _MEMBER_KEYS:
        pred = by_key.get(key, {})
        row[f"cefr_tsar_{key}_label"] = normalize_label(pred.get("label"))
        score = pred.get("score")
        row[f"cefr_tsar_{key}_confidence"] = (
            float(score) if score is not None else None
        )
    row["cefr_tsar_ensemble_label"] = ensemble["label"]
    row["cefr_tsar_ensemble_ordinal"] = ensemble["ordinal"]
    row["cefr_tsar_ensemble_confidence"] = ensemble["score"]
    row["cefr_tsar_disagrees_with_cefr_sp"] = disagrees_with_cefr_sp(
        sp_level, ensemble["label"]
    )
    return row


@register_scorer(TSAR_SCORER_NAME)
def score_cefr_tsar(
    items: pd.DataFrame,
    *,
    device: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Score successful non-empty cleaned_response texts with the TSAR ensemble.

    Never mutates generation runs; never sets meets_a1_criteria.

    Defensive: when ``generation_successful`` is present, failed or empty
    rows are marked ``missing`` rather than trusting upstream filtering.

    ``device`` defaults to auto-detect (via ``predict_member_labels``);
    ``batch_size`` defaults to ``DEFAULT_BATCH_SIZE`` (8).
    """
    if items.empty or "item_id" not in items.columns:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))

    working = items.copy()
    if "cleaned_response" not in working.columns:
        working["cleaned_response"] = ""
    working["cleaned_response"] = (
        working["cleaned_response"].fillna("").astype(str).str.strip()
    )
    if "cefr_sp_level" not in working.columns:
        working["cefr_sp_level"] = None

    rows: List[Dict[str, Any]] = []
    scorable_mask = working["cleaned_response"] != ""
    if "generation_successful" in working.columns:
        scorable_mask = scorable_mask & _bool_series(working["generation_successful"])
    missing = working.loc[~scorable_mask]
    for _, item in missing.iterrows():
        rows.append(
            empty_tsar_row(
                str(item["item_id"]),
                status=STATUS_MISSING,
                cefr_sp_level=item.get("cefr_sp_level"),
            )
        )

    scorable = working.loc[scorable_mask]
    if not scorable.empty:
        texts = scorable["cleaned_response"].tolist()
        item_ids = scorable["item_id"].astype(str).tolist()
        sp_levels = scorable["cefr_sp_level"].tolist()
        try:
            member_rows = predict_member_labels(
                texts,
                device=device,
                batch_size=batch_size,
            )
            for item_id, members, sp in zip(item_ids, member_rows, sp_levels):
                rows.append(_row_from_members(item_id, members, sp))
        except Exception as exc:  # noqa: BLE001 — surface as per-row error state
            err = str(exc)
            for item_id, sp in zip(item_ids, sp_levels):
                rows.append(
                    empty_tsar_row(
                        item_id,
                        status=STATUS_ERROR,
                        error=err,
                        cefr_sp_level=sp,
                    )
                )

    out = pd.DataFrame(rows)
    # Preserve input item order.
    order = working["item_id"].astype(str).tolist()
    out["item_id"] = out["item_id"].astype(str)
    out = out.set_index("item_id").reindex(order).reset_index()
    return out[list(SCORE_COLUMNS)]


def ensure_registered() -> None:
    """Re-register after ``clear_scorers`` (tests) or ensure import side-effect."""
    ensure_scorer_registered(TSAR_SCORER_NAME, score_cefr_tsar)


def tsar_scorer_revision() -> Dict[str, Any]:
    """Pinned metadata for assessment manifests."""
    return {
        "registered": True,
        "aggregation": "confidence_max",
        "ordinal_scheme": "index_plus_one",
        "a1_gate": False,
        "a1_signal_fallback": A1_SIGNAL_FALLBACK_KEY,
        "a1_signal_fallback_model": TRAIN_DOC_EN_MODEL["model_id"],
        "models": [
            {
                "key": m["key"],
                "model_id": m["model_id"],
                "revision": m["revision"],
                "training_split": m["training_split"],
            }
            for m in TSAR_MODELS
        ],
    }


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _format_sweep_key(column: str, value: Any) -> str:
    if column in ("beam_width", "kvl_beam_width", "num_shots", "guided_top_k"):
        return str(int(value))
    return f"{float(value):g}"


def _tsar_metric_block(
    map_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Failure/truncation on all map rows; TSAR metrics on ok scored joins."""
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

    if n == 0 or scores_df.empty:
        stats["cefr_tsar_mean_ordinal"] = None
        stats["cefr_tsar_predicted_a1_rate"] = None
        stats["cefr_tsar_disagreement_rate"] = None
        stats["cefr_tsar_scored_count"] = 0
        return stats

    in_sample = map_df.copy()
    if "in_sample" in in_sample.columns:
        in_sample = in_sample.loc[_bool_series(in_sample["in_sample"])]
    in_sample = in_sample[in_sample["item_id"].astype(str).str.len() > 0]
    merged = in_sample.merge(scores_df, on="item_id", how="left")
    # Dedup to unique items for item-level TSAR rates (shared texts).
    item_level = merged.drop_duplicates(subset=["item_id"], keep="first")
    ok = item_level[item_level.get("cefr_tsar_status", pd.Series(dtype=str)) == STATUS_OK]
    stats["cefr_tsar_scored_count"] = int(len(ok))
    if ok.empty:
        stats["cefr_tsar_mean_ordinal"] = None
        stats["cefr_tsar_predicted_a1_rate"] = None
        stats["cefr_tsar_disagreement_rate"] = None
        return stats

    ordinals = pd.to_numeric(ok["cefr_tsar_ensemble_ordinal"], errors="coerce").dropna()
    stats["cefr_tsar_mean_ordinal"] = (
        float(ordinals.mean()) if not ordinals.empty else None
    )
    # Diagnostic only — never an A1 gate.
    labels = ok["cefr_tsar_ensemble_label"].map(normalize_label)
    stats["cefr_tsar_predicted_a1_rate"] = float((labels == "A1").mean())
    disag = ok["cefr_tsar_disagrees_with_cefr_sp"]
    disag_known = disag.dropna()
    stats["cefr_tsar_disagreement_rate"] = (
        float(disag_known.astype(bool).mean()) if len(disag_known) else None
    )
    return stats


def compute_tsar_assessment_summary(
    scores_df: pd.DataFrame,
    item_map_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Per-model and per-sweep TSAR stats beside failure / truncation rates."""
    summary: Dict[str, Any] = {
        "overall": _tsar_metric_block(item_map_df, scores_df),
        "by_model": {},
        "metadata": {
            "scorer": TSAR_SCORER_NAME,
            "aggregation": "confidence_max",
            "a1_gate": False,
            "note": (
                "cefr_tsar_predicted_a1_rate is diagnostic only; "
                "meets_a1_criteria remains CEFR-SP."
            ),
        },
    }

    if "model" in item_map_df.columns and not item_map_df.empty:
        by_model: Dict[str, Any] = {}
        for model_name, group in item_map_df.groupby("model", sort=True):
            by_model[str(model_name)] = _tsar_metric_block(group, scores_df)
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
            grouped[_format_sweep_key(col, value)] = _tsar_metric_block(
                item_map_df.loc[mask], scores_df
            )
        summary[section] = grouped
        sweep_dimensions.append(col)
        sweep_values[col] = list(grouped.keys())
    # List (not overwritten string) so mixed-family bundles keep all axes.
    summary["metadata"]["sweep_dimension"] = sweep_dimensions
    summary["metadata"]["sweep_values"] = sweep_values

    return summary


# Import-time registration (also re-run via ensure_registered).
ensure_registered()
