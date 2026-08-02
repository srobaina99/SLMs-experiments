"""Automatic-vs-human validation and adequacy non-inferiority (issue #20)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from slm_experiments.core.bool_series import coerce_bool_series
from slm_experiments.evaluation.assessment.analysis_helpers import (
    ADEQUACY_GROUP_KEY_PREFIX,
    ADEQUACY_MARGIN,
    CEFR_BANDS_COLLAPSED,
    CEFR_BANDS_FULL,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CI,
    DEFAULT_RESAMPLES,
    FAMILY_BASELINE,
    HUMAN_BINARY_LABELS,
    EasierOrientation,
    _bootstrap_item_statistic,
    _is_missing,
    arm_key,
    baseline_arm_key,
    group_bootstrap_rng,
    resolve_arm,
)
from slm_experiments.evaluation.assessment.paired_deltas import (
    _stratum_rates,
    build_paired_deltas,
    percentile_bootstrap_ci,
)
from slm_experiments.human.rubric import (
    ANSWER_ADEQUACY,
    OVERALL_SUITABILITY,
    is_human_suitable,
)
from slm_experiments.human.study_export import STUDY_DIRNAME

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
    # Require explicit ok; missing/null status must not treat a bare label as ok.
    if _is_missing(status) or str(status) != "ok":
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
        "inclusion_weight_definition": (
            "inclusion_weight = 1/π_i with π_i = n_s / N_s against the original "
            "pre-calibration stratum pool (unconditional Horvitz–Thompson); "
            "not renormalized over the post-calibration remaining frame"
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
    human_bin = coerce_bool_series(working["human_suitable"]).to_numpy()
    scorer_bin = coerce_bool_series(working[suitable_col]).to_numpy()
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
    # Older bundles may omit cefr_sp_level on item_map; scores.csv still carries it.
    if "cefr_sp_level" not in base.columns:
        score_cols.insert(1, "cefr_sp_level")
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
