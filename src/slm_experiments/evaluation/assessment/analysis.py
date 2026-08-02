"""Paired-delta + validation analysis over assessment bundles.

Exploratory / descriptive framing only: raw 95% CIs, a descriptive
``ci_excludes_zero`` flag, and no confirmatory language.

Also houses automatic-vs-human validation and adequacy non-inferiority
(issue #20): Kendall τ-b / Spearman ρ, binary agreement, CEFR-band
confusion, and the 0.5 adequacy-drop guardrail on the human subset.

Formal analysis should use an unsampled assessment bundle — sampling can
break ``(model, prompt_id)`` pairs, which then drop out of ``n_pairs`` and
appear in ``n_pairs_dropped``. Rates still use all ``item_map`` rows.

This module is the stable public import surface; implementation lives in
``analysis_helpers``, ``paired_deltas``, and ``validation``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from slm_experiments.core.run_store import KIND_ASSESSMENT, RunStore
from slm_experiments.evaluation.assessment.analysis_helpers import (
    ADEQUACY_CSV_FILENAME,
    ADEQUACY_GROUP_KEY_PREFIX,
    ADEQUACY_MARGIN,
    ANALYSIS_DIRNAME,
    ANALYSIS_JSON_FILENAME,
    CEFR_BANDS_COLLAPSED,
    CEFR_BANDS_FULL,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CI,
    DEFAULT_RESAMPLES,
    FAMILY_BASELINE,
    HUMAN_BINARY_LABELS,
    METRIC_SPECS,
    PAIRED_DELTAS_FILENAME,
    VALIDATION_CSV_FILENAME,
    ArmLabel,
    Direction,
    EasierOrientation,
    arm_key,
    baseline_arm_key,
    group_bootstrap_rng,
    resolve_arm,
    resolve_family,
)
from slm_experiments.evaluation.assessment.paired_deltas import (
    _annotate_map,
    _metric_frame,
    _stratum_rates,
    build_paired_deltas,
    ci_excludes_zero,
    direction_from_ci,
    easier_deltas,
    percentile_bootstrap_ci,
)
from slm_experiments.evaluation.assessment.validation import (
    SPEARMAN_WEIGHTED_NOT_REPORTED,
    _load_human_study,
    _validation_metadata,
    adequacy_group_key,
    adequacy_noninferiority,
    average_ranks,
    binary_suitability_agreement,
    build_adequacy_noninferiority_section,
    build_validation_section,
    confusion_by_cefr_band,
    kendall_tau_b,
    orient_ordinal_higher_suitable,
    spearman_rho,
    weighted_kendall_tau_b,
)

__all__ = [
    "ADEQUACY_CSV_FILENAME",
    "ADEQUACY_GROUP_KEY_PREFIX",
    "ADEQUACY_MARGIN",
    "ANALYSIS_DIRNAME",
    "ANALYSIS_JSON_FILENAME",
    "ArmLabel",
    "CEFR_BANDS_COLLAPSED",
    "CEFR_BANDS_FULL",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CI",
    "DEFAULT_RESAMPLES",
    "Direction",
    "EasierOrientation",
    "FAMILY_BASELINE",
    "HUMAN_BINARY_LABELS",
    "METRIC_SPECS",
    "PAIRED_DELTAS_FILENAME",
    "SPEARMAN_WEIGHTED_NOT_REPORTED",
    "VALIDATION_CSV_FILENAME",
    "adequacy_group_key",
    "adequacy_noninferiority",
    "analyze_assessment_bundle",
    "arm_key",
    "average_ranks",
    "baseline_arm_key",
    "binary_suitability_agreement",
    "build_adequacy_noninferiority_section",
    "build_paired_deltas",
    "build_validation_section",
    "ci_excludes_zero",
    "confusion_by_cefr_band",
    "direction_from_ci",
    "easier_deltas",
    "group_bootstrap_rng",
    "kendall_tau_b",
    "orient_ordinal_higher_suitable",
    "percentile_bootstrap_ci",
    "resolve_arm",
    "resolve_family",
    "spearman_rho",
    "weighted_kendall_tau_b",
]


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
    no_baseline_groups: List[str] = []
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
                no_baseline_groups.append(f"{model}/{family}")
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
                no_baseline_groups.append(f"{model}/{family}")
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

    # One aggregated INFO line instead of per-(model, family) UserWarning spam
    # on smoke fixtures that omit a neutral baseline arm. Structured details
    # remain in analysis.json ``warnings``.
    if no_baseline_groups:
        logging.getLogger(__name__).info(
            "no_baseline for %d model/family group(s): %s (details in analysis.json warnings)",
            len(no_baseline_groups),
            ", ".join(no_baseline_groups),
        )

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
