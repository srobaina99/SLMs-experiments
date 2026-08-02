"""Build immutable assessment bundles from generation run full.csv files."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from slm_experiments.core.config_label import config_label
from slm_experiments.core.run_store import (
    KIND_ASSESSMENT,
    RunStore,
    make_run_id,
)
from slm_experiments.evaluation.assessment.scorers import (
    list_scorers,
    score_items,
    scorer_revisions,
)
from slm_experiments.evaluation.assessment.cefr_tsar import (
    DEFAULT_BATCH_SIZE as TSAR_DEFAULT_BATCH_SIZE,
    compute_tsar_assessment_summary,
    ensure_registered as ensure_tsar_registered,
)
from slm_experiments.evaluation.assessment.kvl_v2 import (
    compute_kvl_v2_assessment_summary,
    ensure_registered as ensure_kvl_v2_registered,
)
from slm_experiments.human.rubric import RUBRIC_VERSION

ASSESSMENT_PHASE = "assessment"
ASSESSMENT_EXPERIMENT = "beginner_suitability"

ITEMS_FILENAME = "items.csv"
ITEM_MAP_FILENAME = "item_map.csv"
SCORES_FILENAME = "scores.csv"
SUMMARY_FILENAME = "summary.json"

DEFAULT_SEED = 42

ITEMS_COLUMNS = [
    "item_id",
    "prompt_id",
    "prompt",
    "cleaned_response",
    "inclusion_probability",
]

ITEM_MAP_COLUMNS = [
    "item_id",
    "source_run_id",
    "experiment_id",
    "model",
    "config",
    "prompt_id",
    "generation_successful",
    "hit_max_tokens",
    "in_sample",
    "inclusion_probability",
    "weight_factor",
    "num_shots",
    "guided_top_k",
    "kvl_beam_width",
    "beam_width",
    # Carried from source full.csv so analysis reads bundle artifacts only.
    "cefr_sp_level_ordinal",
    "meets_a1_criteria",
]

_TRACKED_DEPS = (
    "pandas",
    "numpy",
    "textstat",
    "llama-cpp-python",
)


def _dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in _TRACKED_DEPS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    try:
        versions["slm_experiments"] = metadata.version("slm-experiments")
    except metadata.PackageNotFoundError:
        from slm_experiments import __version__

        versions["slm_experiments"] = __version__
    return versions


def probe_code_revision(repo_root: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Best-effort git revision metadata for assessment manifests.

    Never raises: unavailable git / non-checkout → null fields.
    """
    empty: Dict[str, Any] = {
        "git_commit": None,
        "git_describe": None,
        "dirty": None,
    }
    try:
        import subprocess

        if repo_root is not None:
            cwd = str(Path(repo_root))
        else:
            from slm_experiments.models.base import REPO_ROOT as root

            cwd = str(root)

        def _git(*args: str) -> Optional[str]:
            completed = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode != 0:
                return None
            return completed.stdout.strip() or None

        commit = _git("rev-parse", "HEAD")
        if commit is None:
            return empty
        describe = _git("describe", "--tags", "--always", "--dirty")
        porcelain = _git("status", "--porcelain")
        dirty: Optional[bool] = None if porcelain is None else bool(porcelain)
        return {
            "git_commit": commit,
            "git_describe": describe,
            "dirty": dirty,
        }
    except Exception:  # noqa: BLE001 — never fail assess build on git probe
        return empty


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _is_scorable_row(row: pd.Series) -> bool:
    successful = bool(row.get("generation_successful", False))
    cleaned = row.get("cleaned_response", "")
    if cleaned is None or (isinstance(cleaned, float) and pd.isna(cleaned)):
        cleaned = ""
    return successful and str(cleaned).strip() != ""


class AssessmentBundler:
    """Ingest generation runs into an immutable assessment bundle."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        if results_root is not None:
            root = Path(results_root)
        else:
            from slm_experiments.models.base import REPO_ROOT as repo_root

            root = Path(repo_root) / "results"
        self.run_store = RunStore(root)

    def build(
        self,
        source_run_ids: Sequence[str],
        *,
        sample: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        cli_args: Optional[List[str]] = None,
        started_at: Optional[datetime] = None,
        cefr_tsar_device: Optional[str] = None,
        cefr_tsar_batch_size: int = TSAR_DEFAULT_BATCH_SIZE,
    ) -> tuple[str, Path]:
        """
        Build ``results/runs/{ts}_assessment_beginner_suitability/``.

        Reads each source via ``RunStore.read_full_csv`` only — never writes
        back to source runs. Returns ``(run_id, out_dir)``.

        ``cefr_tsar_device`` / ``cefr_tsar_batch_size`` pass through to the
        TSAR scorer (defaults preserve auto-detect device + batch size 8).
        """
        if not source_run_ids:
            raise ValueError("at least one source_run_id is required")

        # Validate + ingest before creating the bundle dir (no orphan empties).
        source_frames: List[pd.DataFrame] = []
        for source_id in source_run_ids:
            source_dir = self.run_store.run_dir(source_id)
            if not source_dir.exists():
                raise FileNotFoundError(f"Source run bundle not found: {source_id}")
            df = self.run_store.read_full_csv(source_id)
            if df.empty:
                continue
            frame = df.copy()
            frame["source_run_id"] = source_id
            source_frames.append(frame)

        if not source_frames:
            raise ValueError("no observations found in source runs")

        combined = pd.concat(source_frames, ignore_index=True)
        items_df, item_map_df, sampling_meta = self._build_items_and_map(
            combined, sample=sample, seed=seed
        )

        started = started_at or datetime.now(timezone.utc)
        run_id = make_run_id(
            ASSESSMENT_PHASE,
            ASSESSMENT_EXPERIMENT,
            started_at=started.replace(tzinfo=None) if started.tzinfo else started,
        )
        out_dir = self.run_store.run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        items_path = out_dir / ITEMS_FILENAME
        item_map_path = out_dir / ITEM_MAP_FILENAME
        scores_path = out_dir / SCORES_FILENAME

        items_df[ITEMS_COLUMNS].to_csv(items_path, index=False)
        item_map_df[ITEM_MAP_COLUMNS].to_csv(item_map_path, index=False)

        # Attach in-pipeline CEFR-SP labels for disagreement (not written to items.csv).
        items_for_scoring = self._attach_cefr_sp_levels(items_df, combined)

        ensure_tsar_registered()
        ensure_kvl_v2_registered()
        scores_df = score_items(
            items_for_scoring,
            scorer_options={
                "cefr_tsar": {
                    "device": cefr_tsar_device,
                    "batch_size": int(cefr_tsar_batch_size),
                }
            },
        )
        if scores_df.empty and "item_id" in items_df.columns:
            scores_df = items_df[["item_id"]].copy()
        scores_df.to_csv(scores_path, index=False)

        # Keep TSAR top-level shape for #14 compat; add scorers namespace additively.
        summary = compute_tsar_assessment_summary(scores_df, item_map_df)
        summary["scorers"] = {
            "cefr_tsar": {
                "overall": summary.get("overall"),
                "by_model": summary.get("by_model"),
                "metadata": summary.get("metadata"),
            },
            "kvl_v2": compute_kvl_v2_assessment_summary(scores_df, item_map_df),
        }
        # Mirror sweep sections under scorers.cefr_tsar when present.
        for key, value in list(summary.items()):
            if key.startswith("by_") and key != "by_model":
                summary["scorers"]["cefr_tsar"][key] = value
        (out_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        completed = datetime.now(timezone.utc)
        successful_source = int(_bool_series(combined["generation_successful"]).sum())
        failed_source = int(len(combined) - successful_source)
        maxed_source = (
            int(_bool_series(combined["hit_max_tokens"]).sum())
            if "hit_max_tokens" in combined.columns
            else 0
        )
        source_total = int(len(combined))

        manifest: Dict[str, Any] = {
            "kind": KIND_ASSESSMENT,
            "run_id": run_id,
            "phase": ASSESSMENT_PHASE,
            "experiment": ASSESSMENT_EXPERIMENT,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "cli_args": cli_args or [],
            "source_run_ids": list(source_run_ids),
            "rubric_version": RUBRIC_VERSION,
            "scorer_revisions": scorer_revisions(),
            "registered_scorers": list_scorers(),
            "dependency_versions": _dependency_versions(),
            "code_revision": probe_code_revision(),
            "sampling": sampling_meta,
            "seed": seed,
            "cefr_tsar": {
                "device": cefr_tsar_device,
                "batch_size": int(cefr_tsar_batch_size),
            },
            "observations": {
                # total / successful / failed / hit_max_tokens = source-row counts
                "total": source_total,
                "successful": successful_source,
                "failed": failed_source,
                "hit_max_tokens": maxed_source,
                "items": int(len(items_df)),
                "item_map_rows": int(len(item_map_df)),
            },
            "artifacts": {
                "items_csv": ITEMS_FILENAME,
                "item_map_csv": ITEM_MAP_FILENAME,
                "scores_csv": SCORES_FILENAME,
                "summary_json": SUMMARY_FILENAME,
            },
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Paired-delta analysis (issue #19); re-runnable via `assess analyze`.
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        analyze_assessment_bundle(run_id, results_root=self.run_store.results_root)
        return run_id, out_dir

    @staticmethod
    def _attach_cefr_sp_levels(
        items_df: pd.DataFrame,
        combined: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join CEFR-SP document levels from source rows onto scoring items."""
        enriched = items_df.copy()
        if "cefr_sp_level" in enriched.columns:
            return enriched
        if "cefr_sp_level" not in combined.columns:
            enriched["cefr_sp_level"] = None
            return enriched

        source = combined.copy()
        if "cleaned_response" not in source.columns:
            source["cleaned_response"] = ""
        source["cleaned_response"] = (
            source["cleaned_response"].fillna("").astype(str).str.strip()
        )
        if "generation_successful" not in source.columns:
            source["generation_successful"] = False
        source = source.loc[
            _bool_series(source["generation_successful"])
            & (source["cleaned_response"] != "")
        ]
        if source.empty:
            enriched["cefr_sp_level"] = None
            return enriched

        # Deterministic for identical text; first successful source row wins.
        levels = (
            source.drop_duplicates(subset=["prompt_id", "cleaned_response"], keep="first")[
                ["prompt_id", "cleaned_response", "cefr_sp_level"]
            ]
        )
        enriched = enriched.merge(
            levels, on=["prompt_id", "cleaned_response"], how="left"
        )
        return enriched

    def _build_items_and_map(
        self,
        combined: pd.DataFrame,
        *,
        sample: Optional[int],
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        working = combined.copy()
        if "cleaned_response" not in working.columns:
            working["cleaned_response"] = ""
        # Strip so dedup key matches scorability ("text" vs "text ").
        working["cleaned_response"] = (
            working["cleaned_response"].fillna("").astype(str).str.strip()
        )
        working["generation_successful"] = _bool_series(
            working.get("generation_successful", pd.Series(False, index=working.index))
        )
        if "hit_max_tokens" in working.columns:
            working["hit_max_tokens"] = _bool_series(working["hit_max_tokens"])
        else:
            working["hit_max_tokens"] = False

        working["config"] = working.apply(
            lambda row: config_label(
                bool(row.get("config_weighting", False)),
                bool(row.get("config_prompting", False)),
            ),
            axis=1,
        )

        scorable_mask = working.apply(_is_scorable_row, axis=1)
        scorable = working.loc[scorable_mask].copy()

        # Dedup key: (prompt_id, cleaned_response) over successful non-empty only.
        # Failures are excluded so (prompt_id, "") never collapses them.
        key_cols = ["prompt_id", "cleaned_response"]
        unique_keys = scorable.drop_duplicates(subset=key_cols, keep="first")

        item_records: List[Dict[str, Any]] = []
        key_to_item: Dict[tuple[Any, str], str] = {}
        for _, row in unique_keys.iterrows():
            item_id = str(uuid.uuid4())
            key = (row["prompt_id"], row["cleaned_response"])
            key_to_item[key] = item_id
            item_records.append(
                {
                    "item_id": item_id,
                    "prompt_id": row["prompt_id"],
                    "prompt": row.get("prompt", ""),
                    "cleaned_response": row["cleaned_response"],
                    "inclusion_probability": 1.0,
                }
            )

        items_df = pd.DataFrame(item_records, columns=ITEMS_COLUMNS)
        items_before = len(items_df)

        sampling_meta: Dict[str, Any] = {
            "sample_requested": sample,
            "sample_seed": seed,
            "items_before_sample": items_before,
            "strategy": "all",
        }
        # Candidate inclusion weight (SRS); 1.0 when keeping all items.
        candidate_inclusion_p = 1.0

        if sample is not None:
            if sample <= 0:
                raise ValueError(f"sample size must be positive, got {sample}")
            if sample < items_before:
                items_df = items_df.sample(n=sample, random_state=seed).reset_index(
                    drop=True
                )
                candidate_inclusion_p = float(sample) / float(items_before)
                items_df["inclusion_probability"] = candidate_inclusion_p
                sampling_meta["strategy"] = "simple_random"
                sampling_meta["inclusion_probability"] = candidate_inclusion_p
            else:
                sampling_meta["strategy"] = "all"
                sampling_meta["inclusion_probability"] = 1.0
        else:
            sampling_meta["inclusion_probability"] = 1.0

        sampling_meta["items_after_sample"] = int(len(items_df))
        kept_ids = set(items_df["item_id"].tolist())

        map_rows: List[Dict[str, Any]] = []
        for _, row in working.iterrows():
            item_id = ""
            in_sample = False
            # Non-items (failures / empty / sampled-out) use in_sample=False.
            if _is_scorable_row(row):
                key = (row["prompt_id"], str(row["cleaned_response"]))
                canonical_id = key_to_item.get(key, "")
                inclusion_probability = candidate_inclusion_p
                if canonical_id and canonical_id in kept_ids:
                    item_id = canonical_id
                    in_sample = True
                else:
                    # Sampled-out success: keep provenance, no item_id.
                    item_id = ""
                    in_sample = False
            else:
                # Failures / empty: no item_id (do not collapse); not sample items.
                item_id = ""
                in_sample = False
                inclusion_probability = 1.0

            raw_a1 = row.get("meets_a1_criteria", pd.NA)
            try:
                meets_a1: Any = pd.NA if pd.isna(raw_a1) else bool(raw_a1)
            except (TypeError, ValueError):
                meets_a1 = pd.NA

            map_rows.append(
                {
                    "item_id": item_id,
                    "source_run_id": row.get("source_run_id", ""),
                    "experiment_id": row.get("experiment_id", ""),
                    "model": row.get("model", ""),
                    "config": row.get("config", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "generation_successful": bool(row.get("generation_successful", False)),
                    "hit_max_tokens": bool(row.get("hit_max_tokens", False)),
                    "in_sample": in_sample,
                    "inclusion_probability": inclusion_probability,
                    "weight_factor": row.get("weight_factor", pd.NA),
                    "num_shots": row.get("num_shots", pd.NA),
                    "guided_top_k": row.get("guided_top_k", pd.NA),
                    "kvl_beam_width": row.get("kvl_beam_width", pd.NA),
                    "beam_width": row.get("beam_width", pd.NA),
                    "cefr_sp_level_ordinal": row.get("cefr_sp_level_ordinal", pd.NA),
                    "meets_a1_criteria": meets_a1,
                }
            )

        item_map_df = pd.DataFrame(map_rows, columns=ITEM_MAP_COLUMNS)
        return items_df, item_map_df, sampling_meta
