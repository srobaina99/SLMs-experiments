"""Import long-format three-rater ratings into an assessment study bundle.

Never writes back to generation runs. Ratings and reliability live under
``results/runs/{assessment_id}/study/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from slm_experiments.core.run_store import KIND_ASSESSMENT, RunStore
from slm_experiments.human.reliability import compute_reliability, consensus_medians
from slm_experiments.human.rubric import (
    NOTES_COLUMN,
    RATING_DIMENSIONS,
    RATING_MAX,
    RATING_MIN,
    RUBRIC_VERSION,
)
from slm_experiments.human.study_export import STUDY_DIRNAME
from slm_experiments.models.base import REPO_ROOT

RATINGS_FILENAME = "ratings.csv"
CONSENSUS_FILENAME = "consensus.csv"
RELIABILITY_FILENAME = "reliability.json"

REQUIRED_RATING_COLUMNS = ["item_id", "rater_id", *RATING_DIMENSIONS]


def _normalize_rating(value) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    number = int(float(value))
    if number < RATING_MIN or number > RATING_MAX:
        raise ValueError(
            f"rating must be an integer in [{RATING_MIN}, {RATING_MAX}], got {value}"
        )
    return number


def validate_ratings(
    ratings: pd.DataFrame,
    *,
    known_item_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Validate long-format ratings.

    Enforces required columns, ordinal 1–4 dimensions, and exactly one row per
    ``(item_id, rater_id)``.
    """
    missing = [col for col in REQUIRED_RATING_COLUMNS if col not in ratings.columns]
    if missing:
        raise ValueError(f"Ratings CSV missing columns: {missing}")

    working = ratings.copy()
    working["item_id"] = working["item_id"].astype(str)
    working["rater_id"] = working["rater_id"].astype(str)

    if working["item_id"].eq("").any() or working["rater_id"].eq("").any():
        raise ValueError("item_id and rater_id must be non-empty")

    dup_mask = working.duplicated(subset=["item_id", "rater_id"], keep=False)
    if dup_mask.any():
        sample = (
            working.loc[dup_mask, ["item_id", "rater_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"exactly one rating per (item_id, rater_id) required; duplicates: {sample}"
        )

    for dim in RATING_DIMENSIONS:
        normalized = []
        for value in working[dim]:
            rating = _normalize_rating(value)
            if rating is None:
                raise ValueError(f"missing required rating in column {dim}")
            normalized.append(rating)
        working[dim] = normalized

    if NOTES_COLUMN in working.columns:
        working[NOTES_COLUMN] = working[NOTES_COLUMN].astype("string")
    else:
        working[NOTES_COLUMN] = pd.Series([pd.NA] * len(working), dtype="string")

    if known_item_ids is not None:
        unknown = set(working["item_id"]) - set(known_item_ids)
        if unknown:
            sample = sorted(unknown)[:5]
            raise ValueError(f"Ratings CSV contains unknown item_id values: {sample}")

    return working[REQUIRED_RATING_COLUMNS + [NOTES_COLUMN]]


def rater_sheet_to_long(sheet: pd.DataFrame, rater_id: str) -> pd.DataFrame:
    """Convert a filled blind rater sheet into long-format rows."""
    if "item_id" not in sheet.columns:
        raise ValueError("rater sheet must include item_id")
    missing = [col for col in RATING_DIMENSIONS if col not in sheet.columns]
    if missing:
        raise ValueError(f"rater sheet missing rating columns: {missing}")

    long_df = sheet[["item_id", *RATING_DIMENSIONS]].copy()
    if NOTES_COLUMN in sheet.columns:
        long_df[NOTES_COLUMN] = sheet[NOTES_COLUMN]
    else:
        long_df[NOTES_COLUMN] = pd.NA
    long_df.insert(1, "rater_id", rater_id)
    return long_df


class StudyImporter:
    """Merge validated ratings into the assessment study bundle and compute reliability."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)

    def import_ratings(
        self,
        assessment_run_id: str,
        ratings_path: Union[Path, str],
        *,
        rater_id: Optional[str] = None,
    ) -> dict:
        """
        Import ratings CSV into ``study/ratings.csv`` and write reliability artifacts.

        If ``rater_id`` is set, treat the CSV as a single blind rater sheet.
        Otherwise expect long-format with a ``rater_id`` column.

        Returns a summary dict (n_ratings, n_items, reliability path).
        """
        run_dir = self.run_store.run_dir(assessment_run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Assessment bundle not found: {run_dir}")

        manifest = self.run_store.read_manifest(assessment_run_id)
        if manifest.get("kind", "generation") != KIND_ASSESSMENT:
            raise ValueError(
                f"run {assessment_run_id} is not an assessment bundle "
                f"(kind={manifest.get('kind')})"
            )

        study_dir = run_dir / STUDY_DIRNAME
        if not study_dir.exists():
            raise FileNotFoundError(
                f"Study directory not found: {study_dir}. Run study-export first."
            )

        ratings_file = Path(ratings_path)
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

        raw = pd.read_csv(ratings_file)
        if rater_id is not None:
            incoming = rater_sheet_to_long(raw, rater_id)
        else:
            incoming = raw

        known_ids: Optional[set[str]] = None
        analysis_path = study_dir / "analysis_items.csv"
        source_key_path = study_dir / "source_key.csv"
        if analysis_path.exists():
            known_ids = set(
                pd.read_csv(analysis_path)["item_id"].astype(str).tolist()
            )
        elif source_key_path.exists():
            key = pd.read_csv(source_key_path)
            if "role" in key.columns:
                analysis_key = key[key["role"] == "analysis"]
            else:
                analysis_key = key
            known_ids = set(analysis_key["item_id"].astype(str).tolist())

        validated = validate_ratings(incoming, known_item_ids=known_ids)

        existing_path = study_dir / RATINGS_FILENAME
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            combined = pd.concat([existing, validated], ignore_index=True)
            # Re-validate uniqueness across the merged table.
            validated = validate_ratings(combined, known_item_ids=known_ids)
        else:
            validated = validated

        validated.to_csv(existing_path, index=False)

        consensus = consensus_medians(validated)
        consensus.to_csv(study_dir / CONSENSUS_FILENAME, index=False)

        reliability = compute_reliability(validated)
        reliability_path = study_dir / RELIABILITY_FILENAME
        reliability_path.write_text(
            json.dumps(reliability, indent=2), encoding="utf-8"
        )

        study_manifest_path = study_dir / "manifest.json"
        study_manifest = {}
        if study_manifest_path.exists():
            study_manifest = json.loads(
                study_manifest_path.read_text(encoding="utf-8")
            )
        study_manifest["rubric_version"] = RUBRIC_VERSION
        study_manifest["imported_at"] = datetime.now(timezone.utc).isoformat()
        study_manifest["n_ratings"] = int(len(validated))
        study_manifest["n_items_rated"] = int(validated["item_id"].nunique())
        study_manifest["n_raters"] = int(validated["rater_id"].nunique())
        artifacts = study_manifest.setdefault("artifacts", {})
        artifacts["ratings_csv"] = RATINGS_FILENAME
        artifacts["consensus_csv"] = CONSENSUS_FILENAME
        artifacts["reliability_json"] = RELIABILITY_FILENAME
        study_manifest_path.write_text(
            json.dumps(study_manifest, indent=2), encoding="utf-8"
        )

        assessment_manifest_path = run_dir / "manifest.json"
        assessment_manifest = json.loads(
            assessment_manifest_path.read_text(encoding="utf-8")
        )
        human_study = assessment_manifest.setdefault("human_study", {})
        human_study["imported_at"] = study_manifest["imported_at"]
        human_study["n_ratings"] = int(len(validated))
        human_study["n_items_rated"] = int(validated["item_id"].nunique())
        assessment_manifest_path.write_text(
            json.dumps(assessment_manifest, indent=2), encoding="utf-8"
        )

        return {
            "n_ratings": int(len(validated)),
            "n_items": int(validated["item_id"].nunique()),
            "n_raters": int(validated["rater_id"].nunique()),
            "ratings_path": existing_path,
            "reliability_path": reliability_path,
            "consensus_path": study_dir / CONSENSUS_FILENAME,
        }
