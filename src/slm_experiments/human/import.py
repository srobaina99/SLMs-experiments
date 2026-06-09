"""Import human review tags back into a run bundle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from slm_experiments.core.run_store import RunStore
from slm_experiments.models.base import REPO_ROOT

TAG_COLUMNS = ["response_appropriateness", "vocabulary_level", "notes"]


def _normalize_tag_value(column: str, value) -> object:
    """Convert reviewer input to stored CSV values."""
    if pd.isna(value) or value == "":
        return None
    if column == "response_appropriateness":
        return float(value)
    return str(value).strip()


class HumanImporter:
    """Merge human tags from a CSV into full.csv of a run bundle."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)

    def import_tags(self, run_id: str, tags_path: Union[Path, str]) -> int:
        """
        Merge tags into full.csv by experiment_id.

        Returns:
            Number of rows updated with at least one tag value.
        """
        run_dir = self.run_store.run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run bundle not found: {run_dir}")

        tags_file = Path(tags_path)
        if not tags_file.exists():
            raise FileNotFoundError(f"Tags file not found: {tags_file}")

        tags_df = pd.read_csv(tags_file)
        if "experiment_id" not in tags_df.columns:
            raise ValueError("Tags CSV must include an experiment_id column")

        missing_tag_cols = [col for col in TAG_COLUMNS if col not in tags_df.columns]
        if missing_tag_cols:
            raise ValueError(f"Tags CSV missing columns: {missing_tag_cols}")

        if tags_df["experiment_id"].duplicated().any():
            raise ValueError("Tags CSV contains duplicate experiment_id values")

        full_df = self.run_store.read_full_csv(run_id)
        if full_df.empty:
            raise ValueError(f"Run bundle {run_id} has no observations in full.csv")

        for col in TAG_COLUMNS:
            if col not in full_df.columns:
                full_df[col] = None
            elif col == "response_appropriateness":
                full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
            else:
                full_df[col] = full_df[col].astype("object")

        known_ids = set(full_df["experiment_id"].astype(str))
        tag_ids = set(tags_df["experiment_id"].astype(str))
        unknown_ids = tag_ids - known_ids
        if unknown_ids:
            sample = sorted(unknown_ids)[:5]
            raise ValueError(
                f"Tags CSV contains unknown experiment_id values: {sample}"
            )

        tags_by_id = tags_df.set_index("experiment_id", drop=False)
        updated_count = 0

        for idx, row in full_df.iterrows():
            exp_id = row["experiment_id"]
            if exp_id not in tags_by_id.index:
                continue

            tag_row = tags_by_id.loc[exp_id]
            changed = False
            for col in TAG_COLUMNS:
                normalized = _normalize_tag_value(col, tag_row[col])
                if normalized is not None:
                    full_df.at[idx, col] = normalized
                    changed = True
            if changed:
                updated_count += 1

        self.run_store.write_full_csv(run_id, full_df)

        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("artifacts", {})["human_review_csv"] = tags_file.name
        manifest["human_eval"] = {
            **manifest.get("human_eval", {}),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "tags_file": str(tags_file.resolve()),
            "tagged_rows": int(len(tags_df)),
            "updated_rows": updated_count,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return updated_count
