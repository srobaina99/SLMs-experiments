"""Provider-neutral LLM-judge seam (export + validated import).

Wired but unimplemented: builds ``judge_input.jsonl``, ships a versioned
rubric aligned with the human study dimensions, documents a strict output
schema, and imports placeholder ``judge_scores.csv`` results.

**No API / provider adapter.** Callers produce scores externally and import
them here. Quality-preservation claims stay human-sample-limited until
imported judge results are shown to agree acceptably with human consensus.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from slm_experiments.core.run_store import KIND_ASSESSMENT, RunStore
from slm_experiments.human.rubric import (
    NOTES_COLUMN,
    RATING_DIMENSIONS,
    RATING_MAX,
    RATING_MIN,
    RUBRIC_VERSION as HUMAN_RUBRIC_VERSION,
)
from slm_experiments.models.base import REPO_ROOT

JUDGE_DIRNAME = "judge"
JUDGE_INPUT_FILENAME = "judge_input.jsonl"
JUDGE_SCORES_FILENAME = "judge_scores.csv"
JUDGE_MANIFEST_FILENAME = "manifest.json"

# Judge rubric is a sibling of the human rubric; same four 1–4 dimensions.
JUDGE_RUBRIC_VERSION = "beginner_suitability_judge_rubric_v0"
JUDGE_SCHEMA_VERSION = "judge_scores_v0"

_ASSETS_DIR = Path(__file__).resolve().parent
JUDGE_RUBRIC_MARKDOWN = (
    _ASSETS_DIR / "judge_rubrics" / f"{JUDGE_RUBRIC_VERSION}.md"
)
JUDGE_SCHEMA_PATH = _ASSETS_DIR / "judge_schema" / f"{JUDGE_SCHEMA_VERSION}.json"

# Strict CSV columns (required). Optional extras allowed after validation.
REQUIRED_SCORE_COLUMNS = ["item_id", *RATING_DIMENSIONS]
OPTIONAL_SCORE_COLUMNS = (NOTES_COLUMN, "judge_id", "rubric_version")

# Documented claim gate (also in ExperimentDesign / docs/metrics).
QUALITY_PRESERVATION_SCOPE = (
    "Quality-preservation claims stay limited to the human sample until "
    "judge results are imported and shown to agree acceptably with human "
    "consensus."
)


def load_judge_schema() -> Dict[str, Any]:
    """Load the committed JSON Schema for judge score rows."""
    if not JUDGE_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Judge schema not found: {JUDGE_SCHEMA_PATH}")
    return json.loads(JUDGE_SCHEMA_PATH.read_text(encoding="utf-8"))


def build_judge_input_records(
    items: pd.DataFrame,
    *,
    rubric_version: str = JUDGE_RUBRIC_VERSION,
) -> List[Dict[str, Any]]:
    """
    Build one judge-input object per assessment item.

    Fields are provider-neutral: ``item_id``, ``prompt``, ``answer``
    (= ``cleaned_response``), plus rubric/dimension metadata for a future
    external judge. No model/config/CEFR/KVL leakage.
    """
    if items.empty:
        return []
    missing = [c for c in ("item_id", "prompt", "cleaned_response") if c not in items.columns]
    if missing:
        raise ValueError(f"items frame missing columns: {missing}")

    records: List[Dict[str, Any]] = []
    for _, row in items.iterrows():
        item_id = str(row["item_id"]).strip()
        if not item_id:
            continue
        answer = row.get("cleaned_response", "")
        if answer is None or (isinstance(answer, float) and pd.isna(answer)):
            answer = ""
        prompt = row.get("prompt", "")
        if prompt is None or (isinstance(prompt, float) and pd.isna(prompt)):
            prompt = ""
        records.append(
            {
                "item_id": item_id,
                "prompt": str(prompt),
                "answer": str(answer),
                "rubric_version": rubric_version,
                "human_rubric_version": HUMAN_RUBRIC_VERSION,
                "dimensions": list(RATING_DIMENSIONS),
                "rating_min": RATING_MIN,
                "rating_max": RATING_MAX,
            }
        )
    return records


def write_judge_input_jsonl(
    records: Sequence[Dict[str, Any]],
    path: Union[Path, str],
) -> int:
    """Write newline-delimited JSON; returns number of lines written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _normalize_score(value: Any, *, column: str) -> int:
    if pd.isna(value) or value == "":
        raise ValueError(f"missing required score in column {column}")
    number = int(float(value))
    if number < RATING_MIN or number > RATING_MAX:
        raise ValueError(
            f"score must be an integer in [{RATING_MIN}, {RATING_MAX}] "
            f"for {column}, got {value}"
        )
    return number


def validate_judge_scores(
    scores: pd.DataFrame,
    *,
    known_item_ids: Optional[set[str]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Validate ``judge_scores.csv`` against the strict seam schema.

    Enforces required columns, ordinal 1–4 dimensions matching the human
    study, and exactly one row per ``item_id``. Optional ``notes``,
    ``judge_id``, and ``rubric_version`` are retained when present.
    """
    # Touch schema so tests / callers confirm the committed contract exists.
    schema = schema if schema is not None else load_judge_schema()
    required_props = schema.get("required", list(REQUIRED_SCORE_COLUMNS))
    missing = [col for col in required_props if col not in scores.columns]
    if missing:
        raise ValueError(f"judge_scores.csv missing columns: {missing}")

    # Reject unknown columns beyond the documented optional set.
    allowed = set(REQUIRED_SCORE_COLUMNS) | set(OPTIONAL_SCORE_COLUMNS)
    extra = [c for c in scores.columns if c not in allowed]
    if extra:
        raise ValueError(
            f"judge_scores.csv has undeclared columns (strict schema): {extra}"
        )

    working = scores.copy()
    working["item_id"] = working["item_id"].astype(str)
    if working["item_id"].str.strip().eq("").any():
        raise ValueError("item_id must be non-empty")

    dup_mask = working.duplicated(subset=["item_id"], keep=False)
    if dup_mask.any():
        sample = (
            working.loc[dup_mask, ["item_id"]]
            .drop_duplicates()
            .head(5)["item_id"]
            .tolist()
        )
        raise ValueError(
            f"exactly one score row per item_id required; duplicates: {sample}"
        )

    for dim in RATING_DIMENSIONS:
        working[dim] = [
            _normalize_score(value, column=dim) for value in working[dim]
        ]

    if NOTES_COLUMN in working.columns:
        working[NOTES_COLUMN] = working[NOTES_COLUMN].astype("string")
    if "judge_id" in working.columns:
        working["judge_id"] = working["judge_id"].astype("string")
    if "rubric_version" in working.columns:
        working["rubric_version"] = working["rubric_version"].astype("string")
        bad = working["rubric_version"].dropna()
        bad = bad[bad.astype(str).str.strip() != ""]
        unexpected = sorted(
            {
                str(v)
                for v in bad
                if str(v).strip() not in {JUDGE_RUBRIC_VERSION, HUMAN_RUBRIC_VERSION}
            }
        )
        if unexpected:
            raise ValueError(
                f"rubric_version must be {JUDGE_RUBRIC_VERSION!r} "
                f"(or human {HUMAN_RUBRIC_VERSION!r}); got {unexpected}"
            )

    if known_item_ids is not None:
        unknown = set(working["item_id"]) - set(known_item_ids)
        if unknown:
            sample = sorted(unknown)[:5]
            raise ValueError(f"judge_scores.csv contains unknown item_id values: {sample}")

    # Preserve column order: required first, then known optionals present.
    ordered = list(REQUIRED_SCORE_COLUMNS)
    for col in OPTIONAL_SCORE_COLUMNS:
        if col in working.columns:
            ordered.append(col)
    return working[ordered]


class JudgeExporter:
    """Write provider-neutral judge input artifacts under an assessment bundle."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)

    def export(self, assessment_run_id: str) -> tuple[Path, int]:
        """
        Build ``{assessment}/judge/judge_input.jsonl`` (+ rubric + schema).

        Returns ``(judge_dir, n_items)``.
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

        items = self.run_store.read_items_csv(assessment_run_id)
        if items.empty:
            raise ValueError(f"Assessment bundle {assessment_run_id} has no items")

        records = build_judge_input_records(items)
        if not records:
            raise ValueError(
                f"No judge-input records built from assessment {assessment_run_id}"
            )

        judge_dir = run_dir / JUDGE_DIRNAME
        if judge_dir.exists():
            shutil.rmtree(judge_dir)
        judge_dir.mkdir(parents=True, exist_ok=True)

        input_path = judge_dir / JUDGE_INPUT_FILENAME
        n_items = write_judge_input_jsonl(records, input_path)

        if JUDGE_RUBRIC_MARKDOWN.exists():
            shutil.copy2(JUDGE_RUBRIC_MARKDOWN, judge_dir / JUDGE_RUBRIC_MARKDOWN.name)
        if JUDGE_SCHEMA_PATH.exists():
            shutil.copy2(JUDGE_SCHEMA_PATH, judge_dir / JUDGE_SCHEMA_PATH.name)

        created_at = datetime.now(timezone.utc).isoformat()
        judge_manifest: Dict[str, Any] = {
            "kind": "llm_judge_seam",
            "assessment_run_id": assessment_run_id,
            "rubric_version": JUDGE_RUBRIC_VERSION,
            "human_rubric_version": HUMAN_RUBRIC_VERSION,
            "schema_version": JUDGE_SCHEMA_VERSION,
            "created_at": created_at,
            "n_items": n_items,
            "dimensions": list(RATING_DIMENSIONS),
            "rating_min": RATING_MIN,
            "rating_max": RATING_MAX,
            "api_adapter": None,
            "provider": None,
            "quality_preservation_scope": QUALITY_PRESERVATION_SCOPE,
            "artifacts": {
                "judge_input_jsonl": JUDGE_INPUT_FILENAME,
                "judge_scores_csv": None,
                "rubric": (
                    JUDGE_RUBRIC_MARKDOWN.name if JUDGE_RUBRIC_MARKDOWN.exists() else None
                ),
                "schema": JUDGE_SCHEMA_PATH.name if JUDGE_SCHEMA_PATH.exists() else None,
            },
        }
        (judge_dir / JUDGE_MANIFEST_FILENAME).write_text(
            json.dumps(judge_manifest, indent=2), encoding="utf-8"
        )

        assessment_manifest_path = run_dir / "manifest.json"
        assessment_manifest = json.loads(
            assessment_manifest_path.read_text(encoding="utf-8")
        )
        artifacts = assessment_manifest.setdefault("artifacts", {})
        artifacts["judge_dir"] = JUDGE_DIRNAME
        assessment_manifest["llm_judge"] = {
            "exported_at": created_at,
            "n_items": n_items,
            "rubric_version": JUDGE_RUBRIC_VERSION,
            "schema_version": JUDGE_SCHEMA_VERSION,
            "api_adapter": None,
            "quality_preservation_scope": QUALITY_PRESERVATION_SCOPE,
        }
        assessment_manifest_path.write_text(
            json.dumps(assessment_manifest, indent=2), encoding="utf-8"
        )

        return judge_dir, n_items


class JudgeImporter:
    """Validate and store externally produced judge_scores.csv (no API calls)."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)

    def import_scores(
        self,
        assessment_run_id: str,
        scores_path: Union[Path, str],
    ) -> dict:
        """
        Import validated ``judge_scores.csv`` into ``{assessment}/judge/``.

        Returns a summary dict. Does not call any LLM provider.
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

        judge_dir = run_dir / JUDGE_DIRNAME
        if not judge_dir.exists():
            raise FileNotFoundError(
                f"Judge directory not found: {judge_dir}. Run judge-export first."
            )

        scores_file = Path(scores_path)
        if not scores_file.exists():
            raise FileNotFoundError(f"Judge scores file not found: {scores_file}")

        known_ids: Optional[set[str]] = None
        input_path = judge_dir / JUDGE_INPUT_FILENAME
        if input_path.exists():
            known_ids = set()
            with input_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    known_ids.add(str(json.loads(line)["item_id"]))
        else:
            items = self.run_store.read_items_csv(assessment_run_id)
            known_ids = set(items["item_id"].astype(str).tolist())

        raw = pd.read_csv(scores_file)
        validated = validate_judge_scores(raw, known_item_ids=known_ids)

        out_path = judge_dir / JUDGE_SCORES_FILENAME
        validated.to_csv(out_path, index=False)

        imported_at = datetime.now(timezone.utc).isoformat()
        judge_manifest_path = judge_dir / JUDGE_MANIFEST_FILENAME
        judge_manifest: Dict[str, Any] = {}
        if judge_manifest_path.exists():
            judge_manifest = json.loads(
                judge_manifest_path.read_text(encoding="utf-8")
            )
        judge_manifest["imported_at"] = imported_at
        judge_manifest["n_scores"] = int(len(validated))
        judge_manifest["n_items_scored"] = int(validated["item_id"].nunique())
        judge_manifest["api_adapter"] = None
        judge_manifest["quality_preservation_scope"] = QUALITY_PRESERVATION_SCOPE
        artifacts = judge_manifest.setdefault("artifacts", {})
        artifacts["judge_scores_csv"] = JUDGE_SCORES_FILENAME
        judge_manifest_path.write_text(
            json.dumps(judge_manifest, indent=2), encoding="utf-8"
        )

        assessment_manifest_path = run_dir / "manifest.json"
        assessment_manifest = json.loads(
            assessment_manifest_path.read_text(encoding="utf-8")
        )
        llm_judge = assessment_manifest.setdefault("llm_judge", {})
        llm_judge["imported_at"] = imported_at
        llm_judge["n_scores"] = int(len(validated))
        llm_judge["n_items_scored"] = int(validated["item_id"].nunique())
        llm_judge["api_adapter"] = None
        llm_judge["quality_preservation_scope"] = QUALITY_PRESERVATION_SCOPE
        assessment_manifest_path.write_text(
            json.dumps(assessment_manifest, indent=2), encoding="utf-8"
        )

        return {
            "n_scores": int(len(validated)),
            "n_items": int(validated["item_id"].nunique()),
            "scores_path": out_path,
            "judge_dir": judge_dir,
        }
