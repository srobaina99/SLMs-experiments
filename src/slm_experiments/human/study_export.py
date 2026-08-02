"""Blind three-rater study export from an assessment bundle.

Writes study artifacts under ``results/runs/{assessment_id}/study/``.
Never mutates generation runs or assessment ``items.csv`` / ``item_map.csv``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from slm_experiments.core.bool_series import coerce_bool_series
from slm_experiments.core.run_store import KIND_ASSESSMENT, RunStore
from slm_experiments.human.rubric import (
    NOTES_COLUMN,
    RATING_DIMENSIONS,
    RUBRIC_MARKDOWN,
    RUBRIC_VERSION,
)
from slm_experiments.models.base import REPO_ROOT

STUDY_DIRNAME = "study"
# Only this subdirectory is safe to distribute to raters (no stratum / arm labels).
RATER_PACKET_DIRNAME = "rater_packet"
RATER_SHEETS_DIRNAME = "rater_sheets"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_CALIBRATION_SIZE = 10
DEFAULT_SEED = 42
DEFAULT_RATERS = ("r1", "r2", "r3")

# Import writes these; re-export must not wipe them without --force.
IMPORTED_STUDY_ARTIFACTS = ("ratings.csv", "consensus.csv", "reliability.json")

# Columns that must never appear in the rater-facing packet.
RATER_PACKET_FORBIDDEN_COLUMNS = frozenset(
    {
        "stratum",
        "family",
        "model",
        "arm",
        "cefr_band",
        "disagreement",
        "truncation",
        "source_run_ids",
        "experiment_ids",
        "configs",
        "inclusion_probability",
        "inclusion_weight",
        "role",
    }
)

BLIND_COLUMNS = ["item_id", "prompt", "answer", *RATING_DIMENSIONS, NOTES_COLUMN]
# Analyst-facing analysis sample (weights + stratum); lives outside rater_packet/.
ANALYSIS_ITEM_COLUMNS = [
    "item_id",
    "prompt",
    "answer",
    "inclusion_probability",
    "inclusion_weight",
    "stratum",
]


def assert_rater_packet_columns_safe(columns: Sequence[str]) -> None:
    """Raise if a rater-packet CSV would leak stratum / arm-identifying columns.

    Complements the ``BLIND_COLUMNS`` whitelist: even if the whitelist is
    accidentally widened, forbidden identifiers must never reach disk in
    ``rater_packet/``.
    """
    leaked = sorted(set(columns) & RATER_PACKET_FORBIDDEN_COLUMNS)
    if leaked:
        raise ValueError(
            "rater packet CSV would leak identifying columns "
            f"(blinding hazard): {leaked}"
        )


def _prepare_rater_sheet_for_write(sheet: pd.DataFrame) -> pd.DataFrame:
    """Select blind columns and fail closed if any forbidden identifier remains."""
    # Guard the whitelist itself so a widened BLIND_COLUMNS cannot slip past
    # a KeyError when the sheet frame lacks the leaked column.
    assert_rater_packet_columns_safe(BLIND_COLUMNS)
    out_sheet = sheet[BLIND_COLUMNS]
    assert_rater_packet_columns_safe(out_sheet.columns.tolist())
    return out_sheet

SOURCE_KEY_COLUMNS = [
    "item_id",
    "prompt_id",
    "prompt",
    "answer",
    "family",
    "model",
    "arm",
    "cefr_band",
    "disagreement",
    "truncation",
    "stratum",
    "inclusion_probability",
    "inclusion_weight",
    "role",
    "source_run_ids",
    "experiment_ids",
    "configs",
]


def parse_experiment_family(source_run_id: str) -> str:
    """Extract experiment family from ``{ts}_{phase}_{experiment}`` run ids."""
    parts = str(source_run_id).split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return "unknown"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def arm_label(row: pd.Series, family: Optional[str] = None) -> str:
    """Classify a map row as baseline vs intervention (family-aware).

    Delegates to ``analysis.resolve_arm`` so weights rows with
    ``num_shots == 0`` are not mis-labelled as baseline.
    """
    from slm_experiments.evaluation.assessment.analysis import resolve_arm

    return resolve_arm(row, family=family)


def cefr_band_label(level: Any) -> str:
    if _is_missing(level):
        return "unknown"
    text = str(level).strip().upper()
    if text == "A1":
        return "A1"
    if text in {"A2", "B1", "B2", "C1", "C2"}:
        return "higher"
    return "unknown"


def disagreement_label(value: Any) -> str:
    if _is_missing(value):
        return "unknown"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return "disagree"
        if lowered in {"false", "0", "no"}:
            return "agree"
        return "unknown"
    return "disagree" if bool(value) else "agree"


def _resolve_disagreement(score: pd.Series) -> Any:
    """Populate TSAR↔CEFR-SP disagreement when TSAR fields are present.

    Prefers the stored ``cefr_tsar_disagrees_with_cefr_sp`` flag; if that is
    missing but both discrete labels exist, recomputes the flag. Returns
    ``None`` when TSAR scores are absent (stratum stays ``unknown``).
    """
    disagree = None
    if "cefr_tsar_disagrees_with_cefr_sp" in score.index:
        disagree = score.get("cefr_tsar_disagrees_with_cefr_sp")
    if not _is_missing(disagree):
        return disagree

    # TSAR absent → leave unknown; TSAR present with both labels → recompute.
    if "cefr_tsar_ensemble_label" not in score.index:
        return None
    ensemble = score.get("cefr_tsar_ensemble_label")
    if _is_missing(ensemble):
        return None
    from slm_experiments.evaluation.assessment.cefr_tsar import (
        disagrees_with_cefr_sp,
    )

    return disagrees_with_cefr_sp(score.get("cefr_sp_level"), ensemble)


def stratified_sample_with_weights(
    df: pd.DataFrame,
    n: int,
    *,
    stratum_col: str = "stratum",
    seed: int = DEFAULT_SEED,
    inclusion_pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Sample up to ``n`` rows stratified by ``stratum_col``.

    Adds ``inclusion_probability`` (``π_i``) and ``inclusion_weight`` (= 1/π_i).

    Sampling allocation uses stratum sizes in ``df``. Inclusion probabilities
    are computed against ``inclusion_pool`` when provided (else ``df``), so a
    post-calibration analysis draw can report unconditional Horvitz–Thompson
    weights vs the original pool: ``π_i = n_s / N_s`` where ``N_s`` is the
    original stratum size.
    """
    if n <= 0:
        raise ValueError(f"sample size must be positive, got {n}")
    if df.empty:
        return df.copy()

    pool = inclusion_pool if inclusion_pool is not None else df
    pool_sizes = {
        name: int(len(group))
        for name, group in pool.groupby(stratum_col, sort=False)
    }

    if len(df) <= n:
        # Census of ``df``: still use unconditional π vs inclusion_pool.
        out = df.copy()
        drawn_sizes = {
            name: int(len(group))
            for name, group in out.groupby(stratum_col, sort=False)
        }

        def _census_p(stratum: Any) -> float:
            pool_n = float(pool_sizes.get(stratum, 0))
            count = float(drawn_sizes.get(stratum, 0))
            return count / pool_n if pool_n > 0 else 0.0

        out["inclusion_probability"] = out[stratum_col].map(_census_p)
        out["inclusion_weight"] = out["inclusion_probability"].map(
            lambda p: (1.0 / p) if p > 0 else float("inf")
        )
        return out.reset_index(drop=True)

    groups = list(df.groupby(stratum_col, sort=False))
    num_groups = len(groups)
    sizes = {name: len(group) for name, group in groups}

    alloc: Dict[Any, int] = {}
    for name, size in sizes.items():
        target = int(round(n * size / len(df)))
        if n >= num_groups:
            target = max(1, target)
        alloc[name] = min(target, size)

    while sum(alloc.values()) > n:
        candidates = [
            name for name in alloc if alloc[name] > (1 if n >= num_groups else 0)
        ]
        if not candidates:
            name = max(alloc, key=alloc.get)  # type: ignore[arg-type]
        else:
            name = max(candidates, key=lambda k: alloc[k])
        alloc[name] -= 1

    while sum(alloc.values()) < n:
        candidates = [name for name in alloc if alloc[name] < sizes[name]]
        if not candidates:
            break
        name = max(candidates, key=lambda k: sizes[k] - alloc[k])
        alloc[name] += 1

    parts: List[pd.DataFrame] = []
    for name, group in groups:
        count = alloc.get(name, 0)
        if count <= 0:
            continue
        sampled = group.sample(n=count, random_state=seed).copy()
        # Unconditional vs inclusion_pool: π = n_drawn / N_original_stratum.
        pool_n = float(pool_sizes.get(name, len(group)))
        p = float(count) / pool_n if pool_n > 0 else 0.0
        sampled["inclusion_probability"] = p
        sampled["inclusion_weight"] = 1.0 / p if p > 0 else float("inf")
        parts.append(sampled)

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts).reset_index(drop=True)


def _load_scores(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "scores.csv"
    if not path.exists():
        return pd.DataFrame(columns=["item_id"])
    return pd.read_csv(path)


def build_item_frame(
    items: pd.DataFrame,
    item_map: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """One row per unique in-sample item with stratification axes."""
    if items.empty:
        return pd.DataFrame()

    in_sample = item_map.copy()
    if "in_sample" in in_sample.columns:
        in_sample = in_sample[coerce_bool_series(in_sample["in_sample"])]
    in_sample = in_sample[in_sample["item_id"].astype(str).str.strip() != ""]
    if in_sample.empty:
        # Fall back to items.csv ids alone (map may omit in_sample).
        in_sample = item_map[item_map["item_id"].astype(str).isin(items["item_id"])]

    score_by_id: Dict[str, pd.Series] = {}
    if not scores.empty and "item_id" in scores.columns:
        for _, row in scores.drop_duplicates("item_id").iterrows():
            score_by_id[str(row["item_id"])] = row

    records: List[Dict[str, Any]] = []
    items_by_id = items.set_index("item_id", drop=False)

    for item_id, group in in_sample.groupby("item_id", sort=False):
        item_id = str(item_id)
        if item_id not in items_by_id.index:
            continue
        item_row = items_by_id.loc[item_id]
        if isinstance(item_row, pd.DataFrame):
            item_row = item_row.iloc[0]

        families = sorted({parse_experiment_family(r) for r in group["source_run_id"]})
        models = sorted({str(m) for m in group["model"].tolist()})
        arms = sorted({arm_label(row) for _, row in group.iterrows()})
        configs = sorted({str(c) for c in group.get("config", pd.Series(dtype=str)).tolist()})
        truncated = bool(
            coerce_bool_series(
                group.get("hit_max_tokens", pd.Series(dtype=bool))
            ).any()
        )

        score = score_by_id.get(item_id)
        if score is not None:
            cefr_level = score.get("cefr_sp_level")
            if _is_missing(cefr_level):
                cefr_level = score.get("cefr_tsar_ensemble_label")
            disagree = _resolve_disagreement(score)
        else:
            cefr_level = None
            disagree = None

        family = families[0] if len(families) == 1 else "multi"
        model = models[0] if len(models) == 1 else "multi"
        arm = arms[0] if len(arms) == 1 else "mixed"
        cefr_band = cefr_band_label(cefr_level)
        disagree_lab = disagreement_label(disagree)
        trunc_lab = "truncated" if truncated else "complete"
        stratum = f"{family}|{model}|{arm}|{cefr_band}|{disagree_lab}|{trunc_lab}"

        records.append(
            {
                "item_id": item_id,
                "prompt_id": item_row.get("prompt_id", ""),
                "prompt": item_row.get("prompt", ""),
                "answer": item_row.get("cleaned_response", ""),
                "family": family,
                "model": model,
                "arm": arm,
                "cefr_band": cefr_band,
                "disagreement": disagree_lab,
                "truncation": trunc_lab,
                "stratum": stratum,
                "source_run_ids": ";".join(
                    sorted({str(x) for x in group["source_run_id"].tolist()})
                ),
                "experiment_ids": ";".join(
                    sorted({str(x) for x in group["experiment_id"].tolist()})
                ),
                "configs": ";".join(configs),
            }
        )

    return pd.DataFrame(records)


class StudyExporter:
    """Export blinded rater sheets + private source key into an assessment study/ dir."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)

    def export(
        self,
        assessment_run_id: str,
        *,
        sample: int = DEFAULT_SAMPLE_SIZE,
        calibration: int = DEFAULT_CALIBRATION_SIZE,
        seed: int = DEFAULT_SEED,
        raters: Sequence[str] = DEFAULT_RATERS,
        force: bool = False,
    ) -> tuple[Path, int]:
        """
        Build study artifacts under ``{assessment}/study/``.

        Returns ``(study_dir, analysis_item_count)``.

        Refuses to wipe an existing study dir that already contains imported
        ratings unless ``force=True``.
        """
        if sample <= 0:
            raise ValueError(f"sample size must be positive, got {sample}")
        if calibration < 0:
            raise ValueError(f"calibration size must be >= 0, got {calibration}")
        if not raters:
            raise ValueError("at least one rater id is required")

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
        item_map = self.run_store.read_item_map_csv(assessment_run_id)
        scores = _load_scores(run_dir)
        if items.empty:
            raise ValueError(f"Assessment bundle {assessment_run_id} has no items")

        frame = build_item_frame(items, item_map, scores)
        if frame.empty:
            raise ValueError(
                f"No in-sample items available for study export in {assessment_run_id}"
            )

        # Calibration pilot drawn first and excluded from the analysis sample.
        # Inclusion weights for both draws use the original pool (pre-calibration)
        # so analysis π_i = n_s / N_s is unconditional Horvitz–Thompson.
        calibration_df = pd.DataFrame(columns=frame.columns)
        remaining = frame
        if calibration > 0 and len(frame) > 0:
            cal_n = min(calibration, len(frame))
            calibration_df = stratified_sample_with_weights(
                frame,
                cal_n,
                stratum_col="stratum",
                seed=seed,
                inclusion_pool=frame,
            )
            calibration_df = calibration_df.copy()
            calibration_df["role"] = "calibration"
            remaining = frame[
                ~frame["item_id"].isin(calibration_df["item_id"])
            ].reset_index(drop=True)

        analysis_n = min(sample, len(remaining))
        if analysis_n == 0:
            raise ValueError(
                "No items left for analysis sample after excluding calibration pilot"
            )
        # Distinct seed offset so analysis draw differs from calibration draw.
        analysis_df = stratified_sample_with_weights(
            remaining,
            analysis_n,
            stratum_col="stratum",
            seed=seed + 1,
            inclusion_pool=frame,
        )
        analysis_df = analysis_df.copy()
        analysis_df["role"] = "analysis"

        study_dir = run_dir / STUDY_DIRNAME
        rater_packet_dir = study_dir / RATER_PACKET_DIRNAME
        sheets_dir = rater_packet_dir / RATER_SHEETS_DIRNAME
        if study_dir.exists():
            imported = [
                name
                for name in IMPORTED_STUDY_ARTIFACTS
                if (study_dir / name).exists()
            ]
            if imported and not force:
                raise FileExistsError(
                    f"Study directory already has imported ratings "
                    f"({', '.join(imported)}). Re-export would delete them. "
                    f"Pass force=True / --force to overwrite, or keep the "
                    f"existing study artifacts."
                )
            shutil.rmtree(study_dir)
        sheets_dir.mkdir(parents=True, exist_ok=True)

        source_key = pd.concat([analysis_df, calibration_df], ignore_index=True)
        # Ensure weight columns exist when calibration empty.
        for col in ("inclusion_probability", "inclusion_weight"):
            if col not in source_key.columns:
                source_key[col] = 1.0
        source_key_path = study_dir / "source_key.csv"
        source_key[SOURCE_KEY_COLUMNS].to_csv(source_key_path, index=False)

        # Analyst-only: stratum / weights stay outside the rater packet.
        analysis_items = analysis_df[ANALYSIS_ITEM_COLUMNS].copy()
        analysis_items.to_csv(study_dir / "analysis_items.csv", index=False)

        if not calibration_df.empty:
            calibration_df[ANALYSIS_ITEM_COLUMNS].to_csv(
                study_dir / "calibration_items.csv", index=False
            )

        for offset, rater_id in enumerate(raters):
            sheet = analysis_df[["item_id", "prompt", "answer"]].copy()
            # Independent per-rater shuffle.
            sheet = sheet.sample(frac=1.0, random_state=seed + 100 + offset).reset_index(
                drop=True
            )
            for dim in RATING_DIMENSIONS:
                sheet[dim] = pd.Series([pd.NA] * len(sheet), dtype="Int64")
            sheet[NOTES_COLUMN] = pd.Series([pd.NA] * len(sheet), dtype="string")
            _prepare_rater_sheet_for_write(sheet).to_csv(
                sheets_dir / f"rater_{rater_id}.csv", index=False
            )

        if RUBRIC_MARKDOWN.exists():
            shutil.copy2(RUBRIC_MARKDOWN, rater_packet_dir / RUBRIC_MARKDOWN.name)

        study_manifest: Dict[str, Any] = {
            "kind": "human_study",
            "assessment_run_id": assessment_run_id,
            "rubric_version": RUBRIC_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sample_requested": sample,
            "sample_drawn": int(len(analysis_df)),
            "calibration_requested": calibration,
            "calibration_drawn": int(len(calibration_df)),
            "seed": seed,
            "raters": list(raters),
            "strata_axes": [
                "family",
                "model",
                "arm",
                "cefr_band",
                "disagreement",
                "truncation",
            ],
            "blind_columns": ["item_id", "prompt", "answer"],
            "rating_dimensions": list(RATING_DIMENSIONS),
            "distribute_to_raters": RATER_PACKET_DIRNAME,
            "artifacts": {
                "source_key_csv": "source_key.csv",
                "analysis_items_csv": "analysis_items.csv",
                "calibration_items_csv": (
                    "calibration_items.csv" if not calibration_df.empty else None
                ),
                "rater_packet_dir": RATER_PACKET_DIRNAME,
                "rater_sheets_dir": f"{RATER_PACKET_DIRNAME}/{RATER_SHEETS_DIRNAME}",
                "rubric": (
                    f"{RATER_PACKET_DIRNAME}/{RUBRIC_MARKDOWN.name}"
                    if RUBRIC_MARKDOWN.exists()
                    else None
                ),
            },
            "pool_size": int(len(frame)),
            "n_strata": int(frame["stratum"].nunique()),
        }
        (study_dir / "manifest.json").write_text(
            json.dumps(study_manifest, indent=2), encoding="utf-8"
        )

        # Point assessment manifest at the study dir (never touch generation runs).
        assessment_manifest_path = run_dir / "manifest.json"
        assessment_manifest = json.loads(
            assessment_manifest_path.read_text(encoding="utf-8")
        )
        artifacts = assessment_manifest.setdefault("artifacts", {})
        artifacts["human_study_dir"] = STUDY_DIRNAME
        assessment_manifest["human_study"] = {
            "exported_at": study_manifest["created_at"],
            "analysis_items": int(len(analysis_df)),
            "calibration_items": int(len(calibration_df)),
            "sample_seed": seed,
            "rubric_version": RUBRIC_VERSION,
        }
        assessment_manifest_path.write_text(
            json.dumps(assessment_manifest, indent=2), encoding="utf-8"
        )

        return study_dir, int(len(analysis_df))
