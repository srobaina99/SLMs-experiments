"""Export a stratified sample from a run bundle for human review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from slm_experiments.core.run_store import RunStore
from slm_experiments.models.base import REPO_ROOT

HUMAN_REVIEW_FILENAME = "human_review.csv"
DEFAULT_SAMPLE_SIZE = 60
DEFAULT_SEED = 42

EXPORT_COLUMNS = [
    "experiment_id",
    "model",
    "config",
    "prompt_id",
    "answer",
    "response_appropriateness",
    "vocabulary_level",
    "notes",
]


def config_label(config_weighting: bool, config_prompting: bool) -> str:
    """Map intervention flags to a config label."""
    if config_weighting and config_prompting:
        return "both"
    if config_weighting:
        return "weighting_only"
    if config_prompting:
        return "prompting_only"
    return "control"


def stratified_sample(
    df: pd.DataFrame, n: int, seed: int = DEFAULT_SEED, config_col: str = "config"
) -> pd.DataFrame:
    """Sample up to n rows, stratified by config label."""
    if n <= 0:
        raise ValueError(f"sample size must be positive, got {n}")
    if len(df) <= n:
        return df.copy()

    groups = list(df.groupby(config_col, sort=False))
    num_groups = len(groups)
    sizes = {name: len(group) for name, group in groups}

    alloc: dict[str, int] = {}
    for name, size in sizes.items():
        target = int(round(n * size / len(df)))
        if n >= num_groups:
            target = max(1, target)
        alloc[name] = min(target, size)

    while sum(alloc.values()) > n:
        candidates = [name for name in alloc if alloc[name] > (1 if n >= num_groups else 0)]
        if not candidates:
            name = max(alloc, key=alloc.get)
        else:
            name = max(candidates, key=lambda k: alloc[k])
        alloc[name] -= 1

    while sum(alloc.values()) < n:
        candidates = [name for name in alloc if alloc[name] < sizes[name]]
        if not candidates:
            break
        name = max(candidates, key=lambda k: sizes[k] - alloc[k])
        alloc[name] += 1

    parts = []
    for name, group in groups:
        count = alloc.get(name, 0)
        if count > 0:
            parts.append(group.sample(n=count, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


class HumanExporter:
    """Sample rows from a run bundle and write human_review.csv."""

    def __init__(self, results_root: Optional[Union[Path, str]] = None):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)

    def export(
        self,
        run_id: str,
        sample: int = DEFAULT_SAMPLE_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> tuple[Path, int]:
        """
        Export a human-review CSV into the run bundle directory.

        Returns:
            (output_path, row_count)
        """
        run_dir = self.run_store.run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run bundle not found: {run_dir}")

        full_df = self.run_store.read_full_csv(run_id)
        if full_df.empty:
            raise ValueError(f"Run bundle {run_id} has no observations in full.csv")

        working = full_df.copy()
        working["config"] = working.apply(
            lambda row: config_label(
                bool(row["config_weighting"]), bool(row["config_prompting"])
            ),
            axis=1,
        )
        working["answer"] = working["response"]

        sampled = stratified_sample(working, sample, seed=seed)

        review_df = sampled[
            ["experiment_id", "model", "config", "prompt_id", "answer"]
        ].copy()
        review_df["response_appropriateness"] = pd.Series(
            [pd.NA] * len(review_df), dtype="Float64"
        )
        review_df["vocabulary_level"] = pd.Series(
            [pd.NA] * len(review_df), dtype="string"
        )
        review_df["notes"] = pd.Series([pd.NA] * len(review_df), dtype="string")

        out_path = run_dir / HUMAN_REVIEW_FILENAME
        review_df[EXPORT_COLUMNS].to_csv(out_path, index=False)

        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.setdefault("artifacts", {})
        artifacts["human_review_csv"] = HUMAN_REVIEW_FILENAME
        manifest["human_eval"] = {
            "exported_rows": len(review_df),
            "sample_requested": sample,
            "sample_seed": seed,
            "source_total": len(full_df),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return out_path, len(review_df)
