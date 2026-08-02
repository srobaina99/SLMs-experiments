"""Run bundle storage: manifest, specification CSV, full CSV, summary JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from slm_experiments.core.bool_series import coerce_bool_series
from slm_experiments.core.config_label import config_label
from slm_experiments.core.result import ExperimentResult

SPEC_COLUMNS = [
    "model",
    "config_weighting",
    "config_prompting",
    "prompt_id",
    "answer",
    "time_spent",
    "generation_successful",
    "hit_max_tokens",
    "meets_a1_criteria",
    "flesch_kincaid_grade",
    "gunning_fog",
    "spache_readability",
    "word_count",
    "difficult_words",
]

NUMERIC_SUMMARY_COLUMNS = [
    "response_time_seconds",
    "flesch_kincaid_grade",
    "gunning_fog",
    "spache_readability",
    "word_count",
    "difficult_words",
    "kvl_mean_score",
    "kvl_min_score",
    "kvl_lookup_coverage",
    "kvl_oov_count",
    "kvl_pct_hard_words",
    "cefr_sp_level_ordinal",
    "cefr_sp_pct_a1",
    "cefr_sp_adjacency",
    "cefr_sp_max_level_ordinal",
    "cefr_sp_expected_level",
]

SWEEP_SUMMARY_SECTIONS = {
    "weights": ("by_weight_factor", "weight_factor"),
    "beam": ("by_beam_width", "beam_width"),
    "kvl_beam": ("by_kvl_beam_width", "kvl_beam_width"),
    "prompting": ("by_num_shots", "num_shots"),
    "guided": ("by_guided_top_k", "guided_top_k"),
}

KIND_GENERATION = "generation"
KIND_ASSESSMENT = "assessment"


def _format_sweep_key(column: str, value: Any) -> str:
    if column in ("beam_width", "kvl_beam_width", "num_shots", "guided_top_k"):
        return str(int(value))
    return f"{float(value):g}"


def _sweep_sort_key(column: str, value: Any) -> float:
    if column in ("beam_width", "kvl_beam_width", "num_shots", "guided_top_k"):
        return float(int(value))
    return float(value)


def _aggregate_metric_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Build count + metric stats for one group, excluding failed generations."""
    success_flags = coerce_bool_series(df["generation_successful"])
    successful = df.loc[success_flags]
    stats: Dict[str, Any] = {
        "count": int(len(df)),
        "generation_successful_count": int(len(successful)),
    }
    if len(df) > 0:
        stats["generation_failure_rate"] = float(1 - len(successful) / len(df))
    else:
        stats["generation_failure_rate"] = 0.0

    if "hit_max_tokens" in df.columns:
        maxed = int(coerce_bool_series(df["hit_max_tokens"]).sum())
        stats["hit_max_tokens_count"] = maxed
        stats["hit_max_tokens_rate"] = float(maxed / len(df)) if len(df) else 0.0
    else:
        stats["hit_max_tokens_count"] = 0
        stats["hit_max_tokens_rate"] = 0.0

    if "meets_a1_criteria" in df.columns:
        a1_flags = coerce_bool_series(df["meets_a1_criteria"])
        a1_pass = int(a1_flags.sum())
        stats["a1_pass_count"] = a1_pass
        stats["a1_pass_rate"] = float(a1_pass / len(df)) if len(df) else 0.0
        if not successful.empty:
            stats["a1_pass_rate_given_valid"] = float(
                coerce_bool_series(successful["meets_a1_criteria"]).sum()
                / len(successful)
            )

    for col in NUMERIC_SUMMARY_COLUMNS:
        if col in successful.columns and not successful.empty:
            col_stats = _metric_stats(successful[col])
            if col_stats is not None:
                stats[col] = col_stats
    return stats


def _build_sweep_grouped(
    df: pd.DataFrame,
    group_column: str,
) -> Optional[Dict[str, Any]]:
    """Aggregate metric stats keyed by sweep hyperparameter values."""
    if group_column not in df.columns:
        return None

    values = df[group_column].dropna().unique()
    if len(values) <= 1:
        return None

    grouped: Dict[str, Any] = {}
    for value in sorted(values, key=lambda item: _sweep_sort_key(group_column, item)):
        key = _format_sweep_key(group_column, value)
        group_df = df[df[group_column] == value]
        grouped[key] = _aggregate_metric_stats(group_df)
    return grouped


def _build_by_config(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate metric stats keyed by intervention config label."""
    if "intervention_config" not in df.columns:
        return {}

    by_config: Dict[str, Any] = {}
    for config in ("control", "weighting_only", "prompting_only", "both"):
        config_df = df[df["intervention_config"] == config]
        if config_df.empty:
            continue
        by_config[config] = _aggregate_metric_stats(config_df)
    return by_config


def _add_sweep_summary(
    summary: Dict[str, Any],
    df: pd.DataFrame,
    experiment: Optional[str],
) -> None:
    if not experiment:
        return

    sweep_spec = SWEEP_SUMMARY_SECTIONS.get(experiment)
    if sweep_spec is None:
        return

    section_key, group_column = sweep_spec
    grouped = _build_sweep_grouped(df, group_column)
    if not grouped:
        return

    summary[section_key] = grouped
    summary["metadata"]["sweep_dimension"] = group_column
    summary["metadata"]["sweep_values"] = sorted(
        grouped.keys(),
        key=lambda item: _sweep_sort_key(group_column, item),
    )


def _add_by_model_summary(
    summary: Dict[str, Any],
    df: pd.DataFrame,
    experiment: Optional[str],
) -> None:
    """Nest per-model aggregates (overall + by_config / sweep) under ``by_model``."""
    if "model" not in df.columns or df.empty:
        return

    by_model: Dict[str, Any] = {}
    sweep_spec = SWEEP_SUMMARY_SECTIONS.get(experiment) if experiment else None

    for model_name in sorted(df["model"].dropna().unique().tolist()):
        model_df = df[df["model"] == model_name]
        model_stats = _aggregate_metric_stats(model_df)

        by_config = _build_by_config(model_df)
        if by_config:
            model_stats["by_config"] = by_config

        if sweep_spec is not None:
            section_key, group_column = sweep_spec
            grouped = _build_sweep_grouped(model_df, group_column)
            if grouped:
                model_stats[section_key] = grouped

        by_model[str(model_name)] = model_stats

    if by_model:
        summary["by_model"] = by_model


def make_run_id(
    phase: Union[int, str], experiment: str, started_at: Optional[datetime] = None
) -> str:
    """Build run ID: {YYYYMMDD_HHMMSS}_{phase}_{experiment}."""
    ts = started_at or datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    phase_label = f"phase{phase}" if isinstance(phase, int) else str(phase)
    return f"{stamp}_{phase_label}_{experiment}"


def _config_label(row: pd.Series) -> str:
    return config_label(bool(row["config_weighting"]), bool(row["config_prompting"]))


def _metric_stats(series: pd.Series) -> Optional[Dict[str, float]]:
    """Aggregate numeric values, ignoring nulls (e.g. disabled CEFR-SP fields)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def compute_summary_stats(
    results: List[ExperimentResult],
    experiment: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate stats; metric means exclude failed generations."""
    if not results:
        return {}

    df = pd.DataFrame([r.to_dict() for r in results])
    success_flags = coerce_bool_series(df["generation_successful"])
    successful_df = df.loc[success_flags]

    summary: Dict[str, Any] = {"overall": {}, "by_config": {}, "metadata": {}}

    for col in NUMERIC_SUMMARY_COLUMNS:
        if col in successful_df.columns and not successful_df.empty:
            col_stats = _metric_stats(successful_df[col])
            if col_stats is not None:
                summary["overall"][col] = col_stats

    if "config_weighting" in df.columns and "config_prompting" in df.columns:
        df = df.copy()
        df["intervention_config"] = df.apply(_config_label, axis=1)
        summary["by_config"] = _build_by_config(df)

    _add_sweep_summary(summary, df, experiment)
    _add_by_model_summary(summary, df, experiment)

    summary["metadata"] = {
        **summary.get("metadata", {}),
        "total_experiments": len(results),
        "successful_experiments": int(successful_df.shape[0]),
        "failed_experiments": int(len(results) - successful_df.shape[0]),
        "generation_failure_rate": float(
            1 - successful_df.shape[0] / len(results)
        )
        if results
        else 0.0,
        "unique_prompts": int(df["prompt"].nunique()),
        "configs_tested": int(df["config_name"].nunique()),
    }
    if "hit_max_tokens" in df.columns:
        maxed = int(coerce_bool_series(df["hit_max_tokens"]).sum())
        summary["metadata"]["hit_max_tokens_count"] = maxed
        summary["metadata"]["hit_max_tokens_rate"] = (
            float(maxed / len(df)) if len(df) else 0.0
        )
    if "model" in df.columns:
        summary["metadata"]["models_tested"] = df["model"].unique().tolist()

    if "meets_a1_criteria" in df.columns:
        a1_flags = coerce_bool_series(df["meets_a1_criteria"])
        summary["metadata"]["a1_pass_experiments"] = int(a1_flags.sum())
        summary["metadata"]["a1_pass_rate"] = float(
            a1_flags.sum() / len(df)
        ) if len(df) else 0.0

    return summary


class RunStore:
    """Write and read run bundles under results/runs/{run_id}/."""

    def __init__(self, results_root: Union[Path, str]):
        self.results_root = Path(results_root)

    def run_dir(self, run_id: str) -> Path:
        return self.results_root / "runs" / run_id

    def write_bundle(
        self,
        run_id: str,
        results: List[ExperimentResult],
        *,
        phase: Union[int, str],
        experiment: str,
        cli_args: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        prompt_count: int = 0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Path:
        """Write manifest, specification.csv, full.csv, and summary.json."""
        out_dir = self.run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        started = started_at or datetime.now(timezone.utc)
        completed = completed_at or datetime.now(timezone.utc)

        successful = sum(1 for r in results if r.generation_successful)
        failed = len(results) - successful
        maxed = sum(1 for r in results if r.hit_max_tokens)

        manifest = {
            "kind": KIND_GENERATION,
            "run_id": run_id,
            "phase": phase,
            "experiment": experiment,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "cli_args": cli_args or [],
            "models": models or sorted({r.model for r in results}),
            "prompt_count": prompt_count,
            "observations": {
                "total": len(results),
                "successful": successful,
                "failed": failed,
                "hit_max_tokens": maxed,
            },
            "artifacts": {
                "specification_csv": "specification.csv",
                "full_csv": "full.csv",
                "summary_json": "summary.json",
            },
        }

        spec_path = out_dir / "specification.csv"
        full_path = out_dir / "full.csv"
        summary_path = out_dir / "summary.json"
        manifest_path = out_dir / "manifest.json"

        self._write_specification_csv(results, spec_path)
        self._write_full_csv(results, full_path)

        summary = compute_summary_stats(results, experiment=experiment)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return out_dir

    def _write_specification_csv(self, results: List[ExperimentResult], path: Path) -> None:
        if not results:
            pd.DataFrame(columns=SPEC_COLUMNS).to_csv(path, index=False, decimal=",")
            return

        df = pd.DataFrame([r.to_dict() for r in results])
        df["time_spent"] = df["response_time_seconds"].round(1)
        df["answer"] = df["response"]

        available = [c for c in SPEC_COLUMNS if c in df.columns]
        df[available].to_csv(path, index=False, decimal=",")

    def _write_full_csv(self, results: List[ExperimentResult], path: Path) -> None:
        if not results:
            pd.DataFrame().to_csv(path, index=False)
            return

        pd.DataFrame([r.to_dict() for r in results]).to_csv(path, index=False)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return manifests for all run bundles, newest first.

        Bundles without an explicit ``kind`` are treated as generation runs
        (legacy manifests written before the assessment discriminator).
        """
        runs_dir = self.results_root / "runs"
        if not runs_dir.exists():
            return []

        manifests: List[Dict[str, Any]] = []
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir():
                continue
            manifest_path = run_path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.setdefault("kind", KIND_GENERATION)
                manifests.append(manifest)

        manifests.sort(key=lambda m: m.get("started_at", ""), reverse=True)
        return manifests

    def bundle_kind(self, run_id: str) -> str:
        """Return the bundle kind (``generation`` or ``assessment``)."""
        return self.read_manifest(run_id).get("kind", KIND_GENERATION)

    def is_assessment(self, run_id: str) -> bool:
        return self.bundle_kind(run_id) == KIND_ASSESSMENT

    def read_manifest(self, run_id: str) -> Dict[str, Any]:
        manifest_path = self.run_dir(run_id) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("kind", KIND_GENERATION)
        return manifest

    def read_summary(self, run_id: str) -> Dict[str, Any]:
        summary_path = self.run_dir(run_id) / "summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def read_items_csv(self, run_id: str) -> pd.DataFrame:
        """Load items.csv from an assessment bundle."""
        path = self.run_dir(run_id) / "items.csv"
        if not path.exists():
            raise FileNotFoundError(f"items.csv not found for run: {run_id}")
        return pd.read_csv(path)

    def read_item_map_csv(self, run_id: str) -> pd.DataFrame:
        """Load item_map.csv from an assessment bundle."""
        path = self.run_dir(run_id) / "item_map.csv"
        if not path.exists():
            raise FileNotFoundError(f"item_map.csv not found for run: {run_id}")
        return pd.read_csv(path)

    def read_full_csv(self, run_id: str) -> pd.DataFrame:
        """Load full.csv from a run bundle."""
        full_path = self.run_dir(run_id) / "full.csv"
        if not full_path.exists():
            raise FileNotFoundError(f"full.csv not found for run: {run_id}")
        try:
            return pd.read_csv(full_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"full.csv is empty (no header/columns) for run: {run_id}"
            ) from exc

    def write_full_csv(self, run_id: str, df: pd.DataFrame) -> Path:
        """Overwrite full.csv in a run bundle."""
        full_path = self.run_dir(run_id) / "full.csv"
        df.to_csv(full_path, index=False)
        return full_path
