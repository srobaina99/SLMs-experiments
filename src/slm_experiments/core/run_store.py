"""Run bundle storage: manifest, specification CSV, full CSV, summary JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from slm_experiments.core.result import ExperimentResult

SPEC_COLUMNS = [
    "model",
    "config_weighting",
    "config_prompting",
    "prompt_id",
    "answer",
    "time_spent",
    "generation_successful",
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
]


def make_run_id(
    phase: Union[int, str], experiment: str, started_at: Optional[datetime] = None
) -> str:
    """Build run ID: {YYYYMMDD_HHMMSS}_{phase}_{experiment}."""
    ts = started_at or datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    phase_label = f"phase{phase}" if isinstance(phase, int) else str(phase)
    return f"{stamp}_{phase_label}_{experiment}"


def _config_label(row: pd.Series) -> str:
    if row["config_weighting"] and row["config_prompting"]:
        return "both"
    if row["config_weighting"]:
        return "weighting_only"
    if row["config_prompting"]:
        return "prompting_only"
    return "control"


def _metric_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def compute_summary_stats(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Aggregate stats; metric means exclude failed generations."""
    if not results:
        return {}

    df = pd.DataFrame([r.to_dict() for r in results])
    successful_df = df[df["generation_successful"] == True]  # noqa: E712

    summary: Dict[str, Any] = {"overall": {}, "by_config": {}, "metadata": {}}

    for col in NUMERIC_SUMMARY_COLUMNS:
        if col in successful_df.columns and not successful_df.empty:
            summary["overall"][col] = _metric_stats(successful_df[col])

    if "config_weighting" in df.columns and "config_prompting" in df.columns:
        df = df.copy()
        df["intervention_config"] = df.apply(_config_label, axis=1)

        for config in ("control", "weighting_only", "prompting_only", "both"):
            config_df = df[df["intervention_config"] == config]
            if config_df.empty:
                continue

            config_success = config_df[config_df["generation_successful"] == True]  # noqa: E712
            summary["by_config"][config] = {"count": int(len(config_df))}

            for col in NUMERIC_SUMMARY_COLUMNS:
                if col in config_success.columns and not config_success.empty:
                    summary["by_config"][config][col] = _metric_stats(config_success[col])

    summary["metadata"] = {
        "total_experiments": len(results),
        "successful_experiments": int(successful_df.shape[0]),
        "failed_experiments": int(len(results) - successful_df.shape[0]),
        "unique_prompts": int(df["prompt"].nunique()),
        "configs_tested": int(df["config_name"].nunique()),
    }
    if "model" in df.columns:
        summary["metadata"]["models_tested"] = df["model"].unique().tolist()

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

        manifest = {
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

        summary = compute_summary_stats(results)
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
        """Return manifests for all run bundles, newest first."""
        runs_dir = self.results_root / "runs"
        if not runs_dir.exists():
            return []

        manifests: List[Dict[str, Any]] = []
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir():
                continue
            manifest_path = run_path / "manifest.json"
            if manifest_path.exists():
                manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

        manifests.sort(key=lambda m: m.get("started_at", ""), reverse=True)
        return manifests

    def read_manifest(self, run_id: str) -> Dict[str, Any]:
        manifest_path = self.run_dir(run_id) / "manifest.json"
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def read_summary(self, run_id: str) -> Dict[str, Any]:
        summary_path = self.run_dir(run_id) / "summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def read_full_csv(self, run_id: str) -> pd.DataFrame:
        """Load full.csv from a run bundle."""
        full_path = self.run_dir(run_id) / "full.csv"
        if not full_path.exists():
            raise FileNotFoundError(f"full.csv not found for run: {run_id}")
        return pd.read_csv(full_path)

    def write_full_csv(self, run_id: str, df: pd.DataFrame) -> Path:
        """Overwrite full.csv in a run bundle."""
        full_path = self.run_dir(run_id) / "full.csv"
        df.to_csv(full_path, index=False)
        return full_path
