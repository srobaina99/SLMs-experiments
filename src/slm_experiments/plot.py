"""Generate boxplots from run bundles."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from slm_experiments.core.run_store import RunStore
from slm_experiments.models.base import REPO_ROOT

sns.set_theme(style="whitegrid")

READABILITY_METRICS = [
    ("flesch_kincaid_grade", "Flesch-Kincaid Grade", 5.0),
    ("gunning_fog", "Gunning Fog", 6.0),
    ("spache_readability", "Spache Readability", 4.0),
]

INTERVENTION_ORDER = ["Control", "Weighting", "Prompting", "Both"]


def _config_label(row: pd.Series) -> str:
    weighting = bool(row.get("config_weighting"))
    prompting = bool(row.get("config_prompting"))
    if weighting and prompting:
        return "Both"
    if weighting:
        return "Weighting"
    if prompting:
        return "Prompting"
    return "Control"


def _load_run_dataframe(run_dir: Path) -> pd.DataFrame:
    full_path = run_dir / "full.csv"
    spec_path = run_dir / "specification.csv"

    if full_path.exists():
        return pd.read_csv(full_path)

    if not spec_path.exists():
        raise FileNotFoundError(f"No CSV artifacts found in {run_dir}")

    try:
        return pd.read_csv(spec_path, decimal=",")
    except Exception:
        return pd.read_csv(spec_path)


def _normalize_successful(df: pd.DataFrame) -> pd.DataFrame:
    if "generation_successful" not in df.columns:
        return df

    success = df["generation_successful"]
    if success.dtype == object:
        success = success.astype(str).str.lower().map({"true": True, "false": False})
    return df[success == True].copy()  # noqa: E712


def _group_column(experiment: str, df: pd.DataFrame) -> Tuple[str, List[str]]:
    if experiment == "factorial":
        df["group"] = df.apply(_config_label, axis=1)
        order = [g for g in INTERVENTION_ORDER if g in df["group"].unique()]
        return "group", order

    if experiment == "weights" and "weight_factor" in df.columns:
        df["group"] = df["weight_factor"].astype(float).map(lambda v: f"{v:g}")
        order = sorted(df["group"].unique(), key=float)
        return "group", order

    if experiment == "beam" and "beam_width" in df.columns:
        df["group"] = df["beam_width"].astype(int).astype(str)
        order = sorted(df["group"].unique(), key=int)
        return "group", order

    if experiment == "kvl_beam" and "kvl_beam_width" in df.columns:
        df["group"] = df["kvl_beam_width"].astype(int).astype(str)
        order = sorted(df["group"].unique(), key=int)
        return "group", order

    if experiment == "prompting" and "num_shots" in df.columns:
        df["group"] = df["num_shots"].astype(int).astype(str)
        order = sorted(df["group"].unique(), key=int)
        return "group", order

    df["group"] = df.apply(_config_label, axis=1)
    order = [g for g in INTERVENTION_ORDER if g in df["group"].unique()]
    return "group", order


def _plot_metric_subplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    target: Optional[float],
    group_col: str,
    group_order: List[str],
    ax,
) -> None:
    hue_col = "model" if "model" in df.columns and df["model"].nunique() > 1 else None

    boxplot_hue = hue_col if hue_col else group_col
    sns.boxplot(
        data=df,
        x=group_col,
        y=metric,
        hue=boxplot_hue,
        order=group_order,
        palette="Set2",
        legend=bool(hue_col),
        ax=ax,
    )
    if hue_col is not None:
        ax.legend(title="Model", fontsize=8, loc="upper right")

    stripplot_kwargs = dict(
        data=df,
        x=group_col,
        y=metric,
        order=group_order,
        alpha=0.35,
        size=3,
        ax=ax,
        legend=False,
    )
    if hue_col:
        stripplot_kwargs["hue"] = hue_col
        stripplot_kwargs["dodge"] = True
        stripplot_kwargs["palette"] = "dark:black"
    else:
        stripplot_kwargs["color"] = "black"

    sns.stripplot(**stripplot_kwargs)

    if target is not None:
        ax.axhline(
            y=target,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"A1 target (≤{target:g})",
        )
        ax.legend(fontsize=8, loc="upper left")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_run(
    run_id: str,
    results_root: Optional[Union[Path, str]] = None,
) -> Path:
    """
    Read manifest + CSV from a run bundle and write readability boxplots.

    Returns:
        Path to the plots directory (results/runs/{run_id}/plots/).
    """
    root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
    store = RunStore(root)
    run_dir = store.run_dir(run_id)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    manifest = store.read_manifest(run_id)
    experiment = manifest.get("experiment", "factorial")

    df = _load_run_dataframe(run_dir)
    df = _normalize_successful(df)

    if df.empty:
        raise ValueError(f"No successful observations to plot for run: {run_id}")

    group_col, group_order = _group_column(experiment, df)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    available = [(m, title, target) for m, title, target in READABILITY_METRICS if m in df.columns]
    if not available:
        raise ValueError(f"No readability metrics found in run data: {run_id}")

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    x_label = {
        "factorial": "Intervention",
        "weights": "Weight factor",
        "beam": "Beam width",
        "kvl_beam": "KVL beam width",
        "prompting": "Prompt shots",
    }.get(experiment, "Configuration")

    for ax, (metric, title, target) in zip(axes, available):
        _plot_metric_subplot(df, metric, title, target, group_col, group_order, ax)
        ax.set_xlabel(x_label, fontsize=9)

    phase = manifest.get("phase", "")
    fig.suptitle(
        f"{run_id}\nPhase {phase} — {experiment} (n={len(df)} successful)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = plots_dir / "boxplot_readability.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for metric, _, _ in available:
        fig_single, ax_single = plt.subplots(figsize=(8, 5))
        metric_title = next(t for m, t, _ in READABILITY_METRICS if m == metric)
        target = next(t for m, _, t in READABILITY_METRICS if m == metric)
        _plot_metric_subplot(
            df, metric, metric_title, target, group_col, group_order, ax_single
        )
        ax_single.set_xlabel(x_label, fontsize=10)
        single_path = plots_dir / f"boxplot_{metric}.png"
        fig_single.savefig(single_path, dpi=200, bbox_inches="tight")
        plt.close(fig_single)

    return plots_dir
