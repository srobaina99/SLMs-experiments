"""Single CLI entry point for all experiment commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EPILOG = """
Quick start
  slm_experiments phase1
  slm_experiments phase1 --prompts all
  slm_experiments runs list
  slm_experiments runs show <run_id>
  slm_experiments plot --run-id <run_id>

Phase 2 sweeps
  slm_experiments phase2 weights
  slm_experiments phase2 prompting --shots 0,1,3
  slm_experiments phase2 guided --top-k-pools 0,5,10,20
  slm_experiments phase2 kvl_beam --widths 1,4,6,8
  # phase2 beam is deprecated (hard-fails at temperature=0)

Human review
  slm_experiments human export --run-id <run_id>
  slm_experiments human import --run-id <run_id> --tags human_review.csv

Results are written to results/runs/<run_id>/.
Run `slm_experiments <command> --help` for command-specific examples.
"""


def _examples(*lines: str) -> str:
    return "Examples:\n" + "\n".join(f"  {line}" for line in lines)


def _add_run_options(
    parser: argparse.ArgumentParser,
    *,
    prompts_default: str = "3",
    models_default: str = "all",
) -> None:
    parser.add_argument(
        "--prompts",
        default=prompts_default,
        metavar="N|all",
        help="Number of A1 prompts to use, or 'all' for all 25 (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        default=models_default,
        metavar="NAMES",
        help="Comma-separated model names or 'all' for Qwen2,Qwen3,TinyLlama,Phi3 "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip automatic boxplot generation after the run",
    )
    parser.add_argument(
        "--enable-cefr-sp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "CEFR-SP sentence difficulty scoring (default: on). "
            "Use --no-enable-cefr-sp to disable. "
            "Requires pip install -e '.[cefr-sp]' and downloaded ckpt."
        ),
    )
    parser.add_argument(
        "--cefr-sp-ckpt",
        default="",
        metavar="PATH",
        help="Path to level_estimator.ckpt (default: data/cefr_sp/level_estimator.ckpt)",
    )
    parser.add_argument(
        "--cefr-sp-device",
        default="cpu",
        metavar="DEVICE",
        help="Torch device for CEFR-SP scoring (default: cpu)",
    )


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    """Accept legacy nested commands (phase1 run, phase2 run weights, ...)."""
    if argv is None:
        return None

    normalized = list(argv)
    if len(normalized) >= 2 and normalized[0] == "phase1" and normalized[1] == "run":
        normalized.pop(1)
    if len(normalized) >= 3 and normalized[0] == "phase2" and normalized[1] == "run":
        normalized.pop(1)
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slm_experiments",
        description=(
            "Evaluate inference-time interventions on small language models "
            "for CEFR A1 English generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser(
        "phase1",
        help="Phase 1 factorial experiment (4 models × 4 intervention configs)",
        description=(
            "Run the Phase 1 2×2 factorial design: control, weighting only, "
            "prompting only, and both interventions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase1",
            "slm_experiments phase1 --prompts all",
            "slm_experiments phase1 --models Qwen3 --prompts 1 --no-plot",
            "slm_experiments phase1 --models Qwen2,Qwen3 --seed 7",
        ),
    )
    _add_run_options(p1)

    p2 = subparsers.add_parser(
        "phase2",
        help="Phase 2 hyperparameter sweeps (weights, beam, guided, kvl_beam, prompting)",
        description="Run one Phase 2 sweep. Pick a sweep type as the next argument.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase2 weights",
            "slm_experiments phase2 prompting --shots 0,1,3 --prompts all",
            "slm_experiments phase2 guided --top-k-pools 0,5,10,20",
            "slm_experiments phase2 kvl_beam --widths 1,4,6,8",
        ),
    )
    p2_sub = p2.add_subparsers(dest="sweep", required=True)

    p2_weights = p2_sub.add_parser(
        "weights",
        help="Sweep logit-bias weight factors (prompting ON)",
        description="Sweep weight_factor values while contextual prompting stays enabled.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase2 weights",
            "slm_experiments phase2 weights --weights 1.0,1.5,2.0,4.0",
            "slm_experiments phase2 weights --models Qwen3 --prompts 3 --no-plot",
        ),
    )
    p2_weights.add_argument(
        "--weights",
        default="1.0,1.3,1.5,2.0,2.5,3.0,4.0",
        metavar="VALUES",
        help="Comma-separated weight_factor values (default: %(default)s)",
    )
    _add_run_options(p2_weights)

    p2_beam = p2_sub.add_parser(
        "beam",
        help="[DEPRECATED] Sweep beam-search widths — use kvl_beam or guided",
        description=(
            "Deprecated: best-of-N beam sweep is superseded by KVL beam and guided "
            "decoding at temperature=0. Prefer ``phase2 kvl_beam`` or ``phase2 guided``."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase2 beam",
            "slm_experiments phase2 beam --widths 4,8,10",
            "slm_experiments phase2 beam --models TinyLlama --prompts 1",
        ),
    )
    p2_beam.add_argument(
        "--widths",
        default="4,8,10",
        metavar="VALUES",
        help="Comma-separated beam widths (default: %(default)s)",
    )
    _add_run_options(p2_beam)

    p2_guided = p2_sub.add_parser(
        "guided",
        help="Sweep guided decoding top-k pool sizes (A1-constrained greedy)",
        description=(
            "Sweep guided_top_k pool size while using top-k A1-constrained greedy decoding."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase2 guided",
            "slm_experiments phase2 guided --top-k-pools 0,5,10,20",
            "slm_experiments phase2 kvl_beam --widths 1,4,6,8",
            "slm_experiments phase2 guided --mode trie --models Qwen3 --prompts all",
        ),
    )
    p2_guided.add_argument(
        "--top-k-pools",
        default="0,5,10,20",
        metavar="VALUES",
        help=(
            "Comma-separated guided top-k pool sizes; 0 = unconstrained in-run "
            "baseline (default: %(default)s)"
        ),
    )
    p2_guided.add_argument(
        "--mode",
        default="flat",
        choices=["flat", "trie"],
        help="Guided decoding mode (default: %(default)s)",
    )
    _add_run_options(p2_guided)

    p2_kvl_beam = p2_sub.add_parser(
        "kvl_beam",
        help="Sweep KVL-scored token-level beam widths",
        description=(
            "Sweep KVL beam width while using running mean GLMM scores "
            "for candidate selection at word boundaries."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase2 kvl_beam",
            "slm_experiments phase2 kvl_beam --widths 1,4,6,8",
            "slm_experiments phase2 kvl_beam --models Qwen3 --prompts 1 --widths 4",
        ),
    )
    p2_kvl_beam.add_argument(
        "--widths",
        default="1,4,6,8",
        metavar="VALUES",
        help=(
            "Comma-separated KVL beam widths; 1 = greedy in-run baseline "
            "(default: %(default)s)"
        ),
    )
    p2_kvl_beam.add_argument(
        "--branch-factor",
        type=int,
        default=10,
        metavar="K",
        help="Top-K logits expanded per beam per step (default: %(default)s)",
    )
    p2_kvl_beam.add_argument(
        "--kvl-l1",
        default="es",
        choices=["es", "de", "cn"],
        help="Learner L1 for KVL lookup (default: %(default)s)",
    )
    _add_run_options(p2_kvl_beam)

    p2_prompting = p2_sub.add_parser(
        "prompting",
        help="Sweep in-context example counts (0, 1, 3 shots)",
        description="Sweep the number of contextual prompting examples shown to the model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments phase2 prompting",
            "slm_experiments phase2 prompting --shots 0,1,3",
            "slm_experiments phase2 prompting --models all --prompts all",
        ),
    )
    p2_prompting.add_argument(
        "--shots",
        default="0,1,3",
        metavar="VALUES",
        help="Comma-separated shot counts (default: %(default)s)",
    )
    _add_run_options(p2_prompting)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate boxplots for an existing run",
        description="Create readability boxplots from a saved run bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments plot --run-id 20260606_120000_phase1_factorial",
            "slm_experiments runs list   # find a run id first",
        ),
    )
    plot_parser.add_argument(
        "--run-id",
        required=True,
        metavar="ID",
        help="Run bundle id under results/runs/",
    )

    runs_parser = subparsers.add_parser(
        "runs",
        help="List or inspect saved run bundles",
        description="Browse experiment outputs under results/runs/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments runs list",
            "slm_experiments runs show 20260606_120000_phase1_factorial",
        ),
    )
    runs_sub = runs_parser.add_subparsers(dest="runs_cmd", required=True)
    runs_sub.add_parser(
        "list",
        help="List all runs with phase, experiment, and observation counts",
    )
    runs_show = runs_sub.add_parser(
        "show",
        help="Show manifest and summary stats for one run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments runs show 20260606_120000_phase1_factorial",
        ),
    )
    runs_show.add_argument(
        "run_id",
        metavar="ID",
        help="Run bundle id under results/runs/",
    )

    human_parser = subparsers.add_parser(
        "human",
        help="Export/import human evaluation tags",
        description="Round-trip human review labels into a run bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments human export --run-id 20260606_120000_phase1_factorial",
            "slm_experiments human export --run-id <id> --sample 60",
            "slm_experiments human import --run-id <id> --tags results/runs/<id>/human_review.csv",
        ),
    )
    human_sub = human_parser.add_subparsers(dest="human_cmd", required=True)
    human_export = human_sub.add_parser(
        "export",
        help="Export a random sample for human review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments human export --run-id 20260606_120000_phase1_factorial",
            "slm_experiments human export --run-id <id> --sample 60",
        ),
    )
    human_export.add_argument(
        "--run-id",
        required=True,
        metavar="ID",
        help="Run bundle id to sample from",
    )
    human_export.add_argument(
        "--sample",
        type=int,
        default=60,
        metavar="N",
        help="Number of rows to export (default: %(default)s)",
    )
    human_import = human_sub.add_parser(
        "import",
        help="Merge completed human_review.csv tags back into full.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples(
            "slm_experiments human import --run-id <id> --tags results/runs/<id>/human_review.csv",
        ),
    )
    human_import.add_argument(
        "--run-id",
        required=True,
        metavar="ID",
        help="Run bundle id to update",
    )
    human_import.add_argument(
        "--tags",
        required=True,
        metavar="PATH",
        help="CSV with experiment_id and tag columns",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    normalized_argv = _normalize_argv(argv)
    parser = _build_parser()
    args = parser.parse_args(normalized_argv)

    if args.command == "phase1":
        from slm_experiments.phase1.runner import FactorialRunner

        cli_args = list(normalized_argv) if normalized_argv is not None else []
        runner = FactorialRunner()
        run_id, out_dir = runner.run(
            prompts=args.prompts,
            models=args.models,
            seed=args.seed,
            no_plot=args.no_plot,
            cli_args=cli_args,
            enable_cefr_sp=args.enable_cefr_sp,
            cefr_sp_ckpt_path=args.cefr_sp_ckpt,
            cefr_sp_device=args.cefr_sp_device,
        )
        print(f"Run complete: {run_id}")
        print(f"Output: {Path(out_dir).resolve()}")
        return

    if args.command == "phase2":
        cli_args = list(normalized_argv) if normalized_argv is not None else []

        if args.sweep == "weights":
            from slm_experiments.phase2.weights import WeightSweepRunner

            runner = WeightSweepRunner()
            run_id, out_dir = runner.run(
                weights=args.weights,
                prompts=args.prompts,
                models=args.models,
                seed=args.seed,
                no_plot=args.no_plot,
                cli_args=cli_args,
                enable_cefr_sp=args.enable_cefr_sp,
                cefr_sp_ckpt_path=args.cefr_sp_ckpt,
                cefr_sp_device=args.cefr_sp_device,
            )
            print(f"Run complete: {run_id}")
            print(f"Output: {Path(out_dir).resolve()}")
            return

        if args.sweep == "prompting":
            from slm_experiments.phase2.prompting import PromptingSweepRunner

            runner = PromptingSweepRunner()
            run_id, out_dir = runner.run(
                shots=args.shots,
                prompts=args.prompts,
                models=args.models,
                seed=args.seed,
                no_plot=args.no_plot,
                cli_args=cli_args,
                enable_cefr_sp=args.enable_cefr_sp,
                cefr_sp_ckpt_path=args.cefr_sp_ckpt,
                cefr_sp_device=args.cefr_sp_device,
            )
            print(f"Run complete: {run_id}")
            print(f"Output: {Path(out_dir).resolve()}")
            return

        if args.sweep == "beam":
            print(
                "ERROR: phase2 beam is deprecated and excluded from thesis claims. "
                "At temperature=0 all best-of-N candidates are identical. "
                "Use phase2 kvl_beam or phase2 guided instead.",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.sweep == "guided":
            from slm_experiments.phase2.guided import GuidedSweepRunner

            runner = GuidedSweepRunner()
            run_id, out_dir = runner.run(
                top_k_pools=args.top_k_pools,
                prompts=args.prompts,
                models=args.models,
                seed=args.seed,
                no_plot=args.no_plot,
                cli_args=cli_args,
                mode=args.mode,
                enable_cefr_sp=args.enable_cefr_sp,
                cefr_sp_ckpt_path=args.cefr_sp_ckpt,
                cefr_sp_device=args.cefr_sp_device,
            )
            print(f"Run complete: {run_id}")
            print(f"Output: {Path(out_dir).resolve()}")
            return

        if args.sweep == "kvl_beam":
            from slm_experiments.phase2.kvl_beam import KvlBeamSweepRunner

            runner = KvlBeamSweepRunner()
            run_id, out_dir = runner.run(
                widths=args.widths,
                branch_factor=args.branch_factor,
                kvl_l1=args.kvl_l1,
                prompts=args.prompts,
                models=args.models,
                seed=args.seed,
                no_plot=args.no_plot,
                cli_args=cli_args,
                enable_cefr_sp=args.enable_cefr_sp,
                cefr_sp_ckpt_path=args.cefr_sp_ckpt,
                cefr_sp_device=args.cefr_sp_device,
            )
            print(f"Run complete: {run_id}")
            print(f"Output: {Path(out_dir).resolve()}")
            return

        print(f"Not implemented: phase2 {args.sweep}", file=sys.stderr)
        sys.exit(1)

    if args.command == "human" and args.human_cmd == "export":
        from slm_experiments.human.export import HumanExporter

        exporter = HumanExporter()
        out_path, row_count = exporter.export(
            run_id=args.run_id,
            sample=args.sample,
        )
        print(f"Exported {row_count} rows for human review")
        print(f"Output: {out_path.resolve()}")
        return

    if args.command == "human" and args.human_cmd == "import":
        from importlib import import_module

        HumanImporter = import_module("slm_experiments.human.import").HumanImporter

        importer = HumanImporter()
        updated = importer.import_tags(run_id=args.run_id, tags_path=args.tags)
        print(f"Imported tags for run: {args.run_id}")
        print(f"Updated {updated} rows in full.csv")
        return

    if args.command == "plot":
        from slm_experiments.plot import plot_run

        plots_dir = plot_run(args.run_id)
        print(f"Plots written to: {plots_dir.resolve()}")
        return

    if args.command == "runs" and args.runs_cmd == "list":
        from slm_experiments.core.run_store import RunStore
        from slm_experiments.models.base import REPO_ROOT

        store = RunStore(Path(REPO_ROOT) / "results")
        manifests = store.list_runs()
        if not manifests:
            print("No runs found.")
            return

        print(f"{'RUN ID':<40} {'PHASE':<6} {'EXPERIMENT':<12} {'STARTED':<26} {'OBS':<12}")
        print("-" * 100)
        for manifest in manifests:
            obs = manifest.get("observations", {})
            obs_label = f"{obs.get('successful', 0)}/{obs.get('total', 0)}"
            print(
                f"{manifest.get('run_id', ''):<40} "
                f"{str(manifest.get('phase', '')):<6} "
                f"{manifest.get('experiment', ''):<12} "
                f"{manifest.get('started_at', ''):<26} "
                f"{obs_label:<12}"
            )
        return

    if args.command == "runs" and args.runs_cmd == "show":
        from slm_experiments.core.run_store import RunStore
        from slm_experiments.models.base import REPO_ROOT

        store = RunStore(Path(REPO_ROOT) / "results")
        try:
            manifest = store.read_manifest(args.run_id)
            summary = store.read_summary(args.run_id)
        except FileNotFoundError:
            print(f"Run not found: {args.run_id}", file=sys.stderr)
            sys.exit(1)

        print(f"Run: {manifest.get('run_id')}")
        print(f"Phase: {manifest.get('phase')}  Experiment: {manifest.get('experiment')}")
        print(f"Started:  {manifest.get('started_at')}")
        print(f"Completed: {manifest.get('completed_at')}")
        print(f"Models: {', '.join(manifest.get('models', []))}")
        print(f"Prompts: {manifest.get('prompt_count')}")
        obs = manifest.get("observations", {})
        print(
            f"Observations: {obs.get('total', 0)} total, "
            f"{obs.get('successful', 0)} successful, "
            f"{obs.get('failed', 0)} failed"
        )
        if manifest.get("cli_args"):
            print(f"CLI args: {' '.join(manifest['cli_args'])}")

        print("\nSummary (successful generations only):")
        overall = summary.get("overall", {})
        for metric in ("flesch_kincaid_grade", "gunning_fog", "spache_readability", "word_count"):
            if metric in overall:
                stats = overall[metric]
                print(
                    f"  {metric}: mean={stats['mean']:.2f}, "
                    f"std={stats['std']:.2f}, "
                    f"min={stats['min']:.2f}, max={stats['max']:.2f}"
                )

        by_config = summary.get("by_config", {})
        if by_config:
            print("\nBy configuration:")
            for config_name, config_stats in by_config.items():
                count = config_stats.get("count", 0)
                fk = config_stats.get("flesch_kincaid_grade", {}).get("mean")
                if fk is not None:
                    print(f"  {config_name}: n={count}, FK mean={fk:.2f}")
                else:
                    print(f"  {config_name}: n={count}")

        sweep_dimension = summary.get("metadata", {}).get("sweep_dimension")
        sweep_sections = {
            "weight_factor": "by_weight_factor",
            "beam_width": "by_beam_width",
            "kvl_beam_width": "by_kvl_beam_width",
            "num_shots": "by_num_shots",
            "guided_top_k": "by_guided_top_k",
        }
        sweep_section = sweep_sections.get(sweep_dimension or "")
        sweep_stats = summary.get(sweep_section or "", {})
        if sweep_stats:
            print(f"\nBy {sweep_dimension} (pooled):")
            for group_name, group_stats in sweep_stats.items():
                count = group_stats.get("count", 0)
                fk = group_stats.get("flesch_kincaid_grade", {}).get("mean")
                a1 = group_stats.get("a1_pass_rate")
                extras = []
                if fk is not None:
                    extras.append(f"FK mean={fk:.2f}")
                if a1 is not None:
                    extras.append(f"a1_pass_rate={a1:.2f}")
                suffix = f", {', '.join(extras)}" if extras else ""
                print(f"  {group_name}: n={count}{suffix}")

        by_model = summary.get("by_model", {})
        if by_model:
            print("\nBy model:")
            for model_name, model_stats in by_model.items():
                count = model_stats.get("count", 0)
                a1 = model_stats.get("a1_pass_rate")
                fail = model_stats.get("generation_failure_rate")
                parts = [f"n={count}"]
                if a1 is not None:
                    parts.append(f"a1_pass_rate={a1:.2f}")
                if fail is not None:
                    parts.append(f"failure_rate={fail:.2f}")
                print(f"  {model_name}: {', '.join(parts)}")
                nested_sweep = model_stats.get(sweep_section or "", {})
                if nested_sweep:
                    for group_name, group_stats in nested_sweep.items():
                        g_count = group_stats.get("count", 0)
                        g_a1 = group_stats.get("a1_pass_rate")
                        g_fk = group_stats.get("flesch_kincaid_grade", {}).get("mean")
                        g_parts = [f"n={g_count}"]
                        if g_a1 is not None:
                            g_parts.append(f"a1_pass_rate={g_a1:.2f}")
                        if g_fk is not None:
                            g_parts.append(f"FK mean={g_fk:.2f}")
                        print(f"    {group_name}: {', '.join(g_parts)}")
                nested_config = model_stats.get("by_config", {})
                if nested_config and not nested_sweep:
                    for config_name, config_stats in nested_config.items():
                        c_count = config_stats.get("count", 0)
                        c_a1 = config_stats.get("a1_pass_rate")
                        c_fk = config_stats.get("flesch_kincaid_grade", {}).get("mean")
                        c_parts = [f"n={c_count}"]
                        if c_a1 is not None:
                            c_parts.append(f"a1_pass_rate={c_a1:.2f}")
                        if c_fk is not None:
                            c_parts.append(f"FK mean={c_fk:.2f}")
                        print(f"    {config_name}: {', '.join(c_parts)}")
        return

    print(f"Not implemented: {args.command}", file=sys.stderr)
    sys.exit(1)
