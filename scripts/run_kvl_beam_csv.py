#!/usr/bin/env python3
"""Run KVL beam (production first-finish decoder) on models × prompts; write CSV."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.prompts import MODEL_CONFIGS
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.base import REPO_ROOT, resolve_gguf_dir
from slm_experiments.phase1.configs import DEFAULT_SYSTEM_PROMPT
from slm_experiments.phase1.runner import parse_models, parse_prompts

CSV_COLUMNS = (
    "model",
    "prompt_index",
    "prompt",
    "response",
    "response_time_seconds",
    "steps_total",
    "words_scored",
    "kvl_beam_running_mean",
    "generation_successful",
    "hit_max_tokens",
    "beam_width",
    "branch_factor",
    "kvl_l1",
)


def build_config(model_name: str, *, beam_width: int, branch_factor: int, kvl_l1: str) -> ExperimentConfig:
    info = MODEL_CONFIGS[model_name]
    return ExperimentConfig(
        model_name=info["model_name"],
        model_id=info["model_id"],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        config_weighting=False,
        config_prompting=True,
        config_kvl_beam=True,
        kvl_beam_width=beam_width,
        kvl_branch_factor=branch_factor,
        kvl_l1=kvl_l1,
        weight_factor=1.0,
        num_shots=0,
        temperature=0.0,
        top_k=50,
        max_new_tokens=200,
        experiment_name=f"{model_name}_kvl_beam_w{beam_width}",
        description=f"KVL beam CSV run: {model_name}",
    )


def result_row(
    model_name: str,
    prompt_index: int,
    prompt: str,
    result,
    *,
    beam_width: int,
    branch_factor: int,
    kvl_l1: str,
    max_new_tokens: int,
) -> dict[str, object]:
    steps = result.kvl_beam_steps_total or 0
    return {
        "model": model_name,
        "prompt_index": prompt_index,
        "prompt": prompt,
        "response": result.response,
        "response_time_seconds": round(result.response_time_seconds, 1),
        "steps_total": steps,
        "words_scored": result.kvl_beam_words_scored or 0,
        "kvl_beam_running_mean": result.kvl_beam_running_mean,
        "generation_successful": result.generation_successful,
        "hit_max_tokens": steps >= max_new_tokens,
        "beam_width": beam_width,
        "branch_factor": branch_factor,
        "kvl_l1": kvl_l1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="3", help="Prompt count or 'all' (default: 3)")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--branch-factor", type=int, default=10)
    parser.add_argument("--kvl-l1", default="es")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path (default: results/kvl_beam_runs/<timestamp>.csv)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompt_list = parse_prompts(args.prompts)
    model_list = parse_models(args.models)
    pipeline = ExperimentPipeline()

    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = REPO_ROOT / "results" / "kvl_beam_runs" / f"kvl_beam_{stamp}.csv"
    else:
        out_path = args.output

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"GGUF dir: {resolve_gguf_dir()}")
    print(f"Models: {', '.join(model_list)}")
    print(f"Prompts: {len(prompt_list)}")
    print(f"Output: {out_path.resolve()}")

    rows: list[dict[str, object]] = []

    for model_name in model_list:
        print(f"\n=== {model_name} ===")
        config = build_config(
            model_name,
            beam_width=args.beam_width,
            branch_factor=args.branch_factor,
            kvl_l1=args.kvl_l1,
        )
        wrapper = get_model_wrapper(model_name, seed=args.seed, timeout_seconds=7200)

        for prompt_index, prompt in enumerate(prompt_list):
            prompt_id = f"P{prompt_index + 1}"
            run_config = replace(config, prompt_id=prompt_id)
            print(f"  [{prompt_id}] {prompt[:60]}...", flush=True)

            result = pipeline.run_kvl_beam(
                prompt=prompt,
                config=run_config,
                model=wrapper,
                beam_width=args.beam_width,
                branch_factor=args.branch_factor,
                experiment_name=run_config.experiment_name,
            )
            row = result_row(
                model_name,
                prompt_index,
                prompt,
                result,
                beam_width=args.beam_width,
                branch_factor=args.branch_factor,
                kvl_l1=args.kvl_l1,
                max_new_tokens=config.max_new_tokens,
            )
            rows.append(row)
            preview = (result.response or "").replace("\n", " ")[:80]
            print(
                f"       ok={result.generation_successful} "
                f"steps={row['steps_total']} "
                f"kvl={row['kvl_beam_running_mean']} "
                f"time={row['response_time_seconds']}s "
                f"preview={preview!r}",
                flush=True,
            )

        if hasattr(wrapper, "cleanup"):
            wrapper.cleanup()
        del wrapper
        gc.collect()

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
