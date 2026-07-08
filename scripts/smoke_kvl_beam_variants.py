#!/usr/bin/env python3
"""
Smoke-test KVL beam stopping / length variants on a real GGUF model.

Each variant tweaks decode policy without changing production code.
Run one variant per invocation, or sweep many configs in one process
(model loaded once).

Usage (single variant, backward compatible):
  SLM_GGUF_DIR=/path/to/gguf PYTHONPATH=src python3 scripts/smoke_kvl_beam_variants.py \\
    --variant baseline --prompt-indices 0,1

Usage (alpha sweep for one variant):
  SLM_GGUF_DIR=/path/to/gguf PYTHONPATH=src python3 scripts/smoke_kvl_beam_variants.py \\
    --sweep stop_length --alphas 0.15,0.25,0.35 \\
    --prompt-indices 0,1 \\
    --output /tmp/kvl_sweep.json

Usage (multi-variant sweep):
  SLM_GGUF_DIR=/path/to/gguf PYTHONPATH=src python3 scripts/smoke_kvl_beam_variants.py \\
    --sweep-configs "stop_length:0.25,length_penalty:0.5,baseline,max80" \\
    --prompt-indices 0,1

Usage (preset: stop_length alpha grid + length_penalty at 0.5/1.0/1.5):
  SLM_GGUF_DIR=/path/to/gguf PYTHONPATH=src python3 scripts/smoke_kvl_beam_variants.py \\
    --sweep-all-length --prompt-indices 0,1

Results are saved by default under results/kvl_beam_smoke/{run_id}/:
  sweep.csv, sweep.json, manifest.json

Variants:
  baseline          Current production KvlBeamDecoder
  always_stop       Always branch stop token IDs (even when top_k masked them out)
  length_penalty    Rank by KVL mean minus alpha * log(generated_len)
  prefer_finished   Final pick prefers finished_pool if mean within epsilon of best active
  stop_pref         always_stop + prefer_finished
  stop_length       always_stop + length_penalty
  max80             baseline decoder but max_tokens=80 cap only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.base import REPO_ROOT
from slm_experiments.models.kvl_beam_decoder import (
    KvlBeamCandidate,
    KvlBeamDecodeResult,
    KvlBeamDecoder,
    make_llamacpp_eval_fn,
    resolve_llamacpp_stop_token_ids,
)

VARIANTS = (
    "baseline",
    "always_stop",
    "length_penalty",
    "prefer_finished",
    "stop_pref",
    "stop_length",
    "max80",
)

DEFAULT_ALPHA_GRID = (0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5)
LENGTH_PENALTY_PRESET_ALPHAS = (0.5, 1.0, 1.5)
DEFAULT_RESULTS_SUBDIR = "kvl_beam_smoke"

CSV_COLUMNS = (
    "variant",
    "length_penalty_alpha",
    "prompt_index",
    "prompt_preview",
    "steps_total",
    "finished_pool_size",
    "words_scored",
    "kvl_beam_running_mean",
    "response_time_seconds",
    "response_chars",
    "hit_max_tokens",
    "response_preview",
)


@dataclass
class VariantPolicy:
    always_branch_stop_tokens: bool = False
    length_penalty_alpha: float = 0.0
    prefer_finished_epsilon: float = 0.0
    max_tokens_override: int | None = None


@dataclass
class SweepConfig:
    variant: str
    length_penalty_alpha: float | None = None


@dataclass
class SmokeContext:
    wrapper: Any
    config: ExperimentConfig
    model_name: str
    beam_width: int
    branch_factor: int


def policy_for(variant: str) -> VariantPolicy:
    if variant == "baseline":
        return VariantPolicy()
    if variant == "always_stop":
        return VariantPolicy(always_branch_stop_tokens=True)
    if variant == "length_penalty":
        return VariantPolicy(length_penalty_alpha=0.15)
    if variant == "prefer_finished":
        return VariantPolicy(prefer_finished_epsilon=0.25)
    if variant == "stop_pref":
        return VariantPolicy(
            always_branch_stop_tokens=True,
            prefer_finished_epsilon=0.25,
        )
    if variant == "stop_length":
        return VariantPolicy(
            always_branch_stop_tokens=True,
            length_penalty_alpha=0.15,
        )
    if variant == "max80":
        return VariantPolicy(max_tokens_override=80)
    raise ValueError(f"Unknown variant: {variant}")


def resolve_policy(entry: SweepConfig) -> VariantPolicy:
    policy = policy_for(entry.variant)
    if entry.length_penalty_alpha is not None:
        policy.length_penalty_alpha = entry.length_penalty_alpha
    return policy


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_sweep_configs(raw: str) -> list[SweepConfig]:
    """Parse comma-separated `variant` or `variant:alpha` entries."""
    entries: list[SweepConfig] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            variant, alpha_raw = part.split(":", 1)
            variant = variant.strip()
            alpha = float(alpha_raw.strip())
            entries.append(SweepConfig(variant=variant, length_penalty_alpha=alpha))
        else:
            entries.append(SweepConfig(variant=part))
    return entries


def load_sweep_configs_from_file(path: Path) -> list[SweepConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Sweep config file must contain a JSON array: {path}")

    entries: list[SweepConfig] = []
    for item in data:
        if isinstance(item, str):
            entries.extend(parse_sweep_configs(item))
        elif isinstance(item, dict):
            variant = item.get("variant")
            if not variant:
                raise ValueError(f"Each sweep config object needs 'variant': {item}")
            alpha = item.get("length_penalty_alpha", item.get("alpha"))
            entries.append(
                SweepConfig(
                    variant=str(variant),
                    length_penalty_alpha=float(alpha) if alpha is not None else None,
                )
            )
        else:
            raise ValueError(f"Unsupported sweep config entry: {item!r}")
    return entries


def build_sweep_entries(args: argparse.Namespace) -> list[SweepConfig]:
    if args.sweep_all_length:
        entries = [
            SweepConfig(variant="stop_length", length_penalty_alpha=alpha)
            for alpha in DEFAULT_ALPHA_GRID
        ]
        entries.extend(
            SweepConfig(variant="length_penalty", length_penalty_alpha=alpha)
            for alpha in LENGTH_PENALTY_PRESET_ALPHAS
        )
        return entries

    if args.sweep_configs:
        path = Path(args.sweep_configs)
        if path.is_file():
            return load_sweep_configs_from_file(path)
        return parse_sweep_configs(args.sweep_configs)

    if args.sweep:
        alphas = (
            parse_float_list(args.alphas)
            if args.alphas
            else list(DEFAULT_ALPHA_GRID)
        )
        return [
            SweepConfig(variant=args.sweep, length_penalty_alpha=alpha)
            for alpha in alphas
        ]

    alpha = args.length_penalty_alpha
    return [SweepConfig(variant=args.variant, length_penalty_alpha=alpha)]


class VariantKvlBeamDecoder(KvlBeamDecoder):
    """KvlBeamDecoder with configurable stop/length/finished policy."""

    def __init__(
        self,
        *,
        policy: VariantPolicy,
        prompt_token_len: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy = policy
        self.prompt_token_len = prompt_token_len

    def _branch_token_ids(self, logits, branch_factor, stop_token_ids=None):
        expanded = super()._branch_token_ids(logits, branch_factor, stop_token_ids)
        if not self.policy.always_branch_stop_tokens or not stop_token_ids:
            return expanded
        seen = set(expanded)
        for token_id in stop_token_ids:
            if token_id not in seen and 0 <= token_id < len(logits):
                expanded.append(token_id)
                seen.add(token_id)
        return expanded

    def _logprob(self, logits, token_id):
        if (
            self.policy.always_branch_stop_tokens
            and token_id < len(logits)
            and logits[token_id] <= float("-inf")
        ):
            return -12.0
        return super()._logprob(logits, token_id)

    def _rank_key(self, candidate: KvlBeamCandidate) -> tuple[float, float]:
        mean = candidate.kvl_running_mean()
        kvl_key = mean if mean is not None else float("-inf")
        alpha = self.policy.length_penalty_alpha
        if alpha > 0 and kvl_key > float("-inf"):
            gen_len = max(1, len(candidate.token_ids) - self.prompt_token_len)
            kvl_key -= alpha * math.log((5 + gen_len) / 6.0)
        return (kvl_key, candidate.cumulative_logprob)

    def decode(self, eval_fn, prompt_token_ids, *, max_tokens, stop, **kwargs):
        if self.policy.max_tokens_override is not None:
            max_tokens = self.policy.max_tokens_override

        stop_token_ids = kwargs.get("stop_token_ids")
        decode_suffix = kwargs.get("decode_suffix")

        active_beams = [
            KvlBeamCandidate(
                token_ids=list(prompt_token_ids),
                text="",
                cumulative_logprob=0.0,
            )
        ]
        finished_pool: list[KvlBeamCandidate] = []
        candidates_pruned = 0
        steps_total = 0

        for _ in range(max_tokens):
            if not active_beams:
                break

            steps_total += 1
            children: list[KvlBeamCandidate] = []

            for beam in active_beams:
                logits, _ = eval_fn(beam.token_ids)
                for token_id in self._branch_token_ids(
                    logits, self.branch_factor, stop_token_ids
                ):
                    child = self._extend_candidate(
                        beam,
                        token_id,
                        self._logprob(logits, token_id),
                        eval_fn,
                        stop,
                        stop_token_ids=stop_token_ids,
                        decode_suffix=decode_suffix,
                    )
                    children.append(child)

            unfinished = [c for c in children if not c.finished]
            finished = [c for c in children if c.finished]
            finished_pool.extend(finished)

            unfinished.sort(key=self._rank_key, reverse=True)
            candidates_pruned += max(0, len(unfinished) - self.beam_width)
            active_beams = unfinished[: self.beam_width]

            if not active_beams and finished_pool:
                break

        for beam in active_beams:
            self._flush_candidate_words(beam)

        self._last_finished_pool_size = len(finished_pool)

        eps = self.policy.prefer_finished_epsilon
        if eps > 0 and finished_pool:
            best_finished = max(finished_pool, key=self._rank_key)
            best_active = max(active_beams, key=self._rank_key) if active_beams else None
            if best_active is None:
                best = best_finished
            else:
                fin_kvl = best_finished.kvl_running_mean()
                act_kvl = best_active.kvl_running_mean()
                fin_kvl = fin_kvl if fin_kvl is not None else float("-inf")
                act_kvl = act_kvl if act_kvl is not None else float("-inf")
                if fin_kvl >= act_kvl - eps:
                    best = best_finished
                else:
                    best = best_active
        else:
            all_candidates = finished_pool + active_beams
            if not all_candidates:
                best = KvlBeamCandidate(
                    token_ids=list(prompt_token_ids),
                    text="",
                    cumulative_logprob=0.0,
                )
            else:
                best = max(all_candidates, key=self._rank_key)

        generated_ids = best.token_ids[len(prompt_token_ids) :]
        result = KvlBeamDecodeResult(
            token_ids=generated_ids,
            text=best.text,
            cumulative_logprob=best.cumulative_logprob,
            steps_total=steps_total,
            words_scored=len(best.kvl_scores),
            running_mean=best.kvl_running_mean(),
            candidates_pruned=candidates_pruned,
        )
        return result, len(finished_pool)


def load_smoke_context(
    *,
    model_name: str = "Qwen3",
    beam_width: int = 4,
    branch_factor: int = 10,
    seed: int = 42,
) -> SmokeContext | dict[str, Any]:
    print(f"Loading model {model_name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    wrapper = get_model_wrapper(model_name, seed=seed)
    if not wrapper.model_loaded:
        return {"error": "model not loaded"}

    config = ExperimentConfig(
        model_name=wrapper.model_name,
        config_prompting=True,
        config_weighting=False,
        temperature=0.0,
        top_k=50,
        max_new_tokens=200,
        kvl_l1="es",
    )
    print(
        f"Model loaded in {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return SmokeContext(
        wrapper=wrapper,
        config=config,
        model_name=model_name,
        beam_width=beam_width,
        branch_factor=branch_factor,
    )


def cleanup_smoke_context(ctx: SmokeContext) -> None:
    if hasattr(ctx.wrapper, "cleanup"):
        ctx.wrapper.cleanup()


def run_config(
    ctx: SmokeContext,
    entry: SweepConfig,
    prompt_indices: list[int],
) -> dict[str, Any]:
    policy = resolve_policy(entry)
    wrapper = ctx.wrapper
    config = ctx.config
    observations = []

    for idx in prompt_indices:
        prompt = STANDARD_PROMPTS[idx]
        final_prompt = wrapper._add_simplification_context(prompt, num_shots=0)
        formatted = wrapper._format_prompt(final_prompt, config.system_prompt)
        prompt_ids = wrapper.llm.tokenize(formatted.encode("utf-8"), add_bos=True)
        stop = wrapper._get_stop_tokens()
        stop_ids = resolve_llamacpp_stop_token_ids(wrapper.llm, stop)
        eval_fn, decode_suffix = make_llamacpp_eval_fn(
            wrapper.llm,
            prompt_ids,
            top_k=config.top_k,
        )

        decoder = VariantKvlBeamDecoder(
            policy=policy,
            prompt_token_len=len(prompt_ids),
            kvl_lookup=wrapper.kvl_lookup,
            l1=config.kvl_l1,
            text_evaluator=wrapper.text_evaluator,
            beam_width=ctx.beam_width,
            branch_factor=ctx.branch_factor,
        )

        t0 = time.time()
        decode_result, finished_pool_size = decoder.decode(
            eval_fn,
            prompt_ids,
            max_tokens=config.max_new_tokens,
            stop=stop,
            stop_token_ids=stop_ids,
            decode_suffix=decode_suffix,
        )
        elapsed = time.time() - t0
        cleaned = wrapper._prepare_beam_scoring_text(decode_result.text)

        observations.append(
            {
                "prompt_index": idx,
                "prompt_preview": prompt[:60],
                "steps_total": decode_result.steps_total,
                "finished_pool_size": finished_pool_size,
                "words_scored": decode_result.words_scored,
                "kvl_beam_running_mean": decode_result.running_mean,
                "response_time_seconds": round(elapsed, 1),
                "response_chars": len(cleaned),
                "response_preview": cleaned[:200].replace("\n", " "),
                "hit_max_tokens": decode_result.steps_total
                >= (policy.max_tokens_override or config.max_new_tokens),
            }
        )

    return {
        "variant": entry.variant,
        "length_penalty_alpha": policy.length_penalty_alpha,
        "policy": policy.__dict__,
        "observations": observations,
    }


def run_sweep(
    entries: list[SweepConfig],
    prompt_indices: list[int],
    *,
    model_name: str = "Qwen3",
    beam_width: int = 4,
    branch_factor: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    loaded = load_smoke_context(
        model_name=model_name,
        beam_width=beam_width,
        branch_factor=branch_factor,
        seed=seed,
    )
    if isinstance(loaded, dict):
        return [{"error": loaded["error"], "observations": []}]

    results: list[dict[str, Any]] = []
    try:
        for i, entry in enumerate(entries, start=1):
            alpha = entry.length_penalty_alpha
            alpha_label = alpha if alpha is not None else resolve_policy(entry).length_penalty_alpha
            print(
                f"[{i}/{len(entries)}] {entry.variant} alpha={alpha_label}",
                file=sys.stderr,
                flush=True,
            )
            results.append(run_config(loaded, entry, prompt_indices))
    finally:
        cleanup_smoke_context(loaded)

    return results


def run_variant(
    variant: str,
    prompt_indices: list[int],
    model_name: str = "Qwen3",
    beam_width: int = 4,
    branch_factor: int = 10,
    seed: int = 42,
    length_penalty_alpha: float | None = None,
) -> dict[str, Any]:
    entry = SweepConfig(variant=variant, length_penalty_alpha=length_penalty_alpha)
    results = run_sweep(
        [entry],
        prompt_indices,
        model_name=model_name,
        beam_width=beam_width,
        branch_factor=branch_factor,
        seed=seed,
    )
    return results[0]


def _fmt_alpha(alpha: float | None) -> str:
    if alpha is None:
        return "-"
    return f"{alpha:g}"


def _fmt_kvl(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def make_smoke_run_id(label: str, started_at: datetime | None = None) -> str:
    ts = started_at or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_").replace("/", "-")
    return f"{stamp}_smoke_{safe_label}"


def default_results_root() -> Path:
    return Path(REPO_ROOT) / "results" / DEFAULT_RESULTS_SUBDIR


def sweep_label_from_args(args: argparse.Namespace, entries: list) -> str:
    if args.sweep_all_length:
        return "all_length"
    if args.sweep_configs:
        return "configs"
    if args.sweep:
        return f"{args.sweep}_sweep"
    if entries:
        entry = entries[0]
        if entry.length_penalty_alpha is not None:
            return f"{entry.variant}_a{entry.length_penalty_alpha:g}"
        return entry.variant
    return "kvl_beam_variants"


def flatten_results_to_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("error"):
            rows.append(
                {
                    "variant": result.get("variant", "?"),
                    "length_penalty_alpha": result.get("length_penalty_alpha"),
                    "prompt_index": None,
                    "prompt_preview": "",
                    "steps_total": None,
                    "finished_pool_size": None,
                    "words_scored": None,
                    "kvl_beam_running_mean": None,
                    "response_time_seconds": None,
                    "response_chars": None,
                    "hit_max_tokens": None,
                    "response_preview": result["error"],
                }
            )
            continue

        alpha = result.get("length_penalty_alpha")
        for obs in result.get("observations", []):
            rows.append(
                {
                    "variant": result["variant"],
                    "length_penalty_alpha": alpha,
                    "prompt_index": obs.get("prompt_index"),
                    "prompt_preview": obs.get("prompt_preview", ""),
                    "steps_total": obs.get("steps_total"),
                    "finished_pool_size": obs.get("finished_pool_size"),
                    "words_scored": obs.get("words_scored"),
                    "kvl_beam_running_mean": obs.get("kvl_beam_running_mean"),
                    "response_time_seconds": obs.get("response_time_seconds"),
                    "response_chars": obs.get("response_chars"),
                    "hit_max_tokens": obs.get("hit_max_tokens"),
                    "response_preview": obs.get("response_preview", ""),
                }
            )
    return rows


def write_results_bundle(
    out_dir: Path,
    *,
    results: list[dict[str, Any]] | dict[str, Any],
    manifest: dict[str, Any],
    is_sweep: bool,
) -> tuple[Path, Path]:
    """Write sweep.json + sweep.csv (+ manifest.json) under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_sweep:
        payload: list[dict[str, Any]] | dict[str, Any] = (
            results if isinstance(results, list) else [results]
        )
    else:
        payload = results if isinstance(results, dict) else results[0]

    json_path = out_dir / ("sweep.json" if is_sweep else "results.json")
    csv_path = out_dir / ("sweep.csv" if is_sweep else "results.csv")
    manifest_path = out_dir / "manifest.json"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result_list = payload if isinstance(payload, list) else [payload]
    rows = flatten_results_to_rows(result_list)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return json_path, csv_path


def print_summary_table(results: list[dict[str, Any]]) -> None:
    headers = (
        "variant",
        "alpha",
        "prompt",
        "steps",
        "chars",
        "words_scored",
        "kvl_mean",
        "hit_max",
        "time_s",
    )
    rows: list[list[str]] = []

    for result in results:
        if result.get("error"):
            rows.append(
                [
                    result.get("variant", "?"),
                    _fmt_alpha(result.get("length_penalty_alpha")),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "ERR",
                ]
            )
            continue

        variant = result["variant"]
        alpha = result.get("length_penalty_alpha")
        for obs in result.get("observations", []):
            rows.append(
                [
                    variant,
                    _fmt_alpha(alpha),
                    str(obs.get("prompt_index", "?")),
                    str(obs.get("steps_total", "?")),
                    str(obs.get("response_chars", "?")),
                    str(obs.get("words_scored", "?")),
                    _fmt_kvl(obs.get("kvl_beam_running_mean")),
                    "Y" if obs.get("hit_max_tokens") else "N",
                    str(obs.get("response_time_seconds", "?")),
                ]
            )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\nSummary")
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser(description="Smoke-test KVL beam variants")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--variant", choices=VARIANTS, help="Single variant to run")
    mode.add_argument(
        "--sweep",
        choices=VARIANTS,
        help="Sweep one variant across multiple length_penalty_alpha values",
    )
    parser.add_argument(
        "--sweep-configs",
        help=(
            "Comma-separated variant or variant:alpha entries, "
            "or path to a JSON file with sweep configs"
        ),
    )
    parser.add_argument(
        "--sweep-all-length",
        action="store_true",
        help=(
            "Preset: stop_length across default alpha grid plus "
            "length_penalty at 0.5, 1.0, 1.5"
        ),
    )
    parser.add_argument(
        "--alphas",
        help=(
            "Comma-separated length_penalty_alpha values for --sweep "
            f"(default: {','.join(str(a) for a in DEFAULT_ALPHA_GRID)})"
        ),
    )
    parser.add_argument(
        "--prompt-indices",
        default="0,1",
        help="Comma-separated STANDARD_PROMPTS indices (default: 0,1)",
    )
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument(
        "--length-penalty-alpha",
        type=float,
        default=None,
        help="Override length_penalty_alpha for length_penalty / stop_length variants",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Base directory for saved results "
            f"(default: results/{DEFAULT_RESULTS_SUBDIR})"
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Run folder name under results-root (default: timestamped auto ID)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional extra JSON copy path (in addition to results bundle)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing results/kvl_beam_smoke/{run_id}/ bundle",
    )
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)

    sweep_modes = (
        args.sweep,
        args.sweep_configs,
        args.sweep_all_length,
    )
    if not args.variant and not any(sweep_modes):
        parser.error(
            "Provide --variant, --sweep, --sweep-configs, or --sweep-all-length"
        )
    if sum(bool(x) for x in sweep_modes) > 1:
        parser.error("Use only one of --sweep, --sweep-configs, or --sweep-all-length")
    if args.variant and args.length_penalty_alpha is not None and any(sweep_modes):
        parser.error("--length-penalty-alpha is only valid with --variant")
    if args.alphas and not args.sweep:
        parser.error("--alphas requires --sweep")

    indices = parse_int_list(args.prompt_indices)
    entries = build_sweep_entries(args)
    for entry in entries:
        if entry.variant not in VARIANTS:
            parser.error(f"Unknown variant: {entry.variant}")

    is_sweep = len(entries) > 1 or any(sweep_modes)

    if is_sweep:
        results = run_sweep(entries, indices, model_name=args.model)
        out_json = json.dumps(results, indent=2)
        print(out_json)
        print_summary_table(results)
    else:
        result = run_variant(
            entries[0].variant,
            indices,
            model_name=args.model,
            length_penalty_alpha=entries[0].length_penalty_alpha,
        )
        results = result
        out_json = json.dumps(result, indent=2)
        print(out_json)

    completed_at = datetime.now(timezone.utc)
    run_id = args.run_id or make_smoke_run_id(
        sweep_label_from_args(args, entries), started_at
    )
    manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "model": args.model,
        "prompt_indices": indices,
        "config_count": len(entries),
        "configs": [
            {
                "variant": e.variant,
                "length_penalty_alpha": e.length_penalty_alpha,
            }
            for e in entries
        ],
        "sweep_mode": {
            "variant": args.variant,
            "sweep": args.sweep,
            "sweep_configs": args.sweep_configs,
            "sweep_all_length": args.sweep_all_length,
            "alphas": args.alphas,
        },
        "artifacts": {
            "json": "sweep.json" if is_sweep else "results.json",
            "csv": "sweep.csv" if is_sweep else "results.csv",
        },
    }

    if not args.no_save:
        results_root = args.results_root or default_results_root()
        out_dir = results_root / run_id
        json_path, csv_path = write_results_bundle(
            out_dir,
            results=results,
            manifest=manifest,
            is_sweep=is_sweep,
        )
        print(f"\nSaved results to {out_dir.resolve()}/", file=sys.stderr)
        print(f"  CSV:  {csv_path.name}", file=sys.stderr)
        print(f"  JSON: {json_path.name}", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json, encoding="utf-8")


if __name__ == "__main__":
    main()
