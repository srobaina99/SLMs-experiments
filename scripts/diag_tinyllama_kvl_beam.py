#!/usr/bin/env python3
"""Diagnose TinyLlama early stopping in KVL beam (first-finish semantics)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.kvl_beam_decoder import (
    KvlBeamDecoder,
    apply_top_k_mask,
    get_last_logits,
    make_llamacpp_eval_fn,
    resolve_llamacpp_stop_token_ids,
)
from slm_experiments.phase1.configs import DEFAULT_SYSTEM_PROMPT


PROMPT = STANDARD_PROMPTS[0]
TOP_K = 50
BEAM_WIDTH = 4
BRANCH_FACTOR = 10


@dataclass
class StepTrace:
    step: int
    beam_idx: int
    token_id: int
    token_text: str
    logprob: float
    suffix: str
    hit_token_stop: bool
    hit_string_stop: bool
    finished: bool


def describe_stop_ids(llm, stop_strings: list[str]) -> dict:
    stop_ids = resolve_llamacpp_stop_token_ids(llm, stop_strings)
    details = {}
    for tid in sorted(stop_ids):
        try:
            piece = llm.detokenize([tid]).decode("utf-8", errors="replace")
        except Exception:
            piece = "<decode error>"
        details[tid] = repr(piece)
    return {"stop_token_ids": sorted(stop_ids), "detokenized": details}


def logits_report(llm, prompt_ids: list[int], top_k: int) -> dict:
    llm.reset()
    llm.eval(prompt_ids)
    raw_logits = get_last_logits(llm)
    masked = apply_top_k_mask(raw_logits, top_k=top_k)

    stop_ids = resolve_llamacpp_stop_token_ids(llm)
    eos = llm.token_eos() if hasattr(llm, "token_eos") else None
    eot = llm.token_eot() if hasattr(llm, "token_eot") else None

    ranked = sorted(
        (
            idx
            for idx, v in enumerate(masked)
            if math.isfinite(v) and v > float("-inf")
        ),
        key=lambda i: masked[i],
        reverse=True,
    )

    def token_info(tid: int) -> dict:
        return {
            "id": tid,
            "raw_logit": raw_logits[tid] if 0 <= tid < len(raw_logits) else None,
            "masked_logit": masked[tid] if 0 <= tid < len(masked) else None,
            "in_top_k": tid in ranked,
            "rank": ranked.index(tid) + 1 if tid in ranked else None,
            "text": llm.detokenize([tid]).decode("utf-8", errors="replace"),
        }

    top10 = [token_info(t) for t in ranked[:10]]
    eos_info = token_info(eos) if eos is not None else None
    eot_info = token_info(eot) if eot is not None else None

    branch_ids = KvlBeamDecoder._branch_token_ids(masked, BRANCH_FACTOR, stop_ids)
    branch_info = [token_info(t) for t in branch_ids]

    return {
        "eos": eos_info,
        "eot": eot_info,
        "top10": top10,
        "branch_token_ids": branch_ids,
        "branch_tokens": branch_info,
        "eos_in_branch": eos in branch_ids if eos is not None else False,
    }


def traced_decode(wrapper, config: ExperimentConfig) -> tuple[list[StepTrace], str, int]:
    """Run decode with per-branch tracing; stop before first finish to capture cause."""
    llm = wrapper.llm
    final_prompt = PROMPT
    if config.config_prompting:
        final_prompt = wrapper._add_simplification_context(PROMPT, num_shots=config.num_shots)

    formatted = wrapper._format_prompt(final_prompt, config.system_prompt)
    prompt_ids = llm.tokenize(formatted.encode("utf-8"), add_bos=True)
    stop_strings = wrapper._get_stop_tokens()
    stop_ids = wrapper._get_stop_token_ids()

    eval_fn, decode_suffix = make_llamacpp_eval_fn(llm, prompt_ids, top_k=config.top_k)
    decoder = KvlBeamDecoder(
        kvl_lookup=wrapper.kvl_lookup,
        l1=config.kvl_l1,
        text_evaluator=wrapper.text_evaluator,
        beam_width=BEAM_WIDTH,
        branch_factor=BRANCH_FACTOR,
    )

    traces: list[StepTrace] = []

    from slm_experiments.models.kvl_beam_decoder import KvlBeamCandidate

    active_beams = [
        KvlBeamCandidate(token_ids=list(prompt_ids), text="", cumulative_logprob=0.0)
    ]
    step = 0
    finish_trace: StepTrace | None = None

    for _ in range(config.max_new_tokens):
        if not active_beams:
            break
        step += 1
        children = []
        for beam_idx, beam in enumerate(active_beams):
            logits, _ = eval_fn(beam.token_ids)
            branch_ids = decoder._branch_token_ids(logits, BRANCH_FACTOR, stop_ids)
            for token_id in branch_ids:
                child = decoder._extend_candidate(
                    beam,
                    token_id,
                    decoder._logprob(logits, token_id),
                    eval_fn,
                    stop_strings,
                    stop_token_ids=stop_ids,
                    decode_suffix=decode_suffix,
                )
                suffix = child.text if child.finished and token_id in stop_ids else decode_suffix(child.token_ids)
                hit_token = token_id in stop_ids
                hit_string = decoder._hits_stop(suffix if not (child.finished and hit_token) else decode_suffix(beam.token_ids), stop_strings)
                if child.finished and hit_token:
                    hit_string = decoder._hits_stop(beam.text, stop_strings)

                trace = StepTrace(
                    step=step,
                    beam_idx=beam_idx,
                    token_id=token_id,
                    token_text=llm.detokenize([token_id]).decode("utf-8", errors="replace"),
                    logprob=decoder._logprob(logits, token_id),
                    suffix=child.text if not hit_token else beam.text,
                    hit_token_stop=hit_token,
                    hit_string_stop=hit_string if not hit_token else False,
                    finished=child.finished,
                )
                traces.append(trace)
                if child.finished and finish_trace is None:
                    finish_trace = trace
                    return traces, child.text, step
                if not child.finished:
                    children.append(child)

        children.sort(key=decoder._rank_key, reverse=True)
        active_beams = children[:BEAM_WIDTH]

    best = active_beams[0] if active_beams else None
    return traces, best.text if best else "", step


def run_model(model_name: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 72}")

    wrapper = get_model_wrapper(model_name, seed=42)
    if not wrapper.model_loaded:
        print("  ERROR: model not loaded")
        return

    llm = wrapper.llm
    config = ExperimentConfig(
        model_name=model_name,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        config_prompting=True,
        config_kvl_beam=True,
        kvl_beam_width=BEAM_WIDTH,
        kvl_branch_factor=BRANCH_FACTOR,
        kvl_l1="es",
        top_k=TOP_K,
        max_new_tokens=200,
    )

    final_prompt = PROMPT
    if config.config_prompting:
        final_prompt = wrapper._add_simplification_context(PROMPT, num_shots=0)
    formatted = wrapper._format_prompt(final_prompt, config.system_prompt)
    prompt_ids = llm.tokenize(formatted.encode("utf-8"), add_bos=True)

    stop_strings = wrapper._get_stop_tokens()
    stop_info = describe_stop_ids(llm, stop_strings)
    print(f"  Stop strings: {stop_strings}")
    print(f"  Stop token IDs: {stop_info['stop_token_ids']}")
    for tid, piece in stop_info["detokenized"].items():
        print(f"    id {tid} -> {piece}")

    print(f"\n  Prompt: {PROMPT!r}")
    print(f"  Formatted prompt tail: ...{formatted[-120:]!r}")

    print("\n  --- Step 1 logits (after prompt) ---")
    rep = logits_report(llm, prompt_ids, TOP_K)
    if rep["eos"]:
        e = rep["eos"]
        print(
            f"  EOS id={e['id']} raw={e['raw_logit']:.4f} "
            f"in_top_k={e['in_top_k']} rank={e['rank']} text={e['text']!r}"
        )
    print("  Top-10 masked tokens:")
    for t in rep["top10"]:
        print(f"    rank {t['rank']:2d} id={t['id']:6d} logit={t['masked_logit']:.4f} {t['text']!r}")
    print(f"  Branch set ({len(rep['branch_token_ids'])} ids): EOS in branch={rep['eos_in_branch']}")
    for t in rep["branch_tokens"]:
        marker = " <-- EOS" if t["id"] == (rep["eos"] or {}).get("id") else ""
        print(f"    id={t['id']:6d} rank={t['rank']} {t['text']!r}{marker}")

    print("\n  --- Traced decode until first finish ---")
    traces, finish_text, finish_step = traced_decode(wrapper, config)
    print(f"  First finish at step {finish_step}, text={finish_text!r}")
    finish = traces[-1]
    print(
        f"  Finishing branch: token_id={finish.token_id} {finish.token_text!r} "
        f"hit_token_stop={finish.hit_token_stop} hit_string_stop={finish.hit_string_stop}"
    )
    print(f"  Suffix at finish (before EOS rollback): {finish.suffix!r}")

    print("\n  Non-finish branches before first finish:")
    for tr in traces:
        if tr.finished:
            continue
        print(
            f"    step {tr.step} beam {tr.beam_idx} id={tr.token_id} "
            f"{tr.token_text!r} -> {tr.suffix!r}"
        )

    print("\n  --- Full generate_kvl_beam ---")
    result = wrapper.generate_kvl_beam(PROMPT, config, beam_width=BEAM_WIDTH, branch_factor=BRANCH_FACTOR)
    print(f"  response={result['response']!r}")
    print(f"  steps={result['kvl_beam_steps_total']} words_scored={result['kvl_beam_words_scored']}")

    wrapper.cleanup()


def main() -> None:
    print("TinyLlama KVL beam early-stop diagnostic")
    print(f"STANDARD_PROMPTS[0] = {PROMPT!r}")
    run_model("TinyLlama")
    run_model("Qwen3")


if __name__ == "__main__":
    main()
