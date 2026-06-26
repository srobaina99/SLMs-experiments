#!/usr/bin/env python3
"""
Spike: manual per-token decode loop on one GGUF model (Phase 0).

Decode loop API (llama-cpp-python >= 0.2, validated against 0.3.31):

  Chosen approach for KVL beam v1
  --------------------------------
  reset + eval(full_prefix) on every eval_fn call.

  Beam search explores W parallel prefixes that diverge at different tokens.
  A single Llama instance keeps one KV cache, so branches cannot share
  incremental state. For each candidate prefix we call llm.reset(), then
  llm.eval(token_ids), then read logits for the last position.

  Logits access
  -------------
  After eval(), read raw logits via llm._ctx.get_logits() (shape n_vocab).
  With default logits_all=False only the last position is valid — sufficient
  for next-token expansion. Alternative: llm.sample(temp=0.0) for greedy
  single-path checks (uses internal sampler chain, not raw logits).

  Tokenization
  ------------
  llm.tokenize(text.encode("utf-8"), add_bos=True)  — prompt prefix
  llm.detokenize([token_id])                        — decode one token
  llm.detokenize(generated_ids, prev_tokens=prompt_ids) — incremental text

  Incremental eval (single-path only)
  -----------------------------------
  llm.reset(); llm.eval(prompt_ids); loop: token = llm.sample(temp=0.0);
  llm.eval([token]). Faster for greedy smoke tests but NOT used for beam
  branches because diverging prefixes require independent KV states.

  One-shot baseline
  -----------------
  llm(prompt, max_tokens=N, temperature=0.0, stop=[...]) — existing wrapper path.

  Stop handling (dual check)
  --------------------------
  1. Token ID — resolve_llamacpp_stop_token_ids() collects EOS, EOT, and Qwen
     chat specials (151643–151648). Qwen3 EOS (151645) detokenizes to "" so
     string checks alone never fire.
  2. String suffix — decoded UTF-8 suffix vs stop strings (secondary guard).

  Stop strings are not tokenized into stop_token_ids (BPE fragments false-trigger).

Temporary spike script — not part of the package API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.kvl_beam_decoder import resolve_llamacpp_stop_token_ids

MAX_TOKENS = 20
DEFAULT_PROMPT = STANDARD_PROMPTS[0]


def get_last_logits(llm) -> np.ndarray:
    """Return logits for the last evaluated position (vocab_size,)."""
    n_vocab = llm.n_vocab()
    return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(n_vocab,)).copy()


def top_k_token_ids(logits: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return top-k (token_id, logit) pairs."""
    if k >= len(logits):
        indices = np.argsort(logits)[::-1]
    else:
        indices = np.argpartition(logits, -k)[-k:]
        indices = indices[np.argsort(logits[indices])[::-1]]
    return [(int(i), float(logits[i])) for i in indices]


def hits_stop(text: str, stop: list[str]) -> bool:
    return any(text.endswith(s) for s in stop)


def hits_stop_token(token_id: int, stop_token_ids: frozenset[int]) -> bool:
    return token_id in stop_token_ids


def decode_incremental_greedy(
    llm,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
    stop: list[str],
    stop_token_ids: frozenset[int],
    top_k: int = 50,
    top_p: float = 0.95,
) -> tuple[list[int], str]:
    """
    Single-path greedy loop: eval prompt once, then eval one token at a time.

    Uses llm.sample(temp=0.0) for token selection (not raw argmax on logits).
    """
    llm.reset()
    llm.eval(prompt_token_ids)
    generated: list[int] = []

    for _ in range(max_tokens):
        next_token = llm.sample(temp=0.0, top_k=top_k, top_p=top_p)
        if hits_stop_token(next_token, stop_token_ids):
            break
        generated.append(next_token)
        llm.eval([next_token])

        text = llm.detokenize(generated, prev_tokens=prompt_token_ids).decode(
            "utf-8", errors="replace"
        )
        if hits_stop(text, stop):
            break

    return generated, llm.detokenize(generated, prev_tokens=prompt_token_ids).decode(
        "utf-8", errors="replace"
    )


def decode_reset_prefix_greedy(
    llm,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
    stop: list[str],
    stop_token_ids: frozenset[int],
) -> tuple[list[int], str]:
    """
    Beam-compatible eval pattern: reset and re-eval the full prefix each step.

    Selects next token via argmax on last-position logits (temperature=0).
    """
    generated: list[int] = []

    for _ in range(max_tokens):
        prefix = prompt_token_ids + generated
        llm.reset()
        llm.eval(prefix)
        logits = get_last_logits(llm)
        next_token = int(np.argmax(logits))
        if hits_stop_token(next_token, stop_token_ids):
            break
        generated.append(next_token)

        text = llm.detokenize(generated, prev_tokens=prompt_token_ids).decode(
            "utf-8", errors="replace"
        )
        if hits_stop(text, stop):
            break

    return generated, llm.detokenize(generated, prev_tokens=prompt_token_ids).decode(
        "utf-8", errors="replace"
    )


def one_shot_generate(
    llm,
    formatted_prompt: str,
    *,
    max_tokens: int,
    stop: list[str],
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    output = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=top_k,
        top_p=top_p,
        stop=stop,
        echo=False,
    )
    return output["choices"][0]["text"]


def run_spike(prompt: str, seed: int = 42) -> int:
    config = ExperimentConfig(
        config_prompting=True,
        temperature=0.0,
        top_k=50,
        top_p=0.95,
        max_new_tokens=MAX_TOKENS,
        system_prompt="You are a helpful English teacher for beginner students.",
    )

    wrapper = get_model_wrapper("Qwen3", seed=seed)
    if not wrapper.model_loaded or wrapper.llm is None:
        print("ERROR: Qwen3 GGUF not loaded.")
        print(f"  Expected path: {wrapper.model_path}")
        print("  Set SLM_GGUF_DIR or place Qwen3-0.6B-Q4_0.gguf under models/gguf/")
        return 1

    llm = wrapper.llm
    final_prompt = wrapper._add_simplification_context(prompt, num_shots=config.num_shots)
    formatted_prompt = wrapper._format_prompt(final_prompt, config.system_prompt)
    stop = wrapper._get_stop_tokens()
    stop_token_ids = resolve_llamacpp_stop_token_ids(llm, stop)
    print(f"Stop token IDs: {sorted(stop_token_ids)}")

    prompt_token_ids = llm.tokenize(formatted_prompt.encode("utf-8"), add_bos=True)
    print(f"Prompt tokens: {len(prompt_token_ids)}")
    print(f"User prompt: {prompt!r}\n")

    # Sanity: logits readable after eval
    llm.reset()
    llm.eval(prompt_token_ids)
    top5 = top_k_token_ids(get_last_logits(llm), k=5)
    print("Top-5 logits after prompt eval:")
    for tid, logit in top5:
        piece = llm.detokenize([tid]).decode("utf-8", errors="replace")
        print(f"  id={tid:6d}  logit={logit:8.3f}  text={piece!r}")

    inc_ids, inc_text = decode_incremental_greedy(
        llm,
        prompt_token_ids,
        max_tokens=MAX_TOKENS,
        stop=stop,
        stop_token_ids=stop_token_ids,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    reset_ids, reset_text = decode_reset_prefix_greedy(
        llm,
        prompt_token_ids,
        max_tokens=MAX_TOKENS,
        stop=stop,
        stop_token_ids=stop_token_ids,
    )
    ones_text = one_shot_generate(
        llm,
        formatted_prompt,
        max_tokens=MAX_TOKENS,
        stop=stop,
        top_k=config.top_k,
        top_p=config.top_p,
    )

    print(f"\nIncremental greedy ({len(inc_ids)} tokens):")
    print(f"  {inc_text!r}")
    print(f"\nReset+prefix greedy ({len(reset_ids)} tokens):")
    print(f"  {reset_text!r}")
    print(f"\nOne-shot llm() ({MAX_TOKENS} max_tokens):")
    print(f"  {ones_text!r}")

    inc_match = inc_text.strip() == ones_text.strip()
    reset_match = reset_text.strip() == ones_text.strip()
    print(f"\nIncremental matches one-shot: {inc_match}")
    print(f"Reset+prefix matches one-shot: {reset_match}")

    if not inc_text.strip() and not reset_text.strip() and not ones_text.strip():
        print("ERROR: all outputs empty")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 KVL beam eval loop spike")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt text (default: first standard prompt)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return run_spike(args.prompt, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
