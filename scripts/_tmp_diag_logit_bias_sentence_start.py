#!/usr/bin/env python
"""Diagnostic for llama.cpp logit_bias coverage of A1 sentence-start token IDs."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from slm_experiments.models.a1_token_index import A1TokenIndex
from slm_experiments.models.base import resolve_gguf_dir
from slm_experiments.models.llamacpp import LlamaCppBaseWrapper


class _DiagnosticWrapper(LlamaCppBaseWrapper):
    """Concrete wrapper shell used only to call inherited _create_logit_bias."""

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        return f"{system_prompt}\n{user_input}"

    def _get_stop_tokens(self):
        return []

    def _extract_response(self, raw_output: str) -> str:
        return raw_output


class MockLlm:
    def __init__(self, token_map: dict[str, list[int]]):
        self.token_map = token_map

    def tokenize(self, data, add_bos=False):
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        if text not in self.token_map:
            raise AssertionError(f"Unexpected tokenize input: {text!r}")
        return self.token_map[text]


def _make_wrapper(llm) -> _DiagnosticWrapper:
    wrapper = object.__new__(_DiagnosticWrapper)
    wrapper.llm = llm
    return wrapper


def _sample(values: Iterable[int], limit: int = 12) -> list[int]:
    return sorted(values)[:limit]


def _assert_and_log(label: str, wrapper: _DiagnosticWrapper, vocab: list[str], weight_factor: float = 1.5) -> dict:
    print(f"\n=== {label} ===")
    index = A1TokenIndex.build(wrapper.llm, vocab, use_trie=False)
    mid = set(index.mid_sentence_ids)
    start = set(index.sentence_start_ids)
    union = mid | start
    mid_only = mid - start
    start_only = start - mid

    bias = wrapper._create_logit_bias(vocab, weight_factor=weight_factor)
    bias_keys = set(bias.keys())
    expected_bias = math.log(weight_factor)

    print(f"vocab count: {len(vocab)}")
    print(f"mid_sentence_ids count: {len(mid)} sample: {_sample(mid)}")
    print(f"sentence_start_ids count: {len(start)} sample: {_sample(start)}")
    print(f"|mid - start| count: {len(mid_only)} sample: {_sample(mid_only)}")
    print(f"|start - mid| count: {len(start_only)} sample: {_sample(start_only)}")
    print(f"bias dict keys count: {len(bias_keys)} sample: {_sample(bias_keys)}")
    print(f"old mid-only behavior would miss start-only IDs: {len(start_only)}")

    missing_mid = mid - bias_keys
    missing_start = start - bias_keys
    extra_bias = bias_keys - union
    wrong_values = {token_id: value for token_id, value in bias.items() if value != expected_bias}

    assert not missing_mid, f"Missing mid IDs from bias: {_sample(missing_mid)}"
    print("ASSERTION PASS: every mid ID is in bias")
    assert not missing_start, f"Missing sentence-start IDs from bias: {_sample(missing_start)}"
    print("ASSERTION PASS: every sentence-start ID is in bias")
    assert bias_keys == union, (
        f"Bias keys are not exactly mid|start; extra={_sample(extra_bias)} "
        f"missing={_sample(union - bias_keys)}"
    )
    print("ASSERTION PASS: bias keys == mid|start")
    assert not wrong_values, f"Unexpected bias values for IDs: {wrong_values}"
    print("ASSERTION PASS: every bias value equals log(weight_factor)")

    return {
        "status": "pass",
        "mid_count": len(mid),
        "start_count": len(start),
        "mid_only_count": len(mid_only),
        "start_only_count": len(start_only),
        "bias_count": len(bias_keys),
    }


def run_mock_check() -> dict:
    token_map = {
        " hello": [10],
        "hello": [11],
        " world": [20],
        "world": [21],
        " cat": [30],
        "cat": [31],
        " good morning": [40, 41],
        "good morning": [42, 43],
    }
    wrapper = _make_wrapper(MockLlm(token_map))
    return _assert_and_log("MOCK TOKENIZER CHECK", wrapper, ["hello", "world", "cat", "good morning"])


def _candidate_ggufs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    roots = [Path(resolve_gguf_dir()), REPO_ROOT / "models" / "gguf"]
    env_dir = os.environ.get("SLM_GGUF_DIR")
    if env_dir:
        roots.insert(0, Path(env_dir))
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.gguf"), key=lambda candidate: candidate.stat().st_size):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    return candidates


def _load_vocab(limit: int = 50) -> list[str]:
    vocab_path = REPO_ROOT / "data" / "vocabularies" / "filtered_starters_vocab.txt"
    words: list[str] = []
    with vocab_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip().lower()
            if word:
                words.append(word)
            if len(words) >= limit:
                break
    return words


def run_real_check() -> dict:
    ggufs = _candidate_ggufs()
    if not ggufs:
        print("\n=== REAL TOKENIZER CHECK ===")
        print("SKIP: no .gguf files found under resolved project model paths")
        return {"status": "skip", "reason": "no gguf found"}

    try:
        from llama_cpp import Llama
    except Exception as exc:
        print("\n=== REAL TOKENIZER CHECK ===")
        print(f"SKIP: llama_cpp import failed: {exc}")
        return {"status": "skip", "reason": f"llama_cpp import failed: {exc}"}

    print("\n=== REAL TOKENIZER CHECK ===")
    last_error = None
    for model_path in ggufs:
        print(f"trying model: {model_path}")
        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=256,
                n_threads=1,
                n_gpu_layers=0,
                seed=42,
                verbose=False,
            )
        except Exception as exc:
            last_error = exc
            print(f"could not load this candidate: {exc}")
            continue

        wrapper = _make_wrapper(llm)
        result = _assert_and_log("REAL TOKENIZER COVERAGE", wrapper, _load_vocab(limit=50))
        result["model_path"] = str(model_path)
        return result

    print(f"SKIP: failed to load all GGUF tokenizer/model candidates; last error: {last_error}")
    return {"status": "skip", "reason": f"all loads failed; last error: {last_error}"}


def main() -> int:
    print("Diagnostic: _create_logit_bias should cover mid_sentence_ids | sentence_start_ids")
    mock_result = run_mock_check()
    real_result = run_real_check()

    print("\n=== SUMMARY ===")
    print(f"mock: {mock_result}")
    print(f"real: {real_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
