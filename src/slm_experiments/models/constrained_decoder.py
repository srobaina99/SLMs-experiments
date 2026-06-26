"""Top-K A1-constrained greedy decoding loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from slm_experiments.models.a1_token_index import A1TokenIndex


@dataclass
class ConstrainedDecodeResult:
    token_ids: List[int]
    text: str
    steps_total: int
    steps_a1_chosen: int
    steps_fallback_argmax: int
    steps_no_a1_in_pool: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0:
        return np.ones_like(logits) / len(logits)
    return exp / total


def _apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= logits.size:
        return logits.copy()
    filtered = logits.copy()
    threshold = np.partition(filtered, -top_k)[-top_k]
    filtered[filtered < threshold] = -np.inf
    return filtered


def _apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return logits.copy()
    filtered = logits.copy()
    sorted_indices = np.argsort(filtered)[::-1]
    sorted_logits = filtered[sorted_indices]
    probs = _softmax(sorted_logits)
    cumulative = np.cumsum(probs)
    cutoff = int(np.searchsorted(cumulative, top_p, side="right"))
    if cutoff < len(sorted_indices):
        remove = set(sorted_indices[cutoff + 1 :].tolist())
        for idx in remove:
            filtered[idx] = -np.inf
    return filtered


def _pool_candidates(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    guided_pool_size: int,
) -> List[int]:
    working = logits.astype(np.float64, copy=True)
    if temperature > 0:
        working = working / temperature
    working = _apply_top_k(working, top_k)
    working = _apply_top_p(working, top_p)
    finite_mask = np.isfinite(working)
    if not finite_mask.any():
        return [int(np.argmax(logits))]
    ranked = np.argsort(working)[::-1]
    pool: List[int] = []
    for idx in ranked:
        if not finite_mask[idx]:
            continue
        pool.append(int(idx))
        if len(pool) >= guided_pool_size:
            break
    return pool


def _argmax_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def _get_next_logits(llm, step_index: int) -> np.ndarray:
    if hasattr(llm, "logits_for_step"):
        return np.asarray(llm.logits_for_step(step_index), dtype=np.float32)

    if hasattr(llm, "get_logits"):
        raw = llm.get_logits()
        return np.asarray(raw, dtype=np.float32)

    if hasattr(llm, "_scores") and hasattr(llm, "n_tokens") and llm.n_tokens > 0:
        return np.asarray(llm._scores[llm.n_tokens - 1, :], dtype=np.float32)

    if hasattr(llm, "_ctx") and llm._ctx is not None and hasattr(llm, "_n_vocab"):
        logits = llm._ctx.get_logits()
        return np.fromiter(logits, dtype=np.float32, count=llm._n_vocab)

    raise AttributeError("llm must expose logits via logits_for_step, get_logits, _scores, or _ctx")


def _eval_tokens(llm, tokens: Sequence[int]) -> None:
    if hasattr(llm, "eval"):
        llm.eval(list(tokens))
        return
    if hasattr(llm, "n_tokens"):
        llm.n_tokens = getattr(llm, "n_tokens", 0) + len(tokens)


def _reset_llm(llm) -> None:
    if hasattr(llm, "reset"):
        llm.reset()


def _decode_generated(llm, prompt_token_ids: Sequence[int], token_ids: Sequence[int]) -> str:
    if hasattr(llm, "detokenize"):
        decoded = llm.detokenize(list(token_ids), prev_tokens=list(prompt_token_ids))
        if isinstance(decoded, bytes):
            return decoded.decode("utf-8", errors="ignore")
        return str(decoded)
    if hasattr(llm, "token_to_text"):
        return "".join(llm.token_to_text(t) for t in token_ids)
    return ""


def _hits_stop(text: str, stop_sequences: Sequence[str]) -> bool:
    return any(stop in text for stop in stop_sequences if stop)


class ConstrainedDecoder:
    """Token-by-token decoder with A1 pool filtering."""

    def decode(
        self,
        llm,
        prompt_token_ids: List[int],
        *,
        max_tokens: int,
        stop: List[str],
        guided_pool_size: int,
        index: A1TokenIndex,
        mode: Literal["flat", "trie"] = "flat",
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> ConstrainedDecodeResult:
        _reset_llm(llm)
        _eval_tokens(llm, prompt_token_ids)

        generated: List[int] = []
        steps_total = 0
        steps_a1_chosen = 0
        steps_fallback_argmax = 0
        steps_no_a1_in_pool = 0
        partial_remaining: Optional[Tuple[int, ...]] = None

        for step_index in range(max_tokens):
            logits = _get_next_logits(llm, step_index)
            generated_text = _decode_generated(llm, prompt_token_ids, generated)
            at_sentence_start = index.at_sentence_start(generated_text)

            chosen: Optional[int] = None
            used_trie_continuation = False

            if mode == "trie" and index.trie is not None and partial_remaining:
                cont_ids = index.trie.continuation_ids(
                    partial_remaining, at_sentence_start
                )
                pool = _pool_candidates(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    guided_pool_size=guided_pool_size,
                )
                trie_hits = [token_id for token_id in pool if token_id in cont_ids]
                if trie_hits:
                    chosen = trie_hits[0]
                    used_trie_continuation = True
                    partial_remaining = index.trie.advance_partial(
                        partial_remaining, chosen
                    )
                else:
                    partial_remaining = None

            if chosen is None:
                pool = _pool_candidates(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    guided_pool_size=guided_pool_size,
                )
                active_set = index.candidate_set_for_context(generated_text)
                a1_hits = [token_id for token_id in pool if token_id in active_set]

                if a1_hits:
                    chosen = a1_hits[0]
                    steps_a1_chosen += 1
                    if (
                        mode == "trie"
                        and index.trie is not None
                        and not used_trie_continuation
                    ):
                        partial_remaining = index.trie.partial_after_token(
                            chosen, at_sentence_start
                        )
                else:
                    chosen = _argmax_token(logits)
                    steps_fallback_argmax += 1
                    if pool and not any(token_id in active_set for token_id in pool):
                        steps_no_a1_in_pool += 1
                    partial_remaining = None

            generated.append(chosen)
            steps_total += 1
            _eval_tokens(llm, [chosen])

            generated_text = _decode_generated(llm, prompt_token_ids, generated)
            if _hits_stop(generated_text, stop):
                break

        text = _decode_generated(llm, prompt_token_ids, generated)
        for stop_seq in stop:
            if stop_seq in text:
                text = text.split(stop_seq, 1)[0]
                break

        return ConstrainedDecodeResult(
            token_ids=generated,
            text=text,
            steps_total=steps_total,
            steps_a1_chosen=steps_a1_chosen,
            steps_fallback_argmax=steps_fallback_argmax,
            steps_no_a1_in_pool=steps_no_a1_in_pool,
        )
