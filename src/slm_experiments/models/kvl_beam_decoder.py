"""
KVL-scored token-level beam decoder (Phase 1+).

Decode loop API
---------------
Production llama.cpp integration uses reset + eval(full_prefix) per candidate:

  llm.reset()
  llm.eval(token_ids)                    # full prefix including prompt + generation
  logits = llm._ctx.get_logits()         # last-position logits, shape (n_vocab,)
  next_id = argmax or top-K after filter  # branch expansion at temperature=0

Tokenization helpers on the same Llama instance:

  llm.tokenize(text.encode("utf-8"), add_bos=True)
  llm.detokenize(token_ids, prev_tokens=prompt_ids)

Stop handling uses two checks after each appended token:

1. Token ID — ``stop_token_ids`` from :func:`resolve_llamacpp_stop_token_ids`
   (EOS, EOT, Qwen chat specials). Required because Qwen3 EOS (151645)
   detokenizes to an empty string, so suffix checks never fire.
2. String suffix — decoded UTF-8 suffix matched with :meth:`_hits_stop` as a
   secondary guard for stop strings that survive detokenization intact.

When stopping on a token ID, that token is stripped from the candidate so EOS
does not appear in the returned text. Stop strings are **not** tokenized into
``stop_token_ids`` (BPE splits like ``"<|im_end|>"`` would false-trigger on
fragments such as token 91 ``"|"``).

Stopping rule — **first-finish** (intentional design, not a bug)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The decode loop returns the **first** candidate that hits a stop condition
(token ID or string suffix) **with non-empty text**, in beam expansion order,
without collecting finished hypotheses and picking ``max(..., key=_rank_key)``.

Why first-finish instead of “best KVL among finished / max-length survivors”:
mean KVL over content words does **not** reward ending a sentence. Without an
early return, the ranker keeps preferring partials that pad toward
``max_tokens``, so the selected text often stretches to the length budget and
becomes incoherent. First-finish lets a completed, naturally stopped answer
win. KVL still steers **which** partials survive each prune step; first-finish
only decides **when** to return. Empty token-ID stops (e.g. EOS detokenizing
to ``""``) are skipped so models with high early EOS probability (TinyLlama)
still beam through content branches. If nothing finishes before
``max_tokens``, the best surviving active beam is returned.

Claim note: width 4 vs 8 is “more exploration before first natural stop,”
not “better final KVL selection among finished hypotheses.”

Incremental eval (eval prompt once, then eval one token) is reserved for
single-path greedy smoke tests only — beam branches require independent KV
state via reset between prefixes.

See scripts/spike_kvl_beam_eval.py for the Phase 0 spike that validated this API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from slm_experiments.evaluation.kvl import KvlLookup
from slm_experiments.evaluation.metrics import TextEvaluator
from slm_experiments.models.word_tracker import WordTracker

# Qwen-style chat template reserved token IDs (im_start / im_end variants).
_QWEN_CHAT_SPECIAL_MIN = 151643
_QWEN_CHAT_SPECIAL_MAX = 151648


def resolve_llamacpp_stop_token_ids(
    llm, stop_strings: list[str] | None = None
) -> frozenset[int]:
    """Collect single-token stop IDs from llama.cpp (EOS + reserved chat specials).

    Always includes ``llm.token_eos()`` and ``llm.token_eot()`` when available.
    For Qwen-style models, also includes reserved IDs 151643–151648 when they
    fall within ``n_vocab`` — these are atomic vocabulary entries for chat
    control tokens.

    ``stop_strings`` is accepted for API symmetry with wrapper stop lists but
    is intentionally **not** tokenized: multi-token stop strings BPE-split into
    fragments (e.g. token 91 ``"|"`` from ``"<|im_end|>"``) that would cause
    false-positive early stops.
    """
    del stop_strings  # reserved for callers; do not tokenize stop strings
    ids: set[int] = set()

    if hasattr(llm, "token_eos") and callable(llm.token_eos):
        ids.add(llm.token_eos())
    if hasattr(llm, "token_eot") and callable(llm.token_eot):
        ids.add(llm.token_eot())

    n_vocab = llm.n_vocab() if hasattr(llm, "n_vocab") and callable(llm.n_vocab) else None
    if n_vocab is not None:
        for token_id in range(_QWEN_CHAT_SPECIAL_MIN, _QWEN_CHAT_SPECIAL_MAX + 1):
            if token_id < n_vocab:
                ids.add(token_id)

    return frozenset(ids)


@dataclass
class KvlBeamCandidate:
    """Single beam hypothesis during KVL-scored decoding."""

    token_ids: list[int]
    text: str
    cumulative_logprob: float
    pending_word: str = ""
    completed_content_words: list[str] = field(default_factory=list)
    kvl_scores: list[float] = field(default_factory=list)
    finished: bool = False

    def kvl_running_mean(self) -> float | None:
        if not self.kvl_scores:
            return None
        return sum(self.kvl_scores) / len(self.kvl_scores)


@dataclass
class KvlBeamDecodeResult:
    """Output of a completed KVL beam decode."""

    token_ids: list[int]
    text: str
    cumulative_logprob: float
    steps_total: int
    words_scored: int
    running_mean: float | None
    candidates_pruned: int


class KvlBeamDecoder:
    """Token-level beam search ranked by running KVL mean."""

    def __init__(
        self,
        *,
        kvl_lookup: KvlLookup,
        l1: str,
        text_evaluator: TextEvaluator,
        beam_width: int = 4,
        branch_factor: int = 10,
    ):
        self.kvl_lookup = kvl_lookup
        self.l1 = l1
        self.text_evaluator = text_evaluator
        self.beam_width = beam_width
        self.branch_factor = branch_factor

    def decode(
        self,
        eval_fn: Callable[[list[int]], tuple[list[float], str]],
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        stop: list[str],
        stop_token_ids: frozenset[int] | None = None,
        decode_suffix: Callable[[list[int]], str] | None = None,
    ) -> KvlBeamDecodeResult:
        """Run KVL-scored beam search.

        eval_fn: given full token prefix, return (logits, decoded_suffix_text).
        Used for tests with mock logits and for llama.cpp in production.
        """
        active_beams = [
            KvlBeamCandidate(
                token_ids=list(prompt_token_ids),
                text="",
                cumulative_logprob=0.0,
            )
        ]
        candidates_pruned = 0
        steps_total = 0
        best_survivor: KvlBeamCandidate | None = None

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
                    if child.finished:
                        # First non-empty finish wins (see module docstring).
                        # Avoids max-length “best KVL” padding that never stops.
                        if child.text.strip():
                            return self._build_decode_result(
                                child,
                                prompt_token_ids,
                                steps_total=steps_total,
                                candidates_pruned=candidates_pruned,
                            )
                        continue
                    children.append(child)

            children.sort(key=self._rank_key, reverse=True)
            candidates_pruned += max(0, len(children) - self.beam_width)
            active_beams = children[: self.beam_width]
            if active_beams:
                best_survivor = max(active_beams, key=self._rank_key)

        for beam in active_beams:
            self._flush_candidate_words(beam)

        if not active_beams:
            if best_survivor is not None:
                best = best_survivor
                self._flush_candidate_words(best)
            else:
                best = KvlBeamCandidate(
                    token_ids=list(prompt_token_ids),
                    text="",
                    cumulative_logprob=0.0,
                )
        else:
            best = max(active_beams, key=self._rank_key)

        return self._build_decode_result(
            best,
            prompt_token_ids,
            steps_total=steps_total,
            candidates_pruned=candidates_pruned,
        )

    @staticmethod
    def _build_decode_result(
        candidate: KvlBeamCandidate,
        prompt_token_ids: list[int],
        *,
        steps_total: int,
        candidates_pruned: int,
    ) -> KvlBeamDecodeResult:
        generated_ids = candidate.token_ids[len(prompt_token_ids) :]
        return KvlBeamDecodeResult(
            token_ids=generated_ids,
            text=candidate.text,
            cumulative_logprob=candidate.cumulative_logprob,
            steps_total=steps_total,
            words_scored=len(candidate.kvl_scores),
            running_mean=candidate.kvl_running_mean(),
            candidates_pruned=candidates_pruned,
        )

    def _extend_candidate(
        self,
        beam: KvlBeamCandidate,
        token_id: int,
        token_logprob: float,
        eval_fn: Callable[[list[int]], tuple[list[float], str]],
        stop: list[str],
        *,
        stop_token_ids: frozenset[int] | None = None,
        decode_suffix: Callable[[list[int]], str] | None = None,
    ) -> KvlBeamCandidate:
        new_ids = beam.token_ids + [token_id]
        if decode_suffix is not None:
            suffix = decode_suffix(new_ids)
        else:
            _, suffix = eval_fn(new_ids)
        token_text = suffix[len(beam.text) :]

        tracker = WordTracker(pending_word=beam.pending_word)
        completed_words = tracker.append_token_text(token_text)

        child = KvlBeamCandidate(
            token_ids=new_ids,
            text=suffix,
            cumulative_logprob=beam.cumulative_logprob + token_logprob,
            pending_word=tracker.pending_word,
            completed_content_words=list(beam.completed_content_words),
            kvl_scores=list(beam.kvl_scores),
        )

        for word in completed_words:
            content_words = self.text_evaluator.extract_content_words(word)
            if not WordTracker.is_content_word(word, content_words):
                continue
            score = self.kvl_lookup.get_score(word, self.l1)
            if score is None:
                continue
            child.completed_content_words.append(word)
            child.kvl_scores.append(score)

        hit_token_stop = (
            stop_token_ids is not None and token_id in stop_token_ids
        )
        hit_string_stop = self._hits_stop(suffix, stop)

        if hit_token_stop or hit_string_stop:
            child.finished = True
            if hit_token_stop:
                child.token_ids = list(beam.token_ids)
                child.text = beam.text
                child.pending_word = beam.pending_word
                child.completed_content_words = list(beam.completed_content_words)
                child.kvl_scores = list(beam.kvl_scores)
            self._flush_candidate_words(child)

        return child

    def _flush_candidate_words(self, candidate: KvlBeamCandidate) -> None:
        tracker = WordTracker(pending_word=candidate.pending_word)
        for word in tracker.flush_pending():
            content_words = self.text_evaluator.extract_content_words(word)
            if not WordTracker.is_content_word(word, content_words):
                continue
            score = self.kvl_lookup.get_score(word, self.l1)
            if score is None:
                continue
            candidate.completed_content_words.append(word)
            candidate.kvl_scores.append(score)
        candidate.pending_word = tracker.pending_word

    @staticmethod
    def _rank_key(candidate: KvlBeamCandidate) -> tuple[float, float]:
        mean = candidate.kvl_running_mean()
        kvl_key = mean if mean is not None else float("-inf")
        return (kvl_key, candidate.cumulative_logprob)

    @staticmethod
    def _top_k_token_ids(logits: list[float], k: int) -> list[int]:
        if not logits or k <= 0:
            return []
        finite = [
            idx
            for idx, value in enumerate(logits)
            if math.isfinite(value) and value > float("-inf")
        ]
        if not finite:
            return []
        finite.sort(key=lambda idx: logits[idx], reverse=True)
        return finite[: min(k, len(finite))]

    @staticmethod
    def _branch_token_ids(
        logits: list[float],
        branch_factor: int,
        stop_token_ids: frozenset[int] | None = None,
    ) -> list[int]:
        """Top-k branch tokens plus any stop IDs with finite logits."""
        expanded = KvlBeamDecoder._top_k_token_ids(logits, branch_factor)
        if not stop_token_ids:
            return expanded
        seen = set(expanded)
        for token_id in stop_token_ids:
            if token_id in seen:
                continue
            if (
                0 <= token_id < len(logits)
                and math.isfinite(logits[token_id])
                and logits[token_id] > float("-inf")
            ):
                expanded.append(token_id)
                seen.add(token_id)
        return expanded

    @staticmethod
    def _logprob(logits: list[float], token_id: int) -> float:
        if token_id < 0 or token_id >= len(logits):
            return float("-inf")
        max_logit = max(logits)
        log_sum = max_logit + math.log(
            sum(math.exp(value - max_logit) for value in logits)
        )
        return logits[token_id] - log_sum

    @staticmethod
    def _hits_stop(text: str, stop: list[str]) -> bool:
        return any(text.endswith(sequence) for sequence in stop)


def get_last_logits(llm) -> list[float]:
    """Return logits for the last evaluated position as a Python list."""
    n_vocab = llm.n_vocab()
    raw = llm._ctx.get_logits()
    return [float(raw[i]) for i in range(n_vocab)]


def apply_top_k_mask(
    logits: list[float],
    *,
    top_k: int,
) -> list[float]:
    """Mask logits outside top_k (temperature=0 expansion set)."""
    n_vocab = len(logits)
    if n_vocab == 0:
        return logits

    ranked = sorted(
        (
            idx
            for idx, value in enumerate(logits)
            if math.isfinite(value) and value > float("-inf")
        ),
        key=lambda idx: logits[idx],
        reverse=True,
    )
    if not ranked:
        return [float("-inf")] * n_vocab

    if top_k > 0:
        ranked = ranked[: min(top_k, len(ranked))]

    masked = [float("-inf")] * n_vocab
    for idx in ranked:
        masked[idx] = logits[idx]
    return masked


def make_llamacpp_eval_fn(
    llm,
    prompt_token_ids: list[int],
    *,
    top_k: int,
) -> tuple[
    Callable[[list[int]], tuple[list[float], str]],
    Callable[[list[int]], str],
]:
    """Build eval_fn and cheap decode_suffix for KvlBeamDecoder."""

    prompt_len = len(prompt_token_ids)

    def decode_suffix(token_ids: list[int]) -> str:
        generated = token_ids[prompt_len:]
        return llm.detokenize(generated, prev_tokens=prompt_token_ids).decode(
            "utf-8",
            errors="replace",
        )

    def eval_fn(token_ids: list[int]) -> tuple[list[float], str]:
        llm.reset()
        llm.eval(token_ids)
        logits = apply_top_k_mask(
            get_last_logits(llm),
            top_k=top_k,
        )
        return logits, decode_suffix(token_ids)

    return eval_fn, decode_suffix
