"""Tests for ConstrainedDecoder with mock logits and stub llm."""

from typing import List, Optional, Sequence

import numpy as np
import pytest

from slm_experiments.models.a1_token_index import A1TokenIndex
from slm_experiments.models.constrained_decoder import ConstrainedDecoder


VOCAB_SIZE = 8
A1_MID = {2, 5}
A1_START = {3, 6}


def _make_index(*, use_trie: bool = False, stop_token_ids: frozenset[int] | None = None) -> A1TokenIndex:
    mid = set(A1_MID)
    start = set(A1_START)
    id_to_words = {2: ("a1",), 3: ("start",), 5: ("multi",), 6: ("start",), 7: ("multi",)}
    for token_id in stop_token_ids or ():
        mid.add(token_id)
        start.add(token_id)
        id_to_words[token_id] = ("<stop>",)
    trie = None
    if use_trie:
        from slm_experiments.models.a1_token_index import A1TokenTrie

        trie = A1TokenTrie(
            mid_sentence_sequences=((5, 7),),
            sentence_start_sequences=((6,),),
        )
    return A1TokenIndex(
        mid_sentence_ids=frozenset(mid),
        sentence_start_ids=frozenset(start),
        id_to_words=id_to_words,
        trie=trie,
    )


class StubLLM:
    """Minimal llama.cpp stand-in for ConstrainedDecoder.decode()."""

    def __init__(
        self,
        logits_sequence: Sequence[np.ndarray],
        *,
        token_text: Optional[dict[int, str]] = None,
    ):
        self.logits_sequence = list(logits_sequence)
        self.token_text = token_text or {
            0: "X",
            1: "Y",
            2: " a1",
            3: "Start",
            4: "Z",
            5: "mul",
            6: "Beg",
            7: "ti",
        }
        self.eval_calls: List[List[int]] = []
        self.reset_called = False
        self.n_tokens = 0

    def reset(self) -> None:
        self.reset_called = True
        self.n_tokens = 0

    def eval(self, tokens: Sequence[int]) -> None:
        self.eval_calls.append(list(tokens))
        self.n_tokens += len(tokens)

    def logits_for_step(self, step_index: int) -> np.ndarray:
        return self.logits_sequence[step_index]

    def detokenize(self, token_ids: Sequence[int], prev_tokens=None) -> bytes:
        return "".join(self.token_text.get(t, "?") for t in token_ids).encode("utf-8")


def _logits_from_scores(scores: dict[int, float], size: int = VOCAB_SIZE) -> np.ndarray:
    arr = np.full(size, -10.0, dtype=np.float32)
    for token_id, score in scores.items():
        arr[token_id] = score
    return arr


class TestConstrainedDecoderFlat:
    def test_prefers_a1_token_in_pool_over_higher_argmax(self):
        # Empty context => sentence-start set {3, 6}. Pool top-3: 4 (best), 1, 3 (A1).
        llm = StubLLM(
            [
                _logits_from_scores({4: 5.0, 1: 4.0, 3: 3.0}),
            ]
        )
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=1,
            stop=[],
            guided_pool_size=3,
            index=_make_index(),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == [3]
        assert result.steps_a1_chosen == 1
        assert result.steps_fallback_argmax == 0

    def test_fallback_to_argmax_when_no_a1_in_pool(self):
        llm = StubLLM(
            [
                _logits_from_scores({4: 5.0, 1: 4.0, 0: 3.0}),
            ]
        )
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=1,
            stop=[],
            guided_pool_size=3,
            index=_make_index(),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == [4]
        assert result.steps_a1_chosen == 0
        assert result.steps_fallback_argmax == 1
        assert result.steps_no_a1_in_pool == 1

    def test_guided_pool_size_limits_candidates(self):
        # Only top-2 in pool: 4, 1 — neither is A1. Fallback argmax is 4.
        llm = StubLLM(
            [
                _logits_from_scores({4: 5.0, 1: 4.0, 2: 3.5}),
            ]
        )
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=1,
            stop=[],
            guided_pool_size=2,
            index=_make_index(),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == [4]
        assert result.steps_no_a1_in_pool == 1

    def test_mid_sentence_set_after_partial_word(self):
        llm = StubLLM(
            [
                _logits_from_scores({4: 5.0}),
                _logits_from_scores({4: 5.0, 2: 2.0}),
            ]
        )
        llm.token_text[4] = "Say"
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=2,
            stop=[],
            guided_pool_size=2,
            index=_make_index(),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids[0] == 4
        assert result.token_ids[1] == 2
        assert result.steps_a1_chosen == 1

    def test_prefers_stop_token_in_pool_when_most_likely(self):
        tok_stop = 4
        llm = StubLLM(
            [
                _logits_from_scores({tok_stop: 5.0, 3: 4.0, 1: 3.0}),
            ]
        )
        llm.token_text[tok_stop] = ""
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=10,
            stop=[],
            stop_token_ids=frozenset({tok_stop}),
            guided_pool_size=3,
            index=_make_index(stop_token_ids=frozenset({tok_stop})),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == []
        assert result.steps_total == 1
        assert result.steps_a1_chosen == 1

    def test_prefers_a1_over_stop_when_a1_is_most_likely(self):
        tok_stop = 4
        llm = StubLLM(
            [
                _logits_from_scores({3: 5.0, tok_stop: 4.0, 1: 3.0}),
            ]
        )
        llm.token_text[tok_stop] = ""
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=1,
            stop=[],
            stop_token_ids=frozenset({tok_stop}),
            guided_pool_size=3,
            index=_make_index(stop_token_ids=frozenset({tok_stop})),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == [3]
        assert result.steps_a1_chosen == 1

    def test_stop_sequence_halts_generation(self):
        llm = StubLLM(
            [
                _logits_from_scores({3: 5.0}),
            ]
        )
        llm.token_text[3] = "STOP"
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=10,
            stop=["STOP"],
            guided_pool_size=5,
            index=_make_index(),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )

        assert result.steps_total == 1
        assert result.text == ""

    def test_temperature_zero_is_deterministic(self):
        logits = [_logits_from_scores({4: 5.0, 1: 4.0, 3: 3.0}) for _ in range(3)]
        llm_a = StubLLM(logits)
        llm_b = StubLLM(logits)

        kwargs = dict(
            prompt_token_ids=[100],
            max_tokens=3,
            stop=[],
            guided_pool_size=3,
            index=_make_index(),
            mode="flat",
            temperature=0.0,
            top_k=50,
        )
        result_a = ConstrainedDecoder().decode(llm_a, **kwargs)
        result_b = ConstrainedDecoder().decode(llm_b, **kwargs)

        assert result_a.token_ids == result_b.token_ids


class TestConstrainedDecoderTrie:
    def test_trie_continues_multi_token_word(self):
        llm = StubLLM(
            [
                _logits_from_scores({4: 5.0}),
                _logits_from_scores({4: 10.0, 5: 1.0}),
                _logits_from_scores({4: 10.0, 7: 1.0}),
            ]
        )
        llm.token_text[4] = "x"
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=3,
            stop=[],
            guided_pool_size=2,
            index=_make_index(use_trie=True),
            mode="trie",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == [4, 5, 7]
        assert result.steps_a1_chosen == 1

    def test_trie_clears_partial_on_failed_continuation(self):
        llm = StubLLM(
            [
                _logits_from_scores({4: 5.0}),
                _logits_from_scores({5: 5.0}),
                _logits_from_scores({4: 10.0, 7: 0.1}),
            ]
        )
        llm.token_text[4] = "x"
        result = ConstrainedDecoder().decode(
            llm,
            prompt_token_ids=[100],
            max_tokens=3,
            stop=[],
            guided_pool_size=1,
            index=_make_index(use_trie=True),
            mode="trie",
            temperature=0.0,
            top_k=50,
        )

        assert result.token_ids == [4, 5, 4]
        assert result.steps_fallback_argmax >= 1
