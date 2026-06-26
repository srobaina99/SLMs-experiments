"""Tests for KVL-scored beam decoder ranking logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_experiments.evaluation.kvl import KvlLookup
from slm_experiments.evaluation.metrics import TextEvaluator
from slm_experiments.models.kvl_beam_decoder import KvlBeamDecoder

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

TOK_FUNDAMENTALLY = 1
TOK_FRIEND = 2
TOK_PLAY = 3
TOK_ESTABLISHMENT = 4
TOK_OOV = 5
TOK_EASY_A = 6
TOK_EASY_B = 7
TOK_PARTIAL_HIGH = 8
TOK_PARTIAL_LOW = 9
TOK_STOP = 99

TOKEN_TEXT = {
    TOK_FUNDAMENTALLY: "fundamentally ",
    TOK_FRIEND: "friend ",
    TOK_PLAY: "play",
    TOK_ESTABLISHMENT: "establishment ",
    TOK_OOV: "notinlookup ",
    TOK_EASY_A: "cat ",
    TOK_EASY_B: "dog ",
    TOK_PARTIAL_HIGH: "aaa",
    TOK_PARTIAL_LOW: "bbb",
    TOK_STOP: "<|eos|>",
}


def _suffix(token_ids: list[int], prompt_len: int) -> str:
    return "".join(TOKEN_TEXT[token_id] for token_id in token_ids[prompt_len:])


@pytest.fixture
def fixture_lookup(tmp_path):
    lookup_dir = tmp_path / "kvl"
    lookup_dir.mkdir()
    source = FIXTURE_DIR / "kvl_lookup_es.json"
    (lookup_dir / "kvl_lookup_es.json").write_text(source.read_text(), encoding="utf-8")
    return KvlLookup(data_dir=str(lookup_dir))


@pytest.fixture
def decoder(fixture_lookup):
    return KvlBeamDecoder(
        kvl_lookup=fixture_lookup,
        l1="es",
        text_evaluator=TextEvaluator(),
        beam_width=2,
        branch_factor=3,
    )


def make_eval_fn(prompt_len: int, logits_by_prefix: dict[tuple[int, ...], dict[int, float]]):
    vocab_size = max(TOKEN_TEXT) + 1
    default_logits = [float("-inf")] * vocab_size

    def eval_fn(token_ids: list[int]) -> tuple[list[float], str]:
        prefix = tuple(token_ids[prompt_len:])
        overrides = logits_by_prefix.get(prefix, {})
        logits = default_logits.copy()
        for token_id, value in overrides.items():
            logits[token_id] = value
        return logits, _suffix(token_ids, prompt_len)

    return eval_fn


class TestKvlBeamDecoderPreference:
    def test_prefers_easy_friend_path_over_hard_word_path(self, decoder):
        prompt = [100]
        prompt_len = len(prompt)

        eval_fn = make_eval_fn(
            prompt_len,
            {
                (): {
                    TOK_FUNDAMENTALLY: 5.0,
                    TOK_FRIEND: 4.0,
                    TOK_ESTABLISHMENT: 3.0,
                },
                (TOK_FUNDAMENTALLY,): {
                    TOK_PLAY: 0.0,
                },
                (TOK_FRIEND,): {
                    TOK_PLAY: 0.0,
                },
            },
        )

        result = decoder.decode(
            eval_fn,
            prompt,
            max_tokens=2,
            stop=[],
        )

        assert result.text == "friend play"
        assert result.running_mean == pytest.approx(1.55)
        assert result.words_scored == 2

    def test_tiebreak_equal_kvl_mean_by_logprob(self, tmp_path):
        lookup_dir = tmp_path / "kvl"
        lookup_dir.mkdir()
        (lookup_dir / "kvl_lookup_es.json").write_text(
            '{"alpha": 1.0, "beta": 1.0}',
            encoding="utf-8",
        )
        decoder = KvlBeamDecoder(
            kvl_lookup=KvlLookup(data_dir=str(lookup_dir)),
            l1="es",
            text_evaluator=TextEvaluator(),
            beam_width=1,
            branch_factor=2,
        )
        prompt = [100]
        prompt_len = len(prompt)

        token_text = {10: "alpha ", 11: "beta "}

        def eval_fn(token_ids: list[int]) -> tuple[list[float], str]:
            suffix = "".join(token_text[tid] for tid in token_ids[prompt_len:])
            logits = [float("-inf")] * 12
            logits[10] = 2.0
            logits[11] = 1.0
            return logits, suffix

        result = decoder.decode(
            eval_fn,
            prompt,
            max_tokens=1,
            stop=[],
        )

        assert result.text == "alpha "
        assert result.running_mean == pytest.approx(1.0)
        assert result.cumulative_logprob > -0.5

    def test_oov_words_excluded_from_running_mean(self, decoder):
        prompt = [100]
        prompt_len = len(prompt)

        eval_fn = make_eval_fn(
            prompt_len,
            {
                (): {
                    TOK_FRIEND: 4.0,
                    TOK_OOV: 5.0,
                },
                (TOK_FRIEND,): {
                    TOK_PLAY: 0.0,
                },
                (TOK_OOV,): {
                    TOK_PLAY: 0.0,
                },
            },
        )

        result = decoder.decode(
            eval_fn,
            prompt,
            max_tokens=2,
            stop=[],
        )

        assert result.text == "friend play"
        assert result.running_mean == pytest.approx(1.55)
        assert result.words_scored == 2

    def test_early_steps_rank_by_logprob_when_no_kvl_scores(self, decoder):
        prompt = [100]
        prompt_len = len(prompt)

        eval_fn = make_eval_fn(
            prompt_len,
            {
                (): {
                    TOK_PARTIAL_HIGH: 5.0,
                    TOK_PARTIAL_LOW: 1.0,
                },
            },
        )

        greedy = KvlBeamDecoder(
            kvl_lookup=decoder.kvl_lookup,
            l1=decoder.l1,
            text_evaluator=decoder.text_evaluator,
            beam_width=1,
            branch_factor=2,
        )
        result = greedy.decode(
            eval_fn,
            prompt,
            max_tokens=1,
            stop=[],
        )

        assert result.text == "aaa"
        assert result.running_mean is None
        assert result.words_scored == 0


class TestKvlBeamDecoderStopTokens:
    def test_stops_on_stop_token_id(self, decoder):
        prompt = [100]
        prompt_len = len(prompt)

        eval_fn = make_eval_fn(
            prompt_len,
            {
                (): {
                    TOK_FRIEND: 4.0,
                    TOK_STOP: 6.0,
                },
            },
        )

        result = decoder.decode(
            eval_fn,
            prompt,
            max_tokens=10,
            stop=[],
            stop_token_ids=frozenset({TOK_STOP}),
        )

        assert result.steps_total < 10
        assert TOK_STOP not in result.token_ids

    def test_stop_token_excluded_from_response_text(self, decoder):
        prompt = [100]
        prompt_len = len(prompt)

        eval_fn = make_eval_fn(
            prompt_len,
            {
                (): {
                    TOK_STOP: 6.0,
                },
            },
        )

        result = decoder.decode(
            eval_fn,
            prompt,
            max_tokens=10,
            stop=[],
            stop_token_ids=frozenset({TOK_STOP}),
        )

        assert "<|eos|>" not in result.text
        assert result.token_ids == []

    def test_stop_token_ids_included_in_branch_expansion(self, fixture_lookup):
        logits = [float("-inf")] * 100
        logits[1] = 10.0
        logits[TOK_STOP] = -500.0

        branch = KvlBeamDecoder._branch_token_ids(logits, 2, frozenset({TOK_STOP}))

        assert 1 in branch
        assert TOK_STOP in branch

        prompt = [100]
        prompt_len = len(prompt)

        def eval_fn(token_ids: list[int]) -> tuple[list[float], str]:
            step_logits = [float("-inf")] * 100
            prefix = tuple(token_ids[prompt_len:])
            if prefix == ():
                step_logits[TOK_FRIEND] = 10.0
                step_logits[TOK_STOP] = -500.0
            elif prefix == (TOK_FRIEND,):
                step_logits[TOK_STOP] = -500.0
            return step_logits, _suffix(token_ids, prompt_len)

        decoder = KvlBeamDecoder(
            kvl_lookup=fixture_lookup,
            l1="es",
            text_evaluator=TextEvaluator(),
            beam_width=2,
            branch_factor=2,
        )

        result = decoder.decode(
            eval_fn,
            prompt,
            max_tokens=2,
            stop=[],
            stop_token_ids=frozenset({TOK_STOP}),
        )

        assert result.steps_total == 2
        assert result.text == "friend "
        assert TOK_STOP not in result.token_ids
