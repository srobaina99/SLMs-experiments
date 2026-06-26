"""Tests for A1TokenIndex tokenization and context switching."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_experiments.models.a1_token_index import A1TokenIndex, A1TokenTrie
from slm_experiments.models.base import REPO_ROOT


VOCAB_PATH = (
    Path(REPO_ROOT) / "data" / "vocabularies" / "filtered_starters_vocab.txt"
)


def _make_tokenizer(token_map: dict[str, list[int]]):
    """Mock llm.tokenize(text.encode(), add_bos=False) -> list[int]."""

    def tokenize(data: bytes, add_bos: bool = True) -> list[int]:
        text = data.decode("utf-8")
        return list(token_map.get(text, [999]))

    llm = MagicMock()
    llm.tokenize.side_effect = tokenize
    return llm


class TestA1TokenIndexBuild:
    def test_mid_and_sentence_start_sets_differ(self):
        llm = _make_tokenizer(
            {
                " hello": [10],
                "hello": [11],
                " cat": [20, 21],
                "cat": [22],
            }
        )
        index = A1TokenIndex.build(llm, ["hello", "cat"], use_trie=False)

        assert index.mid_sentence_ids == frozenset({10, 20, 21})
        assert index.sentence_start_ids == frozenset({11, 22})
        assert index.id_to_words[10] == ("hello",)
        assert index.id_to_words[20] == ("cat",)
        assert index.id_to_words[21] == ("cat",)

    def test_build_with_trie_attaches_trie(self):
        llm = _make_tokenizer(
            {
                " happy": [30, 31],
                "happy": [32],
            }
        )
        index = A1TokenIndex.build(llm, ["happy"], use_trie=True)

        assert index.trie is not None
        assert (30, 31) in index.trie.mid_sentence_sequences
        assert (32,) in index.trie.sentence_start_sequences

    def test_vocab_file_every_word_tokenizes(self):
        if not VOCAB_PATH.is_file():
            pytest.skip("vocab file not present")

        vocab = VOCAB_PATH.read_text(encoding="utf-8").strip().splitlines()
        vocab = [w.strip().lower() for w in vocab if w.strip()]

        token_map = {}
        for i, word in enumerate(vocab):
            token_map[" " + word] = [i + 1]
            token_map[word] = [i + 10001]
        llm = _make_tokenizer(token_map)

        index = A1TokenIndex.build(llm, vocab, use_trie=False)
        indexed_words = {w for words in index.id_to_words.values() for w in words}
        assert indexed_words == set(vocab)


class TestA1TokenIndexCollisions:
    def test_collisions_report_shared_token_ids(self):
        llm = _make_tokenizer(
            {
                " run": [50],
                "run": [50],
                " runner": [50, 51],
                "runner": [52],
            }
        )
        index = A1TokenIndex.build(llm, ["run", "runner"], use_trie=False)
        collisions = index.collisions()

        assert 50 in collisions
        assert set(collisions[50]) == {"run", "runner"}


class TestCandidateSetForContext:
    @pytest.fixture
    def index(self):
        llm = _make_tokenizer(
            {
                " hello": [10],
                "hello": [11],
                " world": [20],
                "world": [21],
            }
        )
        return A1TokenIndex.build(llm, ["hello", "world"], use_trie=False)

    def test_empty_text_uses_sentence_start(self, index):
        assert index.candidate_set_for_context("") == index.sentence_start_ids

    def test_trailing_space_uses_sentence_start(self, index):
        assert index.candidate_set_for_context("Say ") == index.sentence_start_ids

    def test_trailing_newline_uses_sentence_start(self, index):
        assert index.candidate_set_for_context("Line\n") == index.sentence_start_ids

    def test_mid_word_context_uses_mid_sentence(self, index):
        assert index.candidate_set_for_context("Say hel") == index.mid_sentence_ids

    def test_at_sentence_start_helper(self, index):
        assert index.at_sentence_start("") is True
        assert index.at_sentence_start("word ") is True
        assert index.at_sentence_start("partial") is False


class TestA1TokenTrie:
    def test_partial_after_token_returns_remainder(self):
        trie = A1TokenTrie(
            mid_sentence_sequences=((30, 31, 32),),
            sentence_start_sequences=((40, 41),),
        )
        assert trie.partial_after_token(30, at_sentence_start=False) == (31, 32)
        assert trie.partial_after_token(40, at_sentence_start=True) == (41,)
        assert trie.partial_after_token(99, at_sentence_start=False) is None

    def test_continuation_ids_returns_next_token(self):
        trie = A1TokenTrie(
            mid_sentence_sequences=((30, 31),),
            sentence_start_sequences=(),
        )
        assert trie.continuation_ids((31,), at_sentence_start=False) == frozenset({31})
        assert trie.continuation_ids((), at_sentence_start=False) == frozenset()

    def test_advance_partial_tracks_progress(self):
        trie = A1TokenTrie(
            mid_sentence_sequences=((30, 31),),
            sentence_start_sequences=(),
        )
        assert trie.advance_partial((31,), 31) is None
        assert trie.advance_partial((31,), 99) is None
        assert trie.advance_partial((30, 31), 30) == (31,)
