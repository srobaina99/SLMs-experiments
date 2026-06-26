"""Tests for KVL lemma ↔ token ID index."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_experiments.evaluation.kvl import KvlLookup
from slm_experiments.models.kvl_token_index import KvlTokenIndex

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Stub tokenizer: mid-sentence and bare forms, with deliberate collisions.
STUB_TOKEN_MAP = {
    " friend": [10],
    "friend": [10],
    " cat": [20],
    "cat": [21],
    " dog": [20],
    "dog": [22],
    " play": [30],
    "play": [30],
    " like": [40],
    "like": [41],
    " establishment": [50],
    "establishment": [51],
    " fundamentally": [60],
    "fundamentally": [62],
    " person": [70],
    "person": [71],
    " talk": [80],
    "talk": [81],
    " a": [90],
    "a": [91],
    " unknownword": [100],
    "unknownword": [101],
    " xyznotinlookup": [110],
    "xyznotinlookup": [111],
}


@pytest.fixture
def fixture_lookup(tmp_path):
    lookup_dir = tmp_path / "kvl"
    lookup_dir.mkdir()
    source = FIXTURE_DIR / "kvl_lookup_es.json"
    (lookup_dir / "kvl_lookup_es.json").write_text(source.read_text(), encoding="utf-8")
    return KvlLookup(data_dir=str(lookup_dir))


@pytest.fixture
def stub_llm():
    llm = MagicMock()
    llm.tokenize.side_effect = lambda data, add_bos=False: STUB_TOKEN_MAP.get(
        data.decode("utf-8"),
        [999],
    )
    return llm


class TestKvlTokenIndexBuild:
    def test_builds_lemma_and_collision_maps(self, stub_llm, fixture_lookup):
        index = KvlTokenIndex.build(stub_llm, fixture_lookup, "es")

        assert index.lemma_to_token_ids["friend"] == (10,)
        assert index.lemma_to_token_ids["cat"] == (20, 21)
        assert index.id_to_lemmas[20] == ("cat", "dog")

    def test_reports_collision_count(self, stub_llm, fixture_lookup):
        index = KvlTokenIndex.build(stub_llm, fixture_lookup, "es")

        assert index.collision_count() == 1
        stats = index.collision_stats()
        assert stats["lemma_count"] == len(fixture_lookup.load("es"))
        assert stats["collision_token_ids"] == 1
        assert stats["collision_lemma_refs"] == 2
        assert stats["empty_tokenization_lemmas"] == 0

    def test_high_frequency_fixture_words_tokenize_without_error(
        self, stub_llm, fixture_lookup
    ):
        index = KvlTokenIndex.build(stub_llm, fixture_lookup, "es")
        high_frequency_words = [
            "friend",
            "play",
            "like",
            "establishment",
            "fundamentally",
            "person",
            "talk",
        ]

        for word in high_frequency_words:
            token_ids = index.lemma_to_token_ids[word]
            assert token_ids, f"{word!r} produced no token IDs"
            assert all(isinstance(token_id, int) for token_id in token_ids)

    def test_skips_failed_tokenizations(self, fixture_lookup):
        llm = MagicMock()
        llm.tokenize.side_effect = lambda data, add_bos=False: (
            [100] if data.decode("utf-8") == " friend" else (_ for _ in ()).throw(
                RuntimeError("tokenize failed")
            )
        )

        index = KvlTokenIndex.build(llm, fixture_lookup, "es")

        assert index.lemma_to_token_ids["friend"] == (100,)
