"""KVL lemma ↔ token ID index for collision diagnostics and future trie mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_experiments.evaluation.kvl import KvlLookup


@dataclass(frozen=True)
class KvlTokenIndex:
    """Maps KVL lemmas to tokenizer IDs and reports token-level lemma collisions."""

    lemma_to_token_ids: dict[str, tuple[int, ...]]
    id_to_lemmas: dict[int, tuple[str, ...]]

    @classmethod
    def build(cls, llm: Any, kvl_lookup: KvlLookup, l1: str) -> "KvlTokenIndex":
        """Build index from a llama.cpp-like tokenizer and KVL lookup lemmas."""
        lookup = kvl_lookup.load(l1)
        lemma_to_token_ids: dict[str, tuple[int, ...]] = {}
        id_to_lemma_sets: dict[int, set[str]] = {}

        for lemma in lookup:
            token_ids = cls._tokenize_lemma(llm, lemma)
            lemma_to_token_ids[lemma] = token_ids
            for token_id in token_ids:
                id_to_lemma_sets.setdefault(token_id, set()).add(lemma)

        id_to_lemmas = {
            token_id: tuple(sorted(lemmas))
            for token_id, lemmas in sorted(id_to_lemma_sets.items())
        }
        return cls(lemma_to_token_ids=lemma_to_token_ids, id_to_lemmas=id_to_lemmas)

    @staticmethod
    def _tokenize_lemma(llm: Any, lemma: str) -> tuple[int, ...]:
        """Tokenize mid-sentence and bare forms, mirroring logit-bias indexing."""
        seen: set[int] = set()
        ordered: list[int] = []
        for text in (f" {lemma}", lemma):
            try:
                tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
            except Exception:
                continue
            for token_id in tokens:
                if token_id not in seen:
                    seen.add(token_id)
                    ordered.append(token_id)
        return tuple(ordered)

    @property
    def lemma_count(self) -> int:
        return len(self.lemma_to_token_ids)

    @property
    def token_id_count(self) -> int:
        return len(self.id_to_lemmas)

    def collision_token_ids(self) -> dict[int, tuple[str, ...]]:
        """Return token IDs mapped to more than one KVL lemma."""
        return {
            token_id: lemmas
            for token_id, lemmas in self.id_to_lemmas.items()
            if len(lemmas) > 1
        }

    def collision_count(self) -> int:
        return len(self.collision_token_ids())

    def collision_stats(self) -> dict[str, int | float]:
        """Summary stats for audit scripts and diagnostics."""
        collisions = self.collision_token_ids()
        empty_lemmas = sum(
            1 for token_ids in self.lemma_to_token_ids.values() if not token_ids
        )
        return {
            "lemma_count": self.lemma_count,
            "unique_token_ids": self.token_id_count,
            "collision_token_ids": len(collisions),
            "collision_lemma_refs": sum(len(lemmas) for lemmas in collisions.values()),
            "empty_tokenization_lemmas": empty_lemmas,
            "collision_rate": (
                len(collisions) / self.token_id_count if self.token_id_count else 0.0
            ),
        }
