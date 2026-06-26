"""A1 vocabulary token index for guided decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


def _tokenize_word(llm, text: str) -> Tuple[int, ...]:
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
    return tuple(int(t) for t in tokens)


def _record_ids(
    token_ids: Sequence[int],
    word: str,
    target_set: set[int],
    id_to_words: Dict[int, List[str]],
) -> None:
    for token_id in token_ids:
        target_set.add(token_id)
        id_to_words.setdefault(token_id, []).append(word)


@dataclass(frozen=True)
class A1TokenTrie:
    """Trie of full token sequences per A1 word and context."""

    mid_sentence_sequences: Tuple[Tuple[int, ...], ...]
    sentence_start_sequences: Tuple[Tuple[int, ...], ...]

    @classmethod
    def from_word_sequences(
        cls,
        mid_sentence: Dict[str, Tuple[int, ...]],
        sentence_start: Dict[str, Tuple[int, ...]],
    ) -> A1TokenTrie:
        return cls(
            mid_sentence_sequences=tuple(mid_sentence.values()),
            sentence_start_sequences=tuple(sentence_start.values()),
        )

    def _sequences_for_context(self, at_sentence_start: bool) -> Tuple[Tuple[int, ...], ...]:
        if at_sentence_start:
            return self.sentence_start_sequences
        return self.mid_sentence_sequences

    def continuation_ids(
        self, partial_remaining: Tuple[int, ...], at_sentence_start: bool
    ) -> FrozenSet[int]:
        if partial_remaining:
            return frozenset({partial_remaining[0]})
        return frozenset()

    def partial_after_token(
        self, token_id: int, at_sentence_start: bool
    ) -> Optional[Tuple[int, ...]]:
        """Return remaining token IDs if token starts a multi-token A1 word."""
        for sequence in self._sequences_for_context(at_sentence_start):
            if len(sequence) > 1 and sequence[0] == token_id:
                return sequence[1:]
        return None

    def advance_partial(
        self, partial_remaining: Tuple[int, ...], token_id: int
    ) -> Optional[Tuple[int, ...]]:
        if not partial_remaining or partial_remaining[0] != token_id:
            return None
        rest = partial_remaining[1:]
        return rest if rest else None


@dataclass(frozen=True)
class A1TokenIndex:
    """Maps A1 vocabulary words to tokenizer IDs for guided decoding."""

    mid_sentence_ids: FrozenSet[int]
    sentence_start_ids: FrozenSet[int]
    id_to_words: Dict[int, Tuple[str, ...]]
    trie: Optional[A1TokenTrie] = None

    @classmethod
    def build(cls, llm, vocab: List[str], *, use_trie: bool = False) -> A1TokenIndex:
        mid_sentence_ids: set[int] = set()
        sentence_start_ids: set[int] = set()
        id_to_words: Dict[int, List[str]] = {}
        mid_sequences: Dict[str, Tuple[int, ...]] = {}
        start_sequences: Dict[str, Tuple[int, ...]] = {}

        for word in vocab:
            try:
                mid_tokens = _tokenize_word(llm, " " + word)
                if mid_tokens:
                    _record_ids(mid_tokens, word, mid_sentence_ids, id_to_words)
                    mid_sequences[word] = mid_tokens

                start_tokens = _tokenize_word(llm, word)
                if start_tokens:
                    _record_ids(start_tokens, word, sentence_start_ids, id_to_words)
                    start_sequences[word] = start_tokens
            except Exception:
                continue

        id_to_words_final = {
            token_id: tuple(sorted(set(words)))
            for token_id, words in id_to_words.items()
        }

        trie = None
        if use_trie:
            trie = A1TokenTrie.from_word_sequences(mid_sequences, start_sequences)

        return cls(
            mid_sentence_ids=frozenset(mid_sentence_ids),
            sentence_start_ids=frozenset(sentence_start_ids),
            id_to_words=id_to_words_final,
            trie=trie,
        )

    def candidate_set_for_context(self, generated_text: str) -> FrozenSet[int]:
        """Return mid-sentence or sentence-start IDs based on trailing text."""
        if not generated_text or generated_text[-1] in " \n\t\r":
            return self.sentence_start_ids
        return self.mid_sentence_ids

    def collisions(self) -> Dict[int, Tuple[str, ...]]:
        """Return token IDs that map to more than one A1 word."""
        return {
            token_id: words
            for token_id, words in self.id_to_words.items()
            if len(words) > 1
        }

    def at_sentence_start(self, generated_text: str) -> bool:
        return not generated_text or generated_text[-1] in " \n\t\r"
