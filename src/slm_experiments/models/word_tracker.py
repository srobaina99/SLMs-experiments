"""Word-boundary tracking for incremental KVL scoring during beam decode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordTracker:
    """Accumulate decoded token text and emit words at boundaries."""

    pending_word: str = ""

    def append_token_text(self, token_text: str) -> list[str]:
        """Return list of completed words (may be empty)."""
        self.pending_word += token_text
        return self._extract_completed_words()

    def flush_pending(self) -> list[str]:
        """Emit a trailing partial word at end-of-sequence (no boundary yet)."""
        if not self.pending_word:
            return []
        word = self._clean_word(self.pending_word)
        self.pending_word = ""
        return [word] if word else []

    @staticmethod
    def is_content_word(word: str, content_words_set: set[str]) -> bool:
        """Check if cleaned word is a content word."""
        clean = WordTracker._clean_word(word)
        return clean in content_words_set

    @staticmethod
    def _clean_word(word: str) -> str:
        return "".join(c for c in word if c.isalnum()).lower()

    def _extract_completed_words(self) -> list[str]:
        completed: list[str] = []
        while self.pending_word:
            idx = 0
            while idx < len(self.pending_word) and not self.pending_word[idx].isalnum():
                idx += 1
            if idx > 0:
                self.pending_word = self.pending_word[idx:]
            if not self.pending_word:
                break

            end = 0
            while end < len(self.pending_word) and self.pending_word[end].isalnum():
                end += 1

            if end == len(self.pending_word):
                break

            completed.append(self.pending_word[:end].lower())
            self.pending_word = self.pending_word[end:]

        return completed
