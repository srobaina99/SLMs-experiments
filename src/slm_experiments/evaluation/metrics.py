"""Text readability and complexity evaluation."""

from typing import Any, Callable, Dict, Optional, Set, Tuple

import textstat

try:
    import nltk
    from nltk import pos_tag, word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    for resource in ("punkt", "averaged_perceptron_tagger", "wordnet"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource == "punkt"
                else f"taggers/{resource}"
                if resource == "averaged_perceptron_tagger"
                else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


class TextEvaluator:
    """Readability metrics for English text (FK, Gunning Fog, Spache)."""

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None):
        self.tokenizer = tokenizer
        self._pos_cache: Dict[str, Set[str]] = {}

    def extract_content_words(self, text: str) -> Set[str]:
        """Extract content words (nouns, verbs, adjectives, adverbs) from text."""
        if not NLTK_AVAILABLE:
            return self._extract_content_words_fallback(text)

        try:
            if text in self._pos_cache:
                return self._pos_cache[text]

            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            content_pos = {
                "NN", "NNS", "NNP", "NNPS",
                "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                "JJ", "JJR", "JJS",
                "RB", "RBR", "RBS",
            }

            content_words = set()
            for token, pos in pos_tags:
                if pos in content_pos:
                    clean_token = "".join(c for c in token if c.isalnum())
                    if clean_token:
                        content_words.add(clean_token)

            self._pos_cache[text] = content_words
            return content_words
        except Exception:
            return self._extract_content_words_fallback(text)

    def _extract_content_words_fallback(self, text: str) -> Set[str]:
        function_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "are", "am", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should", "could",
            "may", "might", "can", "must", "shall", "it", "its", "this", "that", "these",
            "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "our", "their", "if", "then", "so", "what", "when",
            "where", "why", "how", "all", "each", "every", "no", "not", "only", "just", "very",
        }

        content_words = set()
        for token in text.lower().split():
            clean_token = "".join(c for c in token if c.isalnum())
            if clean_token and len(clean_token) > 2 and clean_token not in function_words:
                content_words.add(clean_token)
        return content_words

    def calculate_a1_word_ratio(
        self, text: str, a1_vocab: Set[str]
    ) -> Tuple[float, int, int]:
        """Calculate ratio of A1 words to content words."""
        content_words = self.extract_content_words(text)
        if not content_words:
            return 0.0, 0, 0

        a1_count = sum(1 for word in content_words if word in a1_vocab)
        ratio = a1_count / len(content_words)
        return ratio, a1_count, len(content_words)

    def get_grade_level_indices(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {"flesch_kincaid_grade": 0.0, "gunning_fog": 0.0}

        return {
            "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
            "gunning_fog": round(textstat.gunning_fog(text), 2),
        }

    def get_readability_scores(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {"spache_readability": 0.0}

        return {"spache_readability": round(textstat.spache_readability(text), 2)}

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        if not text or not text.strip():
            stats: Dict[str, int] = {"word_count": 0, "difficult_words": 0}
            if self.tokenizer is not None:
                stats["token_count"] = 0
            return stats

        stats = {
            "word_count": textstat.lexicon_count(text),
            "difficult_words": textstat.difficult_words(text),
        }
        if self.tokenizer is not None:
            try:
                stats["token_count"] = len(self.tokenizer(text))
            except Exception:
                stats["token_count"] = 0
        return stats

    def evaluate_text_comprehensive(self, text: str) -> Dict[str, Any]:
        return {
            "text_statistics": self.get_text_statistics(text),
            "grade_level_indices": self.get_grade_level_indices(text),
            "readability_scores": self.get_readability_scores(text),
        }
