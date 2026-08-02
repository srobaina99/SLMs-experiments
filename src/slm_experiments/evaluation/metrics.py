"""Text readability and complexity evaluation."""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import textstat

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.stem import WordNetLemmatizer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    WordNetLemmatizer = None  # type: ignore[misc, assignment]

CONTENT_POS_TAGS = frozenset(
    {
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "JJ",
        "JJR",
        "JJS",
        "RB",
        "RBR",
        "RBS",
    }
)

_FUNCTION_WORDS_FALLBACK = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "am",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "can",
        "must",
        "shall",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "if",
        "then",
        "so",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "no",
        "not",
        "only",
        "just",
        "very",
    }
)

if NLTK_AVAILABLE:
    # NLTK 3.8+: averaged_perceptron_tagger_eng; older: averaged_perceptron_tagger.
    # English WordNet lemmatization does not need omw-1.4 (skip; avoids offline
    # download attempts that fail and are unused for EN lemmas).
    for resource, kind in (
        ("punkt", "tokenizers"),
        ("averaged_perceptron_tagger_eng", "taggers"),
        ("averaged_perceptron_tagger", "taggers"),
        ("wordnet", "corpora"),
    ):
        try:
            nltk.data.find(f"{kind}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _treebank_to_wordnet_pos(tag: str) -> str:
    """Map Penn Treebank POS → WordNet POS (n/v/a/r)."""
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("R"):
        return "r"
    return "n"


def _clean_alnum(token: str) -> str:
    return "".join(c for c in token if c.isalnum())


# Resolved once: primary is POS-aware WordNet (nltk core dep).
_LEMMATIZER_BACKEND: Optional[str] = None
_WORDNET_LEMMATIZER: Any = None


def reset_lemmatizer_backend_cache() -> None:
    """Clear cached lemmatizer selection (tests / forced re-probe)."""
    global _LEMMATIZER_BACKEND, _WORDNET_LEMMATIZER
    _LEMMATIZER_BACKEND = None
    _WORDNET_LEMMATIZER = None


def _probe_wordnet_lemmatizer() -> Any:
    """Return a working WordNetLemmatizer, or None if WordNet data is unusable."""
    if not NLTK_AVAILABLE or WordNetLemmatizer is None:
        return None
    try:
        lemmatizer = WordNetLemmatizer()
        # Instantiating succeeds without corpora; probe that WordNet loads.
        lemma = lemmatizer.lemmatize("dogs", "n")
    except LookupError:
        # Missing / unloadable WordNet (or related NLTK data) corpora.
        return None
    if lemma != "dog":
        return None
    return lemmatizer


def resolve_lemmatizer_backend() -> str:
    """Return the active English lemmatizer backend name (cached).

    Primary: ``nltk.WordNetLemmatizer`` (POS-aware) only when WordNet actually
    loads. Fallback: ``simplemma`` when NLTK/WordNet is unavailable. Last
    resort: ``identity``.
    """
    global _LEMMATIZER_BACKEND, _WORDNET_LEMMATIZER
    if _LEMMATIZER_BACKEND is not None:
        return _LEMMATIZER_BACKEND

    lemmatizer = _probe_wordnet_lemmatizer()
    if lemmatizer is not None:
        _WORDNET_LEMMATIZER = lemmatizer
        try:
            import nltk as _nltk

            version = getattr(_nltk, "__version__", "unknown")
        except Exception:
            version = "unknown"
        _LEMMATIZER_BACKEND = f"nltk.WordNetLemmatizer/{version}"
        return _LEMMATIZER_BACKEND

    try:
        import simplemma

        version = getattr(simplemma, "__version__", "unknown")
        _LEMMATIZER_BACKEND = f"simplemma/{version}"
        return _LEMMATIZER_BACKEND
    except ImportError:
        pass

    _LEMMATIZER_BACKEND = "identity"
    return _LEMMATIZER_BACKEND


def lemmatize_english_token(token: str, pos_tag: Optional[str] = None) -> str:
    """Lemmatize an English token for KVL v2 lookup.

    Primary path is POS-aware NLTK ``WordNetLemmatizer`` (uses ``pos_tag``).
    ``simplemma`` is only a fallback when NLTK/WordNet is unavailable.
    """
    surface = token.lower()
    if not surface:
        return surface

    backend = resolve_lemmatizer_backend()
    if backend.startswith("nltk.WordNetLemmatizer") and _WORDNET_LEMMATIZER is not None:
        wn_pos = _treebank_to_wordnet_pos(pos_tag or "NN")
        return _WORDNET_LEMMATIZER.lemmatize(surface, wn_pos)

    if backend.startswith("simplemma"):
        import simplemma

        return simplemma.lemmatize(surface, lang="en").lower()

    return surface


class TextEvaluator:
    """Readability metrics for English text (FK, Gunning Fog, Spache)."""

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None):
        self.tokenizer = tokenizer
        self._token_cache: Dict[Tuple[str, bool], List[str]] = {}

    def extract_content_words(self, text: str) -> Set[str]:
        """Extract unique content-word surface forms (KVL v1 / A1-ratio path)."""
        # Single cache via extract_content_word_tokens — do not fill a second set cache.
        return set(self.extract_content_word_tokens(text, lemmatize=False))

    def extract_content_word_tokens(
        self, text: str, *, lemmatize: bool = False
    ) -> List[str]:
        """Extract content-word tokens in occurrence order (KVL v2 path).

        Unlike ``extract_content_words`` (unique surface-form set), this returns
        every content-word occurrence. When ``lemmatize=True``, each token is
        POS-filtered then lemmatized for lookup (see ``lemmatize_english_token``).
        """
        cache_key = (text, lemmatize)
        if cache_key in self._token_cache:
            return list(self._token_cache[cache_key])

        if not NLTK_AVAILABLE:
            tokens = self._extract_content_word_tokens_fallback(text, lemmatize=lemmatize)
            self._token_cache[cache_key] = tokens
            return list(tokens)

        try:
            raw = word_tokenize(text.lower())
            pos_tags = pos_tag(raw)
            tokens: List[str] = []
            for token, pos in pos_tags:
                if pos not in CONTENT_POS_TAGS:
                    continue
                clean_token = _clean_alnum(token)
                if not clean_token:
                    continue
                if lemmatize:
                    tokens.append(lemmatize_english_token(clean_token, pos))
                else:
                    tokens.append(clean_token)
            self._token_cache[cache_key] = tokens
            return list(tokens)
        except Exception:
            tokens = self._extract_content_word_tokens_fallback(
                text, lemmatize=lemmatize
            )
            self._token_cache[cache_key] = tokens
            return list(tokens)

    def _extract_content_word_tokens_fallback(
        self, text: str, *, lemmatize: bool = False
    ) -> List[str]:
        tokens: List[str] = []
        for token in text.lower().split():
            clean_token = _clean_alnum(token)
            if (
                clean_token
                and len(clean_token) > 2
                and clean_token not in _FUNCTION_WORDS_FALLBACK
            ):
                if lemmatize:
                    tokens.append(lemmatize_english_token(clean_token))
                else:
                    tokens.append(clean_token)
        return tokens

    def _extract_content_words_fallback(self, text: str) -> Set[str]:
        return set(self._extract_content_word_tokens_fallback(text))

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
