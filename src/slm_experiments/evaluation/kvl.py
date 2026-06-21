"""KVL/GLMM learner vocabulary difficulty metrics."""

import json
import os
from typing import Dict, Optional, Set

KVL_HARD_THRESHOLD = -1.0

SUPPORTED_L1S = ("es", "de", "cn")
DEFAULT_KVL_L1 = "es"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PACKAGE_DIR)))
_DEFAULT_KVL_DATA_DIR = os.path.join(_REPO_ROOT, "data", "kvl")


def empty_kvl_metrics(l1: str = DEFAULT_KVL_L1) -> Dict[str, object]:
    """Return safe default KVL metrics for failed or empty generations."""
    return {
        "kvl_l1": l1,
        "kvl_content_word_count": 0,
        "kvl_lookup_count": 0,
        "kvl_oov_count": 0,
        "kvl_lookup_coverage": 0.0,
        "kvl_mean_score": None,
        "kvl_min_score": None,
        "kvl_pct_hard_words": None,
    }


class KvlLookup:
    """Load and query KVL GLMM scores by L1 and English word (lemma-only, v1)."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or _DEFAULT_KVL_DATA_DIR
        self._cache: Dict[str, Dict[str, float]] = {}

    def load(self, l1: str) -> Dict[str, float]:
        if l1 not in SUPPORTED_L1S:
            raise ValueError(f"Unsupported L1: {l1!r}. Must be one of {SUPPORTED_L1S}")

        if l1 in self._cache:
            return self._cache[l1]

        path = os.path.join(self.data_dir, f"kvl_lookup_{l1}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"KVL lookup not found: {path}")

        with open(path, encoding="utf-8") as f:
            lookup = json.load(f)

        self._cache[l1] = lookup
        return lookup

    def get_score(self, word: str, l1: str) -> Optional[float]:
        lookup = self.load(l1)
        return lookup.get(word.lower())


def compute_kvl_metrics(
    text: str,
    l1: str,
    content_words: Optional[Set[str]] = None,
    *,
    kvl_lookup: Optional[KvlLookup] = None,
    hard_threshold: float = KVL_HARD_THRESHOLD,
) -> Dict[str, object]:
    """Compute KVL vocabulary difficulty metrics for text.

    When content_words is omitted, pass an empty set; the caller should supply
    words from TextEvaluator.extract_content_words().
    """
    del text  # scoring uses pre-extracted content_words only
    words = content_words if content_words is not None else set()
    content_word_count = len(words)

    result = empty_kvl_metrics(l1)
    result["kvl_content_word_count"] = content_word_count

    if not content_word_count:
        return result

    lookup = kvl_lookup or KvlLookup()
    try:
        scores_by_word = {
            word: score
            for word in words
            if (score := lookup.get_score(word, l1)) is not None
        }
    except FileNotFoundError:
        result["kvl_oov_count"] = content_word_count
        return result

    lookup_count = len(scores_by_word)
    result["kvl_lookup_count"] = lookup_count
    result["kvl_oov_count"] = content_word_count - lookup_count
    result["kvl_lookup_coverage"] = lookup_count / content_word_count

    if not lookup_count:
        return result

    scores = list(scores_by_word.values())
    result["kvl_mean_score"] = round(sum(scores) / len(scores), 4)
    result["kvl_min_score"] = round(min(scores), 4)
    hard_count = sum(1 for score in scores if score < hard_threshold)
    result["kvl_pct_hard_words"] = round(hard_count / lookup_count, 4)
    return result
