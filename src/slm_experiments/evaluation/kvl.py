"""KVL/GLMM learner vocabulary difficulty metrics."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Set

KVL_HARD_THRESHOLD = -1.0
KVL_V2_LOWER_TAIL_PERCENTILE = 10.0

SUPPORTED_L1S = ("es", "de", "cn")
DEFAULT_KVL_L1 = "es"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PACKAGE_DIR)))
_DEFAULT_KVL_DATA_DIR = os.path.join(_REPO_ROOT, "data", "kvl")


def empty_kvl_metrics(l1: str = DEFAULT_KVL_L1) -> Dict[str, object]:
    """Return safe default KVL v1 metrics for failed or empty generations."""
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


def empty_kvl_v2_metrics(l1: str = DEFAULT_KVL_L1) -> Dict[str, object]:
    """Return safe default KVL v2 metrics (occurrence-level, lemmatized)."""
    return {
        "kvl_v2_l1": l1,
        "kvl_v2_token_count": 0,
        "kvl_v2_lookup_count": 0,
        "kvl_v2_oov_count": 0,
        "kvl_v2_lookup_coverage": 0.0,
        "kvl_v2_mean_score": None,
        "kvl_v2_hard_token_share": None,
        "kvl_v2_lower_tail_score": None,
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


def _percentile_nearest_rank(scores: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile (inclusive). ``percentile`` in [0, 100]."""
    if not scores:
        raise ValueError("scores must be non-empty")
    if percentile <= 0:
        return float(min(scores))
    if percentile >= 100:
        return float(max(scores))
    ordered = sorted(float(s) for s in scores)
    # Nearest-rank: rank = ceil(p/100 * N), 1-indexed.
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def compute_kvl_metrics(
    text: str,
    l1: str,
    content_words: Optional[Set[str]] = None,
    *,
    kvl_lookup: Optional[KvlLookup] = None,
    hard_threshold: float = KVL_HARD_THRESHOLD,
) -> Dict[str, object]:
    """Compute KVL v1 vocabulary difficulty metrics for text.

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


def compute_kvl_v2_metrics(
    text: str,
    l1: str,
    tokens: Optional[Sequence[str]] = None,
    *,
    kvl_lookup: Optional[KvlLookup] = None,
    hard_threshold: float = KVL_HARD_THRESHOLD,
    lower_tail_percentile: float = KVL_V2_LOWER_TAIL_PERCENTILE,
    lemmatize: bool = True,
) -> Dict[str, object]:
    """Compute occurrence-level lemmatized KVL v2 metrics.

    Scores **every** content-word occurrence (not the unique surface-form set).
    When ``tokens`` is omitted, extracts POS-aware lemmatized tokens via
    ``TextEvaluator.extract_content_word_tokens``.

    Aggregates (mean / hard-token share / lower-tail) are computed only over
    looked-up tokens. **Never interpret ``kvl_v2_mean_score`` without also
    reporting ``kvl_v2_lookup_coverage``.**
    """
    if tokens is None:
        from slm_experiments.evaluation.metrics import TextEvaluator

        token_list: List[str] = TextEvaluator().extract_content_word_tokens(
            text, lemmatize=lemmatize
        )
    else:
        token_list = [str(t).lower() for t in tokens if str(t).strip()]

    token_count = len(token_list)
    result = empty_kvl_v2_metrics(l1)
    result["kvl_v2_token_count"] = token_count

    if not token_count:
        return result

    lookup = kvl_lookup or KvlLookup()
    scores: List[float] = []
    try:
        for token in token_list:
            score = lookup.get_score(token, l1)
            if score is not None:
                scores.append(float(score))
    except FileNotFoundError:
        result["kvl_v2_oov_count"] = token_count
        return result

    lookup_count = len(scores)
    result["kvl_v2_lookup_count"] = lookup_count
    result["kvl_v2_oov_count"] = token_count - lookup_count
    result["kvl_v2_lookup_coverage"] = lookup_count / token_count

    if not lookup_count:
        return result

    result["kvl_v2_mean_score"] = round(sum(scores) / lookup_count, 4)
    hard_count = sum(1 for score in scores if score < hard_threshold)
    result["kvl_v2_hard_token_share"] = round(hard_count / lookup_count, 4)
    result["kvl_v2_lower_tail_score"] = round(
        _percentile_nearest_rank(scores, lower_tail_percentile), 4
    )
    return result
