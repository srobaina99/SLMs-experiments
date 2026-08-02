"""Assessment scoring seam — scorers register here for ``assess build``."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pandas as pd

ScorerFn = Callable[[pd.DataFrame], pd.DataFrame]

_REGISTRY: Dict[str, ScorerFn] = {}


def register_scorer(name: str) -> Callable[[ScorerFn], ScorerFn]:
    """Decorator that registers an assessment scorer by name."""

    def decorator(fn: ScorerFn) -> ScorerFn:
        if name in _REGISTRY:
            raise ValueError(f"scorer already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def list_scorers() -> List[str]:
    """Return registered scorer names in registration order."""
    return list(_REGISTRY.keys())


def score_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Run all registered scorers over ``items`` and outer-merge on ``item_id``.

    With an empty registry, returns a frame with only ``item_id`` when that
    column is present, otherwise an empty frame.
    """
    if "item_id" not in items.columns:
        return pd.DataFrame()

    result = items[["item_id"]].drop_duplicates().reset_index(drop=True)
    for name in list_scorers():
        scored = _REGISTRY[name](items)
        if scored is None or scored.empty:
            continue
        if "item_id" not in scored.columns:
            raise ValueError(f"scorer {name!r} must return an item_id column")
        result = result.merge(scored, on="item_id", how="left")
    return result


def clear_scorers() -> None:
    """Remove all registered scorers (test helper)."""
    _REGISTRY.clear()


def scorer_revisions() -> Dict[str, Any]:
    """Pinned scorer metadata for assessment manifests."""
    revisions: Dict[str, Any] = {}
    for name in list_scorers():
        if name == "cefr_tsar":
            try:
                from slm_experiments.evaluation.assessment.cefr_tsar import (
                    tsar_scorer_revision,
                )

                revisions[name] = tsar_scorer_revision()
                continue
            except ImportError:
                pass
        if name == "kvl_v2":
            try:
                from slm_experiments.evaluation.assessment.kvl_v2 import (
                    kvl_v2_scorer_revision,
                )

                revisions[name] = kvl_v2_scorer_revision()
                continue
            except ImportError:
                pass
        revisions[name] = {"registered": True}
    return revisions
