"""Assessment scoring seam — scorers register here for ``assess build``."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

ScorerFn = Callable[..., pd.DataFrame]

_REGISTRY: Dict[str, ScorerFn] = {}


def register_scorer(
    name: str, *, replace: bool = False
) -> Callable[[ScorerFn], ScorerFn]:
    """Decorator that registers an assessment scorer by name.

    When ``replace=False`` (default), a duplicate name raises. Use
    ``ensure_scorer_registered`` for idempotent re-registration after
    ``clear_scorers`` in tests.
    """

    def decorator(fn: ScorerFn) -> ScorerFn:
        if name in _REGISTRY and not replace:
            raise ValueError(f"scorer already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def ensure_scorer_registered(name: str, fn: ScorerFn) -> bool:
    """Idempotent register-or-skip. Returns True if newly registered.

    Public alternative to mutating ``_REGISTRY`` from scorer modules after
    ``clear_scorers()``. Does not replace an existing entry.
    """
    if name in _REGISTRY:
        return False
    _REGISTRY[name] = fn
    return True


def list_scorers() -> List[str]:
    """Return registered scorer names in registration order."""
    return list(_REGISTRY.keys())


def score_items(
    items: pd.DataFrame,
    *,
    scorer_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Run all registered scorers over ``items`` and outer-merge on ``item_id``.

    ``scorer_options`` maps scorer name → kwargs forwarded to that scorer
    (e.g. ``{"cefr_tsar": {"device": "cpu", "batch_size": 8}}``). Scorers that
    do not accept a given kwarg should ignore extras.

    With an empty registry, returns a frame with only ``item_id`` when that
    column is present, otherwise an empty frame.
    """
    if "item_id" not in items.columns:
        return pd.DataFrame()

    options = scorer_options or {}
    result = items[["item_id"]].drop_duplicates().reset_index(drop=True)
    for name in list_scorers():
        kwargs = options.get(name) or {}
        scored = _REGISTRY[name](items, **kwargs) if kwargs else _REGISTRY[name](items)
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
