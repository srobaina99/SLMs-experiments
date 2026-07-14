"""CEFR-SP sentence difficulty metrics (Arase et al., optional secondary metric)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

CEFR_SP_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_SP_NUM_LABELS = 6

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PACKAGE_DIR)))
_DEFAULT_CEFR_SP_DIR = os.path.join(_REPO_ROOT, "data", "cefr_sp")
DEFAULT_CEFR_SP_CKPT = os.path.join(_DEFAULT_CEFR_SP_DIR, "level_estimator.ckpt")


def empty_cefr_sp_metrics(*, enabled: bool = False) -> Dict[str, object]:
    """Return safe default CEFR-SP metrics for disabled, failed, or empty text."""
    return {
        "cefr_sp_enabled": enabled,
        "cefr_sp_sentence_count": 0,
        "cefr_sp_level": None,
        "cefr_sp_level_ordinal": None,
        "cefr_sp_max_level_ordinal": None,
        "cefr_sp_pct_a1": None,
        "cefr_sp_adjacency": None,
        "cefr_sp_expected_level": None,
    }


def ordinal_to_level(ordinal: float) -> str:
    """Map a 0–5 ordinal (possibly fractional mean) to A1–C2 by nearest label."""
    idx = int(round(ordinal))
    idx = max(0, min(CEFR_SP_NUM_LABELS - 1, idx))
    return CEFR_SP_LEVELS[idx]


def sentence_word_lists(sentences: Sequence[str]) -> List[List[str]]:
    """Whitespace-split sentences to match CEFR-SP training tokenization."""
    word_lists: List[List[str]] = []
    for sent in sentences:
        words = sent.split()
        word_lists.append(words if words else [sent])
    return word_lists


def _ensure_nltk_punkt() -> None:
    import nltk

    for resource, find_path in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ):
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(resource, quiet=True)


def _split_sentences(text: str) -> List[str]:
    import nltk

    _ensure_nltk_punkt()
    sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    return sentences


@runtime_checkable
class SentenceScorer(Protocol):
    """Minimal interface for CEFR-SP (or mock) sentence scoring."""

    def score_sentences(
        self, sentences: Sequence[str]
    ) -> List[Dict[str, object]]:
        """Return per-sentence dicts with ``label`` (0–5) and ``probs`` (len 6)."""
        ...


class CefrSpScorer:
    """Lazy-loading wrapper around the official CEFR-SP contrastive checkpoint."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cpu",
    ):
        env_path = os.environ.get("CEFR_SP_CKPT", "").strip()
        self.ckpt_path = ckpt_path or env_path or DEFAULT_CEFR_SP_CKPT
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(
                f"CEFR-SP checkpoint not found: {self.ckpt_path}. "
                "Run: ./venv/bin/python scripts/download_cefr_sp_ckpt.py"
            )

        try:
            import inspect

            import torch
            from slm_experiments.evaluation.cefr_sp_vendor.model import (
                LevelEstimaterContrastive,
            )
        except ImportError as exc:
            raise ImportError(
                "CEFR-SP scoring requires optional extras. "
                'Install with: pip install -e ".[cefr-sp]"'
            ) from exc

        try:
            # Manual load (not Lightning load_from_checkpoint alone):
            # 1) Zenodo hparams bake `../pretrained_model/bert-base-cased/`;
            #    override to the Hub id before constructing the module.
            # 2) Older transformers stored `lm.embeddings.position_ids` as a
            #    buffer; current BertModel does not — drop that key so
            #    strict=True still refuses real missing/unexpected weights.
            # Upstream level_estimator.py is unsafe for inference: after
            # load_from_checkpoint it reinstantiates a fresh untrained model.
            checkpoint = torch.load(
                self.ckpt_path,
                map_location=self.device,
                weights_only=False,
            )
            hp = dict(checkpoint["hyper_parameters"])
            hp["pretrained_model"] = "bert-base-cased"

            sig = inspect.signature(LevelEstimaterContrastive.__init__)
            init_kwargs = {
                name: hp[name]
                for name in sig.parameters
                if name != "self" and name in hp
            }
            model = LevelEstimaterContrastive(**init_kwargs)

            state_dict = {
                key: value
                for key, value in checkpoint["state_dict"].items()
                if key != "lm.embeddings.position_ids"
            }
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CEFR-SP checkpoint from {self.ckpt_path}: {exc}. "
                "If this is a pytorch-lightning / torch mismatch, see "
                "data/cefr_sp/README.md (try pinning pytorch-lightning to a "
                "2.0–2.x line compatible with your torch)."
            ) from exc

        model.eval()
        model.to(self.device)
        self._model = model

    def score_sentences(
        self, sentences: Sequence[str]
    ) -> List[Dict[str, object]]:
        """Score sentences with training-matched whitespace + is_split_into_words."""
        if not sentences:
            return []

        self._ensure_loaded()
        import torch

        model = self._model
        assert model is not None

        word_lists = sentence_word_lists(sentences)

        encoded = model.tokenizer(
            word_lists,
            return_tensors="pt",
            padding=True,
            is_split_into_words=True,
        )
        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            logits = model.contrastive_logits(batch)
            probs = torch.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)

        results: List[Dict[str, object]] = []
        for i in range(len(sentences)):
            prob_row = probs[i].detach().cpu().tolist()
            results.append(
                {
                    "label": int(labels[i].item()),
                    "probs": [float(p) for p in prob_row],
                }
            )
        return results


def compute_cefr_sp_metrics(
    text: str,
    scorer: Optional[SentenceScorer] = None,
    *,
    enabled: bool = True,
    ckpt_path: str = "",
    device: str = "cpu",
) -> Dict[str, object]:
    """Compute document-level CEFR-SP aggregates from sentence scores.

    When ``enabled`` is False, returns the empty schema without importing torch.
    """
    if not enabled:
        return empty_cefr_sp_metrics(enabled=False)

    if not text or not str(text).strip():
        return empty_cefr_sp_metrics(enabled=True)

    sentences = _split_sentences(str(text))
    if not sentences:
        return empty_cefr_sp_metrics(enabled=True)

    active_scorer = scorer or CefrSpScorer(
        ckpt_path=ckpt_path or None,
        device=device or "cpu",
    )
    scored = active_scorer.score_sentences(sentences)
    if not scored:
        return empty_cefr_sp_metrics(enabled=True)
    if len(scored) != len(sentences):
        raise ValueError(
            f"CEFR-SP scorer returned {len(scored)} scores for "
            f"{len(sentences)} sentences"
        )

    labels = [int(item["label"]) for item in scored]
    expected_levels = []
    for item in scored:
        probs = list(item["probs"])  # type: ignore[arg-type]
        if len(probs) != CEFR_SP_NUM_LABELS:
            raise ValueError(
                f"Expected {CEFR_SP_NUM_LABELS} class probs, got {len(probs)}"
            )
        expected_levels.append(sum(i * float(p) for i, p in enumerate(probs)))

    mean_ordinal = sum(labels) / len(labels)
    max_ordinal = max(labels)
    pct_a1 = sum(1 for lv in labels if lv == 0) / len(labels)
    adjacency = sum(1 for lv in labels if lv <= 1) / len(labels)
    mean_expected = sum(expected_levels) / len(expected_levels)

    return {
        "cefr_sp_enabled": True,
        "cefr_sp_sentence_count": len(sentences),
        "cefr_sp_level": ordinal_to_level(mean_ordinal),
        "cefr_sp_level_ordinal": round(mean_ordinal, 4),
        "cefr_sp_max_level_ordinal": int(max_ordinal),
        "cefr_sp_pct_a1": round(pct_a1, 4),
        "cefr_sp_adjacency": round(adjacency, 4),
        "cefr_sp_expected_level": round(mean_expected, 4),
    }
