"""Shared pytest fixtures."""

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

# Make src/ importable without installation during development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slm_experiments.models import get_model_wrapper


class _UnitTestA1CefrSpScorer:
    """Stub scorer: every sentence is A1 (avoids loading the 1.2GB ckpt)."""

    def score_sentences(self, sentences: Sequence[str]) -> List[Dict[str, object]]:
        return [
            {"label": 0, "probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]} for _ in sentences
        ]


def gguf_available() -> bool:
    try:
        wrapper = get_model_wrapper("Qwen3", seed=42)
        return bool(wrapper.model_loaded and wrapper.llm is not None)
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _disable_real_cefr_sp_in_unit_tests(monkeypatch, request):
    """Keep unit tests free of the 1.2GB CEFR-SP load.

    Production / CLI defaults remain ``enable_cefr_sp=True``. Dedicated CEFR-SP
    tests in ``test_cefr_sp_metrics.py`` exercise scoring with mocks or the
    optional real checkpoint.

    Non-CEFR tests resolve to an A1 stub scorer so ``meets_a1_criteria`` (now
    CEFR-SP-gated) still passes on simple successful generations. Inject
    ``ExperimentPipeline(cefr_sp_scorer=...)`` to override.
    """
    path = str(getattr(request.node, "path", "") or request.node.fspath)
    if "test_cefr_sp_metrics" in path:
        return

    from slm_experiments.core.pipeline import ExperimentPipeline

    def _resolve(self, config):
        if self.cefr_sp_scorer is not None:
            return self.cefr_sp_scorer
        if not config.enable_cefr_sp:
            return None
        return _UnitTestA1CefrSpScorer()

    monkeypatch.setattr(ExperimentPipeline, "_resolve_cefr_sp_scorer", _resolve)


@pytest.fixture
def qwen3_wrapper():
    if not gguf_available():
        pytest.skip("GGUF not present")
    wrapper = get_model_wrapper("Qwen3", seed=42)
    yield wrapper
    if hasattr(wrapper, "cleanup"):
        wrapper.cleanup()


@pytest.fixture
def tinyllama_wrapper():
    if not gguf_available():
        pytest.skip("GGUF not present")
    wrapper = get_model_wrapper("TinyLlama", seed=42)
    yield wrapper
    if hasattr(wrapper, "cleanup"):
        wrapper.cleanup()
