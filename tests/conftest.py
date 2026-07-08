"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

# Make src/ importable without installation during development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slm_experiments.models import get_model_wrapper


def gguf_available() -> bool:
    try:
        wrapper = get_model_wrapper("Qwen3", seed=42)
        return bool(wrapper.model_loaded and wrapper.llm is not None)
    except Exception:
        return False


@pytest.fixture
def qwen3_wrapper():
    if not gguf_available():
        pytest.skip("GGUF not present")
    wrapper = get_model_wrapper("Qwen3", seed=42)
    yield wrapper
    if hasattr(wrapper, "cleanup"):
        wrapper.cleanup()
