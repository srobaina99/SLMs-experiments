"""Integration smoke tests for KVL beam search on real GGUF models."""

from __future__ import annotations

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS
from slm_experiments.models import get_model_wrapper


def gguf_available() -> bool:
    try:
        wrapper = get_model_wrapper("Qwen3", seed=42)
        return bool(wrapper.model_loaded and wrapper.llm is not None)
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not gguf_available(), reason="GGUF not present")
def test_generate_kvl_beam_smoke():
    config = ExperimentConfig(
        config_prompting=True,
        temperature=0.0,
        top_k=50,
        top_p=0.95,
        max_new_tokens=20,
        kvl_l1="es",
    )
    wrapper = get_model_wrapper("Qwen3", seed=42)
    result = wrapper.generate_kvl_beam(
        STANDARD_PROMPTS[0],
        config,
        beam_width=2,
        branch_factor=5,
    )

    assert result["generation_successful"] is True
    assert result["response"].strip()
    assert result["kvl_beam_width"] == 2
    assert result["kvl_branch_factor"] == 5
    assert result["kvl_beam_steps_total"] > 0
    assert result["response_time_seconds"] > 0
