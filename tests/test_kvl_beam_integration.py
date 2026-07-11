"""Integration smoke tests for KVL beam search on real GGUF models."""

from __future__ import annotations

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS


@pytest.mark.slow
def test_generate_kvl_beam_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        config_prompting=True,
        temperature=0.0,
        top_k=50,
        max_new_tokens=20,
        kvl_l1="es",
    )
    result = qwen3_wrapper.generate_kvl_beam(
        STANDARD_PROMPTS[0],
        config,
        beam_width=2,
        branch_factor=5,
    )

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["kvl_beam_width"] == 2
    assert result["kvl_branch_factor"] == 5
    assert result["kvl_beam_steps_total"] > 0
    assert result["response_time_seconds"] > 0


@pytest.mark.slow
def test_generate_kvl_beam_tinyllama_not_empty_early_eos(tinyllama_wrapper):
    """TinyLlama assigns high early EOS probability; must not return empty."""
    config = ExperimentConfig(
        config_prompting=True,
        temperature=0.0,
        top_k=50,
        max_new_tokens=50,
        kvl_l1="es",
    )
    result = tinyllama_wrapper.generate_kvl_beam(
        STANDARD_PROMPTS[0],
        config,
        beam_width=4,
        branch_factor=10,
    )

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["kvl_beam_steps_total"] >= 3
