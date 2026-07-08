"""Integration smoke tests for guided decoding on real GGUF models."""

from __future__ import annotations

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS


@pytest.mark.slow
def test_generate_guided_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        config_prompting=True,
        config_guided=True,
        temperature=0.0,
        top_k=50,
        guided_top_k=10,
        guided_mode="flat",
        max_new_tokens=20,
    )
    result = qwen3_wrapper.generate_guided(STANDARD_PROMPTS[0], config)

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["guided_top_k"] == 10
    assert result["guided_steps_total"] > 0
    assert result["response_time_seconds"] > 0
