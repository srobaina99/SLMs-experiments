"""Integration smoke tests for deprecated beam search on real GGUF models."""

from __future__ import annotations

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS


@pytest.mark.slow
def test_generate_beam_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        config_weighting=False,
        config_prompting=True,
        num_shots=0,
        temperature=0.0,
        top_k=50,
        max_new_tokens=20,
    )
    result = qwen3_wrapper.generate_beam(
        STANDARD_PROMPTS[0],
        config,
        beam_width=4,
        selection_method="a1_ratio",
    )

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["beam_width"] == 4
    assert result["response_time_seconds"] > 0
