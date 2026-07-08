"""Integration smoke tests for standard generate() on real GGUF models."""

from __future__ import annotations

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import STANDARD_PROMPTS
from slm_experiments.phase1.configs import DEFAULT_SYSTEM_PROMPT, DEFAULT_WEIGHT_FACTOR


@pytest.mark.slow
def test_generate_phase1_control_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        config_weighting=False,
        config_prompting=False,
        temperature=0.0,
        top_k=50,
        max_new_tokens=20,
    )
    result = qwen3_wrapper.generate(STANDARD_PROMPTS[0], config)

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["response_time_seconds"] > 0


@pytest.mark.slow
def test_generate_phase1_both_interventions_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        config_weighting=True,
        config_prompting=True,
        weight_factor=DEFAULT_WEIGHT_FACTOR,
        temperature=0.0,
        top_k=50,
        max_new_tokens=20,
    )
    result = qwen3_wrapper.generate(STANDARD_PROMPTS[0], config)

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["response_time_seconds"] > 0


@pytest.mark.slow
def test_generate_weights_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        config_weighting=True,
        config_prompting=True,
        weight_factor=2.0,
        num_shots=0,
        temperature=0.0,
        top_k=50,
        max_new_tokens=20,
    )
    result = qwen3_wrapper.generate(STANDARD_PROMPTS[0], config)

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["response_time_seconds"] > 0


@pytest.mark.slow
def test_generate_prompting_smoke(qwen3_wrapper):
    config = ExperimentConfig(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        config_weighting=False,
        config_prompting=True,
        weight_factor=1.0,
        num_shots=1,
        temperature=0.0,
        top_k=50,
        max_new_tokens=20,
    )
    result = qwen3_wrapper.generate(STANDARD_PROMPTS[0], config)

    assert result["generation_successful"] is True, result.get("error_message")
    assert result["response"].strip()
    assert result["response_time_seconds"] > 0
