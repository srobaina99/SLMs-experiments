"""Factorial experiment configuration factory."""

from typing import List

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import MODEL_CONFIGS

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful English teacher for beginner students. "
    "Answer with a paragraph only with plain text"
)

DEFAULT_WEIGHT_FACTOR = 1.5

INTERVENTIONS = [
    (False, False),  # control
    (True, False),   # weighting_only
    (False, True),   # prompting_only
    (True, True),    # both
]


def _config_name(model_name: str, config_weighting: bool, config_prompting: bool) -> str:
    if config_weighting and config_prompting:
        return f"{model_name}_weighted_prompted"
    if config_weighting:
        return f"{model_name}_weighted"
    if config_prompting:
        return f"{model_name}_prompted"
    return f"{model_name}_control"


def create_factorial_configs() -> List[ExperimentConfig]:
    """
    Create all factorial experiment configurations.

    Returns 4 models × 4 intervention combinations = 16 ExperimentConfig objects.
    """
    configs: List[ExperimentConfig] = []

    for model_name, model_info in MODEL_CONFIGS.items():
        for config_weighting, config_prompting in INTERVENTIONS:
            name = _config_name(model_name, config_weighting, config_prompting)
            configs.append(
                ExperimentConfig(
                    model_name=model_info["model_name"],
                    model_id=model_info["model_id"],
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    config_weighting=config_weighting,
                    config_prompting=config_prompting,
                    weight_factor=DEFAULT_WEIGHT_FACTOR,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    max_new_tokens=200,
                    experiment_name=name,
                    description=(
                        f"Factorial experiment: {model_name} with "
                        f"weighting={config_weighting}, prompting={config_prompting}"
                    ),
                )
            )

    return configs


def get_config_by_name(config_name: str) -> ExperimentConfig:
    """Return a single factorial config by experiment_name."""
    for config in create_factorial_configs():
        if config.experiment_name == config_name:
            return config

    available = [c.experiment_name for c in create_factorial_configs()]
    raise ValueError(
        f"Configuration '{config_name}' not found. Available: {available}"
    )


def get_configs_for_model(model_name: str) -> List[ExperimentConfig]:
    """Return all factorial configs for one model."""
    return [c for c in create_factorial_configs() if c.model_name == model_name]
