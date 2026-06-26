"""Experiment configuration dataclass."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    model_name: str = "Qwen3"
    model_id: str = "unsloth/Qwen3-0.6B"
    system_prompt: str = "You are a helpful English teacher for beginner students."

    config_weighting: bool = False
    config_prompting: bool = False
    weight_factor: float = 1.0
    num_shots: int = 0

    prompt_id: str = ""

    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 200

    experiment_name: str = "default_experiment"
    description: str = ""

    kvl_l1: str = "es"

    config_kvl_beam: bool = False
    kvl_beam_width: int = 4
    kvl_branch_factor: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
