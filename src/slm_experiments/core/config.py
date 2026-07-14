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
    config_guided: bool = False
    weight_factor: float = 1.0
    num_shots: int = 0

    guided_top_k: int = 10
    guided_mode: str = "flat"

    prompt_id: str = ""

    temperature: float = 0.0
    top_k: int = 50
    max_new_tokens: int = 200

    experiment_name: str = "default_experiment"
    description: str = ""

    kvl_l1: str = "es"

    config_kvl_beam: bool = False
    kvl_beam_width: int = 4
    kvl_branch_factor: int = 10

    # CEFR-SP secondary metric (default ON; requires [cefr-sp] extras + ckpt)
    enable_cefr_sp: bool = True
    cefr_sp_ckpt_path: str = ""
    cefr_sp_device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


def cefr_sp_config_kwargs(
    *,
    enable_cefr_sp: bool = True,
    cefr_sp_ckpt_path: str = "",
    cefr_sp_device: str = "cpu",
) -> Dict[str, Any]:
    """Fields to pass into ``dataclasses.replace`` for CEFR-SP CLI options."""
    return {
        "enable_cefr_sp": bool(enable_cefr_sp),
        "cefr_sp_ckpt_path": cefr_sp_ckpt_path or "",
        "cefr_sp_device": cefr_sp_device or "cpu",
    }
