"""Model wrapper registry and factory."""

from typing import Dict, Optional, Type

from slm_experiments.models.llamacpp import LlamaCppBaseWrapper
from slm_experiments.models.wrappers.phi3_llamacpp_wrapper import Phi3LlamaCppWrapper
from slm_experiments.models.wrappers.qwen2_llamacpp_wrapper import Qwen2LlamaCppWrapper
from slm_experiments.models.wrappers.qwen3_llamacpp_wrapper import Qwen3LlamaCppWrapper
from slm_experiments.models.wrappers.tinyllama_llamacpp_wrapper import (
    TinyLlamaLlamaCppWrapper,
)

MODEL_REGISTRY: Dict[str, Type[LlamaCppBaseWrapper]] = {
    "Qwen3": Qwen3LlamaCppWrapper,
    "Qwen2": Qwen2LlamaCppWrapper,
    "Phi3": Phi3LlamaCppWrapper,
    "TinyLlama": TinyLlamaLlamaCppWrapper,
}


def get_model_wrapper(
    name: str,
    seed: int = 42,
    model_path: Optional[str] = None,
    timeout_seconds: int = 300,
) -> LlamaCppBaseWrapper:
    """
    Instantiate a registered model wrapper by name.

    Args:
        name: Registry key (Qwen3, Qwen2, Phi3, TinyLlama).
        seed: Random seed passed to llama.cpp for reproducibility.
        model_path: Optional override for GGUF file path.
        timeout_seconds: Generation timeout.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    wrapper_cls = MODEL_REGISTRY[name]
    kwargs = {"seed": seed, "timeout_seconds": timeout_seconds}
    if model_path is not None:
        kwargs["model_path"] = model_path
    return wrapper_cls(**kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "get_model_wrapper",
    "Qwen3LlamaCppWrapper",
    "Qwen2LlamaCppWrapper",
    "Phi3LlamaCppWrapper",
    "TinyLlamaLlamaCppWrapper",
]
