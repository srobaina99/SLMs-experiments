"""Model wrappers for llama.cpp GGUF inference."""

from slm_experiments.models.base import BaseModelWrapper, REPO_ROOT, resolve_gguf_dir
from slm_experiments.models.beam import BeamCandidate, BeamSearchGenerator
from slm_experiments.models.llamacpp import LlamaCppBaseWrapper, default_gguf_path
from slm_experiments.models.wrappers import MODEL_REGISTRY, get_model_wrapper

__all__ = [
    "BaseModelWrapper",
    "REPO_ROOT",
    "resolve_gguf_dir",
    "BeamCandidate",
    "BeamSearchGenerator",
    "LlamaCppBaseWrapper",
    "default_gguf_path",
    "MODEL_REGISTRY",
    "get_model_wrapper",
]
