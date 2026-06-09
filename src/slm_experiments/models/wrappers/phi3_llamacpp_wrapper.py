"""Phi-3-mini llama.cpp wrapper (custom template, GPU offload)."""

from typing import Any, Dict, List, Optional

from slm_experiments.models.llamacpp import LlamaCppBaseWrapper, default_gguf_path


class Phi3LlamaCppWrapper(LlamaCppBaseWrapper):
    """Phi-3-mini-4k-instruct GGUF wrapper."""

    DEFAULT_GGUF = "Phi-3-mini-4k-instruct-q4.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 42,
        timeout_seconds: int = 300,
        n_gpu_layers: int = -1,
    ):
        super().__init__(
            model_name="Phi3",
            model_path=model_path or default_gguf_path(self.DEFAULT_GGUF),
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=n_gpu_layers,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        return (
            f"<|system|>\n"
            f"{system_prompt}<|end|>\n"
            f"<|user|>\n"
            f"{user_input}<|end|>\n"
            f"<|assistant|>\n"
        )

    def _get_stop_tokens(self) -> List[str]:
        return ["<|end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"]

    def _extract_response(self, raw_output: str) -> str:
        response = raw_output.strip()
        for marker in (
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "<|endoftext|>",
        ):
            response = response.replace(marker, "")
        return response.strip()

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
                "quantization": "Q4",
                "template_format": "Phi-3 (custom)",
                "parameters": "3.8B",
            }
        )
        return info
