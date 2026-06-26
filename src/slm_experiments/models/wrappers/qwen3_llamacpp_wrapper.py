"""Qwen3-0.6B llama.cpp wrapper (ChatML, /nothink default)."""

from typing import Any, Dict, List, Optional

from slm_experiments.models.llamacpp import LlamaCppBaseWrapper, default_gguf_path


class Qwen3LlamaCppWrapper(LlamaCppBaseWrapper):
    """Qwen3-0.6B GGUF wrapper with thinking mode disabled via /nothink."""

    DEFAULT_GGUF = "Qwen3-0.6B-Q4_0.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 42,
        timeout_seconds: int = 300,
    ):
        super().__init__(
            model_name="Qwen3",
            model_path=model_path or default_gguf_path(self.DEFAULT_GGUF),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """ChatML template; append /nothink to disable Qwen3 thinking mode."""
        im_end = "<|im_end|>"
        return (
            f"<|im_start|>system\n"
            f"{system_prompt}{im_end}\n"
            f"<|im_start|>user\n"
            f"{user_input} /nothink{im_end}\n"
            f"<|im_start|>assistant\n"
        )

    def _get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>"]

    def _extract_response(self, raw_output: str) -> str:
        response = raw_output.strip()

        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>")
            response = response[think_end + len("</think>") :].strip()

        for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            response = response.replace(marker, "")

        return response.strip()

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_id": "ggml-org/Qwen3-0.6B-GGUF",
                "quantization": "Q4_0",
                "template_format": "ChatML",
                "parameters": "0.6B",
            }
        )
        return info
