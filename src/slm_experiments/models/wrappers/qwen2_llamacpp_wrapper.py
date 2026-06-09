"""Qwen2.5-0.5B llama.cpp wrapper (ChatML)."""

from typing import Any, Dict, List, Optional

from slm_experiments.models.llamacpp import LlamaCppBaseWrapper, default_gguf_path


class Qwen2LlamaCppWrapper(LlamaCppBaseWrapper):
    """Qwen2.5-0.5B-Instruct GGUF wrapper."""

    DEFAULT_GGUF = "qwen2.5-0.5b-instruct-q4_0.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 42,
        timeout_seconds: int = 300,
    ):
        super().__init__(
            model_name="Qwen2",
            model_path=model_path or default_gguf_path(self.DEFAULT_GGUF),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        im_end = "<|im_end|>"
        return (
            f"<|im_start|>system\n"
            f"{system_prompt}{im_end}\n"
            f"<|im_start|>user\n"
            f"{user_input}{im_end}\n"
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
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "quantization": "Q4_0",
                "template_format": "ChatML",
                "parameters": "0.5B",
            }
        )
        return info
