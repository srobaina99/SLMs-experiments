"""TinyLlama-1.1B llama.cpp wrapper."""

from typing import Any, Dict, List, Optional

from slm_experiments.models.llamacpp import LlamaCppBaseWrapper, default_gguf_path


class TinyLlamaLlamaCppWrapper(LlamaCppBaseWrapper):
    """TinyLlama-1.1B-Chat GGUF wrapper."""

    DEFAULT_GGUF = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 42,
        timeout_seconds: int = 300,
    ):
        super().__init__(
            model_name="TinyLlama",
            model_path=model_path or default_gguf_path(self.DEFAULT_GGUF),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        return (
            f"<|system|>\n"
            f"{system_prompt}\n"
            f"<|user|>\n"
            f"{user_input}\n"
            f"<|assistant|>"
        )

    def _get_stop_tokens(self) -> List[str]:
        return ["<|user|>", "<|system|>", "</s>"]

    def _extract_response(self, raw_output: str) -> str:
        response = raw_output.strip()
        for marker in ("<|system|>", "<|user|>", "<|assistant|>", "</s>"):
            response = response.replace(marker, "")
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return " ".join(lines)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "quantization": "Q4_0",
                "template_format": "TinyLlama",
                "parameters": "1.1B",
            }
        )
        return info
