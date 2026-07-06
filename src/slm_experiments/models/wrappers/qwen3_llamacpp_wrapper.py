"""Qwen3-0.6B llama.cpp wrapper (ChatML, enable_thinking=False via Jinja)."""

from typing import Any, Dict, List, Optional

from slm_experiments.models.llamacpp import LlamaCppBaseWrapper, default_gguf_path

try:
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    JINJA_FORMATTER_AVAILABLE = True
except ImportError:
    JINJA_FORMATTER_AVAILABLE = False
    Jinja2ChatFormatter = None  # type: ignore


class Qwen3LlamaCppWrapper(LlamaCppBaseWrapper):
    """Qwen3-0.6B GGUF wrapper with thinking mode disabled via enable_thinking=False."""

    DEFAULT_GGUF = "Qwen3-0.6B-Q4_0.gguf"
    _IM_END = "<|im_end|>"
    _EOS_TOKEN = "<|im_end|>"
    _BOS_TOKEN = ""

    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 42,
        timeout_seconds: int = 300,
    ):
        self._jinja_formatter: Optional[Any] = None
        super().__init__(
            model_name="Qwen3",
            model_path=model_path or default_gguf_path(self.DEFAULT_GGUF),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )

    def _get_chat_template(self) -> Optional[str]:
        if self.llm is None:
            return None
        metadata = getattr(self.llm, "metadata", None) or {}
        template = metadata.get("tokenizer.chat_template")
        if isinstance(template, str) and template.strip():
            return template
        return None

    def _get_jinja_formatter(self) -> Optional[Any]:
        if self._jinja_formatter is not None:
            return self._jinja_formatter
        template = self._get_chat_template()
        if template is None or not JINJA_FORMATTER_AVAILABLE:
            return None
        self._jinja_formatter = Jinja2ChatFormatter(
            template,
            eos_token=self._EOS_TOKEN,
            bos_token=self._BOS_TOKEN,
            add_generation_prompt=True,
        )
        return self._jinja_formatter

    def _format_prompt_fallback(self, user_input: str, system_prompt: str) -> str:
        """Manual ChatML when GGUF metadata or Jinja formatter is unavailable."""
        im_end = self._IM_END
        return (
            f"<|im_start|>system\n"
            f"{system_prompt}{im_end}\n"
            f"<|im_start|>user\n"
            f"{user_input}{im_end}\n"
            f"<|im_start|>assistant\n"
            f"<think>\n\n</think>\n\n"
        )

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """Render ChatML via GGUF Jinja template with enable_thinking=False."""
        formatter = self._get_jinja_formatter()
        if formatter is None:
            return self._format_prompt_fallback(user_input, system_prompt)

        response = formatter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            enable_thinking=False,
        )
        return response.prompt

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
