"""Abstract base class for model wrappers."""

import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.prompts import build_contextual_prompt
from slm_experiments.evaluation.formatter import ResponseFormatter
from slm_experiments.evaluation.metrics import TextEvaluator

try:
    HAS_SIGNAL = hasattr(signal, "SIGALRM")
except ImportError:
    HAS_SIGNAL = False

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PACKAGE_DIR)))
DEFAULT_VOCAB_PATH = os.path.join(
    REPO_ROOT, "data", "vocabularies", "filtered_starters_vocab.txt"
)
_SKIP_VOCAB_ENTRIES = frozenset({"<|im_end|>", "<|endoftext|>"})
LOCAL_GGUF_DIR = os.path.join(REPO_ROOT, "models", "gguf")
THESIS_GGUF_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "..", "SLMs-master-thesis", "Tesis", "Codigo", "models", "gguf")
)


def resolve_gguf_dir() -> str:
    """
    Resolve directory containing GGUF model files.

    Search order:
    1. SLM_GGUF_DIR environment variable
    2. Local models/gguf/ if it contains .gguf files
    3. Sibling thesis repo (SLMs-master-thesis/Tesis/Codigo/models/gguf)
    4. Local models/gguf/ (may be empty)
    """
    env_dir = os.environ.get("SLM_GGUF_DIR")
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)

    if os.path.isdir(LOCAL_GGUF_DIR) and any(
        name.endswith(".gguf") for name in os.listdir(LOCAL_GGUF_DIR)
    ):
        return LOCAL_GGUF_DIR

    if os.path.isdir(THESIS_GGUF_DIR):
        return THESIS_GGUF_DIR

    return LOCAL_GGUF_DIR


class TimeoutError(Exception):
    """Custom timeout exception."""


@contextmanager
def timeout_context(seconds: int):
    """Context manager for generation timeouts (Unix SIGALRM or threading fallback)."""
    if HAS_SIGNAL:
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        start_time = time.time()

        class TimeoutTracker:
            def __init__(self):
                self.timed_out = False
                self.timer = None

            def timeout_callback(self):
                self.timed_out = True

            def start_timer(self):
                self.timer = threading.Timer(seconds, self.timeout_callback)
                self.timer.start()

            def stop_timer(self):
                if self.timer:
                    self.timer.cancel()

        tracker = TimeoutTracker()
        tracker.start_timer()
        try:
            yield
            if tracker.timed_out:
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        finally:
            tracker.stop_timer()


class BaseModelWrapper(ABC):
    """
    Abstract base for all model wrappers.

    Loads A1 vocabulary and provides shared evaluation utilities.
    """

    def __init__(
        self,
        model_name: str,
        timeout_seconds: int = 300,
        vocab_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.vocab_path = vocab_path or DEFAULT_VOCAB_PATH
        self.target_vocabulary = self._load_target_vocabulary()
        self.text_evaluator = TextEvaluator()
        self.response_formatter = ResponseFormatter()

    def _load_target_vocabulary(self) -> List[str]:
        """Load A1 vocabulary for probability weighting."""
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                vocab: List[str] = []
                for line in f:
                    word = line.strip().lower()
                    if not word or word in _SKIP_VOCAB_ENTRIES:
                        continue
                    if len(word) == 1 and not word.isalnum():
                        continue
                    vocab.append(word)
                return vocab
        except FileNotFoundError:
            return []

    def _add_simplification_context(self, prompt: str, num_shots: int = 0) -> str:
        """Prepend simplification instructions and optional shot examples."""
        return build_contextual_prompt(prompt, num_shots=num_shots)

    def generate(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate a response (ModelWrapper protocol).

        Returns response, response_time_seconds, generation_successful.
        """
        start_time = time.time()
        try:
            with timeout_context(self.timeout_seconds):
                result = self._generate_response_impl(prompt, config)
        except TimeoutError:
            return {
                "response": "",
                "response_time_seconds": float(self.timeout_seconds),
                "generation_successful": False,
                "error_message": (
                    f"Generation timed out after {self.timeout_seconds} seconds"
                ),
            }
        except Exception as exc:
            return {
                "response": "",
                "response_time_seconds": time.time() - start_time,
                "generation_successful": False,
                "error_message": str(exc),
            }

        response = result.get("response") or ""
        cleaned = self.response_formatter.clean_response_for_evaluation(response)
        successful = bool(result.get("generation_successful", False)) and bool(
            cleaned.strip()
        )

        return {
            "response": response,
            "response_time_seconds": float(
                result.get("response_time_seconds", time.time() - start_time)
            ),
            "generation_successful": successful,
            "error_message": result.get("error_message", ""),
        }

    @abstractmethod
    def _generate_response_impl(
        self, prompt: str, config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Subclass implementation of generation logic."""
        pass

    @abstractmethod
    def _initialize_model(self):
        """Initialize the underlying model once during wrapper creation."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vocab_size": len(self.target_vocabulary),
            "supports_weighting": True,
            "supports_context_prompting": True,
        }
