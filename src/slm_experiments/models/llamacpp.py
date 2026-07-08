"""Base class for llama.cpp GGUF model wrappers."""

import math
import os
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.evaluation.kvl import KvlLookup
from slm_experiments.evaluation.metrics import TextEvaluator
from slm_experiments.models.base import (
    BaseModelWrapper,
    TimeoutError,
    resolve_gguf_dir,
    timeout_context,
)
from slm_experiments.models.a1_token_index import A1TokenIndex
from slm_experiments.models.beam import BeamSearchGenerator
from slm_experiments.models.constrained_decoder import ConstrainedDecoder
from slm_experiments.models.kvl_beam_decoder import (
    KvlBeamDecoder,
    make_llamacpp_eval_fn,
    resolve_llamacpp_stop_token_ids,
)

try:
    from llama_cpp import Llama

    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    Llama = None  # type: ignore


class LlamaCppBaseWrapper(BaseModelWrapper):
    """
    Shared llama.cpp wrapper: loading, logit_bias weighting, prompt formatting.

    Subclasses implement _format_prompt(), _get_stop_tokens(), _extract_response().
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        timeout_seconds: int = 300,
        seed: int = 42,
        vocab_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.llm = None
        self.model_loaded = False
        self._a1_token_index_cache: Dict[str, A1TokenIndex] = {}
        self._kvl_lookup: Optional[KvlLookup] = None

        super().__init__(model_name, timeout_seconds, vocab_path=vocab_path)
        self._initialize_model()

        if self.llm is not None:
            self.text_evaluator = TextEvaluator(tokenizer=self._tokenize_text)
        else:
            self.text_evaluator = TextEvaluator()

    def _initialize_model(self):
        """Load GGUF model via llama.cpp."""
        if not LLAMACPP_AVAILABLE:
            self.model_loaded = False
            return

        if not os.path.exists(self.model_path):
            self.model_loaded = False
            return

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                verbose=False,
            )
            self.model_loaded = True
        except Exception:
            self.llm = None
            self.model_loaded = False

    def _tokenize_text(self, text: str) -> list:
        if self.llm is None:
            return []
        return self.llm.tokenize(text.encode("utf-8"))

    @abstractmethod
    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """Format prompt with model-specific chat template."""
        pass

    @abstractmethod
    def _get_stop_tokens(self) -> List[str]:
        """Return stop sequences for generation."""
        pass

    @abstractmethod
    def _extract_response(self, raw_output: str) -> str:
        """Extract assistant text from raw model output."""
        pass

    def _get_a1_token_index(self, *, use_trie: bool = False) -> A1TokenIndex:
        cache_key = "trie" if use_trie else "flat"
        if cache_key not in self._a1_token_index_cache:
            self._a1_token_index_cache[cache_key] = A1TokenIndex.build(
                self.llm,
                self.target_vocabulary,
                use_trie=use_trie,
                stop_token_ids=self._get_stop_token_ids(),
            )
        return self._a1_token_index_cache[cache_key]

    def _get_stop_token_ids(self) -> frozenset[int]:
        if not self.llm:
            return frozenset()
        return resolve_llamacpp_stop_token_ids(self.llm, self._get_stop_tokens())

    @property
    def kvl_lookup(self) -> KvlLookup:
        if self._kvl_lookup is None:
            self._kvl_lookup = KvlLookup()
        return self._kvl_lookup

    def _create_logit_bias(
        self, vocab: List[str], weight_factor: float
    ) -> Dict[int, float]:
        """
        Build logit_bias dict for A1 vocabulary weighting.

        llama.cpp logit_bias is additive in log-space; a weight_factor of 1.5
        corresponds to math.log(1.5) bias per matching token.
        """
        if not self.llm or not vocab or weight_factor <= 0:
            return {}

        bias_value = math.log(weight_factor)
        index = A1TokenIndex.build(self.llm, vocab, use_trie=False)
        return {token_id: bias_value for token_id in index.mid_sentence_ids}

    def _prepare_beam_scoring_text(self, raw_output: str) -> str:
        """Extract and clean beam candidate text before A1-ratio scoring."""
        return self.response_formatter.clean_response_for_evaluation(
            self._extract_response(raw_output)
        )

    def _generate_response_impl(
        self, prompt: str, config: ExperimentConfig
    ) -> Dict[str, Any]:
        if not self.model_loaded or self.llm is None:
            return {
                "response": "",
                "response_time_seconds": 0.0,
                "generation_successful": False,
                "error_message": f"{self.model_name} model not loaded",
            }

        start_time = time.time()
        try:
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(
                    prompt, num_shots=config.num_shots
                )

            formatted_prompt = self._format_prompt(final_prompt, config.system_prompt)

            logit_bias: Dict[int, float] = {}
            if config.config_weighting and self.target_vocabulary:
                logit_bias = self._create_logit_bias(
                    self.target_vocabulary, config.weight_factor
                )

            output = self.llm(
                formatted_prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                stop=self._get_stop_tokens(),
                echo=False,
                logit_bias=logit_bias if logit_bias else None,
            )

            elapsed = time.time() - start_time
            raw_response = output["choices"][0]["text"]
            response = self._extract_response(raw_response)
            response = self.response_formatter.clean_response_for_evaluation(response)

            if not response.strip():
                return {
                    "response": response,
                    "response_time_seconds": elapsed,
                    "generation_successful": False,
                    "error_message": "Empty generation",
                }

            return {
                "response": response,
                "response_time_seconds": elapsed,
                "generation_successful": True,
                "error_message": "",
            }
        except Exception as exc:
            return {
                "response": "",
                "response_time_seconds": time.time() - start_time,
                "generation_successful": False,
                "error_message": str(exc),
            }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "backend": "llama.cpp",
                "model_path": self.model_path,
                "model_format": "GGUF",
                "context_window": self.n_ctx,
                "model_loaded": self.model_loaded,
                "seed": self.seed,
            }
        )
        return info

    def generate_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        selection_method: str = "a1_ratio",
    ) -> Dict[str, Any]:
        """
        Generate via beam search with timeout handling.

        Returns response, beam metadata, response_time_seconds, generation_successful.
        """
        start_time = time.time()
        try:
            with timeout_context(self.timeout_seconds):
                result = self._generate_beam_impl(
                    prompt,
                    config,
                    beam_width=beam_width,
                    selection_method=selection_method,
                )
        except TimeoutError:
            return self._beam_failure_response(
                beam_width,
                selection_method,
                float(self.timeout_seconds),
                f"Generation timed out after {self.timeout_seconds} seconds",
            )
        except Exception as exc:
            return self._beam_failure_response(
                beam_width,
                selection_method,
                time.time() - start_time,
                str(exc),
            )

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
            "beam_selection_method": result.get("beam_selection_method", selection_method),
            "beam_a1_ratio": result.get("beam_a1_ratio", 0.0),
            "beam_a1_count": result.get("beam_a1_count", 0),
            "beam_content_word_count": result.get("beam_content_word_count", 0),
            "beam_cumulative_logprob": result.get("beam_cumulative_logprob", 0.0),
            "beam_width": beam_width,
        }

    def _beam_failure_response(
        self,
        beam_width: int,
        selection_method: str,
        elapsed: float,
        error_message: str,
    ) -> Dict[str, Any]:
        return {
            "response": "",
            "response_time_seconds": elapsed,
            "generation_successful": False,
            "error_message": error_message,
            "beam_selection_method": selection_method,
            "beam_a1_ratio": 0.0,
            "beam_a1_count": 0,
            "beam_content_word_count": 0,
            "beam_cumulative_logprob": 0.0,
            "beam_width": beam_width,
        }

    def _generate_beam_impl(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        selection_method: str = "a1_ratio",
    ) -> Dict[str, Any]:
        if not self.model_loaded or self.llm is None:
            return {
                "response": "",
                "response_time_seconds": 0.0,
                "generation_successful": False,
                "error_message": f"{self.model_name} model not loaded",
                "beam_selection_method": selection_method,
                "beam_a1_ratio": 0.0,
                "beam_a1_count": 0,
                "beam_content_word_count": 0,
                "beam_cumulative_logprob": 0.0,
            }

        start_time = time.time()
        try:
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(
                    prompt, num_shots=config.num_shots
                )

            formatted_prompt = self._format_prompt(final_prompt, config.system_prompt)

            beam_generator = BeamSearchGenerator(
                llm=self.llm,
                beam_width=beam_width,
                max_length=config.max_new_tokens,
                length_penalty=1.0,
            )

            beam_results = beam_generator.generate(
                prompt=formatted_prompt,
                temperature=config.temperature,
                top_k=config.top_k,
                stop=self._get_stop_tokens(),
            )

            selection_results = beam_generator.select_best_beams(
                beam_results["beams"],
                a1_vocab=self.target_vocabulary,
                extract_content_words=self.text_evaluator.extract_content_words,
                prepare_scoring_text=self._prepare_beam_scoring_text,
            )

            if selection_method == "a1_ratio":
                selected_beam_data = selection_results["best_by_a1_ratio"]
            else:
                selected_beam_data = selection_results["best_by_probability"]

            if not selected_beam_data:
                return {
                    "response": "",
                    "response_time_seconds": time.time() - start_time,
                    "generation_successful": False,
                    "error_message": "No valid beams generated",
                    "beam_selection_method": selection_method,
                    "beam_a1_ratio": 0.0,
                    "beam_a1_count": 0,
                    "beam_content_word_count": 0,
                    "beam_cumulative_logprob": 0.0,
                }

            response = selected_beam_data["scoring_text"]

            if not response.strip():
                return {
                    "response": response,
                    "response_time_seconds": time.time() - start_time,
                    "generation_successful": False,
                    "error_message": "Empty generation",
                    "beam_selection_method": selection_method,
                    "beam_a1_ratio": selected_beam_data["a1_ratio"],
                    "beam_a1_count": selected_beam_data["a1_count"],
                    "beam_content_word_count": selected_beam_data["content_count"],
                    "beam_cumulative_logprob": selected_beam_data["cumulative_log_prob"],
                }

            return {
                "response": response,
                "response_time_seconds": time.time() - start_time,
                "generation_successful": True,
                "error_message": "",
                "beam_selection_method": selection_method,
                "beam_a1_ratio": selected_beam_data["a1_ratio"],
                "beam_a1_count": selected_beam_data["a1_count"],
                "beam_content_word_count": selected_beam_data["content_count"],
                "beam_cumulative_logprob": selected_beam_data["cumulative_log_prob"],
            }
        except Exception as exc:
            return {
                "response": "",
                "response_time_seconds": time.time() - start_time,
                "generation_successful": False,
                "error_message": str(exc),
                "beam_selection_method": selection_method,
                "beam_a1_ratio": 0.0,
                "beam_a1_count": 0,
                "beam_content_word_count": 0,
                "beam_cumulative_logprob": 0.0,
            }

    def generate_guided(
        self,
        prompt: str,
        config: ExperimentConfig,
    ) -> Dict[str, Any]:
        """
        Generate via top-K A1-constrained greedy decoding.

        Returns response, guided metadata, response_time_seconds, generation_successful.
        """
        start_time = time.time()
        try:
            with timeout_context(self.timeout_seconds):
                result = self._generate_guided_impl(prompt, config)
        except TimeoutError:
            return self._guided_failure_response(
                config,
                float(self.timeout_seconds),
                f"Generation timed out after {self.timeout_seconds} seconds",
            )
        except Exception as exc:
            return self._guided_failure_response(
                config,
                time.time() - start_time,
                str(exc),
            )

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
            "guided_top_k": result.get("guided_top_k", config.guided_top_k),
            "guided_mode": result.get("guided_mode", config.guided_mode),
            "guided_steps_a1_chosen": result.get("guided_steps_a1_chosen", 0),
            "guided_steps_total": result.get("guided_steps_total", 0),
            "guided_steps_fallback_argmax": result.get(
                "guided_steps_fallback_argmax", 0
            ),
            "guided_steps_no_a1_in_pool": result.get(
                "guided_steps_no_a1_in_pool", 0
            ),
            "guided_intervention_rate": result.get("guided_intervention_rate", 0.0),
        }

    def _guided_failure_response(
        self,
        config: ExperimentConfig,
        elapsed: float,
        error_message: str,
    ) -> Dict[str, Any]:
        return {
            "response": "",
            "response_time_seconds": elapsed,
            "generation_successful": False,
            "error_message": error_message,
            "guided_top_k": config.guided_top_k,
            "guided_mode": config.guided_mode,
            "guided_steps_a1_chosen": 0,
            "guided_steps_total": 0,
            "guided_steps_fallback_argmax": 0,
            "guided_steps_no_a1_in_pool": 0,
            "guided_intervention_rate": 0.0,
        }

    def _generate_guided_impl(
        self,
        prompt: str,
        config: ExperimentConfig,
    ) -> Dict[str, Any]:
        if not self.model_loaded or self.llm is None:
            return {
                "response": "",
                "response_time_seconds": 0.0,
                "generation_successful": False,
                "error_message": f"{self.model_name} model not loaded",
                "guided_top_k": config.guided_top_k,
                "guided_mode": config.guided_mode,
                "guided_steps_a1_chosen": 0,
                "guided_steps_total": 0,
                "guided_steps_fallback_argmax": 0,
                "guided_steps_no_a1_in_pool": 0,
                "guided_intervention_rate": 0.0,
            }

        start_time = time.time()
        try:
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(
                    prompt, num_shots=config.num_shots
                )

            formatted_prompt = self._format_prompt(final_prompt, config.system_prompt)
            use_trie = config.guided_mode == "trie"
            index = self._get_a1_token_index(use_trie=use_trie)
            prompt_token_ids = self.llm.tokenize(
                formatted_prompt.encode("utf-8"), add_bos=True
            )

            decode_result = ConstrainedDecoder().decode(
                self.llm,
                list(prompt_token_ids),
                max_tokens=config.max_new_tokens,
                stop=self._get_stop_tokens(),
                stop_token_ids=self._get_stop_token_ids(),
                guided_pool_size=config.guided_top_k,
                index=index,
                mode=config.guided_mode,
                temperature=config.temperature,
                top_k=config.top_k,
            )

            raw_response = decode_result.text
            response = self._extract_response(raw_response)
            response = self.response_formatter.clean_response_for_evaluation(response)

            intervention_rate = (
                decode_result.steps_a1_chosen / decode_result.steps_total
                if decode_result.steps_total > 0
                else 0.0
            )

            if not response.strip():
                return {
                    "response": response,
                    "response_time_seconds": time.time() - start_time,
                    "generation_successful": False,
                    "error_message": "Empty generation",
                    "guided_top_k": config.guided_top_k,
                    "guided_mode": config.guided_mode,
                    "guided_steps_a1_chosen": decode_result.steps_a1_chosen,
                    "guided_steps_total": decode_result.steps_total,
                    "guided_steps_fallback_argmax": decode_result.steps_fallback_argmax,
                    "guided_steps_no_a1_in_pool": decode_result.steps_no_a1_in_pool,
                    "guided_intervention_rate": intervention_rate,
                }

            return {
                "response": response,
                "response_time_seconds": time.time() - start_time,
                "generation_successful": True,
                "error_message": "",
                "guided_top_k": config.guided_top_k,
                "guided_mode": config.guided_mode,
                "guided_steps_a1_chosen": decode_result.steps_a1_chosen,
                "guided_steps_total": decode_result.steps_total,
                "guided_steps_fallback_argmax": decode_result.steps_fallback_argmax,
                "guided_steps_no_a1_in_pool": decode_result.steps_no_a1_in_pool,
                "guided_intervention_rate": intervention_rate,
            }
        except Exception as exc:
            return {
                "response": "",
                "response_time_seconds": time.time() - start_time,
                "generation_successful": False,
                "error_message": str(exc),
                "guided_top_k": config.guided_top_k,
                "guided_mode": config.guided_mode,
                "guided_steps_a1_chosen": 0,
                "guided_steps_total": 0,
                "guided_steps_fallback_argmax": 0,
                "guided_steps_no_a1_in_pool": 0,
                "guided_intervention_rate": 0.0,
            }

    def _kvl_beam_timeout_seconds(
        self,
        max_new_tokens: int,
        beam_width: int,
        branch_factor: int,
    ) -> int:
        """Wall-clock budget for KVL beam (full re-eval per candidate)."""
        # Each decode step runs up to beam_width llama evals (one per active beam).
        evals_per_token = max(beam_width, 1)
        seconds_per_eval = 12.0
        return int(
            max(
                self.timeout_seconds,
                max_new_tokens * evals_per_token * seconds_per_eval + 300,
            )
        )

    def generate_kvl_beam(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        branch_factor: int = 10,
    ) -> Dict[str, Any]:
        """Generate via KVL-scored token-level beam search."""
        start_time = time.time()
        beam_timeout = self._kvl_beam_timeout_seconds(
            config.max_new_tokens, beam_width, branch_factor
        )
        try:
            with timeout_context(beam_timeout):
                result = self._generate_kvl_beam_impl(
                    prompt,
                    config,
                    beam_width=beam_width,
                    branch_factor=branch_factor,
                )
        except TimeoutError:
            return self._kvl_beam_failure_response(
                beam_width,
                branch_factor,
                float(beam_timeout),
                f"Generation timed out after {beam_timeout} seconds",
            )
        except Exception as exc:
            return self._kvl_beam_failure_response(
                beam_width,
                branch_factor,
                time.time() - start_time,
                str(exc),
            )

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
            "kvl_beam_width": beam_width,
            "kvl_branch_factor": branch_factor,
            "kvl_beam_steps_total": result.get("kvl_beam_steps_total", 0),
            "kvl_beam_words_scored": result.get("kvl_beam_words_scored", 0),
            "kvl_beam_running_mean": result.get("kvl_beam_running_mean"),
            "kvl_beam_logprob_tiebreak": result.get("kvl_beam_logprob_tiebreak", 0.0),
            "kvl_beam_candidates_pruned": result.get("kvl_beam_candidates_pruned", 0),
        }

    def _kvl_beam_failure_response(
        self,
        beam_width: int,
        branch_factor: int,
        elapsed: float,
        error_message: str,
    ) -> Dict[str, Any]:
        return {
            "response": "",
            "response_time_seconds": elapsed,
            "generation_successful": False,
            "error_message": error_message,
            "kvl_beam_width": beam_width,
            "kvl_branch_factor": branch_factor,
            "kvl_beam_steps_total": 0,
            "kvl_beam_words_scored": 0,
            "kvl_beam_running_mean": None,
            "kvl_beam_logprob_tiebreak": 0.0,
            "kvl_beam_candidates_pruned": 0,
        }

    def _generate_kvl_beam_impl(
        self,
        prompt: str,
        config: ExperimentConfig,
        beam_width: int = 4,
        branch_factor: int = 10,
    ) -> Dict[str, Any]:
        if not self.model_loaded or self.llm is None:
            return {
                "response": "",
                "response_time_seconds": 0.0,
                "generation_successful": False,
                "error_message": f"{self.model_name} model not loaded",
                "kvl_beam_steps_total": 0,
                "kvl_beam_words_scored": 0,
                "kvl_beam_running_mean": None,
                "kvl_beam_logprob_tiebreak": 0.0,
                "kvl_beam_candidates_pruned": 0,
            }

        start_time = time.time()
        try:
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(
                    prompt, num_shots=config.num_shots
                )

            formatted_prompt = self._format_prompt(final_prompt, config.system_prompt)
            prompt_token_ids = self.llm.tokenize(
                formatted_prompt.encode("utf-8"), add_bos=True
            )

            decoder = KvlBeamDecoder(
                kvl_lookup=self.kvl_lookup,
                l1=config.kvl_l1,
                text_evaluator=self.text_evaluator,
                beam_width=beam_width,
                branch_factor=branch_factor,
            )
            eval_fn, decode_suffix = make_llamacpp_eval_fn(
                self.llm,
                prompt_token_ids,
                top_k=config.top_k,
            )

            decode_result = decoder.decode(
                eval_fn,
                prompt_token_ids,
                max_tokens=config.max_new_tokens,
                stop=self._get_stop_tokens(),
                stop_token_ids=self._get_stop_token_ids(),
                decode_suffix=decode_suffix,
            )

            response = self._prepare_beam_scoring_text(decode_result.text)
            elapsed = time.time() - start_time

            if not response.strip():
                return {
                    "response": response,
                    "response_time_seconds": elapsed,
                    "generation_successful": False,
                    "error_message": "Empty generation",
                    "kvl_beam_steps_total": decode_result.steps_total,
                    "kvl_beam_words_scored": decode_result.words_scored,
                    "kvl_beam_running_mean": decode_result.running_mean,
                    "kvl_beam_logprob_tiebreak": decode_result.cumulative_logprob,
                    "kvl_beam_candidates_pruned": decode_result.candidates_pruned,
                }

            return {
                "response": response,
                "response_time_seconds": elapsed,
                "generation_successful": True,
                "error_message": "",
                "kvl_beam_steps_total": decode_result.steps_total,
                "kvl_beam_words_scored": decode_result.words_scored,
                "kvl_beam_running_mean": decode_result.running_mean,
                "kvl_beam_logprob_tiebreak": decode_result.cumulative_logprob,
                "kvl_beam_candidates_pruned": decode_result.candidates_pruned,
            }
        except TimeoutError:
            raise
        except Exception as exc:
            return {
                "response": "",
                "response_time_seconds": time.time() - start_time,
                "generation_successful": False,
                "error_message": str(exc),
                "kvl_beam_steps_total": 0,
                "kvl_beam_words_scored": 0,
                "kvl_beam_running_mean": None,
                "kvl_beam_logprob_tiebreak": 0.0,
                "kvl_beam_candidates_pruned": 0,
            }

    def cleanup(self):
        """Release llama.cpp model resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.model_loaded = False
        self._a1_token_index_cache.clear()


def default_gguf_path(filename: str) -> str:
    """Return absolute path to a GGUF file (local, thesis repo, or SLM_GGUF_DIR)."""
    return os.path.join(resolve_gguf_dir(), filename)
