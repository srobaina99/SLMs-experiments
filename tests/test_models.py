"""Tests for llama.cpp model wrappers (mocked — no GGUF required)."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline, ModelWrapper
from slm_experiments.models.base import (
    REPO_ROOT,
    THESIS_GGUF_DIR,
    BaseModelWrapper,
    resolve_gguf_dir,
)
from slm_experiments.models.beam import BeamCandidate, BeamSearchGenerator
from slm_experiments.models.llamacpp import LlamaCppBaseWrapper
from slm_experiments.models.wrappers import MODEL_REGISTRY, get_model_wrapper
from slm_experiments.models.wrappers.qwen3_llamacpp_wrapper import Qwen3LlamaCppWrapper


class _StubLlamaCppWrapper(LlamaCppBaseWrapper):
    """Minimal concrete wrapper for testing base behavior."""

    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        return f"SYS:{system_prompt}|USER:{user_input}|ASSIST:"

    def _get_stop_tokens(self):
        return ["STOP"]

    def _extract_response(self, raw_output: str) -> str:
        return raw_output.strip()


def _make_mock_llm(response_text: str = "Hello world.", token_map=None):
    mock = MagicMock()
    mock.return_value = {"choices": [{"text": response_text}]}
    if token_map is None:
        token_map = {" hello": [10], " world": [20]}
    mock.tokenize.side_effect = lambda data, add_bos=True: token_map.get(
        data.decode("utf-8") if isinstance(data, bytes) else data,
        [1, 2, 3],
    )
    return mock


@pytest.fixture
def stub_wrapper(tmp_path):
    """Wrapper with mocked llama.cpp and temp vocab file."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("hello\nworld\ncat\n", encoding="utf-8")
    model_path = tmp_path / "fake.gguf"
    model_path.write_text("fake", encoding="utf-8")

    with patch("slm_experiments.models.llamacpp.Llama") as mock_llama_cls:
        mock_llama_cls.return_value = _make_mock_llm()
        wrapper = _StubLlamaCppWrapper(
            model_name="Stub",
            model_path=str(model_path),
            seed=99,
            vocab_path=str(vocab),
        )
        wrapper.llm = mock_llama_cls.return_value
        wrapper.model_loaded = True
        yield wrapper


class TestModelRegistry:
    def test_registry_has_four_models(self):
        assert set(MODEL_REGISTRY.keys()) == {"Qwen3", "Qwen2", "Phi3", "TinyLlama"}

    def test_get_model_wrapper_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_wrapper("GPT4")


class TestQwen3PromptFormatting:
    @patch("slm_experiments.models.llamacpp.Llama")
    def test_format_prompt_includes_nothink(self, mock_llama, tmp_path):
        model_path = tmp_path / "qwen3.gguf"
        model_path.write_text("fake", encoding="utf-8")
        mock_llama.return_value = _make_mock_llm()

        wrapper = Qwen3LlamaCppWrapper(model_path=str(model_path), seed=42)
        wrapper.llm = mock_llama.return_value
        wrapper.model_loaded = True

        formatted = wrapper._format_prompt("What is a friend?", "Be helpful.")
        assert "/nothink" in formatted
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "<|im_end|>" in formatted

    def test_extract_response_strips_thinking_tags(self):
        wrapper = Qwen3LlamaCppWrapper.__new__(Qwen3LlamaCppWrapper)
        raw = (
            "<think>internal reasoning</think>"
            "A friend is a person you like."
        )
        extracted = wrapper._extract_response(raw)
        assert "internal reasoning" not in extracted
        assert "A friend is a person you like." in extracted


class TestLlamaCppGenerate:
    def test_generate_returns_protocol_keys(self, stub_wrapper):
        config = ExperimentConfig(model_name="Stub")
        result = stub_wrapper.generate("Hi", config)

        assert "response" in result
        assert "response_time_seconds" in result
        assert "generation_successful" in result
        assert result["generation_successful"] is True
        assert result["response"] == "Hello world."

    def test_generate_empty_response_is_failure(self, stub_wrapper):
        stub_wrapper.llm = _make_mock_llm(response_text="   ")
        config = ExperimentConfig(model_name="Stub")
        result = stub_wrapper.generate("Hi", config)
        assert result["generation_successful"] is False

    def test_generate_model_not_loaded(self, tmp_path):
        model_path = tmp_path / "missing.gguf"
        wrapper = _StubLlamaCppWrapper(
            model_name="Stub",
            model_path=str(model_path),
            vocab_path=str(tmp_path / "vocab.txt"),
        )
        (tmp_path / "vocab.txt").write_text("hello\n", encoding="utf-8")

        result = wrapper.generate("Hi", ExperimentConfig())
        assert result["generation_successful"] is False

    def test_config_prompting_adds_context(self, stub_wrapper):
        config = ExperimentConfig(model_name="Stub", config_prompting=True)
        stub_wrapper.generate("What is a friend?", config)
        call_args = stub_wrapper.llm.call_args
        prompt_arg = call_args[0][0]
        assert "# Context" in prompt_arg
        assert "What is a friend?" in prompt_arg

    def test_config_weighting_applies_logit_bias(self, stub_wrapper):
        config = ExperimentConfig(
            model_name="Stub", config_weighting=True, weight_factor=1.5
        )
        stub_wrapper.generate("Hi", config)
        call_kwargs = stub_wrapper.llm.call_args[1]
        logit_bias = call_kwargs["logit_bias"]
        assert logit_bias is not None
        expected_bias = math.log(1.5)
        assert all(v == pytest.approx(expected_bias) for v in logit_bias.values())

    def test_seed_passed_to_llama_constructor(self, tmp_path):
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("hello\n", encoding="utf-8")
        model_path = tmp_path / "fake.gguf"
        model_path.write_text("fake", encoding="utf-8")

        with patch("slm_experiments.models.llamacpp.Llama") as mock_llama_cls:
            mock_llama_cls.return_value = _make_mock_llm()
            _StubLlamaCppWrapper(
                model_name="Stub",
                model_path=str(model_path),
                seed=12345,
                vocab_path=str(vocab),
            )
            mock_llama_cls.assert_called_once()
            assert mock_llama_cls.call_args[1]["seed"] == 12345


class TestLlamaCppGenerateBeam:
    def test_generate_beam_returns_metadata(self, stub_wrapper):
        config = ExperimentConfig(model_name="Stub", config_prompting=True)
        result = stub_wrapper.generate_beam("What is a friend?", config, beam_width=4)

        assert result["generation_successful"] is True
        assert result["beam_width"] == 4
        assert result["beam_selection_method"] == "a1_ratio"
        assert "beam_a1_ratio" in result
        assert result["response"]

    def test_generate_beam_model_not_loaded(self, tmp_path):
        model_path = tmp_path / "missing.gguf"
        wrapper = _StubLlamaCppWrapper(
            model_name="Stub",
            model_path=str(model_path),
            vocab_path=str(tmp_path / "vocab.txt"),
        )
        (tmp_path / "vocab.txt").write_text("hello\n", encoding="utf-8")

        result = wrapper.generate_beam("Hi", ExperimentConfig(), beam_width=8)
        assert result["generation_successful"] is False
        assert result["beam_width"] == 8


class TestWrapperPipelineIntegration:
    def test_stub_wrapper_satisfies_model_wrapper_protocol(self, stub_wrapper):
        assert isinstance(stub_wrapper, ModelWrapper)

    def test_pipeline_with_stub_wrapper(self, stub_wrapper):
        config = ExperimentConfig(model_name="Stub", prompt_id="p01")
        result = ExperimentPipeline().run("What is a friend?", config, stub_wrapper)
        assert result.generation_successful is True
        assert result.word_count > 0

    def test_pipeline_run_beam_with_stub_wrapper(self, stub_wrapper):
        config = ExperimentConfig(
            model_name="Stub", prompt_id="p01", config_prompting=True
        )
        result = ExperimentPipeline().run_beam(
            "What is a friend?", config, stub_wrapper, beam_width=4
        )
        assert result.generation_successful is True
        assert result.beam_width == 4
        assert result.beam_selection_method == "a1_ratio"
        assert result.beam_a1_ratio is not None


class TestBeamSearchGenerator:
    def test_generate_returns_beams(self):
        mock_llm = _make_mock_llm("Simple text.")
        gen = BeamSearchGenerator(mock_llm, beam_width=2, max_length=50)
        result = gen.generate("PROMPT:")
        assert len(result["beams"]) == 2
        assert all(isinstance(b, BeamCandidate) for b in result["beams"])

    def test_select_best_beams(self):
        beams = [
            BeamCandidate([1], -1.0, "the cat runs"),
            BeamCandidate([2], -2.0, "hello friend play"),
        ]
        gen = BeamSearchGenerator(MagicMock(), beam_width=2)
        content = {"cat", "runs", "hello", "friend", "play"}
        a1_vocab = ["cat", "hello", "friend"]
        selection = gen.select_best_beams(beams, a1_vocab, content)
        assert selection["best_by_a1_ratio"] is not None
        assert selection["best_by_probability"] is not None


class TestGgufPathResolution:
    def test_default_gguf_path_uses_thesis_repo_when_local_empty(self):
        from slm_experiments.models.llamacpp import default_gguf_path

        local_ggufs = list(Path(REPO_ROOT, "models", "gguf").glob("*.gguf"))
        if local_ggufs:
            pytest.skip("Local GGUF files present; thesis fallback not tested")

        assert resolve_gguf_dir() == THESIS_GGUF_DIR
        qwen3_path = default_gguf_path("Qwen3-0.6B-Q4_0.gguf")
        assert Path(qwen3_path).exists()

    def test_slm_gguf_dir_env_overrides(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom_gguf"
        custom.mkdir()
        (custom / "test.gguf").write_text("fake", encoding="utf-8")
        monkeypatch.setenv("SLM_GGUF_DIR", str(custom))
        assert resolve_gguf_dir() == str(custom)


class TestVocabLoading:
    def test_default_vocab_path_under_repo_root(self):
        assert REPO_ROOT.endswith("SLMs-experiments") or "SLMs-experiments" in REPO_ROOT

    def test_base_loads_vocab_from_file(self, tmp_path):
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("apple\nbanana\n", encoding="utf-8")

        class MinimalWrapper(BaseModelWrapper):
            def _generate_response_impl(self, prompt, config):
                return {}

            def _initialize_model(self):
                pass

        wrapper = MinimalWrapper("Test", vocab_path=str(vocab))
        assert wrapper.target_vocabulary == ["apple", "banana"]
