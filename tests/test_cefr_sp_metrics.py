"""Tests for optional CEFR-SP sentence difficulty metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import NUMERIC_SUMMARY_COLUMNS, compute_summary_stats
from slm_experiments.evaluation.cefr_sp import (
    CEFR_SP_LEVELS,
    DEFAULT_CEFR_SP_CKPT,
    CefrSpScorer,
    compute_cefr_sp_metrics,
    empty_cefr_sp_metrics,
    ordinal_to_level,
    sentence_word_lists,
)


class MockCefrSpScorer:
    """Fixed per-sentence labels/probs for aggregation tests."""

    def __init__(self, rows: List[Dict[str, object]]):
        self.rows = rows
        self.calls = 0

    def score_sentences(self, sentences: Sequence[str]) -> List[Dict[str, object]]:
        self.calls += 1
        assert len(sentences) == len(self.rows)
        return list(self.rows)


def _a1_row() -> Dict[str, object]:
    return {"label": 0, "probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


def _level_row(label: int) -> Dict[str, object]:
    probs = [0.0] * 6
    probs[label] = 1.0
    return {"label": label, "probs": probs}


class TestEmptySchema:
    def test_empty_defaults(self):
        metrics = empty_cefr_sp_metrics()
        assert metrics["cefr_sp_enabled"] is False
        assert metrics["cefr_sp_sentence_count"] == 0
        assert metrics["cefr_sp_level"] is None
        assert metrics["cefr_sp_level_ordinal"] is None
        assert metrics["cefr_sp_max_level_ordinal"] is None
        assert metrics["cefr_sp_pct_a1"] is None
        assert metrics["cefr_sp_adjacency"] is None
        assert metrics["cefr_sp_expected_level"] is None

    def test_empty_enabled_flag(self):
        metrics = empty_cefr_sp_metrics(enabled=True)
        assert metrics["cefr_sp_enabled"] is True
        assert metrics["cefr_sp_level"] is None

    def test_empty_and_scored_schema_keys_identical(self):
        empty = empty_cefr_sp_metrics(enabled=False)
        blank = compute_cefr_sp_metrics("", enabled=True)
        scored = compute_cefr_sp_metrics(
            "Hello there.",
            scorer=MockCefrSpScorer([_a1_row()]),
            enabled=True,
        )
        assert sorted(empty.keys()) == sorted(blank.keys()) == sorted(scored.keys())


class TestAggregation:
    def test_mocked_scorer_aggregates(self):
        # Sent1: A1, Sent2: A2, Sent3: B1
        rows = [_level_row(0), _level_row(1), _level_row(2)]
        scorer = MockCefrSpScorer(rows)
        metrics = compute_cefr_sp_metrics(
            "One. Two. Three.",
            scorer=scorer,
            enabled=True,
        )

        assert scorer.calls == 1
        assert metrics["cefr_sp_enabled"] is True
        assert metrics["cefr_sp_sentence_count"] == 3
        assert metrics["cefr_sp_level_ordinal"] == pytest.approx(1.0)
        assert metrics["cefr_sp_level"] == "A2"
        assert metrics["cefr_sp_max_level_ordinal"] == 2
        assert metrics["cefr_sp_pct_a1"] == pytest.approx(1 / 3, abs=1e-4)
        # adjacency = fraction of sentences with ordinal <= 1 (A1 or A2)
        assert metrics["cefr_sp_adjacency"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["cefr_sp_expected_level"] == pytest.approx(1.0)

    def test_adjacency_is_fraction_ordinal_le_one(self):
        rows = [_level_row(0), _level_row(1), _level_row(2), _level_row(5)]
        metrics = compute_cefr_sp_metrics(
            "One. Two. Three. Four.",
            scorer=MockCefrSpScorer(rows),
            enabled=True,
        )
        assert metrics["cefr_sp_adjacency"] == pytest.approx(0.5)
        assert metrics["cefr_sp_pct_a1"] == pytest.approx(0.25)
        assert metrics["cefr_sp_level_ordinal"] == pytest.approx(2.0)
        assert metrics["cefr_sp_level"] == "B1"
        assert metrics["cefr_sp_max_level_ordinal"] == 5

    def test_pct_a1_only_counts_label_zero(self):
        rows = [_level_row(0), _level_row(0), _level_row(1)]
        metrics = compute_cefr_sp_metrics(
            "One. Two. Three.",
            scorer=MockCefrSpScorer(rows),
            enabled=True,
        )
        assert metrics["cefr_sp_pct_a1"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["cefr_sp_adjacency"] == pytest.approx(1.0)

    def test_expected_level_from_probs(self):
        rows = [
            {"label": 0, "probs": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]},
        ]
        metrics = compute_cefr_sp_metrics(
            "Hello there.",
            scorer=MockCefrSpScorer(rows),
            enabled=True,
        )
        assert metrics["cefr_sp_expected_level"] == pytest.approx(0.5)
        assert metrics["cefr_sp_level_ordinal"] == 0.0
        assert metrics["cefr_sp_level"] == "A1"

    def test_ordinal_to_level_mapping_a1_through_c2(self):
        for idx, label in enumerate(CEFR_SP_LEVELS):
            assert ordinal_to_level(float(idx)) == label
        assert ordinal_to_level(0.0) == "A1"
        assert ordinal_to_level(0.4) == "A1"
        assert ordinal_to_level(0.6) == "A2"
        assert ordinal_to_level(5.0) == "C2"
        assert ordinal_to_level(-1.0) == "A1"
        assert ordinal_to_level(99.0) == "C2"

    def test_score_count_mismatch_raises(self):
        class ShortScorer:
            def score_sentences(self, sentences: Sequence[str]):
                return [_a1_row()]  # always one row

        with pytest.raises(ValueError, match="returned 1 scores for"):
            compute_cefr_sp_metrics(
                "One. Two. Three.",
                scorer=ShortScorer(),
                enabled=True,
            )

    def test_bad_probs_length_raises(self):
        class BadProbs:
            def score_sentences(self, sentences: Sequence[str]):
                return [{"label": 0, "probs": [1.0, 0.0]} for _ in sentences]

        with pytest.raises(ValueError, match="Expected 6 class probs"):
            compute_cefr_sp_metrics(
                "Hello there.",
                scorer=BadProbs(),
                enabled=True,
            )


class TestTokenization:
    def test_sentence_word_lists_whitespace_split(self):
        assert sentence_word_lists(["Hello world."]) == [["Hello", "world."]]
        assert sentence_word_lists(["nospace"]) == [["nospace"]]
        assert sentence_word_lists([""]) == [[""]]


class TestCefrSpScorerLoad:
    def test_ensure_loaded_overrides_zenodo_pretrained_model_path(
        self, tmp_path, monkeypatch
    ):
        """Zenodo hparams bake '../pretrained_model/bert-base-cased/'; override to Hub id."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("pytorch_lightning")

        ckpt = tmp_path / "level_estimator.ckpt"
        ckpt.write_bytes(b"fake-ckpt")

        captured: dict = {}

        class FakeModel:
            # Signature must match LevelEstimaterContrastive — _ensure_loaded
            # filters init kwargs via inspect.signature.
            def __init__(
                self,
                corpus_path,
                test_corpus_path,
                pretrained_model,
                problem_type,
                with_ib,
                with_loss_weight,
                attach_wlv,
                num_labels,
                word_num_labels,
                num_prototypes,
                alpha,
                ib_beta,
                batch_size,
                learning_rate,
                warmup,
                lm_layer,
            ):
                captured["init_kwargs"] = {
                    "pretrained_model": pretrained_model,
                    "num_prototypes": num_prototypes,
                    "lm_layer": lm_layer,
                }

            def load_state_dict(self, state_dict, strict=True):
                captured["state_dict"] = state_dict
                captured["strict"] = strict
                return None

            def eval(self):
                return self

            def to(self, device):
                return self

        fake_checkpoint = {
            "hyper_parameters": {
                "corpus_path": "/unused",
                "test_corpus_path": "/unused",
                "pretrained_model": "../pretrained_model/bert-base-cased/",
                "problem_type": "classification",
                "with_ib": False,
                "with_loss_weight": True,
                "attach_wlv": False,
                "num_labels": 6,
                "word_num_labels": 6,
                "num_prototypes": 3,
                "alpha": 0.5,
                "ib_beta": 0.0,
                "batch_size": 32,
                "learning_rate": 1e-5,
                "warmup": 0,
                "lm_layer": 11,
            },
            "state_dict": {
                "lm.embeddings.position_ids": torch.arange(512),
                "prototype.weight": torch.zeros(18, 4),
            },
        }

        monkeypatch.setattr(torch, "load", lambda *a, **k: fake_checkpoint)

        from slm_experiments.evaluation.cefr_sp_vendor import model as vendor_model

        monkeypatch.setattr(vendor_model, "LevelEstimaterContrastive", FakeModel)

        scorer = CefrSpScorer(ckpt_path=str(ckpt), device="cpu")
        scorer._ensure_loaded()

        assert captured["init_kwargs"]["pretrained_model"] == "bert-base-cased"
        assert "lm.embeddings.position_ids" not in captured["state_dict"]
        assert "prototype.weight" in captured["state_dict"]
        assert captured["strict"] is True
        assert scorer._model is not None


class TestDisabledPath:
    def test_disabled_does_not_import_torch(self):
        # Ensure torch is not already loaded from an earlier test that needed it.
        # We only assert that disabled compute does not newly require torch.
        before = "torch" in sys.modules
        metrics = compute_cefr_sp_metrics("Hello. World.", enabled=False)
        after = "torch" in sys.modules

        assert metrics["cefr_sp_enabled"] is False
        assert metrics["cefr_sp_sentence_count"] == 0
        assert metrics["cefr_sp_level"] is None
        if not before:
            assert after is False

    def test_disabled_pipeline_default(self):
        class MockSuccessModel:
            def generate(self, prompt: str, config: ExperimentConfig) -> dict:
                return {
                    "response": "A friend is a person you like.",
                    "response_time_seconds": 1.0,
                    "generation_successful": True,
                }

        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=False,
        )
        pipeline = ExperimentPipeline()
        result = pipeline.run("What is a friend?", config, MockSuccessModel())

        assert result.cefr_sp_enabled is False
        assert result.cefr_sp_sentence_count == 0
        assert result.cefr_sp_level is None
        assert result.cefr_sp_level_ordinal is None

    def test_importing_cefr_sp_facade_in_subprocess_avoids_torch(self):
        """Fresh interpreter: façade + disabled compute must not import torch."""
        import subprocess

        script = (
            "import sys\n"
            "from slm_experiments.evaluation.cefr_sp import compute_cefr_sp_metrics\n"
            "import slm_experiments.evaluation.cefr_sp_vendor  # noqa: F401\n"
            "m = compute_cefr_sp_metrics('Hello.', enabled=False)\n"
            "assert m['cefr_sp_enabled'] is False\n"
            "assert 'torch' not in sys.modules\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout


class TestPipelineIntegration:
    class MockSuccessModel:
        def generate(self, prompt: str, config: ExperimentConfig) -> dict:
            return {
                "response": "Hello. How are you?",
                "response_time_seconds": 1.0,
                "generation_successful": True,
            }

    class MockFailureModel:
        def generate(self, prompt: str, config: ExperimentConfig) -> dict:
            return {
                "response": "",
                "response_time_seconds": 0.5,
                "generation_successful": False,
            }

    class MockBeamModel:
        def generate_beam(self, prompt, config, beam_width=4, selection_method="a1_ratio"):
            return {
                "response": "Hello. How are you?",
                "response_time_seconds": 1.0,
                "generation_successful": True,
                "beam_selection_method": selection_method,
                "beam_a1_ratio": 0.5,
                "beam_a1_count": 1,
                "beam_content_word_count": 2,
                "beam_cumulative_logprob": -1.0,
                "beam_width": beam_width,
            }

    class MockGuidedModel:
        def generate_guided(self, prompt, config):
            return {
                "response": "Hello. How are you?",
                "response_time_seconds": 1.0,
                "generation_successful": True,
                "guided_top_k": 10,
                "guided_mode": "flat",
                "guided_steps_a1_chosen": 1,
                "guided_steps_total": 2,
                "guided_intervention_rate": 0.5,
            }

    class MockKvlBeamModel:
        def generate_kvl_beam(self, prompt, config, beam_width=4, branch_factor=10):
            return {
                "response": "Hello. How are you?",
                "response_time_seconds": 1.0,
                "generation_successful": True,
                "kvl_beam_width": beam_width,
                "kvl_branch_factor": branch_factor,
                "kvl_beam_steps_total": 2,
                "kvl_beam_words_scored": 2,
                "kvl_beam_running_mean": 1.0,
                "kvl_beam_logprob_tiebreak": -0.5,
                "kvl_beam_candidates_pruned": 0,
            }

    def test_success_path_with_mock_scorer(self):
        rows = [_level_row(0), _level_row(1)]
        scorer = MockCefrSpScorer(rows)
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=True,
        )
        pipeline = ExperimentPipeline(cefr_sp_scorer=scorer)
        result = pipeline.run("Hi?", config, self.MockSuccessModel())

        assert result.generation_successful is True
        assert result.cefr_sp_enabled is True
        assert result.cefr_sp_sentence_count == 2
        assert result.cefr_sp_level_ordinal == pytest.approx(0.5)
        assert result.cefr_sp_max_level_ordinal == 1
        assert result.cefr_sp_pct_a1 == pytest.approx(0.5)
        assert result.cefr_sp_adjacency == pytest.approx(1.0)
        assert scorer.calls == 1

    def test_failure_path_empty_schema(self):
        scorer = MockCefrSpScorer([])
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p02",
            enable_cefr_sp=True,
        )
        pipeline = ExperimentPipeline(cefr_sp_scorer=scorer)
        result = pipeline.run("Hello?", config, self.MockFailureModel())

        assert result.generation_successful is False
        assert result.cefr_sp_enabled is True
        assert result.cefr_sp_sentence_count == 0
        assert result.cefr_sp_level is None
        assert result.cefr_sp_level_ordinal is None
        assert result.cefr_sp_pct_a1 is None
        assert result.cefr_sp_adjacency is None
        assert scorer.calls == 0

    @pytest.mark.parametrize(
        "method,model_factory,extra_kwargs",
        [
            ("run", lambda s: s.MockSuccessModel(), {}),
            ("run_beam", lambda s: s.MockBeamModel(), {"beam_width": 4}),
            ("run_guided", lambda s: s.MockGuidedModel(), {}),
            (
                "run_kvl_beam",
                lambda s: s.MockKvlBeamModel(),
                {"beam_width": 4, "branch_factor": 10},
            ),
        ],
    )
    def test_all_run_methods_pass_cefr_sp_metrics(
        self, method, model_factory, extra_kwargs
    ):
        rows = [_level_row(0), _level_row(1)]
        scorer = MockCefrSpScorer(rows)
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=True,
        )
        pipeline = ExperimentPipeline(cefr_sp_scorer=scorer)
        runner = getattr(pipeline, method)
        result = runner("Hi?", config, model_factory(self), **extra_kwargs)

        assert isinstance(result, ExperimentResult)
        assert result.cefr_sp_enabled is True
        assert result.cefr_sp_sentence_count == 2
        assert result.cefr_sp_level_ordinal == pytest.approx(0.5)
        assert result.cefr_sp_adjacency == pytest.approx(1.0)
        assert scorer.calls == 1

    def test_meets_a1_criteria_follows_cefr_sp_level(self):
        """Primary gate is CEFR-SP document level A1, not FK/Fog/Spache."""
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=True,
        )

        class SimpleModel:
            def generate(self, prompt, config):
                return {
                    "response": (
                        "A friend is a person you like. You talk to a friend. "
                        "You play with a friend. A friend helps you."
                    ),
                    "response_time_seconds": 1.0,
                    "generation_successful": True,
                }

        hard = MockCefrSpScorer([_level_row(5)] * 4)
        hard_result = ExperimentPipeline(cefr_sp_scorer=hard).run(
            "What is a friend?", config, SimpleModel()
        )
        assert hard_result.cefr_sp_level == "C2"
        assert hard_result.meets_a1_criteria is False

        easy = MockCefrSpScorer([_level_row(0)] * 4)
        easy_result = ExperimentPipeline(cefr_sp_scorer=easy).run(
            "What is a friend?", config, SimpleModel()
        )
        assert easy_result.cefr_sp_level == "A1"
        assert easy_result.meets_a1_criteria is True

    def test_summary_includes_cefr_sp_stats(self):
        rows = [_level_row(0), _level_row(0)]
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=True,
        )
        pipeline = ExperimentPipeline(cefr_sp_scorer=MockCefrSpScorer(rows))
        result = pipeline.run("Hi?", config, self.MockSuccessModel())

        summary = compute_summary_stats([result])
        assert "cefr_sp_level_ordinal" in summary["overall"]
        assert "cefr_sp_pct_a1" in summary["overall"]
        assert "cefr_sp_adjacency" in summary["overall"]
        assert "cefr_sp_max_level_ordinal" in summary["overall"]
        assert "cefr_sp_expected_level" in summary["overall"]

    def test_numeric_summary_columns_include_cefr_sp(self):
        for col in (
            "cefr_sp_level_ordinal",
            "cefr_sp_pct_a1",
            "cefr_sp_adjacency",
            "cefr_sp_max_level_ordinal",
            "cefr_sp_expected_level",
        ):
            assert col in NUMERIC_SUMMARY_COLUMNS

    def test_disabled_cefr_omitted_from_summary_stats(self):
        config = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=False,
        )
        result = ExperimentPipeline().run("Hi?", config, self.MockSuccessModel())
        summary = compute_summary_stats([result])
        assert "cefr_sp_level_ordinal" not in summary["overall"]
        assert "cefr_sp_adjacency" not in summary["overall"]

    def test_scorer_cache_invalidates_on_ckpt_path_change(self):
        pipeline = ExperimentPipeline()
        c1 = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=True,
            cefr_sp_ckpt_path="/tmp/cefr_a.ckpt",
        )
        c2 = ExperimentConfig(
            model_name="Qwen3",
            prompt_id="p01",
            enable_cefr_sp=True,
            cefr_sp_ckpt_path="/tmp/cefr_b.ckpt",
        )
        s1 = pipeline._resolve_cefr_sp_scorer(c1)
        s2 = pipeline._resolve_cefr_sp_scorer(c2)
        assert s1 is not s2
        assert s1.ckpt_path == "/tmp/cefr_a.ckpt"
        assert s2.ckpt_path == "/tmp/cefr_b.ckpt"


@pytest.mark.slow
def test_real_ckpt_smoke():
    """Optional smoke test against the Zenodo checkpoint (skipped if missing)."""
    ckpt = Path(DEFAULT_CEFR_SP_CKPT)
    if not ckpt.is_file():
        pytest.skip(f"CEFR-SP ckpt not found at {ckpt}")

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("pytorch_lightning")
    del torch

    scorer = CefrSpScorer(ckpt_path=str(ckpt), device="cpu")
    easy = compute_cefr_sp_metrics(
        "I like cats. They are soft and nice.",
        scorer=scorer,
        enabled=True,
    )
    hard = compute_cefr_sp_metrics(
        "The epistemological ramifications of ontological pluralism necessitate "
        "a hermeneutic reappraisal of dialectical materialism.",
        scorer=scorer,
        enabled=True,
    )

    assert easy["cefr_sp_enabled"] is True
    assert easy["cefr_sp_sentence_count"] >= 1
    assert easy["cefr_sp_level"] in {"A1", "A2", "B1", "B2", "C1", "C2"}
    assert easy["cefr_sp_level_ordinal"] is not None
    assert 0.0 <= float(easy["cefr_sp_level_ordinal"]) <= 5.0
    assert easy["cefr_sp_max_level_ordinal"] is not None
    assert int(easy["cefr_sp_max_level_ordinal"]) >= 0
    assert easy["cefr_sp_expected_level"] is not None

    assert hard["cefr_sp_enabled"] is True
    assert hard["cefr_sp_level_ordinal"] is not None
    # Easy text should score lower (simpler) than dense academic prose.
    assert float(easy["cefr_sp_level_ordinal"]) < float(hard["cefr_sp_level_ordinal"])
    assert float(easy["cefr_sp_expected_level"]) < float(hard["cefr_sp_expected_level"])
