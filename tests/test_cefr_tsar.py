"""Tests for assessment-only TSAR ModernBERT ensemble cross-check.

No HuggingFace downloads — ensemble forwards are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.evaluation.assessment.bundle import AssessmentBundler
from slm_experiments.evaluation.assessment.cefr_tsar import (
    A1_SIGNAL_FALLBACK_KEY,
    CEFR_TSAR_LEVELS,
    DEFAULT_BATCH_SIZE,
    TSAR_MODELS,
    TSAR_SCORER_NAME,
    aggregate_confidence_max,
    compute_tsar_assessment_summary,
    detect_device,
    disagrees_with_cefr_sp,
    ensure_registered,
    label_to_ordinal,
    resolve_attn_implementation,
    score_cefr_tsar,
)
from slm_experiments.evaluation.assessment.scorers import (
    clear_scorers,
    list_scorers,
    score_items,
)
from slm_experiments.core.result import ExperimentResult


def _make_result(
    *,
    model: str = "Qwen3",
    prompt_id: str = "p01",
    cleaned: str = "A friend is a person you like.",
    successful: bool = True,
    cefr_sp_level: Optional[str] = "A1",
    weight_factor: float = 1.0,
    hit_max_tokens: bool = False,
) -> ExperimentResult:
    text = cleaned if successful else ""
    return ExperimentResult(
        experiment_id="exp",
        timestamp=datetime(2026, 8, 1, 12, 0, 0),
        config_name="weights",
        prompt=f"What about {prompt_id}?",
        system_prompt="",
        response=text,
        model=model,
        model_id=model,
        config_weighting=False,
        config_prompting=True,
        prompt_id=prompt_id,
        weight_factor=weight_factor,
        temperature=0.0,
        response_time_seconds=1.0,
        generation_successful=successful,
        hit_max_tokens=hit_max_tokens,
        meets_a1_criteria=bool(successful and cefr_sp_level == "A1"),
        flesch_kincaid_grade=2.0,
        gunning_fog=3.0,
        spache_readability=2.0,
        word_count=5 if successful else 0,
        difficult_words=0,
        cleaned_response=text,
        cefr_sp_enabled=cefr_sp_level is not None,
        cefr_sp_level=cefr_sp_level if successful else None,
        cefr_sp_level_ordinal=(
            0.0 if cefr_sp_level == "A1" else (2.0 if cefr_sp_level else None)
        ),
    )


class TestPinnedConfig:
    def test_three_pinned_revisions(self):
        assert len(TSAR_MODELS) == 3
        by_key = {m["key"]: m for m in TSAR_MODELS}
        assert by_key["doc_en"]["revision"].startswith("3c29f5f")
        assert by_key["doc_en"]["revision"] == (
            "3c29f5fbcdc753e99bb437ff9303df983486915b"
        )
        assert by_key["doc_sent_en"]["revision"] == (
            "b00d1d4780e46f6410ea8a8509649044dee18298"
        )
        assert by_key["reference_alllang"]["revision"] == (
            "83337437aa82277e96b293665dc3186088a4a839"
        )
        assert A1_SIGNAL_FALLBACK_KEY == "doc_en"
        assert CEFR_TSAR_LEVELS == ("A1", "A2", "B1", "B2", "C1", "C2")

    def test_ordinal_is_index_plus_one(self):
        assert label_to_ordinal("A1") == 1
        assert label_to_ordinal("A2") == 2
        assert label_to_ordinal("B1") == 3
        assert label_to_ordinal("C2") == 6
        assert label_to_ordinal("a1") == 1
        assert label_to_ordinal(None) is None
        assert label_to_ordinal("XX") is None


class TestConfidenceMax:
    def test_picks_highest_softmax_not_majority(self):
        # Majority would be A1 (two votes); confidence-max must pick B2.
        members = [
            {"key": "doc_en", "label": "A1", "score": 0.55},
            {"key": "doc_sent_en", "label": "B2", "score": 0.92},
            {"key": "reference_alllang", "label": "A1", "score": 0.60},
        ]
        best = aggregate_confidence_max(members)
        assert best["label"] == "B2"
        assert best["score"] == pytest.approx(0.92)
        assert best["key"] == "doc_sent_en"
        assert best["ordinal"] == 4

    def test_tie_breaks_by_first_highest(self):
        members = [
            {"key": "doc_en", "label": "A2", "score": 0.80},
            {"key": "doc_sent_en", "label": "B1", "score": 0.80},
            {"key": "reference_alllang", "label": "A1", "score": 0.10},
        ]
        best = aggregate_confidence_max(members)
        assert best["label"] == "A2"
        assert best["key"] == "doc_en"


class TestDisagreement:
    def test_flag_true_when_labels_differ(self):
        assert disagrees_with_cefr_sp("A1", "B1") is True

    def test_flag_false_when_labels_match(self):
        assert disagrees_with_cefr_sp("A1", "A1") is False
        assert disagrees_with_cefr_sp("b2", "B2") is False

    def test_flag_none_when_either_missing(self):
        assert disagrees_with_cefr_sp(None, "A1") is None
        assert disagrees_with_cefr_sp("A1", None) is None
        assert disagrees_with_cefr_sp("", "A1") is None


class TestDeviceAndAttention:
    def test_detect_device_prefers_cuda_then_mps_then_cpu(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": torch_mod}):
            assert detect_device() == "cuda"

            torch_mod.cuda.is_available.return_value = False
            torch_mod.backends.mps.is_available.return_value = True
            assert detect_device() == "mps"

            torch_mod.backends.mps.is_available.return_value = False
            assert detect_device() == "cpu"

    def test_p100_forces_eager_never_flash(self):
        torch_mod = MagicMock()
        torch_mod.cuda.is_available.return_value = True
        torch_mod.cuda.get_device_capability.return_value = (6, 0)  # sm_60
        with patch.dict("sys.modules", {"torch": torch_mod}):
            assert resolve_attn_implementation("cuda") == "eager"

        torch_mod.cuda.get_device_capability.return_value = (8, 0)  # Ampere
        with patch.dict("sys.modules", {"torch": torch_mod}):
            assert resolve_attn_implementation("cuda") == "sdpa"

        assert resolve_attn_implementation("cpu") == "sdpa"
        assert resolve_attn_implementation("mps") == "sdpa"
        for device in ("cuda", "cpu", "mps"):
            assert resolve_attn_implementation(device) != "flash_attention_2"


def _mock_member_outputs() -> Dict[str, List[Dict[str, Any]]]:
    """Per-model top-1 predictions for two texts (confidence-max → B1, A2)."""
    return {
        "doc_en": [
            {"label": "A1", "score": 0.40},
            {"label": "A2", "score": 0.88},
        ],
        "doc_sent_en": [
            {"label": "B1", "score": 0.95},
            {"label": "A1", "score": 0.50},
        ],
        "reference_alllang": [
            {"label": "A2", "score": 0.70},
            {"label": "A2", "score": 0.60},
        ],
    }


class TestMockedEnsembleScorer:
    def setup_method(self):
        clear_scorers()
        ensure_registered()

    def teardown_method(self):
        clear_scorers()

    def test_scorer_registers_as_cefr_tsar(self):
        assert TSAR_SCORER_NAME in list_scorers()

    def test_confidence_max_and_disagreement_without_download(self):
        items = pd.DataFrame(
            {
                "item_id": ["i1", "i2"],
                "cleaned_response": [
                    "A friend is a person you like.",
                    "The quantum chromodynamics Lagrangian.",
                ],
                "cefr_sp_level": ["A1", "A2"],
            }
        )
        outputs = _mock_member_outputs()

        def fake_predict(texts, *, models=None, device=None, batch_size=8):
            assert len(texts) == 2
            # Return list of per-item member dicts.
            rows = []
            for i in range(len(texts)):
                members = []
                for m in TSAR_MODELS:
                    pred = outputs[m["key"]][i]
                    members.append(
                        {
                            "key": m["key"],
                            "label": pred["label"],
                            "score": pred["score"],
                        }
                    )
                rows.append(members)
            return rows

        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            scored = score_cefr_tsar(items)

        assert len(scored) == 2
        # Item 1: confidence-max picks B1 @ 0.95 (not majority A-ish).
        r1 = scored.loc[scored["item_id"] == "i1"].iloc[0]
        assert r1["cefr_tsar_ensemble_label"] == "B1"
        assert r1["cefr_tsar_ensemble_ordinal"] == 3
        assert r1["cefr_tsar_ensemble_confidence"] == pytest.approx(0.95)
        assert r1["cefr_tsar_doc_sent_en_label"] == "B1"
        assert r1["cefr_tsar_status"] == "ok"
        assert r1["cefr_tsar_disagrees_with_cefr_sp"] == True  # noqa: E712  A1 vs B1

        # Item 2: confidence-max picks A2 @ 0.88 from doc_en.
        r2 = scored.loc[scored["item_id"] == "i2"].iloc[0]
        assert r2["cefr_tsar_ensemble_label"] == "A2"
        assert r2["cefr_tsar_ensemble_ordinal"] == 2
        assert r2["cefr_tsar_ensemble_confidence"] == pytest.approx(0.88)
        assert r2["cefr_tsar_disagrees_with_cefr_sp"] == False  # noqa: E712  A2 vs A2

    def test_empty_response_is_missing_not_scored(self):
        items = pd.DataFrame(
            {
                "item_id": ["empty"],
                "cleaned_response": ["   "],
                "cefr_sp_level": ["A1"],
            }
        )
        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels"
        ) as predict:
            scored = score_cefr_tsar(items)
        predict.assert_not_called()
        row = scored.iloc[0]
        assert row["cefr_tsar_status"] == "missing"
        assert pd.isna(row["cefr_tsar_ensemble_label"]) or row[
            "cefr_tsar_ensemble_label"
        ] in (None, "")
        assert row["cefr_tsar_disagrees_with_cefr_sp"] is None or pd.isna(
            row["cefr_tsar_disagrees_with_cefr_sp"]
        )

    def test_failed_generation_marked_missing_even_with_text(self):
        items = pd.DataFrame(
            {
                "item_id": ["i1", "i2"],
                "cleaned_response": ["Hello friend.", "Also text."],
                "cefr_sp_level": ["A1", "A1"],
                "generation_successful": [False, True],
            }
        )

        def fake_predict(texts, *, models=None, device=None, batch_size=8):
            assert texts == ["Also text."]
            return [
                [
                    {"key": m["key"], "label": "A1", "score": 0.9}
                    for m in TSAR_MODELS
                ]
            ]

        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            scored = score_cefr_tsar(items)
        by_id = scored.set_index("item_id")
        assert by_id.loc["i1", "cefr_tsar_status"] == "missing"
        assert by_id.loc["i2", "cefr_tsar_status"] == "ok"

    def test_ensure_registered_uses_public_api_not_private_registry(self):
        from slm_experiments.evaluation.assessment import scorers as scorers_mod

        clear_scorers()
        assert TSAR_SCORER_NAME not in list_scorers()
        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.ensure_scorer_registered",
            wraps=scorers_mod.ensure_scorer_registered,
        ) as ensure_fn:
            ensure_registered()
            ensure_fn.assert_called_once_with(TSAR_SCORER_NAME, score_cefr_tsar)
        assert TSAR_SCORER_NAME in list_scorers()
        # Idempotent — second call is a no-op via the public API.
        assert scorers_mod.ensure_scorer_registered(TSAR_SCORER_NAME, score_cefr_tsar) is False
        assert list_scorers().count(TSAR_SCORER_NAME) == 1

    def test_predict_error_writes_error_state(self):
        items = pd.DataFrame(
            {
                "item_id": ["i1"],
                "cleaned_response": ["Hello friend."],
                "cefr_sp_level": ["A1"],
            }
        )
        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=RuntimeError("no torch"),
        ):
            scored = score_cefr_tsar(items)
        assert scored.iloc[0]["cefr_tsar_status"] == "error"
        assert "no torch" in str(scored.iloc[0]["cefr_tsar_error"])

    def test_unnormalizable_winning_label_is_error_not_ok(self):
        items = pd.DataFrame(
            {
                "item_id": ["i1"],
                "cleaned_response": ["Hello friend."],
                "cefr_sp_level": ["A1"],
            }
        )

        def fake_predict(texts, *, models=None, device=None, batch_size=8):
            return [
                [
                    {"key": "doc_en", "label": "NOT_A_LEVEL", "score": 0.99},
                    {"key": "doc_sent_en", "label": "A1", "score": 0.40},
                    {"key": "reference_alllang", "label": "A2", "score": 0.30},
                ]
            ]

        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            scored = score_cefr_tsar(items)

        row = scored.iloc[0]
        assert row["cefr_tsar_status"] == "error"
        assert "unnormalizable" in str(row["cefr_tsar_error"]).lower()
        assert pd.isna(row["cefr_tsar_ensemble_label"]) or row[
            "cefr_tsar_ensemble_label"
        ] in (None, "")
        assert pd.isna(row["cefr_tsar_ensemble_ordinal"]) or row[
            "cefr_tsar_ensemble_ordinal"
        ] in (None,)
        assert scored.iloc[0]["cefr_tsar_disagrees_with_cefr_sp"] is None or pd.isna(
            scored.iloc[0]["cefr_tsar_disagrees_with_cefr_sp"]
        )

    def test_score_items_merges_tsar_columns(self):
        items = pd.DataFrame(
            {
                "item_id": ["a"],
                "cleaned_response": ["Hi."],
                "cefr_sp_level": ["A1"],
            }
        )

        def fake_predict(texts, **kwargs):
            return [
                [
                    {"key": "doc_en", "label": "A1", "score": 0.9},
                    {"key": "doc_sent_en", "label": "A2", "score": 0.5},
                    {"key": "reference_alllang", "label": "A2", "score": 0.4},
                ]
            ]

        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            merged = score_items(items)
        assert "cefr_tsar_ensemble_label" in merged.columns
        assert merged.iloc[0]["cefr_tsar_ensemble_label"] == "A1"
        assert merged.iloc[0]["cefr_tsar_disagrees_with_cefr_sp"] == False  # noqa: E712


class TestAssessmentSummary:
    def test_summary_beside_failure_and_truncation(self):
        scores = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "cefr_tsar_ensemble_ordinal": 3,
                    "cefr_tsar_ensemble_label": "B1",
                    "cefr_tsar_status": "ok",
                    "cefr_tsar_disagrees_with_cefr_sp": True,
                },
                {
                    "item_id": "i2",
                    "cefr_tsar_ensemble_ordinal": 1,
                    "cefr_tsar_ensemble_label": "A1",
                    "cefr_tsar_status": "ok",
                    "cefr_tsar_disagrees_with_cefr_sp": False,
                },
            ]
        )
        item_map = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "model": "Qwen3",
                    "weight_factor": 1.0,
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                },
                {
                    "item_id": "i1",
                    "model": "TinyLlama",
                    "weight_factor": 2.0,
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                },
                {
                    "item_id": "i2",
                    "model": "Qwen3",
                    "weight_factor": 1.0,
                    "generation_successful": True,
                    "hit_max_tokens": True,
                    "in_sample": True,
                },
                {
                    "item_id": "",
                    "model": "Qwen3",
                    "weight_factor": 1.0,
                    "generation_successful": False,
                    "hit_max_tokens": False,
                    "in_sample": False,
                },
            ]
        )
        summary = compute_tsar_assessment_summary(scores, item_map)
        overall = summary["overall"]
        assert "generation_failure_rate" in overall
        assert "hit_max_tokens_rate" in overall
        assert overall["generation_failure_rate"] == pytest.approx(0.25)
        assert overall["hit_max_tokens_rate"] == pytest.approx(0.25)
        assert overall["cefr_tsar_mean_ordinal"] == pytest.approx(2.0)
        assert overall["cefr_tsar_predicted_a1_rate"] == pytest.approx(0.5)
        assert overall["cefr_tsar_disagreement_rate"] == pytest.approx(0.5)

        assert "Qwen3" in summary["by_model"]
        qwen = summary["by_model"]["Qwen3"]
        assert "generation_failure_rate" in qwen
        assert "cefr_tsar_mean_ordinal" in qwen

        assert "by_weight_factor" in summary
        assert "1" in summary["by_weight_factor"] or "1.0" in summary["by_weight_factor"]
        # Single-family → one-element list (not overwritten string).
        assert summary["metadata"]["sweep_dimension"] == ["weight_factor"]
        assert "weight_factor" in summary["metadata"]["sweep_values"]

    def test_sweep_dimension_list_for_mixed_families(self):
        scores = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "cefr_tsar_ensemble_ordinal": 1,
                    "cefr_tsar_ensemble_label": "A1",
                    "cefr_tsar_status": "ok",
                    "cefr_tsar_disagrees_with_cefr_sp": False,
                },
            ]
        )
        item_map = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "model": "Qwen3",
                    "weight_factor": 1.0,
                    "num_shots": 0,
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                },
                {
                    "item_id": "i1",
                    "model": "Qwen3",
                    "weight_factor": 2.0,
                    "num_shots": 3,
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                },
            ]
        )
        summary = compute_tsar_assessment_summary(scores, item_map)
        dims = summary["metadata"]["sweep_dimension"]
        assert isinstance(dims, list)
        assert "weight_factor" in dims
        assert "num_shots" in dims
        assert set(dims) == set(summary["metadata"]["sweep_values"].keys())
        assert "by_weight_factor" in summary
        assert "by_num_shots" in summary


class TestBundleIntegration:
    def setup_method(self):
        clear_scorers()
        ensure_registered()

    def teardown_method(self):
        clear_scorers()

    def test_assess_build_writes_tsar_scores_and_summary(self, tmp_path: Path):
        store = RunStore(tmp_path)
        started = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        run_id = make_run_id(2, "weights", started_at=started.replace(tzinfo=None))
        results = [
            _make_result(model="Qwen3", prompt_id="p01", cefr_sp_level="A1"),
            _make_result(
                model="TinyLlama",
                prompt_id="p01",
                cleaned="A friend is a person you like.",
                cefr_sp_level="A1",
                weight_factor=2.0,
            ),
            _make_result(
                model="Qwen3",
                prompt_id="p02",
                cleaned="Dogs are animals.",
                cefr_sp_level="A2",
            ),
            _make_result(
                model="Qwen2",
                prompt_id="p01",
                cleaned="",
                successful=False,
                cefr_sp_level=None,
                hit_max_tokens=True,
            ),
        ]
        store.write_bundle(
            run_id,
            results,
            phase=2,
            experiment="weights",
            started_at=started,
        )

        def fake_predict(texts, **kwargs):
            out = []
            for _ in texts:
                out.append(
                    [
                        {"key": "doc_en", "label": "A1", "score": 0.3},
                        {"key": "doc_sent_en", "label": "B1", "score": 0.9},
                        {"key": "reference_alllang", "label": "A2", "score": 0.5},
                    ]
                )
            return out

        bundler = AssessmentBundler(results_root=tmp_path)
        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            assess_id, out_dir = bundler.build([run_id])

        scores = pd.read_csv(out_dir / "scores.csv")
        assert "cefr_tsar_ensemble_label" in scores.columns
        assert (scores["cefr_tsar_ensemble_label"] == "B1").all()
        assert "cefr_tsar_disagrees_with_cefr_sp" in scores.columns

        summary_path = out_dir / "summary.json"
        assert summary_path.exists()
        summary = store.read_summary(assess_id)
        assert "cefr_tsar_mean_ordinal" in summary["overall"]
        assert "generation_failure_rate" in summary["overall"]
        assert "hit_max_tokens_rate" in summary["overall"]
        assert "by_model" in summary

        # Source generation run untouched (no TSAR columns on full.csv).
        source_full = store.read_full_csv(run_id)
        assert "cefr_tsar_ensemble_label" not in source_full.columns

        # C1: meets_a1_criteria still CEFR-SP only on source; TSAR never gates.
        assert "meets_a1_criteria" not in scores.columns


class TestTsarDeviceBatchPassthrough:
    def test_score_cefr_tsar_forwards_device_and_batch_size(self):
        items = pd.DataFrame(
            {
                "item_id": ["i1"],
                "cleaned_response": ["Hello friend."],
                "cefr_sp_level": ["A1"],
                "generation_successful": [True],
            }
        )
        seen: dict = {}

        def fake_predict(texts, *, models=None, device=None, batch_size=8):
            seen["device"] = device
            seen["batch_size"] = batch_size
            return [
                [
                    {"key": "doc_en", "label": "A1", "score": 0.9},
                    {"key": "doc_sent_en", "label": "A1", "score": 0.8},
                    {"key": "reference_alllang", "label": "A1", "score": 0.7},
                ]
            ]

        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            score_cefr_tsar(items, device="cpu", batch_size=16)

        assert seen["device"] == "cpu"
        assert seen["batch_size"] == 16

    def test_score_items_scorer_options_reach_tsar(self):
        from slm_experiments.evaluation.assessment.scorers import (
            clear_scorers,
            score_items,
        )

        clear_scorers()
        ensure_registered()
        items = pd.DataFrame(
            {
                "item_id": ["i1"],
                "cleaned_response": ["Hello friend."],
                "cefr_sp_level": ["A1"],
                "generation_successful": [True],
            }
        )
        seen: dict = {}

        def fake_predict(texts, *, models=None, device=None, batch_size=8):
            seen["device"] = device
            seen["batch_size"] = batch_size
            return [
                [
                    {"key": "doc_en", "label": "A1", "score": 0.9},
                    {"key": "doc_sent_en", "label": "A1", "score": 0.8},
                    {"key": "reference_alllang", "label": "A1", "score": 0.7},
                ]
            ]

        with patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=fake_predict,
        ):
            score_items(
                items,
                scorer_options={"cefr_tsar": {"device": "mps", "batch_size": 4}},
            )

        assert seen == {"device": "mps", "batch_size": 4}
        clear_scorers()

    def test_cli_assess_build_exposes_tsar_flags(self):
        from slm_experiments.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "assess",
                "build",
                "--source-run-ids",
                "run1",
                "--cefr-tsar-device",
                "cuda",
                "--cefr-tsar-batch-size",
                "32",
            ]
        )
        assert args.cefr_tsar_device == "cuda"
        assert args.cefr_tsar_batch_size == 32

        defaults = parser.parse_args(
            ["assess", "build", "--source-run-ids", "run1"]
        )
        assert defaults.cefr_tsar_device is None
        assert defaults.cefr_tsar_batch_size == DEFAULT_BATCH_SIZE
