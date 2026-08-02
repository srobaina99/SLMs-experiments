"""Tests for paired-delta + percentile bootstrap assessment analysis (issue #19)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from slm_experiments.cli import main
from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.evaluation.assessment import AssessmentBundler
from slm_experiments.human.rubric import (
    ANSWER_ADEQUACY,
    OVERALL_SUITABILITY,
    is_human_suitable,
)
from slm_experiments.human.study_export import arm_label


class TestAnalysisImportSurface:
    """Smoke: facade re-exports survive the D1 split."""

    def test_public_symbols_importable_from_facade(self):
        from slm_experiments.evaluation.assessment import analysis
        from slm_experiments.evaluation.assessment import analysis_helpers
        from slm_experiments.evaluation.assessment import paired_deltas
        from slm_experiments.evaluation.assessment import validation

        assert analysis.analyze_assessment_bundle is not None
        assert analysis.build_paired_deltas is paired_deltas.build_paired_deltas
        assert analysis.build_validation_section is validation.build_validation_section
        assert analysis.resolve_arm is analysis_helpers.resolve_arm
        assert analysis.DEFAULT_RESAMPLES == analysis_helpers.DEFAULT_RESAMPLES


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)
ALT_RESPONSE = "A dog is an animal. It is small and friendly."
HARD_RESPONSE = (
    "Photosynthesis constitutes a sophisticated biochemical cascade wherein "
    "chloroplasts orchestrate photon-driven phosphorylation."
)


class MockSuccessModel:
    def __init__(self, response: str = SIMPLE_RESPONSE):
        self.response = response

    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": self.response,
            "response_time_seconds": 2.0,
            "generation_successful": True,
        }


class MockFailureModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.5,
            "generation_successful": False,
        }


def _write_generation_bundle(
    store: RunStore,
    results: list,
    *,
    experiment: str = "weights",
    phase: int = 2,
    started: datetime | None = None,
) -> str:
    started = started or datetime(2026, 6, 6, 14, 30, 22, tzinfo=timezone.utc)
    run_id = make_run_id(phase, experiment, started_at=started.replace(tzinfo=None))
    store.write_bundle(
        run_id,
        results,
        phase=phase,
        experiment=experiment,
        cli_args=["--prompts", "2"],
        models=sorted({r.model for r in results}),
        prompt_count=len({r.prompt_id for r in results}),
        started_at=started,
        completed_at=datetime(2026, 6, 6, 14, 31, 0, tzinfo=timezone.utc),
    )
    return run_id


def _fake_score_items(items: pd.DataFrame, **_kwargs) -> pd.DataFrame:
    """Deterministic fake TSAR + KVL scores keyed by response text length."""
    rows = []
    for _, item in items.iterrows():
        text = str(item.get("cleaned_response", "") or "")
        # Longer / "hard" text → higher ordinal (harder), lower KVL.
        hard = len(text) > 80
        rows.append(
            {
                "item_id": item["item_id"],
                "cefr_tsar_ensemble_ordinal": 4.0 if hard else 1.0,
                "cefr_tsar_ensemble_label": "C1" if hard else "A1",
                "cefr_tsar_status": "ok",
                "cefr_tsar_disagrees_with_cefr_sp": False,
                "kvl_v2_mean_score": -2.0 if hard else 3.0,
                "kvl_v2_lookup_coverage": 1.0,
                "kvl_v2_hard_token_share": 0.4 if hard else 0.05,
                "kvl_v2_lower_tail_score": -3.0 if hard else 1.0,
                "kvl_v2_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _build_weights_assessment(tmp_path: Path) -> tuple[str, RunStore]:
    """Weights sweep: baseline weight=1.0 + intervention weight=2.0, two prompts."""
    pipeline = ExperimentPipeline()
    results = []
    for prompt_id, prompt, base_text, int_text in [
        ("p01", "What is a friend?", SIMPLE_RESPONSE, ALT_RESPONSE),
        ("p02", "What is a dog?", ALT_RESPONSE, HARD_RESPONSE),
    ]:
        for weight, text in [(1.0, base_text), (2.0, int_text)]:
            result = pipeline.run(
                prompt,
                ExperimentConfig(
                    model_name="Qwen3",
                    config_weighting=True,
                    config_prompting=True,
                    prompt_id=prompt_id,
                    weight_factor=weight,
                    num_shots=0,
                ),
                MockSuccessModel(text),
            )
            # Force distinct CEFR-SP ordinals for paired-delta checks.
            if weight == 1.0:
                result.cefr_sp_level_ordinal = 2.0
                result.meets_a1_criteria = False
                result.cefr_sp_level = "A2"
            else:
                result.cefr_sp_level_ordinal = 0.5
                result.meets_a1_criteria = True
                result.cefr_sp_level = "A1"
            results.append(result)

    # One failure on intervention arm (p03) so rates use all rows.
    fail = pipeline.run(
        "What is water?",
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=True,
            config_prompting=True,
            prompt_id="p03",
            weight_factor=2.0,
            num_shots=0,
        ),
        MockFailureModel(),
    )
    fail.cleaned_response = ""
    fail.hit_max_tokens = True
    results.append(fail)

    # Matching baseline success for p03 (pair drops for ordinals; rates keep both).
    base_p03 = pipeline.run(
        "What is water?",
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=True,
            config_prompting=True,
            prompt_id="p03",
            weight_factor=1.0,
            num_shots=0,
        ),
        MockSuccessModel(SIMPLE_RESPONSE),
    )
    base_p03.cefr_sp_level_ordinal = 1.0
    base_p03.meets_a1_criteria = False
    base_p03.cefr_sp_level = "A2"
    results.append(base_p03)

    store = RunStore(tmp_path)
    source_id = _write_generation_bundle(store, results, experiment="weights")
    with patch(
        "slm_experiments.evaluation.assessment.bundle.score_items",
        side_effect=_fake_score_items,
    ):
        assess_id, _ = AssessmentBundler(results_root=tmp_path).build(
            [source_id], seed=42
        )
    return assess_id, store


# ---------------------------------------------------------------------------
# Pure-function tests (no bundle) — written before analysis.py exists
# ---------------------------------------------------------------------------


class TestPercentileBootstrap:
    def test_determinism_under_fixed_seed(self):
        from slm_experiments.evaluation.assessment.analysis import (
            percentile_bootstrap_ci,
        )

        values = np.array([0.1, -0.2, 0.3, -0.05, 0.15], dtype=float)
        a = percentile_bootstrap_ci(values, n_resamples=500, ci=0.95, seed=42)
        b = percentile_bootstrap_ci(values, n_resamples=500, ci=0.95, seed=42)
        assert a == b
        assert a["ci_low"] <= a["point_estimate"] <= a["ci_high"]

    def test_ci_excludes_zero_flag(self):
        from slm_experiments.evaluation.assessment.analysis import (
            ci_excludes_zero,
            direction_from_ci,
            percentile_bootstrap_ci,
        )

        # All-positive easier deltas → CI should exclude 0 toward easier.
        values = np.array([0.5, 0.6, 0.55, 0.7, 0.65], dtype=float)
        result = percentile_bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=7)
        assert ci_excludes_zero(result["ci_low"], result["ci_high"]) is True
        assert direction_from_ci(result["ci_low"], result["ci_high"]) == "easier"

        # Symmetric around 0 → CI includes 0.
        centered = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=float)
        mid = percentile_bootstrap_ci(centered, n_resamples=2000, ci=0.95, seed=7)
        assert ci_excludes_zero(mid["ci_low"], mid["ci_high"]) is False
        assert direction_from_ci(mid["ci_low"], mid["ci_high"]) == "none"

        # All-negative → harder.
        negative = np.array([-0.5, -0.6, -0.55, -0.7, -0.65], dtype=float)
        hard = percentile_bootstrap_ci(negative, n_resamples=2000, ci=0.95, seed=7)
        assert ci_excludes_zero(hard["ci_low"], hard["ci_high"]) is True
        assert direction_from_ci(hard["ci_low"], hard["ci_high"]) == "harder"

    def test_empty_singleton_and_nonpositive_resamples(self):
        from slm_experiments.evaluation.assessment.analysis import (
            ci_excludes_zero,
            direction_from_ci,
            percentile_bootstrap_ci,
        )

        empty = percentile_bootstrap_ci([], n_resamples=100)
        assert np.isnan(empty["point_estimate"])
        assert np.isnan(empty["ci_low"]) and np.isnan(empty["ci_high"])
        assert ci_excludes_zero(empty["ci_low"], empty["ci_high"]) is False
        assert direction_from_ci(empty["ci_low"], empty["ci_high"]) == "none"

        all_nan = percentile_bootstrap_ci([np.nan, np.nan], n_resamples=100)
        assert np.isnan(all_nan["point_estimate"])

        alone = percentile_bootstrap_ci([1.5], n_resamples=500)
        assert alone == {"point_estimate": 1.5, "ci_low": 1.5, "ci_high": 1.5}

        zero_resamples = percentile_bootstrap_ci([1.0, 2.0], n_resamples=0)
        assert zero_resamples["point_estimate"] == pytest.approx(1.5)
        assert zero_resamples["ci_low"] == zero_resamples["ci_high"] == pytest.approx(1.5)

    def test_group_seed_independent_of_other_groups(self):
        """Adding an unrelated group must not change an existing group's CI."""
        from slm_experiments.evaluation.assessment.analysis import (
            group_bootstrap_rng,
            percentile_bootstrap_ci,
        )

        values_a = np.array([0.1, 0.2, 0.15, 0.05, 0.25], dtype=float)
        values_b = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)

        rng_a_alone = group_bootstrap_rng(42, "Qwen3|weights|2.0|cefr_sp_level_ordinal")
        alone = percentile_bootstrap_ci(
            values_a, n_resamples=800, ci=0.95, rng=rng_a_alone
        )

        # Derive RNGs for a sorted key list that also includes another model.
        keys = sorted(
            [
                "Phi3|weights|2.0|cefr_sp_level_ordinal",
                "Qwen3|weights|2.0|cefr_sp_level_ordinal",
            ]
        )
        rngs = {k: group_bootstrap_rng(42, k) for k in keys}
        with_extra = percentile_bootstrap_ci(
            values_a,
            n_resamples=800,
            ci=0.95,
            rng=rngs["Qwen3|weights|2.0|cefr_sp_level_ordinal"],
        )
        # Unrelated group's draw must not affect A.
        _ = percentile_bootstrap_ci(
            values_b,
            n_resamples=800,
            ci=0.95,
            rng=rngs["Phi3|weights|2.0|cefr_sp_level_ordinal"],
        )

        assert alone == with_extra

    def test_ci_excludes_zero_false_for_none_bounds(self):
        """None CI bounds (n_pairs=0 path) must not raise or claim excludes-zero."""
        from slm_experiments.evaluation.assessment.analysis import (
            ci_excludes_zero,
            direction_from_ci,
        )

        assert ci_excludes_zero(None, None) is False  # type: ignore[arg-type]
        assert direction_from_ci(None, None) == "none"  # type: ignore[arg-type]


class TestFamilyAwareArm:
    def test_weights_row_with_num_shots_0_is_intervention(self):
        """Regression: weights rows always carry num_shots=0; must not be baseline."""
        row = pd.Series(
            {
                "source_run_id": "20260606_143022_phase2_weights",
                "weight_factor": 2.0,
                "num_shots": 0,
                "guided_top_k": pd.NA,
                "kvl_beam_width": pd.NA,
                "config": "both",
            }
        )
        assert arm_label(row) == "intervention"

        from slm_experiments.evaluation.assessment.analysis import resolve_arm

        assert resolve_arm(row) == "intervention"
        assert (
            resolve_arm(
                pd.Series(
                    {
                        "source_run_id": "20260606_143022_phase2_weights",
                        "weight_factor": 1.0,
                        "num_shots": 0,
                    }
                )
            )
            == "baseline"
        )

    def test_resolve_arm_per_family(self):
        from slm_experiments.evaluation.assessment.analysis import resolve_arm

        assert (
            resolve_arm(
                pd.Series({"num_shots": 0}),
                family="prompting",
            )
            == "baseline"
        )
        assert (
            resolve_arm(
                pd.Series({"num_shots": 3}),
                family="prompting",
            )
            == "intervention"
        )
        assert (
            resolve_arm(
                pd.Series({"guided_top_k": 0}),
                family="guided",
            )
            == "baseline"
        )
        assert (
            resolve_arm(
                pd.Series({"guided_top_k": 10}),
                family="guided",
            )
            == "intervention"
        )
        assert (
            resolve_arm(
                pd.Series({"kvl_beam_width": 1}),
                family="kvl_beam",
            )
            == "baseline"
        )
        assert (
            resolve_arm(
                pd.Series({"kvl_beam_width": 8}),
                family="kvl_beam",
            )
            == "intervention"
        )


class TestPairedDeltaConstruction:
    def test_paired_deltas_intervention_minus_baseline(self):
        from slm_experiments.evaluation.assessment.analysis import (
            build_paired_deltas,
        )

        # Two prompts: intervention easier on CEFR-SP (lower ordinal).
        baseline = pd.DataFrame(
            {
                "prompt_id": ["p01", "p02"],
                "cefr_sp_level_ordinal": [2.0, 3.0],
            }
        )
        intervention = pd.DataFrame(
            {
                "prompt_id": ["p01", "p02"],
                "cefr_sp_level_ordinal": [1.0, 1.5],
            }
        )
        pairs = build_paired_deltas(
            baseline,
            intervention,
            value_col="cefr_sp_level_ordinal",
        )
        assert list(pairs["prompt_id"]) == ["p01", "p02"]
        assert list(pairs["delta"]) == pytest.approx([-1.0, -1.5])
        assert pairs["n_pairs_dropped"].iloc[0] == 0

    def test_incomplete_pairs_dropped(self):
        from slm_experiments.evaluation.assessment.analysis import (
            build_paired_deltas,
        )

        baseline = pd.DataFrame(
            {
                "prompt_id": ["p01", "p02", "p03"],
                "cefr_sp_level_ordinal": [2.0, 3.0, np.nan],
            }
        )
        intervention = pd.DataFrame(
            {
                "prompt_id": ["p01", "p02"],
                "cefr_sp_level_ordinal": [1.0, np.nan],
            }
        )
        pairs = build_paired_deltas(
            baseline,
            intervention,
            value_col="cefr_sp_level_ordinal",
        )
        assert list(pairs["prompt_id"]) == ["p01"]
        assert pairs["delta"].iloc[0] == pytest.approx(-1.0)
        # p02 missing intervention value, p03 missing baseline value + no int row.
        assert int(pairs["n_pairs_dropped"].iloc[0]) == 2

    def test_conflicting_duplicate_prompt_raises(self):
        from slm_experiments.evaluation.assessment.analysis import (
            build_paired_deltas,
        )

        baseline = pd.DataFrame(
            {
                "prompt_id": ["p01", "p01"],
                "cefr_sp_level_ordinal": [2.0, 4.0],
            }
        )
        intervention = pd.DataFrame(
            {"prompt_id": ["p01"], "cefr_sp_level_ordinal": [1.0]}
        )
        with pytest.raises(ValueError, match="Conflicting cefr_sp_level_ordinal"):
            build_paired_deltas(
                baseline,
                intervention,
                value_col="cefr_sp_level_ordinal",
            )

    def test_identical_duplicate_prompt_keeps_one(self):
        from slm_experiments.evaluation.assessment.analysis import (
            build_paired_deltas,
        )

        baseline = pd.DataFrame(
            {
                "prompt_id": ["p01", "p01"],
                "cefr_sp_level_ordinal": [2.0, 2.0],
            }
        )
        intervention = pd.DataFrame(
            {"prompt_id": ["p01"], "cefr_sp_level_ordinal": [1.0]}
        )
        pairs = build_paired_deltas(
            baseline,
            intervention,
            value_col="cefr_sp_level_ordinal",
        )
        assert len(pairs) == 1
        assert pairs["delta"].iloc[0] == pytest.approx(-1.0)

    def test_status_gating_excludes_non_ok_scores(self):
        from slm_experiments.evaluation.assessment.paired_deltas import (
            _metric_frame,
        )

        map_rows = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                "prompt_id": ["p1", "p2", "p3"],
                "in_sample": [True, True, True],
            }
        )
        scores = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                "cefr_tsar_ensemble_ordinal": [1.0, 2.0, 3.0],
                "cefr_tsar_status": ["ok", "error", "missing"],
                "kvl_v2_mean_score": [0.1, 0.2, 0.3],
                "kvl_v2_status": ["ok", "error", "ok"],
            }
        )
        tsar = _metric_frame(map_rows, scores, "cefr_tsar_ensemble_ordinal")
        assert list(tsar["prompt_id"]) == ["p1"]
        kvl = _metric_frame(map_rows, scores, "kvl_v2_mean_score")
        assert list(kvl["prompt_id"]) == ["p1", "p3"]

    def test_missing_status_column_excludes_secondary_metric(self):
        from slm_experiments.evaluation.assessment.paired_deltas import (
            _metric_frame,
        )

        map_rows = pd.DataFrame(
            {
                "item_id": ["a"],
                "prompt_id": ["p1"],
                "in_sample": [True],
            }
        )
        scores = pd.DataFrame(
            {
                "item_id": ["a"],
                "cefr_tsar_ensemble_ordinal": [1.0],
            }
        )
        tsar = _metric_frame(map_rows, scores, "cefr_tsar_ensemble_ordinal")
        assert tsar.empty

    def test_easier_delta_orientation(self):
        from slm_experiments.evaluation.assessment.analysis import (
            easier_deltas,
        )

        raw = np.array([-1.0, -0.5])  # CEFR-SP: negative raw = easier
        assert list(easier_deltas(raw, easier_orientation="lower")) == pytest.approx(
            [1.0, 0.5]
        )
        assert list(easier_deltas(raw, easier_orientation="higher")) == pytest.approx(
            [-1.0, -0.5]
        )

    def test_nonfinite_values_dropped_from_pairs(self):
        """±inf must not survive into n_pairs (inf−inf → NaN delta + fake CI)."""
        from slm_experiments.evaluation.assessment.analysis import (
            build_paired_deltas,
        )

        baseline = pd.DataFrame(
            {
                "prompt_id": ["p01", "p02"],
                "val": [2.0, np.inf],
            }
        )
        intervention = pd.DataFrame(
            {
                "prompt_id": ["p01", "p02"],
                "val": [1.0, np.inf],
            }
        )
        pairs = build_paired_deltas(baseline, intervention, value_col="val")
        assert list(pairs["prompt_id"]) == ["p01"]
        assert pairs["delta"].iloc[0] == pytest.approx(-1.0)
        assert int(pairs["n_pairs_dropped"].iloc[0]) == 1

    def test_stratum_rates_use_all_rows_as_denominator(self):
        from slm_experiments.evaluation.assessment.paired_deltas import _stratum_rates

        df = pd.DataFrame(
            {
                "generation_successful": [True, True, False, False],
                "hit_max_tokens": [False, True, False, False],
                "meets_a1_criteria": [True, False, False, False],
            }
        )
        rates = _stratum_rates(df)
        assert rates["count"] == 4
        assert rates["generation_failure_rate"] == pytest.approx(0.5)
        assert rates["hit_max_tokens_rate"] == pytest.approx(0.25)
        # 1 A1 among 4 rows (not among 2 successes).
        assert rates["a1_pass_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Bundle integration
# ---------------------------------------------------------------------------


class TestItemMapSchemaExtension:
    def test_item_map_carries_cefr_sp_ordinal_and_a1(self, tmp_path: Path):
        assess_id, store = _build_weights_assessment(tmp_path)
        item_map = store.read_item_map_csv(assess_id)
        assert "cefr_sp_level_ordinal" in item_map.columns
        assert "meets_a1_criteria" in item_map.columns
        # Successful baseline rows should carry the forced ordinal.
        ok = item_map[item_map["generation_successful"] == True]  # noqa: E712
        assert ok["cefr_sp_level_ordinal"].notna().any()
        assert ok["meets_a1_criteria"].notna().any()


class TestAnalyzeBundle:
    def test_analyze_rejects_generation_kind(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        pipeline = ExperimentPipeline()
        result = pipeline.run(
            "What is a friend?",
            ExperimentConfig(
                model_name="Qwen3",
                config_weighting=False,
                config_prompting=True,
                prompt_id="p01",
                weight_factor=1.0,
            ),
            MockSuccessModel(SIMPLE_RESPONSE),
        )
        store = RunStore(tmp_path)
        gen_id = _write_generation_bundle(store, [result])
        with pytest.raises(ValueError, match="not kind=assessment"):
            analyze_assessment_bundle(gen_id, results_root=tmp_path)

    def test_writes_artifacts_and_manifest_analysis_block(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        assess_id, store = _build_weights_assessment(tmp_path)
        out = analyze_assessment_bundle(
            assess_id,
            results_root=tmp_path,
            bootstrap_seed=42,
            resamples=200,
        )
        run_dir = store.run_dir(assess_id)
        assert out == run_dir / "analysis"
        assert (out / "paired_deltas.csv").exists()
        assert (out / "analysis.json").exists()

        manifest = store.read_manifest(assess_id)
        assert "analysis" in manifest
        assert manifest["analysis"]["bootstrap_seed"] == 42
        assert manifest["analysis"]["resamples"] == 200
        assert manifest["analysis"]["ci"] == 0.95
        assert manifest["artifacts"]["paired_deltas_csv"] == "analysis/paired_deltas.csv"
        assert manifest["artifacts"]["analysis_json"] == "analysis/analysis.json"
        # Must not overload sampling seed.
        assert manifest["seed"] == 42

        analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert "by_model" in analysis
        assert "metadata" in analysis
        assert "Qwen3" in analysis["by_model"]
        # Stratify by model first; family nested under model.
        family_block = analysis["by_model"]["Qwen3"]["by_family"]["weights"]
        assert family_block["status"] == "ok"

        deltas = pd.read_csv(out / "paired_deltas.csv")
        assert "model" in deltas.columns
        assert "generation_failure_rate" in deltas.columns
        assert "hit_max_tokens_rate" in deltas.columns
        assert "a1_pass_rate_delta" in deltas.columns
        assert "ci_excludes_zero" in deltas.columns
        assert "direction" in deltas.columns
        assert "easier_delta" in deltas.columns
        # Model-first ordering: first column is model.
        assert list(deltas.columns)[0] == "model"

    def test_no_baseline_recorded_not_hard_fail(self, tmp_path: Path, recwarn):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        pipeline = ExperimentPipeline()
        # Intervention only — no weight=1.0 baseline.
        results = [
            pipeline.run(
                "What is a friend?",
                ExperimentConfig(
                    model_name="Qwen3",
                    config_weighting=True,
                    config_prompting=True,
                    prompt_id="p01",
                    weight_factor=2.0,
                    num_shots=0,
                ),
                MockSuccessModel(SIMPLE_RESPONSE),
            )
        ]
        results[0].cefr_sp_level_ordinal = 1.0
        results[0].meets_a1_criteria = True
        store = RunStore(tmp_path)
        source_id = _write_generation_bundle(store, results, experiment="weights")
        with patch(
            "slm_experiments.evaluation.assessment.bundle.score_items",
            side_effect=_fake_score_items,
        ):
            # Clear warnings from auto-analyze inside build before the lock.
            assess_id, _ = AssessmentBundler(results_root=tmp_path).build(
                [source_id], seed=42
            )
        recwarn.clear()

        analyze_assessment_bundle(
            assess_id, results_root=tmp_path, bootstrap_seed=42, resamples=50
        )
        analysis = json.loads(
            (store.run_dir(assess_id) / "analysis" / "analysis.json").read_text(
                encoding="utf-8"
            )
        )
        block = analysis["by_model"]["Qwen3"]["by_family"]["weights"]
        assert block["status"] == "no_baseline"
        assert "reason" in block
        assert analysis["warnings"]
        assert any("baseline" in w.lower() for w in analysis["warnings"])
        # C8: no UserWarning spam on the no_baseline path (INFO log only).
        baseline_user_warnings = [
            w
            for w in recwarn
            if issubclass(w.category, UserWarning)
            and ("baseline" in str(w.message).lower() or "no_baseline" in str(w.message))
        ]
        assert baseline_user_warnings == []

    def test_analyze_cli_dispatches(self, tmp_path: Path, capsys, monkeypatch):
        monkeypatch.setattr(
            "slm_experiments.models.base.REPO_ROOT",
            str(tmp_path),
        )
        results_root = tmp_path / "results"
        assess_id, _ = _build_weights_assessment(results_root)

        main(
            [
                "assess",
                "analyze",
                "--assessment-run-id",
                assess_id,
                "--bootstrap-seed",
                "11",
                "--resamples",
                "50",
            ]
        )
        out = capsys.readouterr().out
        assert "Analysis complete:" in out
        manifest = json.loads(
            (results_root / "runs" / assess_id / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["analysis"]["bootstrap_seed"] == 11
        assert manifest["analysis"]["resamples"] == 50


# ---------------------------------------------------------------------------
# Issue #20 — automatic-vs-human validation (statistics pinned first)
# ---------------------------------------------------------------------------

# Hand-computed association table (ties on both sides):
# human overall = [1, 1, 2, 3, 4]
# scorer oriented (higher = easier) = [1, 2, 2, 3, 4]
# Kendall τ-b: P=8, Q=0, n0=10, n1=1, n2=1 → 8/9
# Spearman ρ (avg ranks): rx=[1.5,1.5,3,4,5], ry=[1,2.5,2.5,4,5] → 0.9210526315789473
_ASSOC_HUMAN = [1.0, 1.0, 2.0, 3.0, 4.0]
_ASSOC_SCORER = [1.0, 2.0, 2.0, 3.0, 4.0]
_HAND_KENDALL_TAU_B = 8.0 / 9.0
_HAND_SPEARMAN_RHO = 0.9210526315789473


class TestOrdinalAssociation:
    def test_kendall_tau_b_and_spearman_rho_with_ties(self):
        from slm_experiments.evaluation.assessment.analysis import (
            kendall_tau_b,
            spearman_rho,
        )

        tau = kendall_tau_b(_ASSOC_HUMAN, _ASSOC_SCORER)
        rho = spearman_rho(_ASSOC_HUMAN, _ASSOC_SCORER)
        assert tau["value"] == pytest.approx(_HAND_KENDALL_TAU_B)
        assert tau["reason"] is None
        assert rho["value"] == pytest.approx(_HAND_SPEARMAN_RHO)
        assert rho["reason"] is None

    def test_tie_correction_tables(self):
        """Lock τ-b / ρ against independently re-derived reference values."""
        from slm_experiments.evaluation.assessment.analysis import (
            kendall_tau_b,
            spearman_rho,
        )

        tables = [
            # Heavy ties on both sides.
            (
                [1, 1, 1, 2, 2, 2],
                [3, 3, 4, 4, 4, 5],
                0.7035264706814485,
                0.7378647873726218,
            ),
            # Tie on one side only.
            (
                [1, 2, 3, 4, 5],
                [1, 1, 2, 3, 4],
                0.9486832980505138,
                0.9746794344808964,
            ),
            # Perfectly discordant.
            ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0, -1.0),
        ]
        for x, y, exp_tau, exp_rho in tables:
            tau = kendall_tau_b(x, y)
            rho = spearman_rho(x, y)
            assert tau["value"] == pytest.approx(exp_tau)
            assert rho["value"] == pytest.approx(exp_rho)

    def test_weighted_tau_b_reduces_to_unweighted_when_equal_weights(self):
        from slm_experiments.evaluation.assessment.analysis import (
            kendall_tau_b,
            weighted_kendall_tau_b,
        )

        x = _ASSOC_HUMAN
        y = _ASSOC_SCORER
        ones = [1.0] * len(x)
        twos = [2.0] * len(x)
        unweighted = kendall_tau_b(x, y)["value"]
        assert weighted_kendall_tau_b(x, y, ones)["value"] == pytest.approx(unweighted)
        assert weighted_kendall_tau_b(x, y, twos)["value"] == pytest.approx(unweighted)

        # Unequal weights must be allowed to change the estimand.
        unequal = [1.0, 3.0, 1.0, 1.0, 1.0]
        weighted = weighted_kendall_tau_b(x, y, unequal)["value"]
        assert weighted != pytest.approx(unweighted)

    def test_weighted_association_omits_spearman(self):
        from slm_experiments.evaluation.assessment.analysis import (
            SPEARMAN_WEIGHTED_NOT_REPORTED,
        )
        from slm_experiments.evaluation.assessment.validation import (
            _association_block,
        )

        human = np.asarray(_ASSOC_HUMAN, dtype=float)
        scorer = np.asarray(_ASSOC_SCORER, dtype=float)
        weights = np.array([1.0, 3.0, 1.0, 1.0, 1.0])
        block = _association_block(
            human,
            scorer,
            weights,
            bootstrap_seed=42,
            group_key_prefix="test|assoc",
            n_resamples=20,
            ci=0.95,
        )
        assert block["weighted"]["kendall_tau_b"]["value"] is not None
        assert block["weighted"]["spearman_rho"]["status"] == "not_reported"
        assert block["weighted"]["spearman_rho"]["value"] is None
        assert block["weighted"]["spearman_rho"]["reason"] == SPEARMAN_WEIGHTED_NOT_REPORTED
        # Unweighted still carries ρ with status "ok".
        assert block["unweighted"]["spearman_rho"]["status"] == "ok"
        assert block["unweighted"]["spearman_rho"]["value"] == pytest.approx(
            _HAND_SPEARMAN_RHO
        )

    def test_all_tied_returns_null_with_reason(self):
        from slm_experiments.evaluation.assessment.analysis import (
            kendall_tau_b,
            spearman_rho,
            weighted_kendall_tau_b,
        )

        tied = [2.0, 2.0, 2.0]
        other = [1.0, 2.0, 3.0]
        tau = kendall_tau_b(tied, other)
        rho = spearman_rho(tied, other)
        assert tau["value"] is None
        assert "tie" in tau["reason"].lower() or "undefined" in tau["reason"].lower()
        assert rho["value"] is None
        assert "tie" in rho["reason"].lower() or "undefined" in rho["reason"].lower()
        wtau = weighted_kendall_tau_b(tied, other, [1.0, 1.0, 1.0])
        assert wtau["value"] is None

    def test_orient_ordinal_flips_lower_is_easier(self):
        from slm_experiments.evaluation.assessment.analysis import (
            orient_ordinal_higher_suitable,
        )

        raw = np.array([0.0, 1.0, 5.0])
        oriented = orient_ordinal_higher_suitable(raw, easier_orientation="lower")
        assert list(oriented) == pytest.approx([0.0, -1.0, -5.0])
        same = orient_ordinal_higher_suitable(raw, easier_orientation="higher")
        assert list(same) == pytest.approx([0.0, 1.0, 5.0])


class TestBinaryAgreement:
    def test_weighted_vs_unweighted_differ(self):
        from slm_experiments.evaluation.assessment.analysis import (
            binary_suitability_agreement,
        )

        # human: T T F F ; scorer: T F T F ; weights: 1, 3, 1, 1
        # unweighted agree = 2/4 = 0.5
        # weighted agree = (1+1)/(1+3+1+1) = 2/6 ≈ 0.333...
        human = np.array([True, True, False, False])
        scorer = np.array([True, False, True, False])
        weights = np.array([1.0, 3.0, 1.0, 1.0])
        result = binary_suitability_agreement(human, scorer, weights=weights)
        assert result["unweighted"]["percent_agreement"] == pytest.approx(0.5)
        assert result["weighted"]["percent_agreement"] == pytest.approx(1.0 / 3.0)
        assert result["counts"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
        assert result["unweighted"]["sensitivity"] == pytest.approx(0.5)
        assert result["unweighted"]["specificity"] == pytest.approx(0.5)


class TestConfusionMatrices:
    def test_full_and_collapsed_band_matrices(self):
        from slm_experiments.evaluation.assessment.analysis import (
            confusion_by_cefr_band,
        )

        bands = ["A1", "A1", "B1", "C2", "A2"]
        human_suitable = [True, True, False, False, True]
        result = confusion_by_cefr_band(bands, human_suitable)
        full = result["full_band"]
        assert full["bands"] == ["A1", "A2", "B1", "B2", "C1", "C2"]
        assert full["human_labels"] == ["suitable", "not_suitable"]
        # A1 row: 2 suitable, 0 not
        assert full["counts"][0] == [2, 0]
        # A2 row: 1 suitable, 0 not
        assert full["counts"][1] == [1, 0]
        # B1 row: 0 suitable, 1 not
        assert full["counts"][2] == [0, 1]
        # C2 row: 0 suitable, 1 not
        assert full["counts"][5] == [0, 1]
        assert full["row_proportions"][0] == pytest.approx([1.0, 0.0])

        collapsed = result["collapsed_band"]
        assert collapsed["bands"] == ["A1", "higher", "unknown"]
        # A1: 2 suitable; higher (A2+B1+C2): 1 suitable, 2 not
        assert collapsed["counts"][0] == [2, 0]
        assert collapsed["counts"][1] == [1, 2]


class TestAdequacyNonInferiority:
    def test_preserved_when_ci_high_below_margin(self):
        from slm_experiments.evaluation.assessment.analysis import (
            ADEQUACY_MARGIN,
            adequacy_noninferiority,
        )

        # Small drops → CI upper bound well below 0.5
        drops = np.array([0.0, 0.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.1])
        result = adequacy_noninferiority(
            drops,
            bootstrap_seed=42,
            group_key="adequacy|Qwen3|weights|weight_factor=2.0",
            n_resamples=500,
        )
        assert result["margin"] == ADEQUACY_MARGIN == 0.5
        assert result["ci_high"] < 0.5
        assert result["preserved"] is True
        assert result["n_pairs"] == 8

    def test_not_preserved_when_ci_high_at_or_above_margin(self):
        from slm_experiments.evaluation.assessment.analysis import (
            adequacy_noninferiority,
        )

        drops = np.array([0.8, 0.9, 1.0, 0.7, 0.85, 0.75, 0.95, 0.8])
        result = adequacy_noninferiority(
            drops,
            bootstrap_seed=42,
            group_key="adequacy|Qwen3|weights|weight_factor=2.0",
            n_resamples=500,
        )
        assert result["ci_high"] >= 0.5
        assert result["preserved"] is False

    def test_n_pairs_zero_null_cis_not_preserved(self):
        from slm_experiments.evaluation.assessment.analysis import (
            adequacy_noninferiority,
        )

        result = adequacy_noninferiority(
            [],
            bootstrap_seed=42,
            group_key="adequacy|empty|test",
            n_resamples=100,
            n_pairs_dropped=3,
        )
        assert result["n_pairs"] == 0
        assert result["drop"] is None
        assert result["ci_low"] is None
        assert result["ci_high"] is None
        assert result["preserved"] is False
        assert result["n_pairs_dropped"] == 3

    def test_preserved_false_when_ci_high_equals_margin(self, monkeypatch):
        """Strict inequality: ci_high == 0.5 must not count as preserved."""
        import slm_experiments.evaluation.assessment.analysis as analysis_mod
        import slm_experiments.evaluation.assessment.validation as validation_mod

        def _pinned_ci(values, **kwargs):
            return {"point_estimate": 0.4, "ci_low": 0.3, "ci_high": 0.5}

        # Patch the binding used by adequacy_noninferiority (validation module).
        monkeypatch.setattr(validation_mod, "percentile_bootstrap_ci", _pinned_ci)
        result = analysis_mod.adequacy_noninferiority(
            np.array([0.4, 0.4, 0.4, 0.4]),
            bootstrap_seed=42,
            group_key="adequacy|boundary|test",
            n_resamples=10,
        )
        assert result["ci_high"] == 0.5
        assert result["preserved"] is False

    def test_adequacy_group_key_namespace_isolated_from_paired_deltas(self):
        """Adequacy bootstrap must not share RNG stream with #19 paired deltas."""
        from slm_experiments.evaluation.assessment.analysis import (
            group_bootstrap_rng,
            percentile_bootstrap_ci,
        )

        values = np.array([0.1, 0.2, 0.15, 0.05, 0.25], dtype=float)
        paired_key = "Qwen3|weights|weight_factor=2.0|cefr_sp_level_ordinal"
        adequacy_key = "adequacy|Qwen3|weights|weight_factor=2.0"

        a = percentile_bootstrap_ci(
            values, n_resamples=800, ci=0.95, rng=group_bootstrap_rng(42, paired_key)
        )
        b = percentile_bootstrap_ci(
            values, n_resamples=800, ci=0.95, rng=group_bootstrap_rng(42, adequacy_key)
        )
        # Distinct namespaces → distinct streams (almost surely different CIs).
        assert a != b

        # Adequacy namespace is deterministic in isolation.
        b2 = percentile_bootstrap_ci(
            values, n_resamples=800, ci=0.95, rng=group_bootstrap_rng(42, adequacy_key)
        )
        assert b == b2


class TestTsarSuitableStatusGating:
    def test_scorer_suitable_tsar_requires_ok_status(self):
        from slm_experiments.evaluation.assessment.validation import (
            _scorer_suitable_tsar,
        )

        ok = pd.Series(
            {
                "cefr_tsar_status": "ok",
                "cefr_tsar_ensemble_label": "A1",
            }
        )
        assert _scorer_suitable_tsar(ok) is True

        error = pd.Series(
            {
                "cefr_tsar_status": "error",
                "cefr_tsar_ensemble_label": "A1",
            }
        )
        assert _scorer_suitable_tsar(error) is None

        # Missing status column / null status must not treat a bare label as ok.
        missing_col = pd.Series({"cefr_tsar_ensemble_label": "A1"})
        assert _scorer_suitable_tsar(missing_col) is None
        null_status = pd.Series(
            {
                "cefr_tsar_status": pd.NA,
                "cefr_tsar_ensemble_label": "A1",
            }
        )
        assert _scorer_suitable_tsar(null_status) is None


def _attach_human_study(
    store: RunStore,
    assess_id: str,
    *,
    consensus_rows: list[dict],
    source_key_rows: list[dict],
    judge_rows: list[dict] | None = None,
) -> Path:
    """Write synthetic study (+ optional judge) artifacts onto an assessment bundle."""
    run_dir = store.run_dir(assess_id)
    study_dir = run_dir / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(consensus_rows).to_csv(study_dir / "consensus.csv", index=False)
    pd.DataFrame(source_key_rows).to_csv(study_dir / "source_key.csv", index=False)
    if judge_rows is not None:
        judge_dir = run_dir / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(judge_rows).to_csv(judge_dir / "judge_scores.csv", index=False)
    return study_dir


class TestValidationBundleIntegration:
    def test_no_human_study_degrades_gracefully(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        assess_id, store = _build_weights_assessment(tmp_path)
        # Ensure no study dir.
        study = store.run_dir(assess_id) / "study"
        if study.exists():
            import shutil

            shutil.rmtree(study)

        out = analyze_assessment_bundle(
            assess_id, results_root=tmp_path, bootstrap_seed=42, resamples=50
        )
        analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["validation"]["status"] == "no_human_study"
        assert "reason" in analysis["validation"]
        assert analysis["adequacy_noninferiority"]["status"] == "no_human_study"
        # #19 sections still present.
        assert "by_model" in analysis
        assert "Qwen3" in analysis["by_model"]
        assert not (out / "validation.csv").exists()
        assert not (out / "adequacy_noninferiority.csv").exists()

    def test_orphan_consensus_is_insufficient_data(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        assess_id, store = _build_weights_assessment(tmp_path)
        _attach_human_study(
            store,
            assess_id,
            consensus_rows=[
                {
                    "item_id": "orphan-does-not-exist",
                    OVERALL_SUITABILITY: 3.0,
                    "vocabulary_accessibility": 3.0,
                    "syntax_accessibility": 3.0,
                    ANSWER_ADEQUACY: 3.0,
                    "human_suitable": True,
                    "n_raters": 3,
                }
            ],
            source_key_rows=[
                {
                    "item_id": "orphan-does-not-exist",
                    "prompt_id": "p99",
                    "prompt": "p",
                    "answer": "a",
                    "family": "weights",
                    "model": "Qwen3",
                    "arm": "baseline",
                    "cefr_band": "A1",
                    "disagreement": "agree",
                    "truncation": "ok",
                    "stratum": "s",
                    "inclusion_probability": 1.0,
                    "inclusion_weight": 1.0,
                    "role": "analysis",
                    "source_run_ids": "x",
                    "experiment_ids": "weights",
                    "configs": "both",
                }
            ],
        )
        out = analyze_assessment_bundle(
            assess_id, results_root=tmp_path, bootstrap_seed=42, resamples=20
        )
        analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["validation"]["status"] == "insufficient_data"
        assert "overlap" in analysis["validation"]["reason"].lower()
        assert analysis["validation"]["by_model"] == {}
        # Adequacy already treated empty overlap as insufficient_data.
        assert analysis["adequacy_noninferiority"]["status"] == "insufficient_data"

    def test_validation_and_adequacy_artifacts(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        assess_id, store = _build_weights_assessment(tmp_path)
        item_map = store.read_item_map_csv(assess_id)
        # Use successful in-sample items for the human subset.
        usable = item_map[
            (item_map["generation_successful"] == True)  # noqa: E712
            & (item_map["item_id"].astype(str).str.len() > 0)
        ].copy()
        assert len(usable) >= 4

        consensus_rows = []
        source_key_rows = []
        for i, (_, row) in enumerate(usable.iterrows()):
            # Alternate suitability; heavier weight on mismatches for i==1.
            overall = 4.0 if i % 2 == 0 else 2.0
            adequacy = 4.0 if i % 2 == 0 else 2.0
            weight = 3.0 if i == 1 else 1.0
            consensus_rows.append(
                {
                    "item_id": row["item_id"],
                    OVERALL_SUITABILITY: overall,
                    "vocabulary_accessibility": overall,
                    "syntax_accessibility": overall,
                    ANSWER_ADEQUACY: adequacy,
                    "human_suitable": is_human_suitable(overall, adequacy),
                    "n_raters": 3,
                }
            )
            source_key_rows.append(
                {
                    "item_id": row["item_id"],
                    "prompt_id": row["prompt_id"],
                    "prompt": "p",
                    "answer": "a",
                    "family": "weights",
                    "model": row["model"],
                    "arm": "baseline" if float(row["weight_factor"]) == 1.0 else "intervention",
                    "cefr_band": "A1" if row.get("meets_a1_criteria") else "higher",
                    "disagreement": "agree",
                    "truncation": "ok",
                    "stratum": "s",
                    "inclusion_probability": 1.0 / weight,
                    "inclusion_weight": weight,
                    "role": "analysis",
                    "source_run_ids": row["source_run_id"],
                    "experiment_ids": "weights",
                    "configs": row.get("config", ""),
                }
            )

        _attach_human_study(
            store, assess_id, consensus_rows=consensus_rows, source_key_rows=source_key_rows
        )

        out = analyze_assessment_bundle(
            assess_id, results_root=tmp_path, bootstrap_seed=42, resamples=100
        )
        analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["validation"]["status"] == "ok"
        meta = analysis["validation"]["metadata"]
        assert "chance_correction" in meta
        assert "kappa" in meta["chance_correction"].lower() or (
            "omitted" in meta["chance_correction"].lower()
        )
        assert "inclusion_weight_definition" in meta
        assert "unconditional" in meta["inclusion_weight_definition"].lower()
        assert "inclusion_weight_caveat" not in meta
        assert "estimators" in meta
        assert "pair-weighted" in meta["estimators"]["kendall_tau_b_weighted"]
        assert "not reported" in meta["estimators"]["spearman_rho_weighted"].lower()
        assert "Qwen3" in analysis["validation"]["by_model"]
        model_block = analysis["validation"]["by_model"]["Qwen3"]
        assert "generation_failure_rate" in model_block["rates"]
        assert "hit_max_tokens_rate" in model_block["rates"]
        cefr_sp = model_block["scorers"]["cefr_sp"]
        assert cefr_sp["role"] == "primary"
        weighted_assoc = cefr_sp["association"]["overall_suitability"]["weighted"]
        assert "kendall_tau_b" in weighted_assoc
        assert weighted_assoc["spearman_rho"]["status"] == "not_reported"
        assert weighted_assoc["spearman_rho"]["value"] is None
        assert "binary_agreement" in cefr_sp
        assert "confusion" in cefr_sp
        tsar = model_block["scorers"]["cefr_tsar"]
        assert tsar["role"] == "diagnostic"
        assert model_block["scorers"]["judge"]["status"] == "absent"

        assert analysis["adequacy_noninferiority"]["status"] == "ok"
        assert analysis["adequacy_noninferiority"]["margin"] == 0.5
        fam = analysis["adequacy_noninferiority"]["by_model"]["Qwen3"]["by_family"][
            "weights"
        ]
        assert "by_arm" in fam

        val_csv = pd.read_csv(out / "validation.csv")
        adeq_csv = pd.read_csv(out / "adequacy_noninferiority.csv")
        assert list(val_csv.columns)[0] == "model"
        assert list(adeq_csv.columns)[0] == "model"
        assert "generation_failure_rate" in val_csv.columns
        assert "spearman_rho_status" in val_csv.columns
        assert "kendall_tau_b_status" in val_csv.columns
        assert "preserved" in adeq_csv.columns

        # CSV must distinguish omitted-by-design from computed/undefined.
        cefr_rows = val_csv[val_csv["scorer"] == "cefr_sp"]
        weighted = cefr_rows[cefr_rows["weighting"] == "weighted"].iloc[0]
        unweighted = cefr_rows[cefr_rows["weighting"] == "unweighted"].iloc[0]
        assert weighted["spearman_rho_status"] == "not_reported"
        assert pd.isna(weighted["spearman_rho"])
        assert unweighted["spearman_rho_status"] == "ok"
        assert unweighted["kendall_tau_b_status"] == "ok"

        manifest = store.read_manifest(assess_id)
        assert manifest["artifacts"]["validation_csv"] == "analysis/validation.csv"
        assert (
            manifest["artifacts"]["adequacy_noninferiority_csv"]
            == "analysis/adequacy_noninferiority.csv"
        )

    def test_csv_spearman_status_distinguishes_not_reported_from_undefined(self):
        """Blank spearman cells are ambiguous without status — pin the vocabulary."""
        from slm_experiments.evaluation.assessment.analysis import (
            SPEARMAN_WEIGHTED_NOT_REPORTED,
            build_validation_section,
        )
        from slm_experiments.evaluation.assessment.validation import (
            _association_block,
        )

        # Association-level: weighted omission vs unweighted undefined under ties.
        tied_human = np.array([2.0, 2.0, 2.0])
        scorer = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        block = _association_block(
            tied_human,
            scorer,
            weights,
            bootstrap_seed=42,
            group_key_prefix="test|tied",
            n_resamples=10,
            ci=0.95,
        )
        assert block["weighted"]["spearman_rho"]["status"] == "not_reported"
        assert block["weighted"]["spearman_rho"]["reason"] == SPEARMAN_WEIGHTED_NOT_REPORTED
        assert block["unweighted"]["spearman_rho"]["status"] == "undefined"
        assert block["unweighted"]["spearman_rho"]["value"] is None

        # CSV-level: synthetic overlap with constant human overall → undefined ρ.
        annotated = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                "model": ["Qwen3", "Qwen3", "Qwen3"],
                "prompt_id": ["p1", "p2", "p3"],
                "family": ["weights", "weights", "weights"],
                "arm": ["baseline", "baseline", "baseline"],
                "arm_key": ["weight_factor=1", "weight_factor=1", "weight_factor=1"],
                "cefr_sp_level": ["A2", "A2", "A1"],
                "cefr_sp_level_ordinal": [1.0, 2.0, 0.0],
                "meets_a1_criteria": [False, False, True],
                "generation_successful": [True, True, True],
                "hit_max_tokens": [False, False, False],
            }
        )
        consensus = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                OVERALL_SUITABILITY: [2.0, 2.0, 2.0],  # all tied → ρ undefined
                "vocabulary_accessibility": [2.0, 2.0, 2.0],
                "syntax_accessibility": [2.0, 2.0, 2.0],
                ANSWER_ADEQUACY: [2.0, 2.0, 2.0],
                "human_suitable": [False, False, False],
                "n_raters": [3, 3, 3],
            }
        )
        source_key = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                "inclusion_weight": [1.0, 1.0, 1.0],
            }
        )
        scores = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                "cefr_tsar_ensemble_label": ["A2", "B1", "A1"],
                "cefr_tsar_ensemble_ordinal": [2.0, 3.0, 1.0],
                "cefr_tsar_status": ["ok", "ok", "ok"],
            }
        )
        section, csv_df = build_validation_section(
            annotated,
            scores,
            consensus,
            source_key,
            bootstrap_seed=42,
            resamples=20,
            ci=0.95,
        )
        assert section["status"] == "ok"
        cefr = csv_df[csv_df["scorer"] == "cefr_sp"]
        w = cefr[cefr["weighting"] == "weighted"].iloc[0]
        u = cefr[cefr["weighting"] == "unweighted"].iloc[0]
        assert w["spearman_rho_status"] == "not_reported"
        assert pd.isna(w["spearman_rho"])
        assert u["spearman_rho_status"] == "undefined"
        assert pd.isna(u["spearman_rho"])
        # Same blank cell value, different status — the distinction the CSV needs.
        assert w["spearman_rho_status"] != u["spearman_rho_status"]

    def test_judge_block_when_scores_present(self, tmp_path: Path):
        from slm_experiments.evaluation.assessment.analysis import (
            analyze_assessment_bundle,
        )

        assess_id, store = _build_weights_assessment(tmp_path)
        item_map = store.read_item_map_csv(assess_id)
        usable = item_map[
            (item_map["generation_successful"] == True)  # noqa: E712
            & (item_map["item_id"].astype(str).str.len() > 0)
        ].head(4)
        consensus_rows = []
        source_key_rows = []
        judge_rows = []
        for i, (_, row) in enumerate(usable.iterrows()):
            overall = 3.0 if i < 2 else 2.0
            consensus_rows.append(
                {
                    "item_id": row["item_id"],
                    OVERALL_SUITABILITY: overall,
                    "vocabulary_accessibility": overall,
                    "syntax_accessibility": overall,
                    ANSWER_ADEQUACY: overall,
                    "human_suitable": is_human_suitable(overall, overall),
                    "n_raters": 3,
                }
            )
            source_key_rows.append(
                {
                    "item_id": row["item_id"],
                    "prompt_id": row["prompt_id"],
                    "prompt": "p",
                    "answer": "a",
                    "family": "weights",
                    "model": row["model"],
                    "arm": "baseline",
                    "cefr_band": "A1",
                    "disagreement": "agree",
                    "truncation": "ok",
                    "stratum": "s",
                    "inclusion_probability": 1.0,
                    "inclusion_weight": 1.0,
                    "role": "analysis",
                    "source_run_ids": row["source_run_id"],
                    "experiment_ids": "weights",
                    "configs": "both",
                }
            )
            judge_rows.append(
                {
                    "item_id": row["item_id"],
                    OVERALL_SUITABILITY: overall,
                    "vocabulary_accessibility": overall,
                    "syntax_accessibility": overall,
                    ANSWER_ADEQUACY: overall,
                }
            )
        _attach_human_study(
            store,
            assess_id,
            consensus_rows=consensus_rows,
            source_key_rows=source_key_rows,
            judge_rows=judge_rows,
        )
        out = analyze_assessment_bundle(
            assess_id, results_root=tmp_path, bootstrap_seed=42, resamples=50
        )
        analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        judge = analysis["validation"]["by_model"]["Qwen3"]["scorers"]["judge"]
        assert judge["status"] == "ok"
        assert "binary_agreement" in judge
        assert judge["role"] == "secondary"
