"""Tests for three-rater blinded human study export/import + reliability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from slm_experiments.cli import main
from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.evaluation.assessment import AssessmentBundler
from slm_experiments.human.reliability import (
    compute_reliability,
    consensus_medians,
)
from slm_experiments.human.rubric import (
    ANSWER_ADEQUACY,
    OVERALL_SUITABILITY,
    RATING_DIMENSIONS,
    RUBRIC_VERSION,
    is_human_suitable,
)
from slm_experiments.human.study_export import (
    BLIND_COLUMNS,
    RATER_PACKET_DIRNAME,
    RATER_PACKET_FORBIDDEN_COLUMNS,
    RATER_SHEETS_DIRNAME,
    StudyExporter,
    assert_rater_packet_columns_safe,
    arm_label,
    parse_experiment_family,
    stratified_sample_with_weights,
)
from slm_experiments.human.study_import import (
    StudyImporter,
    _normalize_rating,
    validate_ratings,
)


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)
ALT_RESPONSE = "A dog is an animal. It is small and friendly."
THIRD_RESPONSE = "Water is wet. You drink water every day."


class MockSuccessModel:
    def __init__(self, response: str = SIMPLE_RESPONSE):
        self.response = response

    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": self.response,
            "response_time_seconds": 2.0,
            "generation_successful": True,
        }


def _write_generation_bundle(
    store: RunStore,
    results: list[ExperimentResult],
    *,
    experiment: str = "weights",
    phase: int = 2,
) -> str:
    started = datetime(2026, 6, 6, 14, 30, 22, tzinfo=timezone.utc)
    run_id = make_run_id(phase, experiment, started_at=started.replace(tzinfo=None))
    store.write_bundle(
        run_id,
        results,
        phase=phase,
        experiment=experiment,
        cli_args=["--prompts", "3"],
        models=sorted({r.model for r in results}),
        prompt_count=len({r.prompt_id for r in results}),
        started_at=started,
        completed_at=datetime(2026, 6, 6, 14, 31, 0, tzinfo=timezone.utc),
    )
    return run_id


def _build_assessment_bundle(tmp_path: Path) -> tuple[str, RunStore]:
    """Assessment bundle with several distinct scorable items."""
    pipeline = ExperimentPipeline()
    responses = [
        ("p01", "What is a friend?", SIMPLE_RESPONSE, "Qwen3", False, True, 1.0),
        ("p01", "What is a friend?", ALT_RESPONSE, "TinyLlama", True, True, 2.0),
        ("p02", "What is a dog?", THIRD_RESPONSE, "Qwen2", False, False, 1.0),
        ("p02", "What is a dog?", SIMPLE_RESPONSE, "Phi3", True, True, 4.0),
        ("p03", "What is water?", ALT_RESPONSE, "Qwen3", False, True, 1.0),
        ("p03", "What is water?", THIRD_RESPONSE, "TinyLlama", True, False, 2.0),
    ]
    results = []
    for prompt_id, prompt, text, model, weighting, prompting, weight in responses:
        result = pipeline.run(
            prompt,
            ExperimentConfig(
                model_name=model,
                config_weighting=weighting,
                config_prompting=prompting,
                prompt_id=prompt_id,
                weight_factor=weight,
            ),
            MockSuccessModel(text),
        )
        results.append(result)

    store = RunStore(tmp_path)
    source_id = _write_generation_bundle(store, results, experiment="weights")
    assess_id, _ = AssessmentBundler(results_root=tmp_path).build(
        [source_id], seed=42
    )
    return assess_id, store


class TestRubricHelpers:
    def test_human_suitable_gate(self):
        assert is_human_suitable(3.0, 3.0) is True
        assert is_human_suitable(4.0, 3.0) is True
        assert is_human_suitable(2.0, 4.0) is False
        assert is_human_suitable(4.0, 2.0) is False

    def test_parse_family_and_arm(self):
        assert parse_experiment_family("20260606_120000_phase2_weights") == "weights"
        assert parse_experiment_family("20260606_120000_phase2_kvl_beam") == "kvl_beam"
        row = pd.Series(
            {
                "source_run_id": "20260606_120000_phase2_guided",
                "guided_top_k": 0,
                "config": "both",
            }
        )
        assert arm_label(row) == "baseline"
        row = pd.Series(
            {
                "source_run_id": "20260606_120000_phase2_guided",
                "guided_top_k": 10,
                "config": "both",
            }
        )
        assert arm_label(row) == "intervention"
        row = pd.Series(
            {
                "source_run_id": "20260606_120000_phase2_kvl_beam",
                "kvl_beam_width": 1,
            }
        )
        assert arm_label(row) == "baseline"
        row = pd.Series({"config": "control"})
        assert arm_label(row) == "baseline"

    def test_weights_arm_not_confused_by_num_shots(self):
        """Weights rows always carry num_shots=0; family-aware label is required."""
        row = pd.Series(
            {
                "source_run_id": "20260606_120000_phase2_weights",
                "weight_factor": 2.0,
                "num_shots": 0,
                "config": "both",
            }
        )
        assert arm_label(row) == "intervention"
        assert (
            arm_label(
                pd.Series(
                    {
                        "source_run_id": "20260606_120000_phase2_weights",
                        "weight_factor": 1.0,
                        "num_shots": 0,
                    }
                )
            )
            == "baseline"
        )


class TestStratifiedSample:
    def test_assigns_inclusion_weights(self):
        df = pd.DataFrame(
            {
                "stratum": ["a"] * 10 + ["b"] * 10,
                "item_id": [f"i{i}" for i in range(20)],
            }
        )
        sampled = stratified_sample_with_weights(df, n=8, seed=42)
        assert len(sampled) == 8
        assert set(sampled["stratum"]) == {"a", "b"}
        assert (sampled["inclusion_weight"] >= 1.0).all()
        assert (sampled["inclusion_probability"] > 0).all()

    def test_inclusion_weights_unconditional_vs_original_pool(self):
        """π_i = n_drawn / N_original — not renormalized over remaining only."""
        pool = pd.DataFrame(
            {
                "stratum": ["a"] * 10 + ["b"] * 10,
                "item_id": [f"i{i}" for i in range(20)],
            }
        )
        # Simulate post-calibration remaining: 8 of a, 8 of b.
        remaining = pool.iloc[2:10].copy()  # 8 from a
        remaining = pd.concat([remaining, pool.iloc[12:20]], ignore_index=True)  # +8 from b
        assert len(remaining) == 16

        sampled = stratified_sample_with_weights(
            remaining, n=4, seed=42, inclusion_pool=pool
        )
        assert len(sampled) == 4
        for stratum, group in sampled.groupby("stratum"):
            n_drawn = int(len(group))
            # Original pool size for each stratum is 10.
            expected_p = n_drawn / 10.0
            assert group["inclusion_probability"].iloc[0] == pytest.approx(expected_p)
            assert group["inclusion_weight"].iloc[0] == pytest.approx(1.0 / expected_p)
            # Conditional-on-remaining would have been n_drawn/8 — must differ.
            assert expected_p != pytest.approx(n_drawn / 8.0)

    def test_disagreement_stratum_from_tsar_scores(self):
        from slm_experiments.human.study_export import build_item_frame

        items = pd.DataFrame(
            [
                {
                    "item_id": "agree-1",
                    "prompt_id": "p01",
                    "prompt": "Q?",
                    "cleaned_response": "A friend is kind.",
                },
                {
                    "item_id": "disagree-1",
                    "prompt_id": "p02",
                    "prompt": "Q2?",
                    "cleaned_response": "Complex answer.",
                },
            ]
        )
        item_map = pd.DataFrame(
            [
                {
                    "item_id": "agree-1",
                    "source_run_id": "20260601_120000_phase2_weights",
                    "experiment_id": "weights",
                    "model": "Qwen3",
                    "config": "both",
                    "prompt_id": "p01",
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                    "weight_factor": 1.0,
                    "num_shots": 0,
                },
                {
                    "item_id": "disagree-1",
                    "source_run_id": "20260601_120000_phase2_weights",
                    "experiment_id": "weights",
                    "model": "Qwen3",
                    "config": "both",
                    "prompt_id": "p02",
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                    "weight_factor": 2.0,
                    "num_shots": 0,
                },
            ]
        )
        scores = pd.DataFrame(
            [
                {
                    "item_id": "agree-1",
                    "cefr_sp_level": "A1",
                    "cefr_tsar_ensemble_label": "A1",
                    "cefr_tsar_disagrees_with_cefr_sp": False,
                },
                {
                    "item_id": "disagree-1",
                    "cefr_sp_level": "A1",
                    "cefr_tsar_ensemble_label": "B1",
                    "cefr_tsar_disagrees_with_cefr_sp": True,
                },
            ]
        )
        frame = build_item_frame(items, item_map, scores)
        by_id = frame.set_index("item_id")
        assert by_id.loc["agree-1", "disagreement"] == "agree"
        assert by_id.loc["disagree-1", "disagreement"] == "disagree"
        assert "agree" in by_id.loc["agree-1", "stratum"]
        assert "disagree" in by_id.loc["disagree-1", "stratum"]

        # Without TSAR fields, disagreement stays unknown.
        empty_scores = pd.DataFrame({"item_id": ["agree-1", "disagree-1"]})
        unknown_frame = build_item_frame(items, item_map, empty_scores)
        assert set(unknown_frame["disagreement"]) == {"unknown"}


class TestReliability:
    def test_exact_and_adjacent_agreement_plus_medians(self):
        ratings = pd.DataFrame(
            [
                {"item_id": "a", "rater_id": "r1", **{d: 4 for d in RATING_DIMENSIONS}},
                {"item_id": "a", "rater_id": "r2", **{d: 4 for d in RATING_DIMENSIONS}},
                {"item_id": "a", "rater_id": "r3", **{d: 3 for d in RATING_DIMENSIONS}},
                {"item_id": "b", "rater_id": "r1", **{d: 2 for d in RATING_DIMENSIONS}},
                {"item_id": "b", "rater_id": "r2", **{d: 1 for d in RATING_DIMENSIONS}},
                {"item_id": "b", "rater_id": "r3", **{d: 1 for d in RATING_DIMENSIONS}},
            ]
        )
        report = compute_reliability(ratings)
        assert report["rubric_version"] == RUBRIC_VERSION
        assert "not computed" in report["method"]["krippendorff_alpha"].lower()
        for dim in RATING_DIMENSIONS:
            block = report["by_dimension"][dim]
            assert 0.0 <= block["exact_agreement"] <= 1.0
            assert block["adjacent_agreement"] >= block["exact_agreement"]

        consensus = consensus_medians(ratings)
        row_a = consensus[consensus["item_id"] == "a"].iloc[0]
        assert row_a[OVERALL_SUITABILITY] == 4.0
        assert bool(row_a["human_suitable"]) is True
        row_b = consensus[consensus["item_id"] == "b"].iloc[0]
        assert bool(row_b["human_suitable"]) is False


class TestValidateRatings:
    def test_rejects_duplicate_item_rater(self):
        ratings = pd.DataFrame(
            [
                {
                    "item_id": "a",
                    "rater_id": "r1",
                    **{d: 3 for d in RATING_DIMENSIONS},
                },
                {
                    "item_id": "a",
                    "rater_id": "r1",
                    **{d: 2 for d in RATING_DIMENSIONS},
                },
            ]
        )
        with pytest.raises(ValueError, match="exactly one rating"):
            validate_ratings(ratings)

    def test_rejects_out_of_range(self):
        ratings = pd.DataFrame(
            [
                {
                    "item_id": "a",
                    "rater_id": "r1",
                    **{d: 5 for d in RATING_DIMENSIONS},
                },
            ]
        )
        with pytest.raises(ValueError, match="rating must be an integer"):
            validate_ratings(ratings)

    def test_rejects_non_integral_and_bool_ratings(self):
        assert _normalize_rating(3) == 3
        assert _normalize_rating(3.0) == 3
        assert _normalize_rating("3") == 3
        assert _normalize_rating("3.0") == 3
        with pytest.raises(ValueError, match="rating must be an integer"):
            _normalize_rating(3.9)
        with pytest.raises(ValueError, match="rating must be an integer"):
            _normalize_rating("3.5")
        with pytest.raises(ValueError, match="rating must be an integer"):
            _normalize_rating(True)

        ratings = pd.DataFrame(
            [
                {
                    "item_id": "a",
                    "rater_id": "r1",
                    **{d: 3.9 for d in RATING_DIMENSIONS},
                },
            ]
        )
        with pytest.raises(ValueError, match="rating must be an integer"):
            validate_ratings(ratings)


class TestStudyExportImport:
    def test_blind_export_and_private_source_key(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        exporter = StudyExporter(results_root=tmp_path)

        study_dir, n_items = exporter.export(
            assess_id,
            sample=3,
            calibration=1,
            seed=7,
            raters=("r1", "r2", "r3"),
        )

        assert n_items == 3
        assert study_dir == store.run_dir(assess_id) / "study"
        assert (study_dir / "source_key.csv").exists()
        assert (study_dir / "analysis_items.csv").exists()
        assert (study_dir / "calibration_items.csv").exists()
        rater_packet = study_dir / RATER_PACKET_DIRNAME
        assert rater_packet.is_dir()
        # Analyst stratum metadata stays outside the rater packet.
        assert "stratum" in pd.read_csv(study_dir / "analysis_items.csv").columns
        assert not (rater_packet / "analysis_items.csv").exists()
        assert not (rater_packet / "source_key.csv").exists()

        sheet = pd.read_csv(
            rater_packet / RATER_SHEETS_DIRNAME / "rater_r1.csv"
        )
        assert list(sheet.columns) == BLIND_COLUMNS
        for leak in ("model", "config", "experiment_id", "cefr", "kvl", "stratum", "arm"):
            assert not any(leak in c.lower() for c in sheet.columns)

        source_key = pd.read_csv(study_dir / "source_key.csv")
        assert "model" in source_key.columns
        assert "experiment_ids" in source_key.columns
        assert set(source_key["role"]) == {"analysis", "calibration"}

        # Per-rater shuffle: same item set, possibly different order.
        sheet2 = pd.read_csv(
            rater_packet / RATER_SHEETS_DIRNAME / "rater_r2.csv"
        )
        assert set(sheet["item_id"]) == set(sheet2["item_id"])

        study_manifest = json.loads((study_dir / "manifest.json").read_text())
        assert study_manifest["distribute_to_raters"] == RATER_PACKET_DIRNAME

        manifest = json.loads((store.run_dir(assess_id) / "manifest.json").read_text())
        assert manifest["artifacts"]["human_study_dir"] == "study"
        assert manifest["human_study"]["analysis_items"] == 3

        # Never wrote full.csv into the assessment / generation bundles.
        assert not (store.run_dir(assess_id) / "full.csv").exists()

    def test_rater_packet_has_no_stratum_or_arm_columns(self, tmp_path: Path):
        """Rater-facing packet must not leak stratum / arm-identifying columns."""
        assess_id, _store = _build_assessment_bundle(tmp_path)
        study_dir, _ = StudyExporter(results_root=tmp_path).export(
            assess_id, sample=3, calibration=1, seed=11, raters=("r1", "r2")
        )
        packet = study_dir / RATER_PACKET_DIRNAME
        assert packet.is_dir()

        csv_paths = list(packet.rglob("*.csv"))
        assert csv_paths, "expected at least one rater sheet CSV in packet"
        for path in csv_paths:
            columns = set(pd.read_csv(path).columns)
            leaked = columns & RATER_PACKET_FORBIDDEN_COLUMNS
            assert not leaked, f"{path.name} leaked identifying columns: {leaked}"
            for leak in ("stratum", "arm", "model", "family", "cefr_band"):
                assert leak not in columns

        # Analyst files with stratum remain under study/, not inside the packet.
        analysis_cols = set(pd.read_csv(study_dir / "analysis_items.csv").columns)
        assert "stratum" in analysis_cols
        assert (study_dir / "source_key.csv").exists()
        assert "arm" in pd.read_csv(study_dir / "source_key.csv").columns

    def test_assert_rater_packet_columns_safe_rejects_forbidden(self):
        assert_rater_packet_columns_safe(BLIND_COLUMNS)
        with pytest.raises(ValueError, match="blinding hazard"):
            assert_rater_packet_columns_safe([*BLIND_COLUMNS, "stratum"])
        with pytest.raises(ValueError, match="blinding hazard"):
            assert_rater_packet_columns_safe(["item_id", "arm", "model"])

    def test_export_guard_blocks_widened_blind_columns(self, tmp_path: Path):
        """If BLIND_COLUMNS is accidentally widened, export must fail closed."""
        assess_id, _store = _build_assessment_bundle(tmp_path)
        bad_columns = [*BLIND_COLUMNS, "stratum"]
        with patch(
            "slm_experiments.human.study_export.BLIND_COLUMNS",
            bad_columns,
        ):
            with pytest.raises(ValueError, match="blinding hazard"):
                StudyExporter(results_root=tmp_path).export(
                    assess_id, sample=2, calibration=0, seed=1, raters=("r1",)
                )

    def test_import_long_format_and_reliability(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        exporter = StudyExporter(results_root=tmp_path)
        study_dir, _ = exporter.export(
            assess_id, sample=2, calibration=0, seed=3, raters=("r1", "r2", "r3")
        )
        items = pd.read_csv(study_dir / "analysis_items.csv")
        item_ids = items["item_id"].tolist()

        rows = []
        for item_id in item_ids:
            for rater, scores in (
                ("r1", (4, 4, 3, 4)),
                ("r2", (3, 3, 3, 3)),
                ("r3", (4, 3, 2, 3)),
            ):
                rows.append(
                    {
                        "item_id": item_id,
                        "rater_id": rater,
                        OVERALL_SUITABILITY: scores[0],
                        "vocabulary_accessibility": scores[1],
                        "syntax_accessibility": scores[2],
                        ANSWER_ADEQUACY: scores[3],
                        "notes": "",
                    }
                )
        ratings_path = study_dir / "incoming_ratings.csv"
        pd.DataFrame(rows).to_csv(ratings_path, index=False)

        # Snapshot generation source full.csv mtime/content stay untouched.
        source_ids = store.read_manifest(assess_id)["source_run_ids"]
        source_id = source_ids[0]
        source_full_before = store.read_full_csv(source_id).copy()

        importer = StudyImporter(results_root=tmp_path)
        summary = importer.import_ratings(assess_id, ratings_path)

        assert summary["n_ratings"] == 6
        assert summary["n_raters"] == 3
        assert (study_dir / "ratings.csv").exists()
        assert (study_dir / "consensus.csv").exists()
        reliability = json.loads((study_dir / "reliability.json").read_text())
        assert reliability["n_items"] == 2
        assert OVERALL_SUITABILITY in reliability["by_dimension"]

        source_full_after = store.read_full_csv(source_id)
        pd.testing.assert_frame_equal(source_full_before, source_full_after)

    def test_import_rater_sheet_with_rater_id(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        exporter = StudyExporter(results_root=tmp_path)
        study_dir, _ = exporter.export(
            assess_id, sample=2, calibration=0, seed=1, raters=("r1",)
        )
        sheet_path = (
            study_dir / RATER_PACKET_DIRNAME / RATER_SHEETS_DIRNAME / "rater_r1.csv"
        )
        sheet = pd.read_csv(sheet_path)
        for dim in RATING_DIMENSIONS:
            sheet[dim] = 3
        sheet.to_csv(sheet_path, index=False)

        importer = StudyImporter(results_root=tmp_path)
        summary = importer.import_ratings(assess_id, sheet_path, rater_id="r1")
        assert summary["n_ratings"] == 2
        assert summary["n_raters"] == 1

    def test_reexport_refuses_to_wipe_imported_ratings(self, tmp_path: Path):
        assess_id, _store = _build_assessment_bundle(tmp_path)
        exporter = StudyExporter(results_root=tmp_path)
        study_dir, _ = exporter.export(
            assess_id, sample=2, calibration=0, seed=1, raters=("r1", "r2", "r3")
        )
        items = pd.read_csv(study_dir / "analysis_items.csv")
        rows = []
        for item_id in items["item_id"].tolist():
            for rater in ("r1", "r2", "r3"):
                rows.append(
                    {
                        "item_id": item_id,
                        "rater_id": rater,
                        **{d: 3 for d in RATING_DIMENSIONS},
                    }
                )
        ratings_path = tmp_path / "all_ratings.csv"
        pd.DataFrame(rows).to_csv(ratings_path, index=False)
        StudyImporter(results_root=tmp_path).import_ratings(assess_id, ratings_path)
        assert (study_dir / "ratings.csv").exists()

        with pytest.raises(FileExistsError, match="imported ratings"):
            exporter.export(
                assess_id, sample=2, calibration=0, seed=1, raters=("r1", "r2", "r3")
            )
        assert (study_dir / "ratings.csv").exists()

        # --force overwrites.
        exporter.export(
            assess_id,
            sample=2,
            calibration=0,
            seed=1,
            raters=("r1", "r2", "r3"),
            force=True,
        )
        assert not (study_dir / "ratings.csv").exists()
        assert (
            study_dir / RATER_PACKET_DIRNAME / RATER_SHEETS_DIRNAME / "rater_r1.csv"
        ).exists()

    def test_incomplete_rater_coverage_skips_consensus(self, tmp_path: Path):
        assess_id, _store = _build_assessment_bundle(tmp_path)
        exporter = StudyExporter(results_root=tmp_path)
        study_dir, _ = exporter.export(
            assess_id, sample=2, calibration=0, seed=5, raters=("r1", "r2", "r3")
        )
        items = pd.read_csv(study_dir / "analysis_items.csv")
        item_ids = items["item_id"].tolist()

        # Only one rater — must not emit human_suitable consensus rows.
        partial = pd.DataFrame(
            [
                {
                    "item_id": item_id,
                    "rater_id": "r1",
                    **{d: 4 for d in RATING_DIMENSIONS},
                }
                for item_id in item_ids
            ]
        )
        ratings_path = tmp_path / "partial.csv"
        partial.to_csv(ratings_path, index=False)

        summary = StudyImporter(results_root=tmp_path).import_ratings(
            assess_id, ratings_path
        )
        assert summary["n_ratings"] == 2
        assert summary["n_items_consensus"] == 0
        assert summary["n_items_incomplete"] == 2
        assert summary["warning"] is not None
        consensus = pd.read_csv(study_dir / "consensus.csv")
        assert consensus.empty or "human_suitable" not in consensus.columns or len(consensus) == 0
        # Ratings still stored for later merge.
        stored = pd.read_csv(study_dir / "ratings.csv")
        assert len(stored) == 2


class TestCliStudyDispatch:
    @patch("slm_experiments.human.study_export.StudyExporter")
    def test_study_export_dispatches(self, mock_exporter_cls, capsys):
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = (Path("/tmp/run/study"), 100)

        main(
            [
                "human",
                "study-export",
                "--assessment-run-id",
                "20260801_120000_assessment_beginner_suitability",
                "--sample",
                "100",
                "--calibration",
                "10",
            ]
        )

        mock_exporter.export.assert_called_once_with(
            assessment_run_id="20260801_120000_assessment_beginner_suitability",
            sample=100,
            calibration=10,
            seed=42,
            raters=["r1", "r2", "r3"],
            force=False,
        )
        captured = capsys.readouterr()
        assert "100 analysis items" in captured.out

    @patch("slm_experiments.human.study_import.StudyImporter")
    def test_study_import_dispatches(self, mock_importer_cls, capsys):
        mock_importer = mock_importer_cls.return_value
        mock_importer.import_ratings.return_value = {
            "n_ratings": 300,
            "n_items": 100,
            "n_raters": 3,
            "ratings_path": Path("/tmp/run/study/ratings.csv"),
            "reliability_path": Path("/tmp/run/study/reliability.json"),
            "consensus_path": Path("/tmp/run/study/consensus.csv"),
        }

        main(
            [
                "human",
                "study-import",
                "--assessment-run-id",
                "20260801_120000_assessment_beginner_suitability",
                "--ratings",
                "/tmp/ratings.csv",
            ]
        )

        mock_importer.import_ratings.assert_called_once_with(
            assessment_run_id="20260801_120000_assessment_beginner_suitability",
            ratings_path="/tmp/ratings.csv",
            rater_id=None,
        )
        captured = capsys.readouterr()
        assert "300 ratings" in captured.out
