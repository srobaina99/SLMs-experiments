"""Tests for the provider-neutral LLM-judge seam (no live API)."""

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
from slm_experiments.evaluation.assessment.judge import (
    JUDGE_DIRNAME,
    JUDGE_INPUT_FILENAME,
    JUDGE_RUBRIC_VERSION,
    JUDGE_SCHEMA_PATH,
    JUDGE_SCORES_FILENAME,
    QUALITY_PRESERVATION_SCOPE,
    JudgeExporter,
    JudgeImporter,
    build_judge_input_records,
    load_judge_schema,
    validate_judge_scores,
)
from slm_experiments.human.rubric import RATING_DIMENSIONS, RUBRIC_VERSION


SIMPLE_RESPONSE = (
    "A friend is a person you like. You talk to a friend. "
    "You play with a friend. A friend helps you."
)
ALT_RESPONSE = "A dog is an animal. It is small and friendly."


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
        cli_args=["--prompts", "2"],
        models=sorted({r.model for r in results}),
        prompt_count=len({r.prompt_id for r in results}),
        started_at=started,
        completed_at=datetime(2026, 6, 6, 14, 31, 0, tzinfo=timezone.utc),
    )
    return run_id


def _build_assessment_bundle(tmp_path: Path) -> tuple[str, RunStore]:
    pipeline = ExperimentPipeline()
    results = []
    for prompt_id, prompt, text, model in [
        ("p01", "What is a friend?", SIMPLE_RESPONSE, "Qwen3"),
        ("p02", "What is a dog?", ALT_RESPONSE, "TinyLlama"),
    ]:
        result = pipeline.run(
            prompt,
            ExperimentConfig(
                model_name=model,
                config_weighting=False,
                config_prompting=True,
                prompt_id=prompt_id,
                weight_factor=1.0,
            ),
            MockSuccessModel(text),
        )
        results.append(result)

    store = RunStore(tmp_path)
    source_id = _write_generation_bundle(store, results)
    assess_id, _ = AssessmentBundler(results_root=tmp_path).build(
        [source_id], seed=42
    )
    return assess_id, store


class TestJudgeSchema:
    def test_schema_file_exists_and_matches_human_dimensions(self):
        schema = load_judge_schema()
        assert schema["title"] == "judge_scores_v0"
        assert schema.get("additionalProperties") is False
        required = schema["required"]
        assert required[0] == "item_id"
        for dim in RATING_DIMENSIONS:
            assert dim in required
            prop = schema["properties"][dim]
            assert prop["minimum"] == 1
            assert prop["maximum"] == 4

    def test_no_provider_adapter_in_module(self):
        import slm_experiments.evaluation.assessment.judge as judge_mod

        source = Path(judge_mod.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                lowered = stripped.lower()
                assert "openai" not in lowered
                assert "anthropic" not in lowered
                assert "litellm" not in lowered
                assert "httpx" not in lowered
                assert "requests" not in lowered
        assert "api_adapter" in source
        assert judge_mod.QUALITY_PRESERVATION_SCOPE
        assert "human sample" in judge_mod.QUALITY_PRESERVATION_SCOPE.lower()


class TestValidateJudgeScores:
    def test_accepts_strict_placeholder_rows(self):
        scores = pd.DataFrame(
            [
                {
                    "item_id": "a",
                    **{d: 3 for d in RATING_DIMENSIONS},
                    "notes": "placeholder",
                },
                {
                    "item_id": "b",
                    **{d: 2 for d in RATING_DIMENSIONS},
                },
            ]
        )
        validated = validate_judge_scores(scores, known_item_ids={"a", "b"})
        assert list(validated["item_id"]) == ["a", "b"]
        assert validated.loc[0, RATING_DIMENSIONS[0]] == 3

    def test_rejects_out_of_range(self):
        scores = pd.DataFrame(
            [{"item_id": "a", **{d: 5 for d in RATING_DIMENSIONS}}]
        )
        with pytest.raises(ValueError, match="score must be an integer"):
            validate_judge_scores(scores)

    def test_rejects_duplicate_item_id(self):
        scores = pd.DataFrame(
            [
                {"item_id": "a", **{d: 3 for d in RATING_DIMENSIONS}},
                {"item_id": "a", **{d: 2 for d in RATING_DIMENSIONS}},
            ]
        )
        with pytest.raises(ValueError, match="exactly one score row"):
            validate_judge_scores(scores)

    def test_rejects_undeclared_columns(self):
        scores = pd.DataFrame(
            [
                {
                    "item_id": "a",
                    **{d: 3 for d in RATING_DIMENSIONS},
                    "provider": "openai",
                }
            ]
        )
        with pytest.raises(ValueError, match="undeclared columns"):
            validate_judge_scores(scores)

    def test_rejects_unknown_item_id(self):
        scores = pd.DataFrame(
            [{"item_id": "missing", **{d: 3 for d in RATING_DIMENSIONS}}]
        )
        with pytest.raises(ValueError, match="unknown item_id"):
            validate_judge_scores(scores, known_item_ids={"a"})

    def test_rejects_non_integral_and_bool_scores(self):
        from slm_experiments.evaluation.assessment.judge import _normalize_score

        dim = RATING_DIMENSIONS[0]
        assert _normalize_score(3, column=dim) == 3
        assert _normalize_score(3.0, column=dim) == 3
        assert _normalize_score("3", column=dim) == 3
        assert _normalize_score("3.0", column=dim) == 3
        with pytest.raises(ValueError, match="must be an integer"):
            _normalize_score(3.9, column=dim)
        with pytest.raises(ValueError, match="must be an integer"):
            _normalize_score("3.5", column=dim)
        with pytest.raises(ValueError, match="must be an integer"):
            _normalize_score(True, column=dim)

        scores = pd.DataFrame(
            [{"item_id": "a", **{d: 3.9 for d in RATING_DIMENSIONS}}]
        )
        with pytest.raises(ValueError, match="must be an integer"):
            validate_judge_scores(scores)

    def test_schema_declared_maximum_constrains_import(self):
        """Editing schema max/min must change validation — not hardcoded-only."""
        schema = load_judge_schema()
        # Tighten maximum to 3 while keeping type integer.
        tight = json.loads(json.dumps(schema))
        for dim in RATING_DIMENSIONS:
            tight["properties"][dim]["maximum"] = 3

        ok = pd.DataFrame(
            [{"item_id": "a", **{d: 3 for d in RATING_DIMENSIONS}}]
        )
        validate_judge_scores(ok, schema=tight)

        too_high = pd.DataFrame(
            [{"item_id": "a", **{d: 4 for d in RATING_DIMENSIONS}}]
        )
        with pytest.raises(ValueError, match=r"\[1, 3\]"):
            validate_judge_scores(too_high, schema=tight)


class TestJudgeExportImport:
    def test_export_builds_jsonl_and_copies_assets(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        judge_dir, n_items = JudgeExporter(results_root=tmp_path).export(assess_id)

        assert n_items >= 2
        assert judge_dir == store.run_dir(assess_id) / JUDGE_DIRNAME
        input_path = judge_dir / JUDGE_INPUT_FILENAME
        assert input_path.exists()

        lines = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == n_items
        record = lines[0]
        assert set(record.keys()) >= {
            "item_id",
            "prompt",
            "answer",
            "rubric_version",
            "dimensions",
        }
        assert record["rubric_version"] == JUDGE_RUBRIC_VERSION
        assert record["human_rubric_version"] == RUBRIC_VERSION
        assert record["dimensions"] == list(RATING_DIMENSIONS)
        for leak in ("model", "config", "cefr", "kvl", "experiment_id"):
            assert leak not in record

        assert (judge_dir / f"{JUDGE_RUBRIC_VERSION}.md").exists()
        assert (judge_dir / JUDGE_SCHEMA_PATH.name).exists()

        manifest = json.loads((store.run_dir(assess_id) / "manifest.json").read_text())
        assert manifest["llm_judge"]["api_adapter"] is None
        assert QUALITY_PRESERVATION_SCOPE in manifest["llm_judge"]["quality_preservation_scope"]

    def test_placeholder_import_against_strict_schema(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        exporter = JudgeExporter(results_root=tmp_path)
        judge_dir, n_items = exporter.export(assess_id)

        item_ids = []
        with (judge_dir / JUDGE_INPUT_FILENAME).open(encoding="utf-8") as handle:
            for line in handle:
                item_ids.append(json.loads(line)["item_id"])

        placeholder = pd.DataFrame(
            [
                {
                    "item_id": item_id,
                    **{d: 3 for d in RATING_DIMENSIONS},
                    "judge_id": "placeholder",
                    "rubric_version": JUDGE_RUBRIC_VERSION,
                }
                for item_id in item_ids
            ]
        )
        scores_path = tmp_path / "placeholder_judge_scores.csv"
        placeholder.to_csv(scores_path, index=False)

        summary = JudgeImporter(results_root=tmp_path).import_scores(
            assess_id, scores_path
        )
        assert summary["n_scores"] == n_items
        assert summary["n_items"] == n_items
        assert summary["n_items_expected"] == n_items
        assert summary["coverage_complete"] is True
        stored = pd.read_csv(judge_dir / JUDGE_SCORES_FILENAME)
        assert set(stored["item_id"]) == set(item_ids)
        for dim in RATING_DIMENSIONS:
            assert (stored[dim] == 3).all()

        assess_manifest = json.loads(
            (store.run_dir(assess_id) / "manifest.json").read_text()
        )
        assert assess_manifest["llm_judge"]["n_scores"] == n_items
        assert assess_manifest["llm_judge"]["coverage_complete"] is True
        assert assess_manifest["llm_judge"]["api_adapter"] is None

    def test_partial_coverage_warns_and_records_stats(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        judge_dir, n_items = JudgeExporter(results_root=tmp_path).export(assess_id)
        assert n_items >= 2

        item_ids = []
        with (judge_dir / JUDGE_INPUT_FILENAME).open(encoding="utf-8") as handle:
            for line in handle:
                item_ids.append(json.loads(line)["item_id"])

        # Import scores for only the first item.
        partial = pd.DataFrame(
            [
                {
                    "item_id": item_ids[0],
                    **{d: 3 for d in RATING_DIMENSIONS},
                    "judge_id": "partial",
                }
            ]
        )
        scores_path = tmp_path / "partial_judge_scores.csv"
        partial.to_csv(scores_path, index=False)

        with pytest.warns(UserWarning, match="Partial judge coverage"):
            summary = JudgeImporter(results_root=tmp_path).import_scores(
                assess_id, scores_path
            )

        assert summary["n_scores"] == 1
        assert summary["n_items"] == 1
        assert summary["n_items_expected"] == n_items
        assert summary["coverage_complete"] is False

        judge_manifest = json.loads((judge_dir / "manifest.json").read_text())
        assert judge_manifest["n_items_scored"] == 1
        assert judge_manifest["n_items_expected"] == n_items
        assert judge_manifest["coverage_complete"] is False

        assess_manifest = json.loads(
            (store.run_dir(assess_id) / "manifest.json").read_text()
        )
        assert assess_manifest["llm_judge"]["coverage_complete"] is False
        assert assess_manifest["llm_judge"]["n_items_expected"] == n_items

    def test_reexport_refuses_to_wipe_imported_scores(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        exporter = JudgeExporter(results_root=tmp_path)
        judge_dir, n_items = exporter.export(assess_id)

        item_ids = []
        with (judge_dir / JUDGE_INPUT_FILENAME).open(encoding="utf-8") as handle:
            for line in handle:
                item_ids.append(json.loads(line)["item_id"])
        placeholder = pd.DataFrame(
            [
                {"item_id": item_id, **{d: 3 for d in RATING_DIMENSIONS}}
                for item_id in item_ids
            ]
        )
        scores_path = tmp_path / "scores.csv"
        placeholder.to_csv(scores_path, index=False)
        JudgeImporter(results_root=tmp_path).import_scores(assess_id, scores_path)
        assert (judge_dir / JUDGE_SCORES_FILENAME).exists()

        with pytest.raises(FileExistsError, match="imported scores"):
            exporter.export(assess_id)
        assert (judge_dir / JUDGE_SCORES_FILENAME).exists()

        exporter.export(assess_id, force=True)
        assert not (store.run_dir(assess_id) / JUDGE_DIRNAME / JUDGE_SCORES_FILENAME).exists()
        assert (store.run_dir(assess_id) / JUDGE_DIRNAME / JUDGE_INPUT_FILENAME).exists()

    def test_export_fails_loudly_if_rubric_missing(self, tmp_path: Path):
        assess_id, _store = _build_assessment_bundle(tmp_path)
        missing = tmp_path / "missing_rubric.md"
        with patch(
            "slm_experiments.evaluation.assessment.judge.JUDGE_RUBRIC_MARKDOWN",
            missing,
        ):
            with pytest.raises(FileNotFoundError, match="Judge rubric source missing"):
                JudgeExporter(results_root=tmp_path).export(assess_id)

    def test_export_fails_loudly_if_schema_missing(self, tmp_path: Path):
        assess_id, _store = _build_assessment_bundle(tmp_path)
        missing = tmp_path / "missing_schema.json"
        with patch(
            "slm_experiments.evaluation.assessment.judge.JUDGE_SCHEMA_PATH",
            missing,
        ):
            with pytest.raises(FileNotFoundError, match="Judge schema source missing"):
                JudgeExporter(results_root=tmp_path).export(assess_id)

    def test_export_rejects_generation_kind(self, tmp_path: Path):
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
        with pytest.raises(ValueError, match="not an assessment bundle"):
            JudgeExporter(results_root=tmp_path).export(gen_id)

    def test_export_rejects_empty_items(self, tmp_path: Path):
        assess_id, store = _build_assessment_bundle(tmp_path)
        items_path = store.run_dir(assess_id) / "items.csv"
        cols = pd.read_csv(items_path).columns
        pd.DataFrame(columns=cols).to_csv(items_path, index=False)
        with pytest.raises(ValueError, match="has no items"):
            JudgeExporter(results_root=tmp_path).export(assess_id)

    def test_build_records_from_items_frame(self):
        items = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "prompt_id": "p01",
                    "prompt": "Hello?",
                    "cleaned_response": "Hi.",
                    "inclusion_probability": 1.0,
                }
            ]
        )
        records = build_judge_input_records(items)
        assert len(records) == 1
        assert records[0]["answer"] == "Hi."
        assert records[0]["prompt"] == "Hello?"


class TestJudgeCli:
    @patch("slm_experiments.evaluation.assessment.judge.JudgeExporter")
    def test_judge_export_dispatches(self, mock_exporter_cls, capsys):
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = (Path("/tmp/run/judge"), 42)

        main(
            [
                "assess",
                "judge-export",
                "--assessment-run-id",
                "20260801_120000_assessment_beginner_suitability",
            ]
        )

        mock_exporter.export.assert_called_once_with(
            "20260801_120000_assessment_beginner_suitability",
            force=False,
        )
        captured = capsys.readouterr()
        assert "42" in captured.out

    @patch("slm_experiments.evaluation.assessment.judge.JudgeImporter")
    def test_judge_import_dispatches(self, mock_importer_cls, capsys):
        mock_importer = mock_importer_cls.return_value
        mock_importer.import_scores.return_value = {
            "n_scores": 42,
            "n_items": 42,
            "n_items_expected": 42,
            "coverage_complete": True,
            "scores_path": Path("/tmp/run/judge/judge_scores.csv"),
            "judge_dir": Path("/tmp/run/judge"),
        }

        main(
            [
                "assess",
                "judge-import",
                "--assessment-run-id",
                "20260801_120000_assessment_beginner_suitability",
                "--scores",
                "/tmp/judge_scores.csv",
            ]
        )

        mock_importer.import_scores.assert_called_once_with(
            "20260801_120000_assessment_beginner_suitability",
            "/tmp/judge_scores.csv",
        )
        captured = capsys.readouterr()
        assert "42 judge scores" in captured.out
