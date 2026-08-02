"""Tests for immutable assessment bundles and kind-aware runs."""

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
from slm_experiments.core.run_store import (
    KIND_ASSESSMENT,
    KIND_GENERATION,
    RunStore,
    make_run_id,
)
from slm_experiments.evaluation.assessment import AssessmentBundler
from slm_experiments.evaluation.assessment.cefr_tsar import DEFAULT_BATCH_SIZE
from slm_experiments.evaluation.assessment.scorers import (
    clear_scorers,
    list_scorers,
    register_scorer,
    score_items,
)


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


class MockFailureModel:
    def generate(self, prompt: str, config: ExperimentConfig) -> dict:
        return {
            "response": "",
            "response_time_seconds": 0.5,
            "generation_successful": False,
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


def _pipeline_results_with_failures(tmp_path: Path) -> tuple[str, RunStore]:
    """Two successful identical texts + two distinct failures (empty cleaned)."""
    pipeline = ExperimentPipeline()
    ok_a = pipeline.run(
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
    ok_b = pipeline.run(
        "What is a friend?",
        ExperimentConfig(
            model_name="TinyLlama",
            config_weighting=True,
            config_prompting=True,
            prompt_id="p01",
            weight_factor=2.0,
        ),
        MockSuccessModel(SIMPLE_RESPONSE),
    )
    # Distinct successful text for a second item.
    ok_c = pipeline.run(
        "What is a dog?",
        ExperimentConfig(
            model_name="Qwen3",
            config_weighting=False,
            config_prompting=False,
            prompt_id="p02",
        ),
        MockSuccessModel(ALT_RESPONSE),
    )
    fail_a = pipeline.run(
        "What is a friend?",
        ExperimentConfig(
            model_name="Qwen2",
            config_weighting=False,
            config_prompting=False,
            prompt_id="p01",
        ),
        MockFailureModel(),
    )
    fail_b = pipeline.run(
        "What is a dog?",
        ExperimentConfig(
            model_name="Phi3",
            config_weighting=True,
            config_prompting=False,
            prompt_id="p02",
        ),
        MockFailureModel(),
    )
    # Force empty cleaned_response on failures (formatter may leave empty anyway).
    fail_a.cleaned_response = ""
    fail_b.cleaned_response = ""
    fail_a.hit_max_tokens = True

    store = RunStore(tmp_path)
    run_id = _write_generation_bundle(
        store, [ok_a, ok_b, ok_c, fail_a, fail_b], experiment="weights"
    )
    return run_id, store


class TestAssessmentBundle:
    def test_make_run_id_assessment_shape(self):
        ts = datetime(2026, 8, 1, 12, 0, 0)
        run_id = make_run_id("assessment", "beginner_suitability", started_at=ts)
        assert run_id == "20260801_120000_assessment_beginner_suitability"

    def test_bundle_immutability_source_full_csv_unchanged(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        source_full = store.run_dir(source_id) / "full.csv"
        before = source_full.read_bytes()

        with patch.object(store, "write_full_csv", wraps=store.write_full_csv) as spy:
            bundler = AssessmentBundler(results_root=tmp_path)
            # Reuse same RunStore instance so the spy applies.
            bundler.run_store = store
            assess_id, out_dir = bundler.build(
                [source_id],
                started_at=datetime(2026, 8, 1, 15, 0, 0, tzinfo=timezone.utc),
            )

        assert spy.call_count == 0
        assert source_full.read_bytes() == before
        assert assess_id.endswith("_assessment_beginner_suitability")
        assert (out_dir / "items.csv").exists()
        assert (out_dir / "item_map.csv").exists()
        assert (out_dir / "manifest.json").exists()
        # Assessment summary is TSAR/cross-check shaped — not generation FK/Fog/Spache.
        assert (out_dir / "summary.json").exists()
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        assert "cefr_tsar_mean_ordinal" in summary.get("overall", {})
        assert "flesch_kincaid_grade" not in summary.get("overall", {})

    def test_dedup_failures_not_collapsed(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        bundler = AssessmentBundler(results_root=tmp_path)
        assess_id, _ = bundler.build([source_id])

        items = store.read_items_csv(assess_id)
        item_map = store.read_item_map_csv(assess_id)
        manifest = store.read_manifest(assess_id)

        assert manifest["kind"] == KIND_ASSESSMENT
        assert manifest["phase"] == "assessment"
        assert manifest["experiment"] == "beginner_suitability"
        assert source_id in manifest["source_run_ids"]
        assert "rubric_version" in manifest
        assert "scorer_revisions" in manifest
        assert "dependency_versions" in manifest
        assert "code_revision" in manifest
        rev = manifest["code_revision"]
        assert set(rev) >= {"git_commit", "git_describe", "dirty"}
        # Null-safe shape: values are str/bool or None (never raises).
        assert rev["git_commit"] is None or isinstance(rev["git_commit"], str)
        assert rev["git_describe"] is None or isinstance(rev["git_describe"], str)
        assert rev["dirty"] is None or isinstance(rev["dirty"], bool)
        assert "sampling" in manifest
        assert "seed" in manifest
        assert "cefr_tsar" in manifest
        assert manifest["cefr_tsar"]["batch_size"] == DEFAULT_BATCH_SIZE
        assert manifest["cefr_tsar"]["device"] is None

        obs = manifest["observations"]
        assert obs["total"] == 5  # all source rows
        assert obs["successful"] == 3
        assert obs["failed"] == 2
        assert obs["total"] == obs["successful"] + obs["failed"]
        assert obs["items"] == 2
        assert obs["item_map_rows"] == 5
        assert "hit_max_tokens" in obs

        # Two unique successful texts (p01 shared by two models + p02), not 3.
        assert len(items) == 2
        assert set(items["prompt_id"]) == {"p01", "p02"}
        assert "" not in set(items["cleaned_response"].tolist())

        # Failures appear in item_map with empty item_id — two separate rows.
        failure_rows = item_map[item_map["generation_successful"] == False]  # noqa: E712
        assert len(failure_rows) == 2
        assert (failure_rows["item_id"].fillna("") == "").all()
        assert failure_rows["hit_max_tokens"].astype(bool).sum() == 1
        assert (failure_rows["in_sample"] == False).all()  # noqa: E712

        # Shared successful text → one item_id, two source rows.
        p01_item = items.loc[items["prompt_id"] == "p01", "item_id"].iloc[0]
        p01_map = item_map[item_map["item_id"] == p01_item]
        assert len(p01_map) == 2
        assert set(p01_map["model"]) == {"Qwen3", "TinyLlama"}
        assert (p01_map["in_sample"] == True).all()  # noqa: E712

    def test_sample_keeps_all_source_rows_in_item_map(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        source_rows = len(store.read_full_csv(source_id))
        bundler = AssessmentBundler(results_root=tmp_path)
        assess_id, _ = bundler.build([source_id], sample=1, seed=42)

        items = store.read_items_csv(assess_id)
        item_map = store.read_item_map_csv(assess_id)
        manifest = store.read_manifest(assess_id)
        obs = manifest["observations"]

        assert len(items) == 1
        assert len(item_map) == source_rows
        assert obs["item_map_rows"] == source_rows
        assert obs["total"] == source_rows
        assert obs["total"] == obs["successful"] + obs["failed"]
        assert obs["items"] == 1

        kept_id = items["item_id"].iloc[0]
        in_sample = item_map[item_map["in_sample"] == True]  # noqa: E712
        out_sample_success = item_map[
            (item_map["generation_successful"] == True)  # noqa: E712
            & (item_map["in_sample"] == False)  # noqa: E712
        ]
        failures = item_map[item_map["generation_successful"] == False]  # noqa: E712

        assert len(in_sample) >= 1
        assert (in_sample["item_id"] == kept_id).all()
        assert len(out_sample_success) >= 1
        assert (out_sample_success["item_id"].fillna("") == "").all()
        assert len(failures) == 2
        assert (failures["item_id"].fillna("") == "").all()
        assert (failures["in_sample"] == False).all()  # noqa: E712
        # SRS candidate weight on scorable rows; failures stay at 1.0.
        assert float(manifest["sampling"]["inclusion_probability"]) == 0.5
        scorable = item_map[item_map["generation_successful"] == True]  # noqa: E712
        assert (scorable["inclusion_probability"] == 0.5).all()
        assert (failures["inclusion_probability"] == 1.0).all()

    def test_never_calls_write_full_csv_on_source(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        bundler = AssessmentBundler(results_root=tmp_path)

        with patch(
            "slm_experiments.core.run_store.RunStore.write_full_csv"
        ) as write_spy:
            bundler.build([source_id])
        write_spy.assert_not_called()


class TestScoringSeam:
    def setup_method(self):
        clear_scorers()

    def teardown_method(self):
        clear_scorers()

    def test_empty_registry(self):
        assert list_scorers() == []
        items = pd.DataFrame({"item_id": ["a", "b"], "cleaned_response": ["x", "y"]})
        scored = score_items(items)
        assert list(scored.columns) == ["item_id"]
        assert len(scored) == 2

    def test_register_scorer(self):
        @register_scorer("dummy")
        def dummy(items: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"item_id": items["item_id"], "dummy_score": [1.0] * len(items)}
            )

        assert list_scorers() == ["dummy"]
        items = pd.DataFrame({"item_id": ["a"], "cleaned_response": ["hi"]})
        scored = score_items(items)
        assert scored.loc[0, "dummy_score"] == 1.0

    def test_duplicate_register_without_replace_raises(self):
        @register_scorer("dup")
        def first(items: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"item_id": items["item_id"]})

        with pytest.raises(ValueError, match="scorer already registered: dup"):

            @register_scorer("dup")
            def second(items: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"item_id": items["item_id"]})

    def test_scorer_missing_item_id_raises(self):
        @register_scorer("no_id")
        def no_id(items: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"score": [1.0] * len(items)})

        items = pd.DataFrame({"item_id": ["a"], "cleaned_response": ["hi"]})
        with pytest.raises(ValueError, match="must return an item_id column"):
            score_items(items)


class TestAssessmentBuildGuards:
    def test_build_empty_source_list_raises(self, tmp_path: Path):
        bundler = AssessmentBundler(results_root=tmp_path)
        with pytest.raises(ValueError, match="at least one source_run_id"):
            bundler.build([])

    def test_build_missing_source_run_raises(self, tmp_path: Path):
        bundler = AssessmentBundler(results_root=tmp_path)
        with pytest.raises(FileNotFoundError, match="Source run bundle not found"):
            bundler.build(["missing_run_id"])

    def test_build_empty_source_csvs_raises(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        cols = store.read_full_csv(source_id).columns
        store.write_full_csv(source_id, pd.DataFrame(columns=cols))
        bundler = AssessmentBundler(results_root=tmp_path)
        with pytest.raises(ValueError, match="no observations found in source runs"):
            bundler.build([source_id])

    def test_build_blank_full_csv_raises_clear_error(self, tmp_path: Path):
        store = RunStore(tmp_path)
        run_id = "20260606_143022_phase2_weights"
        run_dir = store.run_dir(run_id)
        run_dir.mkdir(parents=True)
        (run_dir / "full.csv").write_text("")
        (run_dir / "manifest.json").write_text(
            '{"kind":"generation","phase":2,"experiment":"weights"}'
        )
        with pytest.raises(ValueError, match="full.csv is empty"):
            store.read_full_csv(run_id)
        bundler = AssessmentBundler(results_root=tmp_path)
        with pytest.raises(ValueError, match="full.csv is empty"):
            bundler.build([run_id])

    def test_whitespace_only_success_not_scorable(self, tmp_path: Path):
        pipeline = ExperimentPipeline()
        result = pipeline.run(
            "What is a friend?",
            ExperimentConfig(
                model_name="Qwen3",
                config_weighting=False,
                config_prompting=True,
                prompt_id="p01",
                weight_factor=1.0,
                enable_cefr_sp=False,
            ),
            MockSuccessModel("   \t\n"),
        )
        assert result.generation_successful is False
        store = RunStore(tmp_path)
        started = datetime(2026, 6, 6, 14, 30, 22, tzinfo=timezone.utc)
        run_id = make_run_id(2, "weights", started_at=started.replace(tzinfo=None))
        store.write_bundle(
            run_id,
            [result],
            phase=2,
            experiment="weights",
            cli_args=["--prompts", "1"],
            models=["Qwen3"],
            prompt_count=1,
            started_at=started,
            completed_at=started,
        )
        # Force a stale successful flag with whitespace-only text (hand-edit path).
        full = store.read_full_csv(run_id)
        full["cleaned_response"] = full["cleaned_response"].astype(object)
        full.loc[0, "generation_successful"] = True
        full.loc[0, "cleaned_response"] = "   "
        store.write_full_csv(run_id, full)
        assess_id, _ = AssessmentBundler(results_root=tmp_path).build([run_id])
        items = store.read_items_csv(assess_id)
        item_map = store.read_item_map_csv(assess_id)
        assert items.empty
        assert bool(item_map.iloc[0]["in_sample"]) is False
        item_id = item_map.iloc[0]["item_id"]
        assert item_id is None or (isinstance(item_id, float) and pd.isna(item_id)) or str(item_id).strip() in {"", "nan"}

    def test_build_sample_non_positive_raises(self, tmp_path: Path):
        source_id, _ = _pipeline_results_with_failures(tmp_path)
        bundler = AssessmentBundler(results_root=tmp_path)
        with pytest.raises(ValueError, match="sample size must be positive"):
            bundler.build([source_id], sample=0)
        with pytest.raises(ValueError, match="sample size must be positive"):
            bundler.build([source_id], sample=-1)

    def test_build_auto_writes_analysis_artifacts(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        bundler = AssessmentBundler(results_root=tmp_path)
        assess_id, out_dir = bundler.build([source_id])

        analysis_dir = out_dir / "analysis"
        assert analysis_dir.is_dir()
        assert (analysis_dir / "paired_deltas.csv").exists() or (
            analysis_dir / "analysis.json"
        ).exists()
        manifest = store.read_manifest(assess_id)
        assert "analysis" in manifest

    def test_build_multi_source_concat_and_cross_run_dedup(self, tmp_path: Path):
        """Identical (prompt_id, cleaned_response) across runs collapses to one item."""
        pipeline = ExperimentPipeline()
        shared = pipeline.run(
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
        other = pipeline.run(
            "What is a dog?",
            ExperimentConfig(
                model_name="TinyLlama",
                config_weighting=True,
                config_prompting=True,
                prompt_id="p02",
                weight_factor=2.0,
            ),
            MockSuccessModel(ALT_RESPONSE),
        )
        store = RunStore(tmp_path)
        started_a = datetime(2026, 6, 6, 14, 30, 22, tzinfo=timezone.utc)
        started_b = datetime(2026, 6, 6, 15, 0, 0, tzinfo=timezone.utc)
        run_a = make_run_id(2, "weights", started_at=started_a.replace(tzinfo=None))
        run_b = make_run_id(2, "prompting", started_at=started_b.replace(tzinfo=None))
        store.write_bundle(
            run_a,
            [shared],
            phase=2,
            experiment="weights",
            cli_args=["--prompts", "1"],
            models=["Qwen3"],
            prompt_count=1,
            started_at=started_a,
            completed_at=started_a,
        )
        # Same SIMPLE_RESPONSE / p01 plus a distinct second text.
        shared_b = pipeline.run(
            "What is a friend?",
            ExperimentConfig(
                model_name="Phi3",
                config_weighting=False,
                config_prompting=False,
                prompt_id="p01",
                weight_factor=1.0,
            ),
            MockSuccessModel(SIMPLE_RESPONSE),
        )
        store.write_bundle(
            run_b,
            [shared_b, other],
            phase=2,
            experiment="prompting",
            cli_args=["--prompts", "2"],
            models=["Phi3", "TinyLlama"],
            prompt_count=2,
            started_at=started_b,
            completed_at=started_b,
        )

        assess_id, _ = AssessmentBundler(results_root=tmp_path).build([run_a, run_b])
        items = store.read_items_csv(assess_id)
        item_map = store.read_item_map_csv(assess_id)
        manifest = store.read_manifest(assess_id)

        assert set(manifest["source_run_ids"]) == {run_a, run_b}
        # Two unique texts → two items (shared text deduped across runs).
        assert len(items) == 2
        shared_maps = item_map[
            (item_map["prompt_id"] == "p01")
            & (item_map["in_sample"] == True)  # noqa: E712
        ]
        assert set(shared_maps["source_run_id"]) == {run_a, run_b}
        assert shared_maps["item_id"].nunique() == 1


class TestKindAwareRuns:
    def test_list_runs_includes_assessment_kind(self, tmp_path: Path):
        source_id, store = _pipeline_results_with_failures(tmp_path)
        bundler = AssessmentBundler(results_root=tmp_path)
        assess_id, _ = bundler.build([source_id])

        manifests = store.list_runs()
        by_id = {m["run_id"]: m for m in manifests}
        assert by_id[source_id]["kind"] == KIND_GENERATION
        assert by_id[assess_id]["kind"] == KIND_ASSESSMENT
        assert store.is_assessment(assess_id) is True
        assert store.is_assessment(source_id) is False

    def test_runs_list_and_show_assessment(
        self, tmp_path: Path, capsys, monkeypatch
    ):
        monkeypatch.setattr(
            "slm_experiments.models.base.REPO_ROOT",
            str(tmp_path),
        )
        # Bundles live under results/ when REPO_ROOT is tmp_path.
        results_root = tmp_path / "results"
        source_id, store = _pipeline_results_with_failures(results_root)
        bundler = AssessmentBundler(results_root=results_root)
        assess_id, _ = bundler.build([source_id])

        main(["runs", "list"])
        list_out = capsys.readouterr().out
        assert assess_id in list_out
        assert "assessment" in list_out
        assert "beginner_suitability" in list_out
        assert "KIND" in list_out

        main(["runs", "show", assess_id])
        show_out = capsys.readouterr().out
        assert f"Run: {assess_id}" in show_out
        assert "Kind: assessment" in show_out
        assert source_id in show_out
        assert "Rubric:" in show_out
        # Must not crash looking for generation summary metrics.
        assert "flesch_kincaid_grade" not in show_out

    def test_cli_assess_build_dispatches(self, tmp_path: Path, capsys, monkeypatch):
        monkeypatch.setattr(
            "slm_experiments.models.base.REPO_ROOT",
            str(tmp_path),
        )
        results_root = tmp_path / "results"
        source_id, _ = _pipeline_results_with_failures(results_root)

        main(
            [
                "assess",
                "build",
                "--source-run-ids",
                source_id,
                "--sample",
                "1",
                "--seed",
                "7",
                "--no-plot",
            ]
        )
        out = capsys.readouterr().out
        assert "Assessment bundle complete:" in out
        assert "_assessment_beginner_suitability" in out

    def test_cli_assess_build_forwards_tsar_options(self, monkeypatch, capsys):
        """CLI assess build passes --cefr-tsar-* into AssessmentBundler.build."""
        captured: dict = {}

        def fake_build(self, source_run_ids, **kwargs):
            captured["source_run_ids"] = list(source_run_ids)
            captured.update(kwargs)
            return "fake_assessment_id", Path("/tmp/fake_assessment")

        monkeypatch.setattr(AssessmentBundler, "build", fake_build)
        main(
            [
                "assess",
                "build",
                "--source-run-ids",
                "run_a",
                "run_b",
                "--cefr-tsar-device",
                "cpu",
                "--cefr-tsar-batch-size",
                "16",
                "--no-plot",
            ]
        )
        out = capsys.readouterr().out
        assert "Assessment bundle complete: fake_assessment_id" in out
        assert captured["source_run_ids"] == ["run_a", "run_b"]
        assert captured["cefr_tsar_device"] == "cpu"
        assert captured["cefr_tsar_batch_size"] == 16

    def test_probe_code_revision_shape_and_fallback(self, monkeypatch):
        from slm_experiments.evaluation.assessment.bundle import probe_code_revision

        # Happy path with mocked git.
        responses = {
            ("rev-parse", "HEAD"): "abc123def",
            ("describe", "--tags", "--always", "--dirty"): "v0-1-gabc123def-dirty",
            ("status", "--porcelain"): " M file.py\n",
        }

        def fake_run(args, cwd=None, capture_output=None, text=None, timeout=None, check=None):
            key = tuple(args[1:])
            out = responses.get(key, "")
            class Done:
                returncode = 0 if key in responses else 1
                stdout = out
                stderr = ""
            return Done()

        monkeypatch.setattr(
            "subprocess.run",
            fake_run,
        )
        rev = probe_code_revision(repo_root="/tmp")
        assert rev["git_commit"] == "abc123def"
        assert rev["git_describe"] == "v0-1-gabc123def-dirty"
        assert rev["dirty"] is True

        # Unavailable git → null-safe empty shape, never raises.
        def boom(*_a, **_k):
            raise FileNotFoundError("git missing")

        monkeypatch.setattr("subprocess.run", boom)
        empty = probe_code_revision(repo_root="/tmp")
        assert empty == {
            "git_commit": None,
            "git_describe": None,
            "dirty": None,
        }
