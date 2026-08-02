"""Tests for occurrence-level lemmatized KVL v2 metrics and assessment scorer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pandas as pd
import pytest

from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.evaluation.assessment.bundle import AssessmentBundler
from slm_experiments.evaluation.assessment.kvl_v2 import (
    KVL_V2_SCORER_NAME,
    compute_kvl_v2_assessment_summary,
    ensure_registered,
    kvl_v2_scorer_revision,
    score_kvl_v2,
)
from slm_experiments.evaluation.assessment.scorers import (
    clear_scorers,
    list_scorers,
)
from slm_experiments.evaluation.kvl import (
    KVL_V2_LOWER_TAIL_PERCENTILE,
    KvlLookup,
    compute_kvl_metrics,
    compute_kvl_v2_metrics,
    empty_kvl_metrics,
    empty_kvl_v2_metrics,
)
from slm_experiments.evaluation.metrics import (
    TextEvaluator,
    lemmatize_english_token,
    resolve_lemmatizer_backend,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_lookup(tmp_path):
    lookup_dir = tmp_path / "kvl"
    lookup_dir.mkdir()
    source = FIXTURE_DIR / "kvl_lookup_es.json"
    (lookup_dir / "kvl_lookup_es.json").write_text(source.read_text(), encoding="utf-8")
    return KvlLookup(data_dir=str(lookup_dir))


class TestKvlV2TokenSemantics:
    def test_occurrence_level_counts_repeats(self, fixture_lookup):
        # Three "friend" occurrences → v2 token_count 3; v1 unique set → 1.
        tokens = ["friend", "friend", "friend"]
        v2 = compute_kvl_v2_metrics(
            "ignored",
            "es",
            tokens=tokens,
            kvl_lookup=fixture_lookup,
        )
        v1 = compute_kvl_metrics(
            "ignored",
            "es",
            content_words=set(tokens),
            kvl_lookup=fixture_lookup,
        )

        assert v2["kvl_v2_token_count"] == 3
        assert v2["kvl_v2_lookup_count"] == 3
        assert v2["kvl_v2_lookup_coverage"] == 1.0
        assert v2["kvl_v2_mean_score"] == 2.1
        assert v1["kvl_content_word_count"] == 1
        assert v1["kvl_lookup_count"] == 1

    def test_evaluator_occurrence_vs_unique_surface_set(self):
        text = "Friends help friends. Friends play."
        evaluator = TextEvaluator()
        unique = evaluator.extract_content_words(text)
        tokens = evaluator.extract_content_word_tokens(text, lemmatize=False)

        assert len(tokens) > len(unique)
        assert "friends" in tokens or "friend" in tokens
        # Unique set collapses repeats of the same surface form.
        assert tokens.count("friends") >= 2 or tokens.count("friend") >= 2

    def test_extract_content_words_single_cache(self):
        evaluator = TextEvaluator()
        text = "Friends help friends."
        assert not hasattr(evaluator, "_pos_cache")
        _ = evaluator.extract_content_words(text)
        assert (text, False) in evaluator._token_cache
        # Second call hits the same token cache (no parallel set cache).
        before = len(evaluator._token_cache)
        _ = evaluator.extract_content_words(text)
        assert len(evaluator._token_cache) == before

    def test_playing_lemmatizes_to_play_and_hits_kvl(self, fixture_lookup):
        """Blocker regression: POS-aware WordNet must map playing→play for lookup."""
        assert lemmatize_english_token("playing", "VBG") == "play"
        assert lemmatize_english_token("playing", "VBG") != "playing"
        assert lemmatize_english_token("playing", "VBG") != "playe"

        # Surface "playing" is absent from the fixture; lemma "play" is present.
        assert fixture_lookup.get_score("playing", "es") is None
        assert fixture_lookup.get_score("play", "es") == 1.0

        surface = compute_kvl_v2_metrics(
            "ignored",
            "es",
            tokens=["playing"],
            kvl_lookup=fixture_lookup,
        )
        lemmatized = compute_kvl_v2_metrics(
            "ignored",
            "es",
            tokens=[lemmatize_english_token("playing", "VBG")],
            kvl_lookup=fixture_lookup,
        )
        assert surface["kvl_v2_lookup_count"] == 0
        assert surface["kvl_v2_mean_score"] is None
        assert lemmatized["kvl_v2_lookup_count"] == 1
        assert lemmatized["kvl_v2_mean_score"] == 1.0
        assert lemmatized["kvl_v2_lookup_coverage"] == 1.0

        # End-to-end extract+lemmatize path (Kids / playing → play hit).
        text = "Kids are playing."
        surface_tokens = TextEvaluator().extract_content_word_tokens(
            text, lemmatize=False
        )
        lemma_tokens = TextEvaluator().extract_content_word_tokens(
            text, lemmatize=True
        )
        assert "playing" in surface_tokens
        assert "play" in lemma_tokens
        assert "playing" not in lemma_tokens

        end_to_end = compute_kvl_v2_metrics(
            text,
            "es",
            kvl_lookup=fixture_lookup,
            lemmatize=True,
        )
        surface_e2e = compute_kvl_v2_metrics(
            text,
            "es",
            tokens=surface_tokens,
            kvl_lookup=fixture_lookup,
        )
        assert end_to_end["kvl_v2_lookup_count"] > surface_e2e["kvl_v2_lookup_count"]
        assert end_to_end["kvl_v2_mean_score"] is not None

    def test_lemmatize_english_token_verbs(self):
        assert lemmatize_english_token("running", "VBG") == "run"
        assert lemmatize_english_token("talking", "VBG") == "talk"
        assert lemmatize_english_token("running", "VBG") != "running"

    def test_nn_tagged_nouns_not_forced_to_verb_lemmas(self):
        """Regression: no noun→verb retry (reading/clothing/painting stay nouns)."""
        from nltk.stem import WordNetLemmatizer

        wn = WordNetLemmatizer()
        for surface in ("reading", "clothing", "painting"):
            noun_lemma = wn.lemmatize(surface, "n")
            verb_lemma = wn.lemmatize(surface, "v")
            assert verb_lemma != noun_lemma  # verb path would corrupt these
            got = lemmatize_english_token(surface, "NN")
            assert got == noun_lemma
            assert got != verb_lemma
            assert got not in ("read", "clothe")

    def test_revision_records_actual_wordnet_backend(self):
        backend = resolve_lemmatizer_backend()
        assert backend.startswith("nltk.WordNetLemmatizer/")
        rev = kvl_v2_scorer_revision()
        assert rev["lemmatizer"] == backend
        assert rev["lower_tail_percentile"] == KVL_V2_LOWER_TAIL_PERCENTILE
        assert "simplemma" not in rev["lemmatizer"]

    def test_missing_wordnet_does_not_claim_nltk(self):
        """WordNet probe failure must fall through (simplemma if present, else identity)."""
        import builtins
        import slm_experiments.evaluation.metrics as metrics_mod

        class BrokenLemmatizer:
            def lemmatize(self, *_args, **_kwargs):
                raise LookupError("Resource wordnet not found.")

        real_import = builtins.__import__

        def _import_no_simplemma(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "simplemma" or name.startswith("simplemma."):
                raise ImportError("simplemma not installed (test stub)")
            return real_import(name, globals, locals, fromlist, level)

        metrics_mod.reset_lemmatizer_backend_cache()
        try:
            with patch.object(
                metrics_mod, "WordNetLemmatizer", BrokenLemmatizer
            ), patch.object(metrics_mod, "NLTK_AVAILABLE", True), patch(
                "builtins.__import__", side_effect=_import_no_simplemma
            ):
                backend = resolve_lemmatizer_backend()
            assert backend == "identity"
            assert not backend.startswith("nltk.")
        finally:
            metrics_mod.reset_lemmatizer_backend_cache()

        # When simplemma is importable, probe failure selects it (no hard dep).
        fake_simplemma = type(
            "simplemma",
            (),
            {"__version__": "0.0-test", "lemmatize": staticmethod(lambda t, lang="en": t)},
        )()
        metrics_mod.reset_lemmatizer_backend_cache()
        try:
            with patch.object(
                metrics_mod, "WordNetLemmatizer", BrokenLemmatizer
            ), patch.object(metrics_mod, "NLTK_AVAILABLE", True), patch.dict(
                "sys.modules", {"simplemma": fake_simplemma}
            ):
                backend = resolve_lemmatizer_backend()
            assert backend == "simplemma/0.0-test"
        finally:
            metrics_mod.reset_lemmatizer_backend_cache()

    def test_omw_not_in_import_time_resource_loop(self):
        import slm_experiments.evaluation.metrics as metrics_mod
        import inspect

        source = inspect.getsource(metrics_mod)
        # Comment may mention omw-1.4; the download/find resource tuple must not.
        assert '("omw-1.4", "corpora")' not in source
        assert "(\"omw-1.4\", \"corpora\")" not in source


class TestKvlV2CoverageAndAggregates:
    def test_oov_excluded_from_mean_but_counted_in_coverage(self, fixture_lookup):
        metrics = compute_kvl_v2_metrics(
            "ignored",
            "es",
            tokens=["friend", "notinlookup", "friend"],
            kvl_lookup=fixture_lookup,
        )
        assert metrics["kvl_v2_token_count"] == 3
        assert metrics["kvl_v2_lookup_count"] == 2
        assert metrics["kvl_v2_oov_count"] == 1
        assert metrics["kvl_v2_lookup_coverage"] == pytest.approx(2 / 3)
        assert metrics["kvl_v2_mean_score"] == 2.1
        assert metrics["kvl_v2_hard_token_share"] == 0.0

    def test_zero_coverage_leaves_mean_none(self, fixture_lookup):
        metrics = compute_kvl_v2_metrics(
            "ignored",
            "es",
            tokens=["zzzabsent", "yyynotfound"],
            kvl_lookup=fixture_lookup,
        )
        assert metrics["kvl_v2_lookup_coverage"] == 0.0
        assert metrics["kvl_v2_mean_score"] is None
        assert metrics["kvl_v2_hard_token_share"] is None
        assert metrics["kvl_v2_lower_tail_score"] is None

    def test_hard_token_share_and_lower_tail(self, fixture_lookup):
        # Scores: 2.1, -2.3, -3.1 → mean hard share 2/3; 10th pct near hardest.
        metrics = compute_kvl_v2_metrics(
            "ignored",
            "es",
            tokens=["friend", "establishment", "fundamentally"],
            kvl_lookup=fixture_lookup,
        )
        assert metrics["kvl_v2_hard_token_share"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["kvl_v2_lower_tail_score"] == -3.1
        assert metrics["kvl_v2_mean_score"] == pytest.approx(
            round((2.1 + (-2.3) + (-3.1)) / 3, 4)
        )

    def test_empty_defaults(self):
        metrics = empty_kvl_v2_metrics("es")
        assert metrics["kvl_v2_token_count"] == 0
        assert metrics["kvl_v2_lookup_coverage"] == 0.0
        assert metrics["kvl_v2_mean_score"] is None

    def test_v1_fields_unchanged_shape(self):
        """v1 empty helpers keep legacy keys for old-run compatibility."""
        v1 = empty_kvl_metrics("es")
        assert set(v1) == {
            "kvl_l1",
            "kvl_content_word_count",
            "kvl_lookup_count",
            "kvl_oov_count",
            "kvl_lookup_coverage",
            "kvl_mean_score",
            "kvl_min_score",
            "kvl_pct_hard_words",
        }
        assert "kvl_v2_mean_score" not in v1


class TestKvlV2AssessmentScorer:
    @pytest.fixture(autouse=True)
    def _registry(self):
        clear_scorers()
        ensure_registered()
        yield
        clear_scorers()

    def test_registered(self):
        assert KVL_V2_SCORER_NAME in list_scorers()

    def test_ensure_registered_uses_public_api_not_private_registry(self):
        from slm_experiments.evaluation.assessment import scorers as scorers_mod

        clear_scorers()
        assert KVL_V2_SCORER_NAME not in list_scorers()
        with patch(
            "slm_experiments.evaluation.assessment.kvl_v2.ensure_scorer_registered",
            wraps=scorers_mod.ensure_scorer_registered,
        ) as ensure_fn:
            ensure_registered()
            ensure_fn.assert_called_once_with(KVL_V2_SCORER_NAME, score_kvl_v2)
        assert KVL_V2_SCORER_NAME in list_scorers()
        assert scorers_mod.ensure_scorer_registered(KVL_V2_SCORER_NAME, score_kvl_v2) is False
        assert list_scorers().count(KVL_V2_SCORER_NAME) == 1

    def test_score_items_adds_v2_columns(self, fixture_lookup):
        items = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "cleaned_response": "A friend is a person you like.",
                },
                {"item_id": "i2", "cleaned_response": ""},
            ]
        )
        with patch(
            "slm_experiments.evaluation.assessment.kvl_v2.KvlLookup",
            return_value=fixture_lookup,
        ):
            scored = score_kvl_v2(items)

        assert "kvl_v2_lookup_coverage" in scored.columns
        assert "kvl_v2_mean_score" in scored.columns
        assert "kvl_v2_hard_token_share" in scored.columns
        assert "kvl_v2_lower_tail_score" in scored.columns
        row1 = scored.set_index("item_id").loc["i1"]
        assert row1["kvl_v2_status"] == "ok"
        assert row1["kvl_v2_token_count"] > 0
        row2 = scored.set_index("item_id").loc["i2"]
        assert row2["kvl_v2_status"] == "missing"
        assert row2["kvl_v2_mean_score"] is None or pd.isna(row2["kvl_v2_mean_score"])

    def test_kvl_l1_column_ignored_always_es(self, fixture_lookup):
        """Assessment KVL v2 is fixed to Spanish; kvl_l1 on items is ignored."""
        items = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "cleaned_response": "A friend is a person you like.",
                    "kvl_l1": "de",
                },
            ]
        )
        with patch(
            "slm_experiments.evaluation.assessment.kvl_v2.KvlLookup",
            return_value=fixture_lookup,
        ):
            scored = score_kvl_v2(items)

        row = scored.iloc[0]
        assert row["kvl_v2_status"] == "ok"
        assert row["kvl_v2_l1"] == "es"

    def test_summary_namespaced_under_scorers(self, fixture_lookup):
        scores = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "kvl_v2_status": "ok",
                    "kvl_v2_lookup_coverage": 1.0,
                    "kvl_v2_mean_score": 1.5,
                    "kvl_v2_hard_token_share": 0.0,
                    "kvl_v2_lower_tail_score": 1.0,
                }
            ]
        )
        item_map = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "model": "Qwen3",
                    "generation_successful": True,
                    "hit_max_tokens": False,
                    "in_sample": True,
                    "weight_factor": 1.0,
                },
                {
                    "item_id": "",
                    "model": "Qwen3",
                    "generation_successful": False,
                    "hit_max_tokens": False,
                    "in_sample": False,
                    "weight_factor": 1.0,
                },
            ]
        )
        block = compute_kvl_v2_assessment_summary(scores, item_map)
        assert block["overall"]["generation_failure_rate"] == 0.5
        assert block["overall"]["kvl_v2_mean_coverage"] == 1.0
        assert block["overall"]["kvl_v2_mean_score"] == 1.5
        assert "Qwen3" in block["by_model"]
        # No multi-value sweep column here → empty list (not a string).
        assert block["metadata"]["sweep_dimension"] == []
        assert block["metadata"]["sweep_values"] == {}

    def test_sweep_dimension_list_for_mixed_families(self):
        scores = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "kvl_v2_status": "ok",
                    "kvl_v2_lookup_coverage": 1.0,
                    "kvl_v2_mean_score": 1.5,
                    "kvl_v2_hard_token_share": 0.0,
                    "kvl_v2_lower_tail_score": 1.0,
                }
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
        summary = compute_kvl_v2_assessment_summary(scores, item_map)
        dims = summary["metadata"]["sweep_dimension"]
        assert isinstance(dims, list)
        assert "weight_factor" in dims
        assert "num_shots" in dims
        assert set(dims) == set(summary["metadata"]["sweep_values"].keys())
        assert "by_weight_factor" in summary
        assert "by_num_shots" in summary


def _make_result(
    *,
    model: str = "Qwen3",
    prompt_id: str = "p01",
    cleaned: str = "A friend is a person you like.",
    successful: bool = True,
    cefr_sp_level: Optional[str] = "A1",
    weight_factor: float = 1.0,
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
        hit_max_tokens=False,
        meets_a1_criteria=bool(successful and cefr_sp_level == "A1"),
        flesch_kincaid_grade=2.0,
        gunning_fog=3.0,
        spache_readability=2.0,
        word_count=5 if successful else 0,
        difficult_words=0,
        cleaned_response=text,
        cefr_sp_enabled=cefr_sp_level is not None,
        cefr_sp_level=cefr_sp_level if successful else None,
        cefr_sp_level_ordinal=0.0 if cefr_sp_level == "A1" else None,
        kvl_l1="es",
        kvl_content_word_count=3 if successful else 0,
        kvl_lookup_count=2 if successful else 0,
        kvl_oov_count=1 if successful else 0,
        kvl_lookup_coverage=0.67 if successful else 0.0,
        kvl_mean_score=1.2 if successful else None,
        kvl_min_score=0.5 if successful else None,
        kvl_pct_hard_words=0.0 if successful else None,
    )


class TestKvlV2BundleIntegration:
    def test_assess_build_writes_v2_scores_and_namespaced_summary(
        self, tmp_path: Path, fixture_lookup
    ):
        store = RunStore(tmp_path / "results")
        started = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        run_id = make_run_id(2, "weights", started_at=started.replace(tzinfo=None))
        store.write_bundle(
            run_id,
            [_make_result(), _make_result(model="TinyLlama", prompt_id="p02")],
            phase=2,
            experiment="weights",
            cli_args=["--prompts", "2"],
            models=["Qwen3", "TinyLlama"],
            prompt_count=2,
            started_at=started,
            completed_at=datetime(2026, 8, 1, 10, 1, 0, tzinfo=timezone.utc),
        )

        clear_scorers()
        from slm_experiments.evaluation.assessment.cefr_tsar import (
            ensure_registered as ensure_tsar,
        )

        ensure_tsar()
        ensure_registered()

        def _fake_predict(texts, **kwargs):
            return [
                [
                    {"key": "doc_en", "label": "A1", "score": 0.9},
                    {"key": "doc_sent_en", "label": "A1", "score": 0.8},
                    {"key": "reference_alllang", "label": "A1", "score": 0.7},
                ]
                for _ in texts
            ]

        bundler = AssessmentBundler(results_root=tmp_path / "results")
        with patch(
            "slm_experiments.evaluation.assessment.kvl_v2.KvlLookup",
            return_value=fixture_lookup,
        ), patch(
            "slm_experiments.evaluation.assessment.cefr_tsar.predict_member_labels",
            side_effect=_fake_predict,
        ):
            assess_id, out_dir = bundler.build([run_id], seed=0)

        scores = pd.read_csv(out_dir / "scores.csv")
        assert "kvl_v2_lookup_coverage" in scores.columns
        assert "kvl_v2_mean_score" in scores.columns
        assert scores["kvl_v2_status"].notna().all()
        assert "cefr_tsar_status" in scores.columns

        summary = store.read_summary(assess_id)
        assert "scorers" in summary
        assert "kvl_v2" in summary["scorers"]
        assert "cefr_tsar" in summary["scorers"]
        assert "kvl_v2_mean_coverage" in summary["scorers"]["kvl_v2"]["overall"]
        # #14 compat: TSAR metrics remain at top-level overall.
        assert "cefr_tsar_mean_ordinal" in summary["overall"]
        assert "generation_failure_rate" in summary["overall"]
        assert summary["scorers"]["kvl_v2"]["metadata"]["scorer"] == "kvl_v2"
