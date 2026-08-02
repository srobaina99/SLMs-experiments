# Evaluation-stack-plan — stale-claim audit (corrected factual baseline)

**Ticket:** #3 (Correct the plan's Current State + CEFR framing to match live code) · **Map:** #2
**Audited document:** `thoughts/plans/evaluation-stack-plan.md`
**Date:** 2026-07-19 · **Method:** read live code, no changes made.

This is the corrected factual baseline the revised plan must adopt. The plan was written as if CEFR were being **introduced from scratch** and the headline were still FK/Fog/Spache. Neither is true: **CEFR-SP (Arase et al. contrastive BERT) is already the live A1 gate**, computed in-pipeline and enabled by default, with FK/Fog/Spache already demoted to descriptive metrics and documented as such in the sibling docs.

Every stale claim below is enumerated with (a) the plan location, (b) what it says, (c) why it is stale, and (d) corrected text.

---

## Ground truth (verified in code)

- **A1 gate:** `evaluation/a1_criteria.py::meets_a1_criteria` returns `True` iff `generation_valid AND cefr_sp_enabled AND cefr_sp_level == "A1"`. It does **not** reference FK/Fog/Spache.
- **CEFR-SP scorer:** `evaluation/cefr_sp.py::CefrSpScorer` lazy-loads the Arase et al. contrastive checkpoint (`level_estimator.ckpt`) via the vendored module `evaluation/cefr_sp_vendor/model.py::LevelEstimaterContrastive` (BERT `bert-base-cased`). Document-level aggregation in `compute_cefr_sp_metrics` (mean-ordinal → nearest label). Fields emitted: `cefr_sp_enabled`, `cefr_sp_sentence_count`, `cefr_sp_level`, `cefr_sp_level_ordinal`, `cefr_sp_max_level_ordinal`, `cefr_sp_pct_a1`, `cefr_sp_adjacency`, `cefr_sp_expected_level`.
- **Wired into the pipeline, on by default:** `core/pipeline.py::_success_text_and_a1` computes CEFR-SP and gates A1 on it. CLI flag `--enable-cefr-sp` **defaults to on**; `--no-enable-cefr-sp` disables it (`cli.py`). With CEFR-SP disabled, `meets_a1_criteria` is always `False`.
- **Dependency posture:** torch / transformers / pytorch-lightning are declared as the **optional `[cefr-sp]` extra** in `pyproject.toml` (`torch>=2.0.0`, `transformers>=4.36.0`, `pytorch-lightning>=2.0.0,<3`), installed via `pip install -e ".[cefr-sp]"`. `requirements.txt` stays torch-free; `tests/test_scaffold.py::TestScaffold.test_requirements` asserts `"torch"` and `"transformers"` are absent from it. CEFR-SP runs **in-process in the main venv** — there is no `venv-eval` and no `requirements-eval.txt` today.
- **FK/Fog/Spache:** already descriptive-only. `docs/metrics.md` (lines 7, 111–181) and `ExperimentDesign.md` (lines 9, 103–111) both state the primary A1 gate is CEFR-SP and the readability formulas are recorded but do not decide the gate.

---

## Stale claims (enumerated, with corrected text)

### S1 — Overview: "replace FK/Fog/Spache as the headline outcome"
- **Location:** Overview, line 5.
- **Says:** "Replace the legacy FK/Fog/Spache conjunction (`meets_a1_criteria`) as the **headline outcome** with a layered, validated assessment…"
- **Stale because:** `meets_a1_criteria` is **no longer** an FK/Fog/Spache conjunction. It already resolves to CEFR-SP document-level A1. That replacement has shipped.
- **Corrected:** *"The headline outcome (`meets_a1_criteria`) is already CEFR-SP document-level A1 (`evaluation/a1_criteria.py`). This effort **layers** validated assessment on top of that live gate — adequacy guardrail, human validation, occurrence-level KVL, and an LLM-judge seam — and revisits **which** CEFR scorer backs the gate (see S2). It does not introduce CEFR or retire an FK/Fog/Spache headline; that already happened."*

### S2 — Overview point 1: TSAR ensemble framed as introducing CEFR
- **Location:** Overview point 1, line 7 (and the whole "The TSAR CEFR scorer" section, lines 41–57).
- **Says:** "scores answer difficulty with the **official TSAR 2025 three-model ModernBERT CEFR ensemble** (primary scalable measure)".
- **Stale because:** a CEFR primary measure already exists (Arase CEFR-SP). The TSAR ModernBERT ensemble is a **candidate replacement/augmentation**, not the introduction of CEFR scoring.
- **Corrected:** *"The primary CEFR measure already exists (Arase CEFR-SP). The open question is whether to keep it, swap in the TSAR 2025 three-model ModernBERT ensemble, or run both — deferred to the scorer-comparison and scorer-decision tickets (#4/#5). Present the TSAR ensemble as a **candidate**, not as the CEFR debut."*

### S3 — Status: "not yet implemented"
- **Location:** Overview "Status" line 15.
- **Says:** "Plan only — **assessment complete, not yet implemented**."
- **Stale because:** the CEFR-SP scorer and the A1 gate **are** implemented and live in the pipeline. "Not yet implemented" is only true for the assessment bundle, isolated evaluator, KVL v2, human study, and judge seam.
- **Corrected:** *"The CEFR-SP scorer (`evaluation/cefr_sp.py`) and the CEFR-SP A1 gate (`evaluation/a1_criteria.py`, on by default) are **already implemented and live in the pipeline**. Still unbuilt: the immutable assessment bundle, any isolated-evaluator refactor, occurrence-level KVL v2, the three-rater human study, and the LLM-judge seam."*

### S4 — Current State table, "Headline outcome" row *(the flatly-wrong one)*
- **Location:** Current State, line 65.
- **Says:** "`meets_a1_criteria` = FK ≤ 5 ∧ Fog ≤ 6 ∧ Spache ≤ 4 on a valid generation (`evaluation/a1_criteria.py`). Historical name; **not** CEFR."
- **Stale because:** the current `meets_a1_criteria` implementation contains no FK/Fog/Spache thresholds. It is CEFR-SP.
- **Corrected:** *"`meets_a1_criteria` = valid generation ∧ CEFR-SP enabled ∧ `cefr_sp_level == "A1"` (`evaluation/a1_criteria.py`). Uses Arase et al. CEFR-SP document-level aggregate; **not** FK/Fog/Spache. False when CEFR-SP is disabled or the level is null."*

### S5 — Current State table, "Deps" row (incomplete)
- **Location:** Current State, line 71.
- **Says:** "`requirements.txt` has **no torch/transformers**; `tests/test_scaffold.py` asserts their absence."
- **Stale because:** still true as far as it goes, but omits that CEFR-SP deps now exist as the optional `[cefr-sp]` extra — implying torch is entirely out of the project.
- **Corrected:** *"`requirements.txt` is torch-free (asserted by `tests/test_scaffold.py::TestScaffold.test_requirements`). CEFR-SP dependencies live in the optional `[cefr-sp]` extra in `pyproject.toml` (`torch>=2.0.0`, `transformers>=4.36.0`, `pytorch-lightning>=2.0.0,<3`), installed via `pip install -e ".[cefr-sp]"`."*

### S6 — Current State table, "Docs" row
- **Location:** Current State, line 72.
- **Says:** "`ExperimentDesign.md` / `docs/metrics.md` lock the headline to the readability proxy; `docs/cites-to-include.md` already sketches the CEFR upgrade."
- **Stale because:** both docs already define the headline as CEFR-SP A1 and demote FK/Fog/Spache to descriptive. They do not lock the headline to the readability proxy, and CEFR is not a future "upgrade" in them.
- **Corrected:** *"`ExperimentDesign.md` (lines 9, 103–111) and `docs/metrics.md` (lines 7, 111–181) already document CEFR-SP as the primary A1 gate with FK/Fog/Spache descriptive. Remaining doc reconciliation is minor wording (e.g. `ExperimentDesign.md` line 285 still calls the guided/KVL secondary question a 'readability proxy' improvement)."*

### S7 — Phase 1 endpoint hierarchy: `meets_a1_criteria` listed as legacy
- **Location:** Phase 1, line 123.
- **Says:** "**legacy** = FK/Fog/Spache + `meets_a1_criteria`".
- **Stale because:** `meets_a1_criteria` is the **current primary** gate (CEFR-SP), not legacy. Only FK/Fog/Spache are legacy/descriptive.
- **Corrected:** *"legacy = FK/Fog/Spache (descriptive only). `meets_a1_criteria` is the **live CEFR-SP gate**; the proposed 'primary = ensemble ordinal CEFR' endpoint would refine which scorer backs it (see S2), not introduce it."*

### S8 — Phase 3 / interview decision: `venv-eval` + `requirements-eval.txt` + subprocess presented as the mechanism
- **Location:** Interview Decisions line 29; Desired End State point 2 (line 79); Critical-path/Phase 3 (lines 146–152); Files-Touched "New" (`requirements-eval.txt`).
- **Says:** CEFR runs in an **isolated `venv-eval`** via a **new `requirements-eval.txt`**, with `cli.py` **subprocessing into `venv-eval/bin/python`**.
- **Stale because:** the shipped mechanism is **in-process** scoring in the main venv, gated behind the optional `[cefr-sp]` extra in `pyproject.toml`. No `venv-eval`, no `requirements-eval.txt`, no subprocess exist today. The plan's torch-free-`requirements.txt` goal is **already met** by the optional extra.
- **Corrected:** *"Current state: CEFR-SP scores **in-process** in the main venv behind the optional `[cefr-sp]` extra; `requirements.txt` stays torch-free via that extra, not via a separate venv. The subprocess/`venv-eval`/`requirements-eval.txt` design is a **proposed** change, not the status quo — its adoption is the evaluator-execution-model decision (ticket #6). The revised plan must present it as a decision, and if it keeps the in-process extra, drop `requirements-eval.txt` from 'New files'."*

---

## Rows verified ACCURATE (no change needed — recorded to save synthesis effort)

- **KVL row** (line 66): accurate. `evaluation/kvl.py` is lemma-only v1 over the **unique surface-form set** (`content_words` is a `Set`), POS filtering via `TextEvaluator.extract_content_words`, no lemmatization; fields include coverage/mean/min/pct-hard (plus `kvl_oov_count`, `kvl_content_word_count`, `kvl_lookup_count`). The plan's KVL-v2 (occurrence-level, lemmatized) is correctly framed as a future change.
- **Answer text row** (line 67): accurate. Eval runs on `cleaned_response` (`evaluation/formatter.py`); `ExperimentResult` stores both raw `response` and `cleaned_response` (`core/result.py`); `answer` in `SPEC_COLUMNS` is the raw response.
- **Human eval row** (line 68): accurate. Single-rater `human/export.py` → `human_review.csv`; `human/import.py` is loaded via `importlib` in `cli.py:511` because `import` is a reserved word.
- **Run store row** (line 69) and **CLI row** (line 70): accurate against `core/run_store.py` (`SPEC_COLUMNS`, sweep sections, run-ID format) and `cli.py` subcommands.

---

## Net effect for synthesis (#10)

The revised plan's framing shifts from **"introduce CEFR / replace FK-Fog-Spache headline"** to **"the CEFR-SP A1 gate is already live and default; this effort validates it, decides whether to keep or replace the scorer (#4/#5), decides the evaluator execution model (#6), and layers adequacy/human/KVL-v2/judge on top."** The Current State table must be rewritten per S4–S6, endpoint hierarchy per S7, and the execution-model sections flagged as a decision per S8.
