# Beginner Suitability Evaluation Stack — Implementation Plan

## Overview

Replace the legacy FK/Fog/Spache conjunction (`meets_a1_criteria`) as the **headline outcome** with a layered, validated assessment of **suitability for Spanish-L1 CEFR A1 learners**. The stack:

1. scores answer difficulty with the **official TSAR 2025 three-model ModernBERT CEFR ensemble** (primary scalable measure),
2. compares each intervention against its in-run baseline on CEFR, guarded by an **answer-adequacy** criterion,
3. keeps a **corrected occurrence-level Spanish KVL** diagnostic,
4. validates automatic scores against **three blinded human raters**, and
5. leaves a **provider-neutral LLM-judge seam** wired but unimplemented.

**Construct (locked):** *a correct, coherent answer that a Spanish-speaking CEFR A1 learner can understand without help.*

**Status:** Plan only — **assessment complete, not yet implemented**. This document is the go/no-go blueprint. Implementation is gated on regenerating a usable Phase 2 pool (see *Critical-path dependency*).

**Model outputs are English.** The four SLMs answer in English; CEFR difficulty is measured on the English answer, while KVL measures Spanish-L1 lexical familiarity. These are complementary, not redundant.

---

## Interview Decisions (locked)

| Question | Decision |
|----------|----------|
| Scope of first session | **Assess only.** Build happens in a later, explicitly green-lit session. |
| Where the "beneficial intervention" guardrail binds | **CEFR delta on the full automated set**; adequacy guardrail applies **only on the human-rated subset**. Headline quality-preservation claims are **explicitly limited to the human sample** until the judge is validated. |
| Statistical margins | I **propose defaults** (below); user approves/adjusts at build time. |
| Source runs | **Regenerate first.** Current Phase 2 runs are too degraded (maxed failure/truncation) to yield a usable CEFR-scorable pool. |
| `venv-eval` execution model | **Either** acceptable → plan adopts **subprocess-into-`venv-eval`** (keeps `cli.py` runnable from the torch-free generation venv and preserves the "no torch in `requirements.txt`" test). |
| CEFR hardware | **Support both** local Apple Silicon (CPU/MPS auto-detect) and ClusterUY GPU — no hard-coded `device=0`. |
| Human raters + rubric | **I author** the 1–4 anchored rubric + A1-suitability thresholds for approval, and build the 3-rater tooling. Real raters TBD; validation runs on placeholder ratings until then. |

### Proposed statistical margins (pending approval)

- **CEFR "decrease" that counts:** the paired per-`(model, prompt_id)` mean ordinal CEFR delta vs baseline has a **percentile bootstrap CI that excludes 0** (not merely "any decrease").
- **Adequacy non-inferiority margin:** intervention quality is "preserved" if the **median human adequacy (1–4) drops by < 0.5** relative to baseline on the human-rated subset.
- **Multiplicity:** Benjamini–Hochberg FDR control **within each exploratory sweep family** (weights, prompting, guided, kvl_beam), not across families.

---

## The TSAR CEFR scorer (resolved — fully specified)

The "official TSAR three-model ensemble" is concrete and reproducible. Source: [`tsar-2025-shared-task/code/tsar2025_evaluation_script.py`](https://github.com/tsar-workshop/tsar-2025-shared-task/blob/main/code/tsar2025_evaluation_script.py); findings paper [ACL 2025.tsar-1.8](https://aclanthology.org/2025.tsar-1.8/).

**Three checkpoints (HF IDs):**

| HF model ID | Training split | Current tip SHA (pin this) |
|-------------|----------------|-----------------------------|
| `AbdullahBarayan/ModernBERT-base-doc_en-Cefr` | English doc-level | `3c29f5fbcdc753e99bb437ff9303df983486915b` |
| `AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr` | English sent+doc | `b00d1d4780e46f6410ea8a8509649044dee18298` |
| `AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2` | multilingual ref | `83337437aa82277e96b293665dc3186088a4a839` |

- All three output `id2label` `0..5` = `A1..C2`. Ordinal value = index + 1 (`A1=1 … C2=6`).
- **Aggregation = confidence-based** (NOT majority vote): take each model's top-1 `{label, score}`, keep the label of the model with the **highest softmax score**.
- Reported official quality (confidence-based, test set): weighted F1 ≈ **0.89**, adjacent accuracy ≈ **0.99**, RMSE ≈ **0.34**.
- Deps: `transformers>=4.55`, `torch`. Upstream assumes `device=0` (CUDA) — we replace with auto-detect.
- Upstream does **not** pin revisions; we pin the SHAs above.

---

## Current State (before change)

| Concern | Today |
|---------|-------|
| Headline outcome | `meets_a1_criteria` = FK ≤ 5 ∧ Fog ≤ 6 ∧ Spache ≤ 4 on a valid generation (`evaluation/a1_criteria.py`). Historical name; **not** CEFR. |
| KVL | `evaluation/kvl.py` — **unique surface-form set** lookup (es/de/cn), POS only for content-word filtering, **no lemmatization**. Fields: coverage, mean, min, pct-hard. |
| Answer text | Eval runs on `cleaned_response` (`evaluation/formatter.py`); `full.csv`/`answer` store raw `response`. |
| Human eval | Single-rater: `human/export.py` → `human_review.csv`; `human/import.py` merges 3 tag columns onto `full.csv` by `experiment_id`. Note `human/import.py` uses a reserved name (loaded via `importlib`). |
| Run store | `core/run_store.py` writes `manifest.json` / `specification.csv` (`SPEC_COLUMNS`, European decimals) / `full.csv` (all `ExperimentResult` fields) / `summary.json` (overall, by_config, sweep sections, by_model). Run ID `{YYYYMMDD_HHMMSS}_{phase}_{experiment}`. |
| CLI | `cli.py` dispatch-only: `phase1`, `phase2 {weights,prompting,beam,guided,kvl_beam}`, `plot`, `runs {list,show}`, `human {export,import}`. |
| Deps | `requirements.txt` has **no torch/transformers**; `tests/test_scaffold.py` asserts their absence. |
| Docs | `ExperimentDesign.md` / `docs/metrics.md` lock the headline to the readability proxy; `docs/cites-to-include.md` already sketches the CEFR upgrade. |

---

## Desired End State

1. A new immutable **assessment bundle** at `results/runs/{timestamp}_assessment_beginner_suitability/` links ≥1 source run without mutating source `full.csv`.
2. CEFR ordinal scores (`cefr_scores.csv`) produced by the pinned three-model ensemble in an **isolated `venv-eval`**, with explicit missing/error states.
3. Versioned **occurrence-level lemmatized KVL** metrics alongside preserved legacy fields.
4. **Three-rater** blinded long-format human study with ordinal Krippendorff's α, consensus medians, and a fixed rubric.
5. Provider-neutral **judge seam** (`judge_input.jsonl`, rubric, schema, `judge_scores.csv` importer) — **no API adapter**.
6. Paired per-`(model, prompt_id)` CEFR/KVL deltas with bootstrap CIs; validation of CEFR (and future judge) vs weighted human consensus.
7. Updated methodological contract (`ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md`, `AGENTS.md`, new `docs/human-eval.md`).
8. Mocked tests covering every new seam; **no GGUF and no live API required by tests**.

---

## What We're NOT Doing

- **Not** deleting FK/Fog/Spache/`meets_a1_criteria` — they stay as **legacy diagnostics**.
- **Not** creating a weighted composite outcome score. ("Weighted" in analysis = inclusion/sampling weights, not a composite.)
- **Not** implementing the LLM-judge API adapter (seam only).
- **Not** changing KVL **beam decoding** behavior — this is an evaluation refactor only.
- **Not** adding torch/transformers to the generation `requirements.txt`.
- **Not** overwriting or regenerating source-run artifacts inside the assessment bundle.
- **Not** claiming quality-preservation beyond the human sample until judge↔human agreement is shown.

---

## Critical-path dependency

**Regenerate Phase 2 first.** The handoff `thoughts/shared/handoffs/general/2026-07-12_10-32-58_phase2-failure-maxed-rates.md` documents maxed failure/truncation rates. The CEFR scorer only ingests successful, non-empty `cleaned_response`; a degraded pool yields near-empty scores. Practical sequence:

1. Fix/regenerate Phase 2 → usable scorable pool of valid, non-truncated generations.
2. Build the assessment stack (all mock-tested, GGUF/GPU/API-free).
3. Run CEFR + KVL-v2 over the bundle.
4. Human study (needs real raters) → reliability + validation.

The stack in phases 1–7 below can be **built and unit-tested on synthetic/mocked runs** before step 1 completes; only real numbers are blocked.

---

## Implementation Phases

Phases map 1:1 to the original plan's todos.

### Phase 1 — Lock the protocol (`lock-protocol`)

Document, **before inspecting any new scores**:

- Construct definition (above).
- Endpoint hierarchy: **primary** = ensemble ordinal CEFR; **guardrail** = human adequacy (sample-limited); **secondary** = Spanish KVL v2; **legacy** = FK/Fog/Spache + `meets_a1_criteria`; always beside failure & truncation rates.
- Baseline = in-run, paired by `(model, prompt_id)`.
- Beneficial-intervention rule + the proposed margins (once approved).
- Explicit statement that quality-preservation claims are **limited to the human sample** until judge validation.
- Multiplicity policy (BH-FDR within sweep family).

**Files:** `ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md`.

---

### Phase 2 — Immutable assessment bundle + CLI (`assessment-bundle`)

- New module under `src/slm_experiments/evaluation/` (e.g. `assessment/`), dispatched from `cli.py` via a new top-level `assess` command (dispatch-only, mirrors `human`/`runs`).
- Bundle dir `results/runs/{timestamp}_assessment_beginner_suitability/` referencing one or more `source_run_ids`. Read source `full.csv` copies only; **never write back**.
- **Item deduplication:** store unique `(prompt_id, cleaned_response)` items in `items.csv` separately from a one-to-many `item_map.csv` (`item_id → [source experiment_id, model, config, sweep value, truncation state, …]`). Duplicate deterministic outputs scored/rated once; analyses join back to every condition.
- **Assessment manifest** records: scorer revisions (pinned SHAs), dependency versions, rubric version, source run IDs, sampling probabilities, timestamps, seeds.

**Files:** `src/slm_experiments/evaluation/assessment/*`, `cli.py`, `core/run_store.py` (bundle helpers), tests.

---

### Phase 3 — Isolated CEFR scorer (`cefr-ensemble`)

- New **evaluator-only** requirements file (e.g. `requirements-eval.txt`: `transformers>=4.55`, `torch`, plus lemmatizer/reliability deps) installed into a separate **`venv-eval`**. Generation `requirements.txt` stays torch-free.
- `cli.py` (main venv) **subprocesses into `venv-eval/bin/python`** for the scoring step.
- Reproduce the three pinned checkpoints + confidence-based aggregation. **Device auto-detect** (CUDA → MPS → CPU), no hard-coded `device=0`.
- Batch only successful, non-empty `cleaned_response`; write `cefr_scores.csv` with per-model label/confidence, ensemble label, ordinal value, disagreement flag, and explicit **missing/error** states for skipped/failed items.
- Summaries: mean ordinal CEFR, predicted-A1 rate, ensemble disagreement — **per model and per sweep value**, reported **beside all-output failure and truncation rates** (never conditional-only).

**Files:** `requirements-eval.txt`, `src/slm_experiments/evaluation/assessment/cefr.py`, subprocess runner, tests (mocked ensemble — no model download).

---

### Phase 4 — Correct KVL evaluation (`kvl-v2`)

- In `evaluation/metrics.py` + `evaluation/kvl.py`, add a **versioned occurrence-level** path using POS-aware **lemmatized content-word tokens** (all occurrences, not the unique surface-form set). Spanish lemmatizer dependency lives in `requirements-eval.txt`.
- Report: token lookup **coverage**, **mean** score, **hard-token share**, and a **lower-tail** score (e.g. 10th percentile). **Never interpret a mean without coverage.**
- **Preserve** all legacy KVL fields for old-run compatibility. **Do not** touch KVL beam decoding.

**Files:** `evaluation/kvl.py`, `evaluation/metrics.py`, tests (`tests/test_kvl_metrics.py` extension for v2 semantics + OOV coverage).

---

### Phase 5 — Three-rater human criterion (`human-study`)

- Replace the single-rater round trip (`human/export.py`, `human/import.py`) with a **study-level long-format** flow. New module names must avoid the reserved `import` collision.
- Default **120 unique items** after a small **excluded calibration pilot**. Stratify by experiment family, model, baseline/intervention level, CEFR prediction/disagreement, truncation state; retain **inclusion weights**.
- All three raters get the **same items in independently randomized order**, showing only **opaque item ID, prompt, answer**. Collect four anchored **1–4** ratings — overall A1 suitability, vocabulary accessibility, syntax accessibility, answer adequacy — plus optional notes.
- Store the **private source key separately**. Validate exactly one rating per `(item_id, rater_id)`. Compute **ordinal Krippendorff's α** and consensus **medians**. Derive **human suitability** only from **median overall suitability + adequacy thresholds** fixed in the rubric.
- **Rubric:** I author the 1–4 anchor descriptions + suitability/adequacy thresholds → user approves → version it (`rubric_version`).

**Files:** `src/slm_experiments/human/study_export.py`, `study_import.py`, `reliability.py`, rubric file, `cli.py`, tests.

---

### Phase 6 — LLM-judge seam (`judge-seam`)

- Provider-neutral `judge_input.jsonl`, a versioned rubric matching the human dimensions, a **strict output schema**, and a validated `judge_scores.csv` importer.
- **No API adapter.** Document that quality-preservation claims stay limited to the human sample until judge results are imported and shown to agree acceptably with human consensus.

**Files:** `src/slm_experiments/evaluation/assessment/judge.py`, schema/rubric files, tests (placeholder import).

---

### Phase 7 — Analysis, verification, tests, docs (`analysis-tests-docs`)

- Per-model **paired prompt deltas** + **percentile bootstrap CIs** for ordinal CEFR and KVL; report quality deltas separately; **BH-FDR within each sweep family**.
- Validate CEFR (and future judge) vs **weighted human consensus**: ordinal association, binary suitable/not-suitable agreement, confusion by CEFR band, uncertainty intervals.
- Update `core/run_store.py`, `runs list`/`runs show` (surface assessment bundles), `AGENTS.md`, and new `docs/human-eval.md` (assessment bundle + isolated-evaluator policy).
- **Mocked tests** for: bundle immutability/dedup, official CEFR aggregation (confidence-based), KVL v2 token semantics + OOV coverage, blind three-rater export/import, rating validation, reliability, judge placeholder import, paired summaries, backward compatibility. **No GGUF / no live API.**

**Files:** `evaluation/assessment/analysis.py`, `core/run_store.py`, `cli.py`, `AGENTS.md`, `docs/human-eval.md`, tests across the above.

---

## Files Touched (summary)

### New
- `requirements-eval.txt`
- `src/slm_experiments/evaluation/assessment/` (bundle, cefr, judge, analysis)
- `src/slm_experiments/human/study_export.py`, `study_import.py`, `reliability.py`
- rubric + judge schema files
- `docs/human-eval.md`
- new tests per phase

### Modified
- `src/slm_experiments/cli.py` (dispatch: `assess`, new human study commands)
- `src/slm_experiments/core/run_store.py` (bundle helpers, `runs list/show`)
- `src/slm_experiments/evaluation/kvl.py`, `metrics.py` (KVL v2 path)
- `ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md`, `AGENTS.md`

### Explicitly unchanged
- `requirements.txt` (stays torch-free)
- KVL beam decoder + all generation/decoding paths
- Source-run artifacts (read-only from the bundle)

---

## Open Items (resolve at build time)

1. **Approve the proposed statistical margins** (CEFR CI-excludes-0; adequacy < 0.5 median drop; BH-FDR per family).
2. **Approve the human rubric anchors + suitability/adequacy thresholds** (to be drafted in Phase 5).
3. **Spanish lemmatizer choice** for KVL v2 (lightweight default to be proposed; goes in `requirements-eval.txt`).
4. **Confirm Phase 2 regeneration** produces a usable scorable pool before running real CEFR scoring.
5. **Rater logistics** — recruit three real raters when ready; until then validation runs on placeholder ratings.

---

## References

- Official TSAR script + repo: <https://github.com/tsar-workshop/tsar-2025-shared-task>
- Findings paper: <https://aclanthology.org/2025.tsar-1.8/>
- HF collection (3 CEFR evaluators): <https://huggingface.co/collections/AbdullahBarayan/tsar-2025-shared-task-on-rcts-cefr-evaluators>
- Repo research notes: `thoughts/shared/research/2026-07-14_cefr-l2-grading-sota-2026.md`, `thoughts/shared/research/2026-07-14_cefr-sp-sota-status.md`
- Phase 2 failure handoff: `thoughts/shared/handoffs/general/2026-07-12_10-32-58_phase2-failure-maxed-rates.md`
