# Beginner Suitability Evaluation Stack — Implementation Plan

## Overview

**The CEFR A1 gate is already live.** `meets_a1_criteria` already resolves to **Arase et al. CEFR-SP document-level A1** (`cefr_sp_level == "A1"`), scored in-pipeline by `evaluation/cefr_sp.py` and enabled by default; FK/Fog/Spache are already demoted to descriptive metrics (`evaluation/a1_criteria.py`, `docs/metrics.md`, `ExperimentDesign.md`). This plan does **not** introduce CEFR or retire an FK/Fog/Spache headline — that already shipped.

Instead, this effort **layers a validated assessment on top of the live gate**:

1. keeps **CEFR-SP document-level A1 as the primary in-pipeline gate** (unchanged),
2. adds a **committed, assessment-only TSAR ModernBERT ensemble** as a post-hoc ordinal / disagreement cross-check (never an A1 gate, never on generation runs),
3. compares each intervention against its **in-run neutral baseline** on the paired CEFR-SP mean-ordinal delta, guarded by a human **answer-adequacy** criterion,
4. adds a **corrected occurrence-level Spanish KVL** diagnostic (v2) alongside the preserved v1 fields,
5. validates automatic scores against **three blinded human raters** with an approved Spanish rubric, and
6. leaves a **provider-neutral LLM-judge seam** wired but unimplemented.

**Construct (locked):** *a correct, coherent English answer that a Spanish-L1 CEFR A1 learner can understand without help.*

**Status:** Plan only — a spec to hand off. The CEFR-SP scorer and the A1 gate **are already implemented and live**; still unbuilt are the immutable assessment bundle, the TSAR assessment-only cross-check, occurrence-level KVL v2, the three-rater human study, and the LLM-judge seam. Real assessment numbers are gated on regenerating a usable Phase 2 pool (see *Critical-path dependency*); the stack itself can be built and mock-tested before then.

**Model outputs are English.** The four SLMs answer in English; CEFR difficulty is measured on the English answer, while KVL measures Spanish-L1 lexical familiarity. Complementary, not redundant.

**Provenance.** Every decision below is locked by a ticket on the wayfinder map [Wayfinder: revise the beginner-suitability evaluation-stack plan](https://github.com/srobaina99/SLMs-experiments/issues/2); the driving ticket is cited inline where the decision lands.

---

## Current State (already shipped — the corrected baseline)

Corrected per the stale-claim audit `thoughts/plans/evaluation-stack-plan-stale-claims-audit.md` ([Correct the plan's Current State + CEFR framing](https://github.com/srobaina99/SLMs-experiments/issues/3)).

| Concern | Today (verified in code) |
|---------|--------------------------|
| **Primary A1 gate** | `meets_a1_criteria` = valid generation ∧ CEFR-SP enabled ∧ `cefr_sp_level == "A1"` (`evaluation/a1_criteria.py`). Arase et al. CEFR-SP document-level aggregate; **not** FK/Fog/Spache. False when CEFR-SP is disabled or the level is null. |
| **CEFR-SP scorer** | `evaluation/cefr_sp.py::CefrSpScorer` lazy-loads the Arase contrastive ckpt (`level_estimator.ckpt`) via vendored `evaluation/cefr_sp_vendor/model.py` (BERT `bert-base-cased`). Document aggregation = mean-ordinal → nearest label. Wired into `core/pipeline.py::_success_text_and_a1`; CLI `--enable-cefr-sp` **defaults ON**. |
| **CEFR-SP deps** | Optional `[cefr-sp]` extra in `pyproject.toml` (`torch>=2.0.0`, `transformers>=4.36.0`, `pytorch-lightning>=2.0.0,<3`), installed via `pip install -e ".[cefr-sp]"`. Runs **in-process in the main venv** — no `venv-eval`, no `requirements-eval.txt`. |
| **`requirements.txt`** | Torch-free; `tests/test_scaffold.py::TestScaffold.test_requirements` asserts `torch`/`transformers` absent. Invariant is met by the optional extra, not a separate venv. |
| **FK / Fog / Spache** | Descriptive only (`evaluation/metrics.py`); documented as such in `docs/metrics.md` and `ExperimentDesign.md`. |
| **KVL (v1)** | `evaluation/kvl.py` — lemma-only over the **unique surface-form set** (es/de/cn), POS filtering, **no lemmatization**. Fields: `kvl_lookup_coverage`, `kvl_mean_score`, `kvl_min_score`, `kvl_pct_hard_words`, `kvl_oov_count`, `kvl_content_word_count`, `kvl_lookup_count`. |
| **Answer text** | Eval runs on `cleaned_response` (`evaluation/formatter.py`); `full.csv` stores both raw `response` and `cleaned_response`; spec.csv `answer` = raw `response`. |
| **Human eval** | Single-rater: `human/export.py` → `human_review.csv`; `human/import.py` (loaded via `importlib`, name is reserved) merges tag columns onto source `full.csv` by `experiment_id`. **Not blind**, omits prompt text, rates raw `response`, mutates the source run. |
| **Run store** | `core/run_store.py` writes `manifest.json` / `specification.csv` (`SPEC_COLUMNS`, European decimals) / `full.csv` / `summary.json` (overall, by_config, sweep sections, by_model). Run ID `{YYYYMMDD_HHMMSS}_{phase}_{experiment}`; `make_run_id("assessment", "beginner_suitability")` already yields the target bundle name. |
| **CLI** | `cli.py` dispatch-only: `phase1`, `phase2 {weights,prompting,beam,guided,kvl_beam}`, `plot`, `runs {list,show}`, `human {export,import}`. Flat if-chain with nested subparsers. |

**What this means:** the framing is *"the CEFR-SP A1 gate is live and default; validate it, add a committed assessment-only TSAR cross-check, and layer adequacy / human / KVL-v2 / judge on top,"* **not** *"introduce CEFR / replace an FK-Fog-Spache headline."*

---

## Decision 1 — Scorer: keep CEFR-SP primary, add committed assessment-only TSAR

Locked by [DECIDE: primary CEFR scorer](https://github.com/srobaina99/SLMs-experiments/issues/5) and grounded on gold-standard evidence by [RESEARCH: which CEFR scorer as PRIMARY](https://github.com/srobaina99/SLMs-experiments/issues/11).

- **Primary A1 gate: unchanged.** `meets_a1_criteria` = `cefr_sp_level == "A1"` (Arase CEFR-SP, in-pipeline, default ON). Historical run continuity preserved.
- **Secondary automatic signal: a committed TSAR three-model ModernBERT ensemble**, running **only on the assessment-bundle path** (post-hoc), never on generation runs. It is an **ordinal / disagreement cross-check**, **never an A1 gate**.

**Why CEFR-SP stays primary (the decisive A1-class fact):** the TSAR 2025 **confidence-based ensemble** — the exact config published as "the TSAR evaluator" — scores **A1 F1 = 0.00 (test) / 0.40 (val)** (Findings Tables 1 & 8). Its headline weighted-F1 ≈ 0.89 / adjacent-acc ≈ 0.99 / RMSE ≈ 0.34 is carried by B1–C2 on a Cambridge-Exams/ELG **document** set where A1 is near-absent. CEFR-SP is explicitly rebalanced for A1 scarcity (**A1 F1 ≈ 0.78**; macro-F1 0.845, QWK 0.628) and its sentence granularity degrades more gracefully to ~10-word answers. On the metric this thesis reports (binary A1), the deployed TSAR ensemble is A1-blind; CEFR-SP is validated and engineered for it.

**Hard conditions for implementation (from #11):**
- **C1 — TSAR is ordinal/direction-only.** `cefr_tsar_*` fields are a directional cross-check; they never gate A1.
- **C2 — pin exactly.** Pin the three HF SHAs + confidence-max aggregation (upstream is unpinned). If any A1 TSAR signal is ever wanted, use the **single `TRAIN_DOC_EN` model** (A1 F1 = 0.80), **not** the ensemble.

**TSAR ensemble (concrete, reproducible).** Source: [`tsar2025_evaluation_script.py`](https://github.com/tsar-workshop/tsar-2025-shared-task/blob/main/code/tsar2025_evaluation_script.py); findings [ACL 2025.tsar-1.8](https://aclanthology.org/2025.tsar-1.8/).

| HF model ID | Training split | Tip SHA (pin this) |
|-------------|----------------|--------------------|
| `AbdullahBarayan/ModernBERT-base-doc_en-Cefr` | English doc-level | `3c29f5fbcdc753e99bb437ff9303df983486915b` |
| `AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr` | English sent+doc | `b00d1d4780e46f6410ea8a8509649044dee18298` |
| `AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2` | multilingual ref | `83337437aa82277e96b293665dc3186088a4a839` |

- All three: `id2label` `0..5` = `A1..C2`; ordinal = index + 1.
- **Aggregation = confidence-based** (NOT majority vote): take each model's top-1 `{label, score}`, keep the label with the **highest softmax score**.
- Deps: `transformers>=4.55`, `torch`. Device **auto-detect** (no hard-coded `device=0`).

**Assessment-bundle output shape:** `cefr_tsar_*` fields (per-model + ensemble ordinal/label/confidence) plus a **CEFR-SP ↔ TSAR disagreement flag**; analysis reports agreement with humans for **both** scorers. CEFR-SP = the gate; TSAR = ordinal/disagreement cross-check.

**Assets:** `thoughts/shared/research/2026-07-19_cefr-sp-vs-tsar-ensemble.md`, `thoughts/shared/research/2026-07-19_primary-cefr-scorer-gold-standard-grounding.md`.

---

## Decision 2 — Protocol (endpoints, statistic, baseline, margins, multiplicity)

Locked by [DECIDE: protocol lock](https://github.com/srobaina99/SLMs-experiments/issues/7). Documented **before inspecting any new scores**.

**Endpoint hierarchy**
- **Primary:** CEFR-SP document-level A1 — the live `meets_a1_criteria` gate (`cefr_sp_level == "A1"`).
- **Guardrail:** human answer-adequacy — **human-rated subset only**.
- **Secondary:** TSAR ModernBERT ensemble ordinal (assessment-only directional cross-check, never an A1 gate) + Spanish KVL v2.
- **Legacy / descriptive:** FK / Fog / Spache.
- Every endpoint is **always reported beside generation-failure and hit-max-tokens rates** — never conditional-only.

**Primary statistic (form)**
- The "beneficial intervention" test runs on the **paired CEFR-SP mean-ordinal delta** vs baseline (more power than the coarse 25-prompt binary), in the **difficulty-decreasing (easier) direction**.
- The **binary A1 pass-rate change** is reported alongside as the **construct-level headline**.
- The **TSAR ensemble ordinal delta** is reported the same way but only as a **secondary directional cross-check**.

**Baseline**
- **In-run neutral / identity point of each sweep**, paired by `(model, prompt_id)`, identical decoding: weights → weight 1.0 (no bias); prompting → zero-shot; guided → unconstrained greedy; kvl_beam → width 1 / plain greedy. Where a grid lacks its neutral point, **add it** so every run carries its own baseline.

**Adequacy non-inferiority margin**
- Margin = **0.5** on the 1–4 scale, **human subset only**: adequacy is "preserved" if the paired drop (baseline − intervention) in mean consensus adequacy has a bootstrap CI whose **upper bound stays below 0.5**.
- The item-level human-suitable gate (median overall ≥ 3 **and** median adequacy ≥ 3) is fixed in the rubric (Decision 4).

**Bootstrap**
- **Percentile bootstrap, 95% two-sided CI**, resampling over the paired `prompt_id`s within each model (the pairing unit), **10,000 resamples**, **fixed seed recorded in the assessment manifest**. Same machinery for the ordinal delta and the adequacy-drop check. A **descriptive** "CI excludes 0" flag marks easier-direction shifts (and the < 0.5 adequacy check) — no confirmatory language.

**Multiplicity**
- **No multiplicity correction.** Report **raw 95% CIs**; frame all sweeps (weights, prompting, guided, kvl_beam) as **exploratory / descriptive** — observed trends, not confirmatory significance claims. (This supersedes the earlier draft's within-family FDR proposal.)

Quality-preservation claims stay **limited to the human sample** until the judge seam is implemented and shown to agree acceptably with human consensus.

---

## Decision 3 — Execution model: in-process behind optional extras

Locked by [DECIDE: evaluator execution model](https://github.com/srobaina99/SLMs-experiments/issues/6). **Drop** the isolated `venv-eval` + `requirements-eval.txt` + `cli.py`-subprocess design entirely.

**Why (facts, not preference):** CEFR-SP already ships in-process behind `[cefr-sp]`; no dependency conflict forces isolation (TSAR `transformers>=4.55` ⊇ CEFR-SP `>=4.36`; `torch 2.x` satisfies both; Lightning does not constrain `transformers`); the torch-free-`requirements.txt` invariant — the subprocess design's only justification — is already met by the optional extra.

**Deps posture:** add TSAR + Spanish-lemmatizer (KVL v2) deps as optional extras in `pyproject.toml` (extend `[cefr-sp]` or add a sibling `[cefr-tsar]`): `torch`, `transformers>=4.55`, lemmatizer. `requirements.txt` unchanged; scaffold test still passes.

**Two runnable environments, one code path:**
- **Cluster = the real full run (primary).** ClusterUY has no venvs / no runtime `pip install -e .`, and compute nodes may be air-gapped, so the extra becomes a **baked-image recipe**: a separate heavier **`slm-thesis-eval` Singularity image** (Docker → Hub → `.sif`) with the eval extras + **pinned ModernBERT SHAs** + the **CEFR-SP `.ckpt`** baked in / bind-mounted. The **generation image stays lean and torch-free.** Run `assess` with `TRANSFORMERS_OFFLINE=1`, `HF_HOME` on a staged path (no Hub fetch at run time), `PYTHONPATH=/workspace/src`, `--no-plot`, via `sbatch`. **Device auto-detect** CUDA → MPS → CPU, and **P100-safe attention**: never `flash_attention_2` (Ampere+ only) — force `sdpa`/eager on the P100 (sm_60).
- **Local = smoke test only (secondary).** The same in-process `assess` code runs on the Mac (MPS/CPU) over a tiny synthetic/sample pool to verify wiring before submitting the cluster job. Not the real numbers.

Building the image is implementation (out of this planning effort's scope); the plan only specifies the recipe.

---

## Decision 4 — Three-rater human study + approved rubric

Locked by [VALIDATE + draft: three-rater human study](https://github.com/srobaina99/SLMs-experiments/issues/9). Asset: `thoughts/plans/human-study-rubric-draft-v0.md`. Shipped by [#16](https://github.com/srobaina99/SLMs-experiments/issues/16); reliability approach reconciled in the cross-doc ticket (#21).

- **Replace the single-rater round trip.** The current `human export/import` is not blind (leaks model/config), omits prompt text, rates the raw `response`, and mutates the source `full.csv`. Build **study-level long-format** modules (`study_export` / `study_import` / `reliability.py`), not an in-place patch; avoid the reserved `import` name.
- **Blind sheet:** `item_id`, `prompt`, `answer` (= `cleaned_response`) only — never `model` / `config` / `experiment_id` / CEFR / KVL. The private source key stays in the study bundle; **never write back to generation runs**.
- **Sample:** default analysis **n = 100** unique items after an **excluded calibration pilot**; stratify on assessment `item_map` axes (family, model, baseline/intervention, CEFR band/disagreement, truncation) with inclusion weights. **Caveat (shipped):** weights are conditional on the post-calibration frame, not unconditional against the original pool.
- **Ratings:** four ordinal **1–4** dimensions — overall suitability, vocabulary, syntax, answer adequacy — plus optional notes. One shared item set, per-rater randomized order, exactly one rating per `(item_id, rater_id)`.
- **Reliability & consensus (shipped):** mean pairwise **exact + adjacent (±1) percent agreement** per dimension + consensus **medians**. **Krippendorff's α was deliberately dropped** (no chance-corrected agreement coefficient). See `human/reliability.py` and `docs/human-eval.md`.
- **Rubric `beginner_suitability_rubric_v0` (approved).** Spanish anchors (human-edited), English column keys retained. **Binary human-suitable** iff median overall ≥ 3 **and** median adequacy ≥ 3 (vocab/syntax diagnostic only). Bump `rubric_version` on future edits.

---

## Desired End State

1. A new immutable **assessment bundle** at `results/runs/{timestamp}_assessment_beginner_suitability/` links ≥ 1 source run without mutating source `full.csv`.
2. **TSAR ensemble** ordinal scores (`cefr_tsar_*`) produced in-process behind the eval extra by the pinned three-model ensemble (confidence-max), with explicit missing/error states, plus a CEFR-SP ↔ TSAR disagreement flag. (CEFR-SP itself already scores in-pipeline.)
3. Versioned **occurrence-level lemmatized KVL v2** metrics alongside preserved v1 fields.
4. **Three-rater** blinded long-format human study with exact + adjacent (±1) percent-agreement reliability (no Krippendorff's α), consensus medians, and the fixed rubric.
5. Provider-neutral **judge seam** (`judge_input.jsonl`, rubric, schema, `judge_scores.csv` importer) — **no API adapter**.
6. Paired per-`(model, prompt_id)` CEFR-SP / TSAR / KVL deltas with percentile bootstrap CIs; validation of CEFR-SP (and future judge) vs weighted human consensus.
7. Reconciled methodological contract across `ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md`, `AGENTS.md`, new `docs/human-eval.md` (see the *Cross-doc reconciliation* appendix).
8. Mocked tests covering every new seam; **no GGUF and no live API required by tests**.

---

## What We're NOT Doing

- **Not** deleting FK/Fog/Spache — they stay as legacy descriptive diagnostics.
- **Not** changing the primary A1 gate — CEFR-SP stays the gate; TSAR never gates A1.
- **Not** running TSAR (or any eval scorer) on generation runs — assessment-path only.
- **Not** creating a `venv-eval` / `requirements-eval.txt` / subprocess runner — in-process behind optional extras.
- **Not** adding torch/transformers to the generation `requirements.txt`.
- **Not** creating a weighted composite outcome score.
- **Not** implementing the LLM-judge API adapter (seam only).
- **Not** changing KVL **beam decoding** — this is an evaluation refactor only.
- **Not** overwriting or regenerating source-run artifacts inside the assessment bundle.
- **Not** claiming quality-preservation beyond the human sample until judge↔human agreement is shown.

---

## Critical-path dependency (gates real numbers only)

**The Phase 2 pool must be regenerated before real assessment numbers can be trusted** — but building and mock-testing the whole stack is **decoupled** from it.

The handoff `thoughts/shared/handoffs/general/2026-07-12_10-32-58_phase2-failure-maxed-rates.md` documents maxed failure/truncation rates. Both scorers ingest only successful, non-empty `cleaned_response`; a degraded pool yields near-empty scores.

- **Decoupled (do now, no pool needed):** build the assessment bundle, `assess` CLI, TSAR cross-check, KVL v2, three-rater study modules, and judge seam, each **unit-tested on synthetic/mocked runs** (no GGUF / GPU / API).
- **Hard-gated on real numbers only:** running the real TSAR/KVL-v2 scores and the human study requires a **regenerated, healthy Phase 2 pool** (valid, non-truncated generations). Confirm pool health before submitting the cluster `assess` job.

Practical sequence: (1) fix/regenerate Phase 2 → healthy scorable pool; (2) build + mock-test the stack (parallel to 1); (3) run CEFR-SP-in-pipeline + TSAR + KVL-v2 over the bundle on the cluster eval image; (4) human study (needs real raters) → reliability + validation.

---

## Implementation Phases

Reorganized around the corrected mental model: the gate is live; everything below **layers assessment on top**. Each phase is buildable and mock-testable independently of the Phase 2 regeneration (only real numbers wait).

### Phase A — Lock the protocol (`lock-protocol`)

Write Decision 2 into the methodological contract before inspecting scores: construct, endpoint hierarchy, primary statistic (paired CEFR-SP mean-ordinal delta + binary A1 headline), in-run baseline, adequacy margin (0.5, human subset), bootstrap (10k, percentile, seeded), and the **no-multiplicity / exploratory** framing. State that quality-preservation claims are human-sample-limited until judge validation.

**Files:** `ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md` (see appendix for exact edits).

### Phase B — Immutable assessment bundle + `assess` CLI (`assessment-bundle`)

Validated against live `core/run_store.py` / `cli.py` by [VALIDATE: assessment bundle + assess CLI + dedup](https://github.com/srobaina99/SLMs-experiments/issues/8).

- New module under `src/slm_experiments/evaluation/assessment/`, dispatched from `cli.py` via a new top-level `assess` command (dispatch-only, mirrors `human`/`runs`). **`assess` sub-commands must avoid reserved words** and must **not** reuse `_add_run_options` (which injects generation/CEFR-SP flags); `assess` takes its own source-run-oriented options (source run ids, sample).
- Bundle dir `results/runs/{timestamp}_assessment_beginner_suitability/` (already the output of `make_run_id("assessment", "beginner_suitability")`) referencing one or more `source_run_ids`. Ingest via `RunStore.read_full_csv`; **never call `write_full_csv` on a source run**.
- **Kind discriminator:** the assessment `manifest.json` carries `kind: assessment` plus the base keys `list_runs()` needs (`phase`, `experiment`, `started_at`, `observations`). Make `run_store.list_runs` + `runs list` / `runs show` **branch on `kind`** so assessment bundles are surfaced without pretending to be generation runs (today `runs show` hard-requires a generation-shaped `summary.json` and `sys.exit(1)`s on missing file).
- **Item deduplication (source = `full.csv`, not spec.csv):** the dedup key `cleaned_response` exists only in `full.csv`. Store unique `(prompt_id, cleaned_response)` items over **successful, non-empty** responses in `items.csv`; a one-to-many `item_map.csv` (`item_id → [source experiment_id, model, config, sweep value, truncation state, …]`) carries failure/truncation state. `(prompt_id, "")` must **not** collapse all failures into one item.
- **Assessment manifest** records: scorer revisions (pinned SHAs), dependency versions, rubric version, source run IDs, sampling probabilities, timestamps, seeds.

**Files:** `src/slm_experiments/evaluation/assessment/*`, `cli.py`, `core/run_store.py` (bundle helpers + `kind` branching), tests.

### Phase C — Assessment-only scorers: TSAR ensemble + KVL v2 (`assessment-scorers`)

**TSAR ensemble cross-check (Decision 1):**
- Reproduce the three **pinned** checkpoints + **confidence-max** aggregation in-process behind the eval extra. **Device auto-detect**; **P100-safe attention** (no flash-attn). Batch only successful, non-empty `cleaned_response`.
- Write `cefr_tsar_*` per-model label/confidence, ensemble label + ordinal, and a CEFR-SP ↔ TSAR **disagreement flag**, with explicit **missing/error** states.
- Summaries: mean ordinal, predicted-A1 rate (diagnostic only, never a gate), disagreement — **per model and per sweep value**, always **beside all-output failure and truncation rates**.
- **C1/C2 honored:** ordinal/direction only; SHAs + confidence-max pinned; `TRAIN_DOC_EN` single-model is the only A1-signal fallback.

**KVL v2 (`kvl-v2`, kept from prior scoping):**
- In `evaluation/kvl.py` + `evaluation/metrics.py`, add a **versioned occurrence-level** path using POS-aware **lemmatized English content-word tokens** (all occurrences, not the unique surface-form set), looked up in the **Spanish-L1** KVL table by default. (`[kvl-v2]` is an empty install marker; NLTK WordNet is a core dep — not a Spanish lemmatizer.)
- Report: token lookup **coverage**, **mean** score, **hard-token share**, and a **lower-tail** score (e.g. 10th percentile). **Never interpret a mean without coverage.**
- **Preserve** all v1 KVL fields for old-run compatibility. **Do not** touch KVL beam decoding.

**Files:** `src/slm_experiments/evaluation/assessment/cefr_tsar.py`, `evaluation/kvl.py`, `evaluation/metrics.py`, `pyproject.toml` (eval extra), tests (mocked ensemble — no model download; KVL v2 token semantics + OOV coverage).

### Phase D — Three-rater human study (`human-study`)

Implement Decision 4: `study_export` / `study_import` / `reliability.py` + the versioned rubric file; blind long-format sheets; private source key in the study bundle; validation of one rating per `(item_id, rater_id)`; **exact + adjacent (±1) percent agreement** per dimension + consensus medians (**no Krippendorff's α** — dropped by design in #16); binary human-suitable from median overall + adequacy. Leave the legacy single-rater `human export/import` in place for smoke tests until explicitly retired.

**Files:** `src/slm_experiments/human/study_export.py`, `study_import.py`, `reliability.py`, rubric file, `cli.py`, tests.

### Phase E — LLM-judge seam (`judge-seam`, kept)

Provider-neutral `judge_input.jsonl`, a versioned rubric matching the human dimensions, a **strict output schema**, and a validated `judge_scores.csv` importer. **No API adapter.** Document that quality-preservation claims stay human-sample-limited until judge results are imported and shown to agree acceptably with human consensus.

**Files:** `src/slm_experiments/evaluation/assessment/judge.py`, schema/rubric files, tests (placeholder import).

### Phase F — Analysis, validation, tests, cross-doc reconciliation (`analysis-tests-docs`)

- Per-model **paired prompt deltas** + **percentile bootstrap CIs** (Decision 2) for CEFR-SP ordinal, TSAR ordinal (secondary), and KVL; report quality deltas separately; **raw CIs, no multiplicity correction**.
- Validate CEFR-SP (and future judge) vs **weighted human consensus**: ordinal association, binary suitable/not-suitable agreement, confusion by CEFR band, uncertainty intervals. Report the same agreement for TSAR as a diagnostic.
- Surface assessment bundles in `runs list` / `runs show` (kind-aware); add new `docs/human-eval.md`.
- Execute the **cross-doc reconciliation** (appendix) across `ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md`, `AGENTS.md`.
- **Mocked tests** for: bundle immutability/dedup, kind-aware `runs` readers, TSAR confidence-max aggregation, KVL v2 token semantics + OOV coverage, blind three-rater export/import, rating validation, reliability, judge placeholder import, paired summaries, backward compatibility. **No GGUF / no live API.**

**Files:** `evaluation/assessment/analysis.py`, `core/run_store.py`, `cli.py`, `AGENTS.md`, `docs/human-eval.md`, sibling docs, tests.

---

## Files Touched (summary)

### New
- `src/slm_experiments/evaluation/assessment/` (bundle, `cefr_tsar`, judge, analysis)
- `src/slm_experiments/human/study_export.py`, `study_import.py`, `reliability.py`
- rubric + judge schema files
- `docs/human-eval.md`
- cluster `slm-thesis-eval` image recipe + `assess` sbatch script (specified here; built at implementation time)
- new tests per phase

### Modified
- `pyproject.toml` (optional extras: `[cefr-sp]`, `[cefr-tsar]`; empty `[kvl-v2]` marker)
- `src/slm_experiments/cli.py` (dispatch: `assess`, study commands)
- `src/slm_experiments/core/run_store.py` (bundle helpers, `kind` branching in `list_runs` / `runs show`)
- `src/slm_experiments/evaluation/kvl.py`, `metrics.py` (KVL v2 path)
- `ExperimentDesign.md`, `docs/metrics.md`, `docs/cites-to-include.md`, `AGENTS.md` (reconciliation appendix)

### Explicitly unchanged
- `requirements.txt` (stays torch-free; scaffold test still passes)
- `evaluation/a1_criteria.py` + `evaluation/cefr_sp.py` (the live gate — no change)
- KVL beam decoder + all generation/decoding paths
- Source-run artifacts (read-only from the bundle)

---

## Cross-doc reconciliation (comprehensive)

**Status:** executed in [#21](https://github.com/srobaina99/SLMs-experiments/issues/21). The four sibling docs plus new `docs/human-eval.md` and the Decision 4 / rubric-draft reliability wording were reconciled to shipped code. Line numbers below were as of the pre-implementation audit and may no longer match.

The four sibling docs were **already largely correct** on the CEFR-SP framing; the drift below is what the plan committed to fixing. Enumerated line-references are as of the audit.

### `ExperimentDesign.md` — mostly correct; minor drift + future additions
- **`README`-level framing correct:** line 9 (primary = CEFR-SP A1) and lines 103–111 (A1 gate = CEFR-SP; FK/Fog/Spache descriptive; SMOG excluded) need **no change**.
- **Drift — "readability proxy" wording (line 285):** Phase-2 RQ3 "Does guided top-k or KVL beam width improve the **readability proxy** / KVL metrics" predates the CEFR-SP gate. Reword to reference CEFR-SP / KVL, not "readability proxy."
- **Future additions (post-implementation):** the *Output Format* section (spec.csv columns lines 216–220 are Phase-1-shaped; full.csv "beam metadata" line 224 is deprecated-beam) and the single-rater note (line 111) will need the **assessment bundle** (`kind: assessment`, `items.csv`/`item_map.csv`, `cefr_tsar_*`) and the **three-rater study** documented once they ship.

### `docs/metrics.md` — correct on the gate; relabel legacy + add new sections
- **Relabel — heading "Primary Metrics (3)" (line 5):** FK/Fog/Spache are labeled "Primary" then immediately "no longer decide `meets_a1_criteria`." Rename to **"Descriptive readability metrics"** to remove the contradiction.
- **Relabel — "A1 threshold" / "TARGET for A1" (lines 22, 26, 38, 41, 54, 57):** these thresholds no longer gate anything; keep as historical proxy context but mark descriptive.
- **CEFR-SP section (lines 111–159) is accurate** — no change.
- **KVL section (lines 68–109) is accurate v1** — add **KVL v2** occurrence-level/lemmatized columns once shipped.
- **Add (post-implementation):** a **TSAR ensemble** subsection (assessment-only, ordinal/disagreement, A1-blind caveat, pinned SHAs) and a pointer to the assessment bundle / human study.

### `docs/cites-to-include.md` — one flatly-stale line + one framing fix
- **Stale (line 3):** "upgrading evaluation beyond the current US readability proxy (`meets_a1_criteria`: Flesch–Kincaid, Gunning Fog, Spache)". `meets_a1_criteria` is **CEFR-SP**, not FK/Fog/Spache — correct this.
- **Framing (line 19):** "Sentence CEFR (CEFR-SP / ModernBERT) | Primary neural upgrade" reads as future work; note **CEFR-SP is already live**, and the **TSAR ModernBERT ensemble is the assessment-only cross-check being added**.
- **Framing (line 197):** "Keep FK/Fog/Spache; document them as a proxy **gate**" — they are no longer the gate; descriptive.
- The reading list itself is sound and directly supports the plan (Arase CEFR-SP, Ace-CEFR, TSAR/ModernBERT, Matsumura & Arase, UniversalCEFR).

### `AGENTS.md` — correct now; additions once the stack ships
- Mission (primary outcome = CEFR-SP A1) is **correct** — no change.
- **Add (post-implementation):** `assess` in §5 (How to Run) and the CLI map; the **assessment bundle** artifact + `kind: assessment` in §6 (Results Contract); the **three-rater study** in §10 (Human Evaluation); extend §13 "Do NOT" (`torch/transformers` only via optional extras) to cover the new **`[cefr-tsar]`/eval extra**.

---

## Open at build time (only what genuinely remains)

Everything substantive is locked by the map. Left to resolve at implementation / ops time:

1. **Confirm the regenerated Phase 2 pool is healthy** (valid, non-truncated) before running real assessment numbers.
2. **Recruit three real raters**; until then, validation runs on placeholder ratings.

Resolved during shipping (recorded so the open list is not re-litigated):

3. **Lemmatizer for KVL v2:** English POS-aware `nltk.WordNetLemmatizer` (core dep) looking up the Spanish-L1 KVL table — not a Spanish lemmatizer; `[kvl-v2]` is an empty marker.
4. **Eval extras:** sibling `[cefr-tsar]` (torch + transformers≥4.55) alongside `[cefr-sp]`; `requirements.txt` stays torch-free.

---

## References

- Arase et al. 2022 — CEFR-SP: <https://aclanthology.org/2022.emnlp-main.416/> · ckpt DOI [10.5281/zenodo.7234096](https://doi.org/10.5281/zenodo.7234096)
- TSAR 2025 findings + script + evaluators: <https://aclanthology.org/2025.tsar-1.8/> · <https://github.com/tsar-workshop/tsar-2025-shared-task> · <https://huggingface.co/collections/AbdullahBarayan/tsar-2025-shared-task-on-rcts-cefr-evaluators>
- Repo research notes: `thoughts/shared/research/2026-07-19_cefr-sp-vs-tsar-ensemble.md`, `thoughts/shared/research/2026-07-19_primary-cefr-scorer-gold-standard-grounding.md`, `thoughts/shared/research/2026-07-14_cefr-l2-grading-sota-2026.md`, `thoughts/shared/research/2026-07-14_cefr-sp-sota-status.md`
- Stale-claim audit: `thoughts/plans/evaluation-stack-plan-stale-claims-audit.md`
- Human study + rubric: `thoughts/plans/human-study-rubric-draft-v0.md`
- Phase 2 failure handoff: `thoughts/shared/handoffs/general/2026-07-12_10-32-58_phase2-failure-maxed-rates.md`
