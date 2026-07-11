# Experiment Setup Recommendations

Recommendations from the July 2026 multi-model adversarial review of the master-thesis experimental setup. Scope: whether each experiment’s configuration is internally consistent, whether baselines and grids support thesis-defensible claims, and what to change (or reframe) before publishing results.

**Reviewers:** Claude Opus 4.8, GPT-5.5 Medium, Composer 2.5 Fast  
**Related:** earlier June 2026 review in `[improvements.md](../improvements.md)` (implementation bugs; many fixed). This document focuses on **design validity and reporting**, including issues that appeared after `temperature=0.0` and the guided / KVL beam phases.

---



## How to read this document

Recommendations are grouped by urgency:


| Tier             | Meaning                                                         |
| ---------------- | --------------------------------------------------------------- |
| **Must do**      | Blocks or seriously weakens formal thesis claims if ignored     |
| **Should do**    | Methodological gaps; address or state explicitly as limitations |
| **Nice to have** | Improves rigor and reproducibility; not blockers                |


Each item explains **why it matters**, **which files are involved**, and **what to modify**.

---



## Overall assessment

The framework is a solid engineering setup for comparing inference-time interventions on four SLMs (Qwen2, Qwen3, TinyLlama, Phi3).


| Experiment                | Config factory                                | Runner                              | Assessment                           |
| ------------------------- | --------------------------------------------- | ----------------------------------- | ------------------------------------ |
| Phase 1 factorial         | `src/slm_experiments/phase1/configs.py`       | `phase1/runner.py`                  | Sound 2×2; strongest causal design   |
| Phase 2 weights           | `phase2/weights.py` (`create_weight_configs`) | same                                | Valid as “both ON, vary w”           |
| Phase 2 prompting         | `phase2/prompting.py`                         | same                                | Cleanest Phase 2 sweep               |
| Phase 2 guided            | `phase2/guided.py`                            | same                                | Implemented; in-run baseline `k0`    |
| Phase 2 KVL beam          | `phase2/kvl_beam.py`                          | same + `models/kvl_beam_decoder.py` | Implemented; baseline `w1` + first-finish |
| Phase 2 beam (deprecated) | `phase2/beam.py`                              | `models/beam.py`                    | Void at temp=0; do not cite          |


Shared defaults live in `src/slm_experiments/core/config.py` (`temperature=0.0`, `top_k=50`, `max_new_tokens=200`). CLI entry: `src/slm_experiments/cli.py`.

**Bottom line:** Phase 1 and Phase 2 prompting are closest to thesis-ready. Weight sweep needs clearer framing; guided and KVL now include in-run baselines. Readability pass rates are a useful **proxy**, not proof of CEFR A1 competence.

---



## Must do



### 1. Run formal claims with `--prompts all` (25 prompts) *DONE* *nothing to do*

**Why.** The CLI default of 3 prompts is intentional for smoke tests (48 Phase 1 observations). That sample is too small for stable pass rates or intervention effect sizes. Under greedy decoding the prompt set is the only within-condition variance source.

**Files involved**


| File                                      | Role                                                                |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `src/slm_experiments/core/prompts.py`     | `STANDARD_PROMPTS` — the 25 CEFR-themed items                       |
| `src/slm_experiments/phase1/runner.py`    | `parse_prompts()`, default `prompts="3"` in `FactorialRunner.run()` |
| `src/slm_experiments/phase2/weights.py`   | Same default on `WeightSweepRunner.run()`                           |
| `src/slm_experiments/phase2/prompting.py` | Same on `PromptingSweepRunner.run()`                                |
| `src/slm_experiments/phase2/guided.py`    | Same on `GuidedSweepRunner.run()`                                   |
| `src/slm_experiments/phase2/kvl_beam.py`  | Same on `KvlBeamSweepRunner.run()`                                  |
| `src/slm_experiments/cli.py`              | `--prompts` argument defaults                                       |
| `scripts/clusteruy/run_phase2_*.sh`       | Cluster jobs (prefer `--prompts all` for formal runs)               |
| `ExperimentDesign.md`                     | Documents 3 vs 25 observation counts                                |


**What to modify / do**

- **Usage (no code change required for claims):** always pass `--prompts all` for published results.
- **Docs:** keep the n=3 default in runners/CLI as a smoke guard; in `ExperimentDesign.md` and the thesis, state that n=3 is development-only.
- **Cluster:** ensure `scripts/clusteruy/run_phase2_weights.sh`, `run_phase2_prompting.sh`, `run_phase2_kvl_beam.sh` (and any guided job you add) use `--prompts all`.

```bash
python -m slm_experiments phase1 --prompts all
python -m slm_experiments phase2 weights --prompts all
# … prompting, guided, kvl_beam
```

---



### 2. Sync documentation with the code (`temperature=0.0`, no `top_p`) *DONE*

**Why.** Written protocol still describes temp 0.7 / top-p 0.95 and “planned” guided decoding. Methods copied from those docs would be wrong. Pre-migration runs are not comparable to current greedy runs.

**Status (2026-07-11):** Docs and HTML flow updated to match `core/config.py` (`temperature=0.0`, no `top_p`). Guided and KVL marked implemented. See checklist.

### 3. Exclude deprecated `phase2 beam` from thesis claims *DONE*

**Why.** At `temperature=0.0`, best-of-N repeats identical greedy decodes. Beam width cannot change the answer, but CSVs still look real.

**Status (2026-07-11):** CLI hard-fails on `phase2 beam`. Cluster script exits 1. `ExperimentDesign.md` §2b marked deprecated/excluded. Historical code kept in `phase2/beam.py` / `models/beam.py`.

### 4. Resolve or disclose KVL beam “first-finish” behavior *DONE* (Option B)

**Why it was flagged.** Width sweeps are usually read as “wider → better selected hypothesis.” The decoder returns the **first** finished non-empty candidate, not the best KVL-ranked one among finished hyps.

**Why first-finish was kept (design rationale).** Mean KVL over content words does **not** reward finishing a sentence. Collecting finished (or max-length) hypotheses and returning `max(..., key=KVL)` made the “best” answer stretch to `max_new_tokens`, often as incoherent padding — KVL never “benefits” from a clean stop. First-finish returns as soon as a non-empty beam hits EOS/stop so a completed sentence can win. KVL still ranks which partials survive each prune; first-finish only decides when to return. Width 4 vs 8 is therefore “more exploration before first natural stop,” not “better final KVL pick.”

**Status (2026-07-11):** Option B — disclosed and claimed carefully. Option A (collect finished → max KVL) left unused on purpose.

**Files involved**


| File                                                      | Role                                                                                                           |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `src/slm_experiments/models/kvl_beam_decoder.py`          | Decode loop early `return` on first finished child; module docstring states rationale                          |
| `src/slm_experiments/models/llamacpp.py`                  | `generate_kvl_beam()` wraps the decoder                                                                        |
| `src/slm_experiments/phase2/kvl_beam.py`                  | Sweep factory / runner (`DEFAULT_KVL_BEAM_WIDTH_GRID = [4, 8]`)                                                |
| `src/slm_experiments/core/pipeline.py`                    | `run_kvl_beam()` — evaluation after decode                                                                     |
| `docs/kvl_beamsearch.md`                                  | First-finish + rationale                                                                                       |
| `ExperimentDesign.md`                                     | §2e stopping rule + rationale                                                                                  |
| `thoughts/shared/research/2026-07-04-kvl-finish-token.md` | Older research note on stop tokens (may predate first-finish; prefer `kvl_beamsearch.md`)                      |


**What was done (Option B)**

- Documented first-finish and the length-padding failure mode in `docs/kvl_beamsearch.md` and `ExperimentDesign.md` §2e.
- Module docstring + inline comment in `kvl_beam_decoder.py` state the same rationale for code reviewers.
- Thesis/methods: do **not** interpret width 4 vs 8 as “better KVL selection among finished hyps”; prefer KVL columns in `full.csv` / `summary.json` as primary endpoints; readability secondary.

**Option A (not taken):** collect finished hypotheses and return `max(..., key=_rank_key)` — would require re-runs and would reintroduce max-length stretch unless a length/stop bonus were added.



### 5. Fix or document logit-bias sentence-start coverage *DONE*

**Why.** Guided decoding boosts both mid-sentence and sentence-start A1 token IDs. Weighting previously biased only mid-sentence IDs, so sentence-initial A1 words were under-boosted vs the documented “boost A1 vocabulary” intent.

**Files involved**


| File                                                | Role                                                                        |
| --------------------------------------------------- | --------------------------------------------------------------------------- |
| `src/slm_experiments/models/llamacpp.py`            | `_create_logit_bias()` — now biases `mid_sentence_ids | sentence_start_ids` |
| `src/slm_experiments/models/a1_token_index.py`      | Builds both `mid_sentence_ids` and `sentence_start_ids` (~L91–127)          |
| `src/slm_experiments/models/constrained_decoder.py` | Uses context-aware start vs mid sets (guided)                               |
| `docs/interventions.md`                             | § Building logit_bias — documents union of both contexts                    |
| `ExperimentDesign.md`                               | Probability weighting — mid + sentence-start tokenization                   |
| `tests/test_models.py`                              | Unit + `generate()` path assertions for union coverage                      |
| `src/slm_experiments/phase1/configs.py`             | Weighting arms — **re-run required** under new bias                         |
| `src/slm_experiments/phase2/weights.py`             | Weight sweep — **re-run required** under new bias                           |


**Status (2026-07-11):** Preferred fix applied and verified. Code biases
`mid_sentence_ids | sentence_start_ids`. Docs updated. Tests cover both the
helper and the `generate()` → `logit_bias` path. Real-tokenizer diagnostic
(`scripts/_tmp_diag_logit_bias_sentence_start.py`) confirmed coverage on
Qwen2.5 GGUF (50 vocab words: 54 start-only IDs that mid-only would have
missed). **Still required:** re-run Phase 1 (weighting / both) and Phase 2
weights with `--prompts all` before citing results produced under the old
mid-sentence-only bias.

**What was modified**

```python
# In llamacpp.py _create_logit_bias — union both ID sets
return {
    token_id: bias_value
    for token_id in (index.mid_sentence_ids | index.sentence_start_ids)
}
```

- Docs: `docs/interventions.md`, `ExperimentDesign.md`, `docs/guided-decoding.md`.
- Tests:
  - `test_create_logit_bias_includes_sentence_start_ids` — helper returns mid ∪ start
  - `test_config_weighting_applies_logit_bias` — `generate()` passes the same ID set
- Diagnostic: `scripts/_tmp_diag_logit_bias_sentence_start.py` (mock + real GGUF).
- **Still required:** re-run Phase 1 (weighting / both) and Phase 2 weights with `--prompts all`.

---



### 6. Stratify summaries and thesis tables by model *DONE*

**Why.** Sweep sections in `summary.json` pool all four models. A pooled “optimal” hyperparameter can reflect one model’s failures or baseline shift (Simpson’s paradox). Plots already hue by model.

**Status (2026-07-11):** `compute_summary_stats()` writes top-level `by_model` with nested `by_config` / sweep keys. Pooled sections remain as overview. `runs show` prints per-model lines. Thesis tables should still lead with per-model cells.

### 7. Reframe “A1 success” as an automated readability proxy *DONE*

**Why.** The binary success flag is three US readability formulas, not CEFR communicative descriptors. The metrics are near-collinear.

**Status (2026-07-11):** `ExperimentDesign.md` and `docs/metrics.md` (plus AGENT framing) describe `meets_a1_criteria` as a readability proxy pass. Field name unchanged. Human export/import remains the agreement layer.

## Should do



### 8. Treat the study as deterministic; use paired-by-prompt statistics

**Why.** `temperature=0.0` makes `(model, config, prompt)` deterministic. Seed replications do not add sampling noise.

**Files involved**


| File                                                  | Role                                                                             |
| ----------------------------------------------------- | -------------------------------------------------------------------------------- |
| `src/slm_experiments/core/config.py`                  | `temperature = 0.0`                                                              |
| `src/slm_experiments/cli.py`                          | `--seed` (default 42) — still useful for library init, not for sampling variance |
| `src/slm_experiments/models/wrappers/__init__.py`     | `get_model_wrapper(..., seed=seed)`                                              |
| `ExperimentDesign.md` / `improvements.md`             | Drop or qualify “3–5 seed replications” advice under temp=0                      |
| Analysis notebooks / thesis scripts (outside package) | Implement paired-by-prompt tests                                                 |


**What to modify**

- Docs: state deterministic greedy decoding; analysis = paired Δ by `prompt_id` within `model`.
- Do **not** change temperature back to 0.7 unless you intentionally want a stochastic arm (and then re-enable diversity for any best-of-N design).
- Optional: add a small analysis helper under `src/slm_experiments/` or `scripts/` that joins two conditions on `(model, prompt_id)` — not required for generation correctness.

---



### 9. Always report failure rate next to readability means

**Why.** Means in `summary.json` use only `generation_successful==True` rows; interventions that empty hard prompts look better on conditional FK.

**Files involved**


| File                                    | Role                                                                                                                                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/slm_experiments/core/run_store.py` | `_aggregate_metric_stats()` already computes `generation_failure_rate` and `a1_pass_rate` over all rows; means still exclude failures (~L63–87) |
| `src/slm_experiments/core/pipeline.py`  | Sets `generation_successful` / empty cleaned text                                                                                               |
| `ExperimentDesign.md`                   | § summary.json — tell readers which fields are primary                                                                                          |


**What to modify**

- **Thesis tables:** for every cell print `a1_pass_rate`, `generation_failure_rate`, then conditional means.
- **Optional code:** in `run_store.py` / `cli.py` `runs show`, print failure rate next to means by default so it is hard to miss.
- **KVL:** ensure timeout failures from `models/base.py` / `llamacpp.py` remain `generation_successful=False` and are counted (see item 13).

---



### 10. Frame Phase 2 weight sweep precisely

**Why.** Every grid point has prompting ON. `weight_factor=1.0` is zero bias, not a Phase 1 control.

**Files involved**


| File                                     | Role                                                                                            |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `src/slm_experiments/phase2/weights.py`  | `create_weight_configs()` sets `config_weighting=True`, `config_prompting=True` for all factors |
| `src/slm_experiments/phase1/configs.py`  | `weighting_only` has prompting **OFF** — different carrier                                      |
| `src/slm_experiments/models/llamacpp.py` | `_create_logit_bias` → `log(1.0)=0` for factor 1.0                                              |
| `ExperimentDesign.md`                    | §2a — must say “weighting + prompting ON”                                                       |
| `docs/interventions.md`                  | Clarify Phase 2a vs Phase 1 weighting_only                                                      |


**What to modify**

- **Docs only (if design kept):** rewrite §2a in `ExperimentDesign.md` and the docstring in `weights.py` to: “combined intervention, vary weight factor.”
- **Do not** label `1.0` as “control” in plots (`plot.py`) or thesis captions — use “prompting + zero bias.”
- **Optional experiment:** add a second grid with `config_prompting=False` if you need an isolated weight dose–response (new factory or flag in `weights.py`).

---



### 11. Add in-run baselines for guided and KVL sweeps *DONE*

**Why.** Without an intervention-off point in the same bundle, “does guided/KVL help?” requires fragile cross-run joins.

**Status (2026-07-11):** In-run baselines added on the fixed carrier (`config_prompting=True`, `num_shots=0`, `config_weighting=False`):

| Sweep | Baseline | Intervention grid | Decode for baseline |
| ----- | -------- | ----------------- | ------------------- |
| Guided | `guided_top_k=0`, `config_guided=False`, `{model}_guided_k0` | `5,10,20` | `pipeline.run()` |
| KVL | width `1`, `config_kvl_beam=False`, `{model}_kvl_beam_w1` | `4,6,8` | `pipeline.run()` (not width-1 beam) |

Defaults: guided `--top-k-pools 0,5,10,20`; KVL `--widths 1,4,6,8`. Baseline rows stamp sweep keys so `by_guided_top_k` / `by_kvl_beam_width` include them. `run_guided` now passes KVL metrics for parity with baseline rows.

**Files modified:** `phase2/guided.py`, `phase2/kvl_beam.py`, `cli.py`, `pipeline.py`, `result.py`, `ExperimentDesign.md`, guided/KVL docs, cluster script, tests.

---


### 12. Scope KVL claims to Spanish L1 (or sweep L1)

**Why.** Default L1 is Spanish; results do not auto-generalize to `de` / `cn`.

**Files involved**


| File                                             | Role                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------- |
| `src/slm_experiments/core/config.py`             | `kvl_l1: str = "es"`                                                |
| `src/slm_experiments/phase2/kvl_beam.py`         | `create_kvl_beam_configs(..., kvl_l1="es")`; runner CLI passthrough |
| `src/slm_experiments/cli.py`                     | `--kvl-l1` choices `es|de|cn`                                       |
| `data/kvl/kvl_lookup_es.json` (and `_de`, `_cn`) | Lookup tables                                                       |
| `src/slm_experiments/evaluation/kvl.py`          | `compute_kvl_metrics`, `KvlLookup`                                  |
| `ExperimentDesign.md` / `docs/kvl_beamsearch.md` | Scope claims                                                        |


**What to modify**

- **Thesis:** claim Spanish L1 only, **or**
- **Code/runs:** invoke `--kvl-l1 de` / `cn` (or extend the sweep factory to loop L1s) and report separately.
- Prefer KVL metrics as primary outcome for this arm (`run_store.py` already includes them in `NUMERIC_SUMMARY_COLUMNS`).

---



### 13. Bound and report KVL beam timeouts

**Why.** Runner uses `timeout_seconds=7200`, but per-generation budget can grow to ~`max_new_tokens × beam_width × 12 + 300` seconds.

**Files to modify**


| File                                                     | What to change                                                                                                                                     |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/slm_experiments/models/llamacpp.py`                 | `_kvl_beam_timeout_seconds()` (~L597–612) — replace open-ended `max(...)` with an explicit cap; make `seconds_per_eval` a named constant or config |
| `src/slm_experiments/phase2/kvl_beam.py`                 | `get_model_wrapper(..., timeout_seconds=7200)` (~L135) — align with the capped per-call budget                                                     |
| `src/slm_experiments/models/base.py`                     | Timeout → `generation_successful=False` path                                                                                                       |
| `src/slm_experiments/core/result.py` / `full.csv` fields | Optional: `timeout=True` or error_message class for summary counts                                                                                 |
| `src/slm_experiments/core/run_store.py`                  | Optional: `timeout_count` / `timeout_rate` in `_aggregate_metric_stats`                                                                            |
| `docs/kvl_beamsearch.md`, `ExperimentDesign.md`          | Document wall-clock budget                                                                                                                         |


---



### 14. Label Phase 2 as exploratory (multiple comparisons)

**Why.** Many grid × model × metric contrasts without FDR.

**Files involved**


| File                                                   | Role                                        |
| ------------------------------------------------------ | ------------------------------------------- |
| `ExperimentDesign.md`                                  | § Statistical Analysis / Research Questions |
| `docs/experiment-setup-recommendations.md` (this file) | Checklist                                   |
| Thesis methods (not in repo)                           | Pre-register primary endpoints              |


**What to modify:** documentation and thesis text only — e.g. primary = per-model `a1_pass_rate` + mean FK; other contrasts exploratory unless FDR applied.

---



### 15. Acknowledge Phase 1’s fixed weight strength

**Why.** Weighting arms always use `weight_factor=1.5`.

**Files involved**


| File                                    | Role                                                             |
| --------------------------------------- | ---------------------------------------------------------------- |
| `src/slm_experiments/phase1/configs.py` | `DEFAULT_WEIGHT_FACTOR = 1.5` applied for all cells (~L13, ~L51) |
| `ExperimentDesign.md`                   | Phase 1 table — state “at weight_factor=1.5”                     |
| `src/slm_experiments/phase2/weights.py` | Dose–response under combined setting                             |


**What to modify:** docs/thesis wording; optionally only set `weight_factor` when `config_weighting=True` in `configs.py` to avoid misleading metadata on control rows (cosmetic).

---



## Nice to have



### 16. Record model file and library provenance in the run manifest

**Files to modify**


| File                                                 | Change                                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| `src/slm_experiments/core/run_store.py`              | `write_bundle()` / manifest schema — add GGUF path, file hash, package versions |
| `src/slm_experiments/models/llamacpp.py` or wrappers | Expose resolved GGUF path after load                                            |
| `src/slm_experiments/evaluation/metrics.py`          | Log whether NLTK resources loaded vs heuristic POS fallback                     |
| `pyproject.toml` / `requirements.txt`                | Pin versions used in thesis runs                                                |


---



### 17. Stratify prompt analysis by thematic type

**Files involved**


| File                                                 | Role                                        |
| ---------------------------------------------------- | ------------------------------------------- |
| `src/slm_experiments/core/prompts.py`                | `STANDARD_PROMPTS` with CEFR theme comments |
| Analysis / thesis (or optional `run_store` grouping) | Bucket definition vs description vs how-to  |


**What to modify:** add a small `PROMPT_CATEGORIES` map in `prompts.py` (or analysis script) and report pass rates by category; no change required to generation.

---



### 18. Document model asymmetry

**Files to modify**


| File                                                           | Role                           |
| -------------------------------------------------------------- | ------------------------------ |
| `src/slm_experiments/core/prompts.py`                          | `MODEL_CONFIGS` registry       |
| `docs/models.md`                                               | GGUF names, GPU notes for Phi3 |
| `src/slm_experiments/models/wrappers/phi3_llamacpp_wrapper.py` | GPU layers if used             |
| `ExperimentDesign.md`                                          | Models table                   |


---



### 19. Soften or rename guided “trie” mode in thesis text

**Files involved**


| File                                           | Role                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| `src/slm_experiments/models/a1_token_index.py` | `A1TokenTrie.continuation_ids` — per-word continuation, not full vocab trie |
| `src/slm_experiments/phase2/guided.py`         | `guided_mode` / `--mode`                                                    |
| `docs/guided-decoding.md`                      | Describe actual mechanism                                                   |


---



### 20. Remove dual A1-ratio implementations with deprecated beam

**Files to modify**


| File                                            | Role                                      |
| ----------------------------------------------- | ----------------------------------------- |
| `src/slm_experiments/models/beam.py`            | Separate A1-ratio formula (weighted 1.5×) |
| `src/slm_experiments/evaluation/metrics.py`     | `calculate_a1_word_ratio`                 |
| `src/slm_experiments/phase2/beam.py` + `cli.py` | Remove together with item 3               |
| Tests importing deprecated beam path            | Delete or rewrite                         |


---



## File index (quick lookup)


| Concern                    | Primary files                                                                |
| -------------------------- | ---------------------------------------------------------------------------- |
| Shared generation defaults | `src/slm_experiments/core/config.py`                                         |
| Phase 1 factory            | `src/slm_experiments/phase1/configs.py`                                      |
| Phase 2 factories/runners  | `src/slm_experiments/phase2/{weights,prompting,guided,kvl_beam,beam}.py`     |
| CLI                        | `src/slm_experiments/cli.py`                                                 |
| Logit bias                 | `src/slm_experiments/models/llamacpp.py` (`_create_logit_bias`)              |
| A1 token IDs               | `src/slm_experiments/models/a1_token_index.py`                               |
| Guided decode              | `src/slm_experiments/models/constrained_decoder.py`                          |
| KVL beam decode            | `src/slm_experiments/models/kvl_beam_decoder.py`                             |
| Deprecated best-of-N       | `src/slm_experiments/models/beam.py`                                         |
| Pipeline / A1 flag         | `src/slm_experiments/core/pipeline.py`, `evaluation/a1_criteria.py`          |
| Summaries                  | `src/slm_experiments/core/run_store.py`                                      |
| Prompts / models           | `src/slm_experiments/core/prompts.py`                                        |
| Formal protocol doc        | `ExperimentDesign.md`                                                        |
| Intervention docs          | `docs/interventions.md`, `docs/guided-decoding.md`, `docs/kvl_beamsearch.md` |
| Cluster jobs               | `scripts/clusteruy/*.sh`, `docs/clusteruy.md`                                |


---



## Recommended methods framing (paste-friendly)

> We evaluate whether inference-time interventions improve **automated readability** of English answers from four small instruction-tuned models under **deterministic greedy decoding** (`temperature = 0`, `top_k = 50`, max 200 new tokens; see `src/slm_experiments/core/config.py`). Primary binary outcome is a **proxy pass** (`meets_a1_criteria` in `evaluation/a1_criteria.py`) requiring Flesch–Kincaid ≤ 5, Gunning Fog ≤ 6, and Spache ≤ 4 on valid generations. This is not a CEFR A1 proficiency test; human ratings on a subsample assess agreement with the proxy. Phase 1 (`phase1/configs.py`) uses a 2×2 factorial of logit bias (weight factor 1.5) and contextual prompting. Phase 2 sweeps one hyperparameter at a time on fixed carriers in `phase2/*.py`. All published results use the full 25-prompt set (`STANDARD_PROMPTS` in `core/prompts.py`). Best-of-N beam (`phase2 beam`) at temperature 0 is excluded. KVL beam (`models/kvl_beam_decoder.py`) uses first-finish stopping (mean KVL does not reward EOS; without it, outputs pad to max length) and is scored for Spanish L1 vocabulary difficulty unless other L1s are swept.

---



## Priority checklist before defending results

- [ ] All cited runs use `--prompts all` on **current** temp=0 code (`phase1/runner.py` / `phase2/*.py` defaults overridden via CLI)
- [x] `ExperimentDesign.md`, `docs/interventions.md`, `docs/guided-decoding.md`, `docs/kvl_beamsearch.md` match `core/config.py` (temp=0.0, no top_p; guided/KVL marked implemented)
- [x] Deprecated beam excluded (`cli.py` hard-fail / `scripts/clusteruy/run_phase2_beam.sh` exits 1 / thesis docs exclude)
- [x] KVL first-finish **kept by design** and disclosed (`docs/kvl_beamsearch.md` / `ExperimentDesign.md` §2e / `kvl_beam_decoder.py` docstring). Rationale: mean KVL does not reward stop; without first-finish, outputs pad to `max_new_tokens`. Width ≠ “better final KVL pick.”
- [x] Logit bias sentence-start fixed in `llamacpp.py` `_create_logit_bias` (union mid + start IDs); helper + `generate()` tests + real-GGUF diag pass; **re-run** Phase 1 weighting / both and Phase 2 weights with `--prompts all` before citing results
- [x] `by_model` added in `run_store.py`; thesis tables should stratify by model
- [x] “A1 pass” framed as readability proxy in `ExperimentDesign.md` / `docs/metrics.md`
- [ ] Phase 2a framed as combined prompting + weight in `ExperimentDesign.md` / `weights.py` docstring
- [ ] Failure rates from `summary.json` reported beside conditional means
- [x] Guided / KVL in-run baselines in `phase2/guided.py` and `phase2/kvl_beam.py` (defaults `0,5,10,20` and `1,4,6,8`)
- [ ] KVL claims scoped to L1 = Spanish (`config.kvl_l1`) or multi-L1 sweeps completed

---



## Review metadata


| Field            | Value                                                                   |
| ---------------- | ----------------------------------------------------------------------- |
| Review date      | 2026-07-08                                                              |
| Document written | 2026-07-11                                                              |
| Document updated | 2026-07-11 (items 2–7, 11 attended; item 4 Option B + first-finish rationale) |
| Command          | `/multi-model-review`                                                   |
| Focus            | Experiment configs, baselines, grids, evaluation validity               |
| Prior review     | `[improvements.md](../improvements.md)` (2026-06, implementation fixes) |


