# Experiment Improvements

Multi-model adversarial review of the SLM experimentation framework (Claude Opus 4.8, Composer 2.5 Fast, GPT-5.5 Medium). Scope: experiment design, intervention correctness, metrics, and fixes landed through 2026-06-14.

**Intent:** Evaluate whether inference-time interventions help four SLMs produce CEFR A1 English. Phase 1 is a 4×4 factorial; Phase 2 sweeps `weight_factor`, `beam_width`, and `num_shots`.

---

## Overall Verdict (updated 2026-06-14)

The experimental structure is sound — a 2×2 factorial for Phase 1 and one-at-a-time sweeps for Phase 2 are appropriate designs. **Core implementation gaps that invalidated weighting, beam, and A1 measurement have been fixed in git** (see [Fixes landed in git](#fixes-landed-in-git)).

Remaining work is mostly **methodological rigor and reporting** (model-stratified summaries, beam candidate diversity, doc/code alignment on temperature), not broken inference paths.

| Aspect | Assessment |
|--------|------------|
| Phase 1 factorial design | Correct structure; default n=3 is intentional for smoke runs (see #6) |
| Phase 2 sweep design | Sensible one-at-a-time grids; baselines and cross-model pooling need care |
| Logit bias | Fixed — `log(weight_factor)` + vocab cleaned at load time |
| Beam search | Fixed — stop tokens, cleaned scoring, per-beam content words, response-only text |
| Summary / sweep buckets | Sweep sections + `by_model` stratification; proxy pass rates |
| Publishable claims today | Use `--prompts all` (25) for formal claims. Weighting and beam paths are trustworthy after re-run. A1 pass rate is now computed in code. |

---

## Fixes Landed in Git

| Commit | What changed |
|--------|--------------|
| `9666a01` | Strip polluted vocab tokens + load-time skip; beam `sequence_text=response_text`; Phase 2 sweep buckets; `num_shots` on `ExperimentResult`; docs aligned with `log(weight_factor)` |
| `7a6b105` | Beam stop tokens; A1-ratio on extracted/cleaned text; per-beam content words (no cross-candidate union) |
| `1207c3d` | `meets_a1_criteria` (FK≤5 ∧ Fog≤6 ∧ Spache≤4); `a1_pass_rate` / `a1_pass_count` in summaries; `generation_successful_count` + `generation_failure_rate` |
| `ecefbb6` | Integration tests for A1 criteria (pipeline, beam, CSV, summary pass rates) |

`log(weight_factor)` was already correct in the initial commit; docs were corrected in `9666a01`.

**Re-run note:** Any weighting or beam experiments run before these commits should be discarded and re-run on the fixed code.

---

## Consensus Findings

### Act on — likely to invalidate results

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | **A1 success criteria never implemented** | **Fixed** (`1207c3d`) | `meets_a1_criteria` in pipeline + pass rates in `summary.json` |
| 2 | **Beam path missing stop tokens** | **Fixed** (`7a6b105`) | `stop=self._get_stop_tokens()` passed to beam generator |
| 3 | **A1-ratio scored on raw text, not cleaned text** | **Fixed** (`7a6b105`) | `_prepare_beam_scoring_text()` before selection |
| 4 | **Cross-candidate content-word union biases beam selection** | **Fixed** (`7a6b105`) | Per-beam `extract_content_words` |
| 5 | **Vocabulary list includes punctuation & stop tokens** | **Fixed** (`9666a01`) | Removed from file + `_SKIP_VOCAB_ENTRIES` at load time |
| 6 | **Default n=3 prompts is underpowered** | **By design** | See [Default prompt count](#default-prompt-count-n3) below |
| 7 | **Summary `count` includes failures; means exclude them** | **Partial** (`1207c3d`) | Added `generation_successful_count` and `generation_failure_rate`; `count` still means total observations |
| 8 | **Phase 2 summaries pool across models** | **Fixed** | `summary.json` includes `by_model` with nested `by_config` / sweep keys |

### Consider — methodological weaknesses

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 9 | **Prompting few-shot leakage** | **Fixed** | Varied shot examples (definition, how-to, descriptive); none overlap evaluation prompts |
| 10 | **Stochastic RNG advances across conditions** | **Superseded** | All experiments use `temperature=0.0` (greedy); seed replications no longer add sampling noise — use paired-by-prompt analysis |
| 11 | **Phase 2 weight sweep: docs say greedy, code uses temp=0.7** | **Superseded** | Code and docs aligned on `temperature=0.0`; `top_p` removed |
| 12 | **`weight_factor=1.0` still passes zero-bias dict** | Open | Unlike `logit_bias=None` when weighting off; baseline in weight sweep ≠ Phase 1 `prompting_only` |
| 13 | **Failure exclusion → survivorship bias** | Open | Interventions that cause more empty outputs look better on conditional means; mitigated partially by `generation_failure_rate` |
| 14 | **Beam optimizes same vocab used for logit bias; outcomes judged by readability** | Open | Circular optimization; no fluency/coherence check |
| 15 | **Best-of-N may produce identical candidates** | Open | llama.cpp may reset RNG to model seed each call; verify on your llama.cpp version |

---

## Default Prompt Count (n=3)

Reviewers flagged n=3 as **statistically underpowered** for publishable effect sizes — that concern stands for formal claims.

The **default is intentional**, not a bug: it prevents accidentally launching a full run (`--prompts all` → 25 prompts × all configs × all models) when doing a quick smoke test.

| Use case | Command | Observations (Phase 1) |
|----------|---------|------------------------|
| Smoke / dev (default) | `phase1` or `--prompts 3` | 4 × 4 × 3 = **48** |
| Formal / publishable claims | `--prompts all` | 4 × 4 × 25 = **400** |

Documented in `ExperimentDesign.md` and `AGENTS.md`. For any claim about intervention effect size, use `--prompts all`. Under `temperature=0.0`, prefer paired-by-prompt contrasts within model rather than multi-seed sampling replications.

---

## Lone-Model Findings

| Finding | Model | Category |
|---------|-------|----------|
| `cumulative_log_prob` is fake (`-log(temp)×length`) — longest output wins | Opus | Consider — dead code path, misleading in `full.csv` |
| 1.5× A1 weight is cosmetic for argmax selection | Opus | Noted — doesn't change which beam wins |
| Logit bias is approximate, not exact probability multiplier | Opus | Noted — docs slightly overclaim |
| Tokenization misses sentence-initial forms (`" " + word` only) | Composer | Consider |
| Phase 2 weight sweep confounds with prompting ON (≠ Phase 1 `weighting_only`) | Composer | Consider — frame Phase 2a as “both ON, vary w” |
| Two A1-ratio implementations (`beam.py` vs `metrics.py`) | Composer | Consider |
| Phi3 GPU vs others CPU | Composer | Noted — document asymmetry |
| No multiple-comparison correction | All | Noted — expected for exploratory work |
| Short-text readability metrics are noisy | Opus | Noted — design already excludes SMOG for this reason |
| Sweep key `1.0`→`"1"` vs experiment name `1_0` | Opus | Noted — minor join friction |
| NLTK legacy resource names | Opus | Noted — silent fallback to heuristic POS |

---

## Dismissed / Lower Priority

- **Phi3 GPU asymmetry** — real but unlikely to flip conclusions; document it
- **Sweep key formatting (`"1"` vs `1_0`)** — cosmetic
- **Phase 1 omitting explicit `num_shots=0`** — harmless (default is 0)
- **`prompt_id` P1 vs p01 in tests** — no runtime impact
- **Default n=3 underpowered for claims** — not a code defect; use `--prompts all` when claiming results

---

## Prioritized Roadmap

### Tier 1 — Fix before trusting weighting/beam results

| # | Item | Status |
|---|------|--------|
| 1 | Strip punctuation and special tokens from bias vocabulary | **Done** (`9666a01`) — re-run old weighting experiments |
| 2 | Implement `meets_a1_criteria` and report pass rate | **Done** (`1207c3d`, `ecefbb6`) |
| 3 | Pass stop tokens into beam generation; score cleaned text | **Done** (`7a6b105`) |
| 4 | Per-beam content words — drop cross-candidate union | **Done** (`7a6b105`) |
| 5 | Verify beam candidate diversity (`seed + i`, N>1 differ test) | **Open** |

### Tier 2 — Strengthen experimental validity

| # | Item | Status |
|---|------|--------|
| 6 | Full sample size for claims — `--prompts all` (25); seed replications | **Usage policy** (default n=3 stays for safety) |
| 7 | Per-observation seeds — `hash(model, prompt, config, rep)` | **Open** |
| 8 | Report failure rate alongside means | **Partial** (`1207c3d`) — `generation_successful_count` added |
| 9 | Add `by_model` × sweep breakdown in `summary.json` | **Done** |
| 10 | Disjoint few-shot examples from evaluation prompts | **Done** |
| 11 | Align weight sweep with docs — greedy temp or update docs | **Open** |

### Tier 3 — Analysis & reporting

| # | Item | Status |
|---|------|--------|
| 12 | Pre-register primary comparisons; apply FDR for Phase 2 grids | **Open** |
| 13 | Report effect sizes with CIs (paired by prompt within model) | **Open** |
| 14 | For beam: selected vs runner-up readability; width=1 baseline | **Open** |
| 15 | Unify A1-ratio computation in `TextEvaluator` | **Open** |

---

## Key File References

| Area | Files |
|------|-------|
| Experiment spec | `ExperimentDesign.md`, `docs/interventions.md`, `docs/metrics.md` |
| Guided decoding (planned) | `docs/guided-decoding.md` |
| A1 pass criteria | `src/slm_experiments/evaluation/a1_criteria.py`, `core/pipeline.py`, `core/result.py` |
| Logit bias | `src/slm_experiments/models/llamacpp.py`, `models/base.py`, `data/vocabularies/filtered_starters_vocab.txt` |
| Beam / best-of-N | `src/slm_experiments/models/beam.py`, `models/llamacpp.py` (`_generate_beam_impl`) |
| Summaries | `src/slm_experiments/core/run_store.py` |
| Prompting shots | `src/slm_experiments/core/prompts.py` (`SHOT_EXAMPLES`, `STANDARD_PROMPTS`) |
| Phase runners | `src/slm_experiments/phase1/runner.py`, `phase2/weights.py`, `phase2/beam.py`, `phase2/prompting.py` |

---

## Review Metadata

- **Review date:** 2026-06-09
- **Status update:** 2026-06-14 (git commits `9666a01`–`ecefbb6`)
- **Reviewers:** Claude Opus 4.8 (thinking high), Composer 2.5 Fast, GPT-5.5 Medium
- **Scope:** Full experiment framework + fixes through 2026-06-14
- **Command:** `/multi-model-review`
