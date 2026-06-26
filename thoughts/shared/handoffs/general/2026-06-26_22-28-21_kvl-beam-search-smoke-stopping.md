---
date: 2026-06-26T22:28:21+00:00
researcher: Cursor Agent
git_commit: d442285104445c37fac62d3e2728fc864d50f172
branch: cursor/98798fd4
repository: SLMs-experiments
topic: "KVL-Scored Beam Search — Implementation + Stopping Investigation"
tags: [implementation, kvl-beam, beam-search, llama-cpp, stopping, length-penalty, smoke-test]
status: complete
last_updated: 2026-06-26
last_updated_by: Cursor Agent
type: implementation_strategy
---

# Handoff: KVL beam search implemented; stopping/length policy not yet in production

## Task(s)

| Task | Status |
|------|--------|
| Implement plan `thoughts/shared/plans/2026-06-21-kvl-beam-search.md` Phases 0–5 | **Completed** (via Composer 2.5 subagents) |
| Real-model smoke test (Qwen3, 3 prompts) | **Completed** — run `20260622_110058_phase2_kvl_beam` |
| Diagnose 200-step max-token behavior | **Completed** — root cause documented |
| Parallel variant smoke tests (stopping / length penalty) | **Completed** |
| Length-penalty alpha sweep (0.15–1.5) | **Completed** |
| Full sweep reproducibility verification (16 configs × 2 prompts) | **Completed** — 32/32 match vs reference JSONs |
| Wire winning stop policy into production decoder | **Not started** |
| Phase 6 (best-of-N + KVL rerank baseline) | **Not started** (optional in plan) |
| Manual verification checkboxes in plan | **Mostly unchecked** — need human sign-off on real GGUF runs |
| Git commit of KVL beam work | **Not done** — all changes uncommitted on `cursor/98798fd4` |

**Plan phase status:** Automated criteria for Phases 0–5 are checked off in the plan. Manual verification items (spike vs one-shot, smoke-run bundle inspection, collision report review) remain open.

## Critical References

- `thoughts/shared/plans/2026-06-21-kvl-beam-search.md` — master implementation plan
- `docs/kvl_beamsearch.md` — design decisions (objective, OOV, length norm)
- `docs/kvl-beam-flow.html` — visual toy example for decoder ranking

## Recent changes

Core implementation (all new unless noted):

- `src/slm_experiments/models/word_tracker.py` — word-boundary tracking for KVL scoring
- `src/slm_experiments/models/kvl_beam_decoder.py` — `KvlBeamDecoder`, `make_llamacpp_eval_fn`, stop-token ID resolution, dual stop handling (token ID + string suffix)
- `src/slm_experiments/models/kvl_token_index.py` — lemma↔token index + collision diagnostics (Phase 5)
- `src/slm_experiments/models/llamacpp.py` — `generate_kvl_beam()`, lazy `kvl_lookup`, extended timeout for beam re-eval
- `src/slm_experiments/phase2/kvl_beam.py` — `KvlBeamSweepRunner`, config grid
- `src/slm_experiments/core/config.py` — `config_kvl_beam`, `kvl_beam_width`, `kvl_branch_factor`
- `src/slm_experiments/core/pipeline.py` — `KvlBeamModelWrapper`, `run_kvl_beam()`
- `src/slm_experiments/core/result.py` — `kvl_beam_*` fields, `create_from_kvl_beam_response()`
- `src/slm_experiments/core/run_store.py` — `by_kvl_beam_width` summary bucket
- `src/slm_experiments/cli.py` — `phase2 kvl_beam` subcommand
- `src/slm_experiments/plot.py` — plot grouping by `kvl_beam_width`
- `src/slm_experiments/models/wrappers/qwen3_llamacpp_wrapper.py` — stop-token handling adjustments

Scripts & smoke infrastructure:

- `scripts/spike_kvl_beam_eval.py` — Phase 0 llama.cpp decode-loop spike
- `scripts/smoke_kvl_beam_variants.py` — variant/sweep harness (single, `--sweep`, `--sweep-configs`, `--sweep-all-length`)
- `scripts/compare_kvl_smoke_results.py` — diff sweep JSON vs reference runs
- `scripts/audit_kvl_token_index.py` — collision stats CLI
- `scripts/clusteruy/run_phase2_kvl_beam.sh` — cluster job template

Tests (all passing in last full run: 169 passed, 2 skipped):

- `tests/test_word_tracker.py`, `tests/test_kvl_beam_decoder.py`, `tests/test_kvl_beam_integration.py` (skipped without GGUF)
- `tests/test_pipeline_kvl_beam.py`, `tests/test_phase2_kvl_beam.py`, `tests/test_kvl_token_index.py`
- `tests/test_run_store.py` — extended for `by_kvl_beam_width`
- `tests/test_models.py` — skip thesis GGUF fallback when sibling thesis dir missing

## Learnings

### 1. Why every observation hits 200 steps

Three compounding causes:

1. **EOS masked by top_k:** `apply_top_k_top_p_mask(top_k=50)` sets EOS/im_end logits to `-inf`. `_branch_token_ids` only injects stop IDs when logits are finite → **`finished_pool_size=0`** in baseline runs.
2. **KVL objective favors continuation:** Ranking is `(kvl_running_mean, cumulative_logprob)`. Stopping freezes the mean; adding easy content words keeps or improves it. No length penalty in production decoder.
3. **Search structure:** Only unfinished beams stay in `active_beams`. Final pick is `max(finished_pool + active_beams)` — long active beams beat shorter finished ones.

See `src/slm_experiments/models/kvl_beam_decoder.py:160-206` (decode loop) and `:289-293` (`_rank_key`).

### 2. Variant smoke-test conclusions (Qwen3, prompts 0–1, seed=42)

| Policy | Effect |
|--------|--------|
| `length_penalty` alone (any α) | **No effect** — all beams same length each step; `finished_pool` empty |
| `always_stop` | Creates ~4782 finished candidates/run but **same 735-char output** as baseline |
| `prefer_finished` alone | **No effect** — empty finished pool |
| `stop_pref` | Pool populated; still 200 steps; minor char diff on P0 (599 vs 735) |
| `stop_length` α=0.25 | **Best balance** — P0 173 chars, P1 409 chars; still 200 decode steps |
| `stop_length` α≥0.75 | Over-truncated, incomplete answers |
| `max80` | Only variant that cuts **steps** (80) and runtime (~47s vs ~112s/prompt) |

**Key insight:** Length penalty changes *which path wins* but not decode depth unless combined with early stop selection or lower `max_new_tokens`. α=0.15 alone is too weak; α=0.25–0.35 shortens output text meaningfully.

### 3. Reproducibility

Full 16-config sweep reproduced prior subagent results exactly (32 observations): `scripts/compare_kvl_smoke_results.py` → **ALL MATCH** on steps, chars, words_scored, kvl_mean, previews. Reference JSONs were in `/tmp/kvl_*`. Verified run: `results/kvl_beam_smoke/verify_all_params/`.

### 4. Qwen3 output quality

All variants show template artifacts (`thought`, `|imend|`, `[name]` repetition). Not a KVL pipeline bug — small-model + chat-template decode quirk. `_prepare_beam_scoring_text()` strips some but not all.

### 5. Runtime

KVL beam at W=4, K=10, max_tokens=200: ~70–160s/observation depending on CPU contention. Full prefix `reset()+eval()` per candidate per step — design accepts this for v1.

### 6. Environment

- Set `SLM_GGUF_DIR=/Users/tsis/Documents/SLMs-master-thesis/Tesis/Codigo/models/gguf` for Qwen3
- Run tests with `PYTHONPATH=src` if editable install unavailable
- Sandbox blocks GGUF load unless `required_permissions: ["all"]`

## Artifacts

**Plan & design:**
- `thoughts/shared/plans/2026-06-21-kvl-beam-search.md`
- `docs/kvl_beamsearch.md`
- `docs/kvl-beam-flow.html`

**Production code:**
- `src/slm_experiments/models/kvl_beam_decoder.py`
- `src/slm_experiments/models/word_tracker.py`
- `src/slm_experiments/models/kvl_token_index.py`
- `src/slm_experiments/models/llamacpp.py`
- `src/slm_experiments/phase2/kvl_beam.py`
- `src/slm_experiments/core/{config,pipeline,result,run_store}.py`
- `src/slm_experiments/cli.py`

**Smoke / spike scripts:**
- `scripts/smoke_kvl_beam_variants.py`
- `scripts/compare_kvl_smoke_results.py`
- `scripts/spike_kvl_beam_eval.py`
- `scripts/audit_kvl_token_index.py`

**Verified sweep results:**
- `results/kvl_beam_smoke/verify_all_params/sweep.json`
- `results/kvl_beam_smoke/verify_all_params/sweep.csv`
- `results/kvl_beam_smoke/verify_all_params/manifest.json`

**Prior real-model run (3 prompts):**
- `results/runs/20260622_110058_phase2_kvl_beam/` (if present on disk)

**Reference JSONs from subagent runs (may still be in `/tmp/`):**
- `/tmp/kvl_variant_*.json`, `/tmp/kvl_alpha_*.json`

## Action Items & Next Steps

1. **Decide production stopping policy** — recommended starting point:
   - `always_branch_stop_tokens=True` in `KvlBeamDecoder` / `make_llamacpp_eval_fn` path
   - `length_penalty_alpha=0.25` on rank key
   - Optionally lower default `max_new_tokens` to 80–100 for sweeps (only thing that cuts compute)
   - Alternative: final selection from `finished_pool` only once any beam finishes with acceptable mean

2. **Implement chosen policy in production** — currently only in `scripts/smoke_kvl_beam_variants.py` (`VariantKvlBeamDecoder`). Port to `src/slm_experiments/models/kvl_beam_decoder.py` + config fields (`kvl_beam_length_penalty_alpha`, `kvl_beam_always_stop_branch`).

3. **Re-smoke after production change:**
   ```bash
   export SLM_GGUF_DIR=... PYTHONPATH=src
   python -m slm_experiments phase2 kvl_beam --models Qwen3 --prompts 3 --widths 4 --no-plot
   ```

4. **Complete manual verification** in plan (Phases 0, 2, 4, 5 manual checkboxes).

5. **Run publishable sweep** when satisfied:
   ```bash
   python -m slm_experiments phase2 kvl_beam --prompts all --widths 4,8 --seed 42 --no-plot
   ```

6. **Commit** — all KVL beam work is uncommitted; user did not request commit during session.

7. **Phase 6 (optional):** best-of-N + KVL rerank baseline in `src/slm_experiments/models/beam.py`.

8. **Register `slow` pytest mark** in `pytest.ini` to silence integration-test warning.

## Other Notes

**CLI usage:**
```bash
python -m slm_experiments phase2 kvl_beam --models Qwen3 --prompts 3 --widths 4,8 --branch-factor 10 --kvl-l1 es --no-plot
```

**Smoke variant sweep:**
```bash
python3 scripts/smoke_kvl_beam_variants.py --sweep-all-length --prompt-indices 0,1
python3 scripts/compare_kvl_smoke_results.py results/kvl_beam_smoke/<run_id>/sweep.json
```

**Decode loop API (locked):** `reset()` + `eval(full_prefix)` per candidate; logits via `llm._ctx.get_logits()`; stop via token IDs (EOS + Qwen 151643–151648) because EOS detokenizes to empty string. Documented in `kvl_beam_decoder.py` header and `scripts/spike_kvl_beam_eval.py`.

**Subagent IDs from this session (for transcript lookup):** Phase implementers and smoke testers ran as Composer 2.5 fast subagents; final verification sweep run ID `verify_all_params`.

**`humanlayer thoughts sync`:** not available in this environment (`humanlayer` CLI not found). Handoff written to disk only.
