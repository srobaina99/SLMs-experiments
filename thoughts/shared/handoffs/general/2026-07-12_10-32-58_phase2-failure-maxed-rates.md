---
date: 2026-07-12T10:32:58-03:00
researcher: Cursor Agent
git_commit: 76b638f6dc40ff8a30859f2b9352cb78b0eb16dc
branch: main
repository: SLMs-experiments
topic: "Phase 2 thesis readiness — failure/maxed-out rates + recommendations"
tags: [implementation, phase2, thesis, hit_max_tokens, failure-rate, recommendations, validation]
status: complete
last_updated: 2026-07-12
last_updated_by: Cursor Agent
type: implementation_strategy
---

# Handoff: Phase 2 thesis readiness (failure + maxed-out rates)

## Task(s)

| Task | Status |
|------|--------|
| Assess whether Phase 2 experiments are ready / useful for master thesis | **Completed** — yes, implementation-ready; no formal `--prompts all` bundles in repo yet |
| Update `docs/experiment-setup-recommendations.md` for Phase-2-only thesis; `--prompts all` as formal policy (n=3 = smoke guardrail) | **Completed** |
| Attend failure-rate survivorship issue: report failure rates **and** maxed-out (`hit_max_tokens`) rates | **Completed** end-to-end |
| Grok 4.5 subagent quality review of that work | **Completed** — verdict: core wiring solid; doc/script follow-ups remain |
| Add `scripts/_tmp_*` validators for the new rates | **Completed** — 3 offline scripts pass; optional GGUF smoke skips if model unloadable |
| Fix High findings from Grok review (README/AGENTS Phase-1 framing; ad-hoc `steps>=budget` heuristic) | **Not started** (offered, not requested) |
| Formal Phase 2 cluster runs (`--prompts all`) | **Not started** — still required before citing results |
| Re-run Phase 2 weights after logit-bias mid∪start fix | **Still required** (checklist) |
| Git commit of this work | **Not done** — all changes uncommitted on `main` |

Thesis scope decision (user): **Phase 1 will not be run**; focus is Phase 2 only.

## Critical References

- `docs/experiment-setup-recommendations.md` — thesis checklist, Phase-2-only policy, item 9 (failure + maxed rates)
- `ExperimentDesign.md` — Phase 2 sweep spec, summary schema, research questions
- `docs/metrics.md` — Failed Generations and Maxed-Out Outputs

## Recent changes

### Rates / `hit_max_tokens` (code)

- `src/slm_experiments/core/result.py` — `hit_max_tokens: bool` on `ExperimentResult`; factories accept/pass it
- `src/slm_experiments/core/pipeline.py` — propagates flag on `run` / `run_beam` / `run_guided` / `run_kvl_beam`
- `src/slm_experiments/models/base.py` — `generate()` passes through `hit_max_tokens`; timeouts/exceptions set `False`
- `src/slm_experiments/models/llamacpp.py` — greedy: `finish_reason == "length"`; guided/KVL return decoder flag
- `src/slm_experiments/models/constrained_decoder.py` — for-else on budget exhaust → `hit_max_tokens`
- `src/slm_experiments/models/kvl_beam_decoder.py` — first-finish → `False`; post-loop survivor after budget → `True`
- `src/slm_experiments/core/run_store.py` — `SPEC_COLUMNS` includes `hit_max_tokens`; `_aggregate_metric_stats` adds `generation_failure_rate`, `hit_max_tokens_count/rate`; metadata + manifest `observations.hit_max_tokens`
- `src/slm_experiments/cli.py` — `runs show` prints `failure_rate` / `maxed_out_rate` for sweep + by_model
- `tests/test_run_store.py`, `tests/test_kvl_beam_decoder.py` — aggregation + KVL flag asserts

### Docs

- `docs/experiment-setup-recommendations.md` — Phase 2–only thesis; item 1/9/15; checklist; methods framing
- `ExperimentDesign.md` — spec column + summary example + rates note
- `docs/metrics.md` — failed + maxed-out section

### Validation scripts (untracked)

- `scripts/_tmp_validate_hit_max_tokens_decoders.py`
- `scripts/_tmp_validate_hit_max_tokens_greedy_wiring.py`
- `scripts/_tmp_validate_hit_max_tokens_run_store.py`
- `scripts/_tmp_smoke_hit_max_tokens_gguf.py` (optional; skipped here when model failed to load)

## Learnings

1. **Two independent signals.** `generation_successful=False` (empty/timeout/error) and `hit_max_tokens=True` (budget exhausted without natural stop) can both be true. Means stay conditional on success; both rates use all rows as denominator.
2. **Detection differs by path.** Greedy = llama.cpp `finish_reason`; guided/KVL = Python `for/else` on decode loops. First-finish KVL must **not** be scored as maxed when `steps_total == max_tokens` but a natural stop won — ad-hoc scripts that use `steps >= budget` are wrong for that case.
3. **`overall` in `summary.json` has metric means only** — rates live in `metadata`, `by_config`, sweep sections, and `by_model`. Docs saying “every bucket” are slightly overstated for top-level `overall`.
4. **Deprecated `phase2 beam` never sets `hit_max_tokens`** — defaults `False`. Fine if excluded from thesis; gap if anyone still runs it.
5. **Thesis policy:** formal claims = `--prompts all` (25 prompts); CLI default `n=3` is intentional smoke guardrail. Cluster scripts already pass `--prompts all`.
6. **Phase 2 weights re-run still required** after mid∪start logit-bias fix before citing weight results. Phase 1 re-run is N/A (out of scope).
7. **Grok review High docs gap:** README/AGENTS still lead with Phase 1 as primary experiment path — contradicts Phase-2-only thesis decision.

## Artifacts

### Modified (uncommitted)

- `docs/experiment-setup-recommendations.md`
- `ExperimentDesign.md`
- `docs/metrics.md`
- `src/slm_experiments/cli.py`
- `src/slm_experiments/core/pipeline.py`
- `src/slm_experiments/core/result.py`
- `src/slm_experiments/core/run_store.py`
- `src/slm_experiments/models/base.py`
- `src/slm_experiments/models/constrained_decoder.py`
- `src/slm_experiments/models/kvl_beam_decoder.py`
- `src/slm_experiments/models/llamacpp.py`
- `tests/test_kvl_beam_decoder.py`
- `tests/test_run_store.py`

### New (untracked)

- `scripts/_tmp_validate_hit_max_tokens_decoders.py`
- `scripts/_tmp_validate_hit_max_tokens_greedy_wiring.py`
- `scripts/_tmp_validate_hit_max_tokens_run_store.py`
- `scripts/_tmp_smoke_hit_max_tokens_gguf.py`
- `docs/cites-to-include.md` (unrelated; pre-existing untracked — do not assume part of this work)

### Related prior docs

- `docs/clusteruy.md` — how to sbatch Phase 2
- `docs/kvl_beamsearch.md` — first-finish rationale
- `AGENTS.md` / `README.md` — still need Phase-2-only framing update

### Validate locally

```bash
./venv/bin/python scripts/_tmp_validate_hit_max_tokens_decoders.py
./venv/bin/python scripts/_tmp_validate_hit_max_tokens_greedy_wiring.py
./venv/bin/python scripts/_tmp_validate_hit_max_tokens_run_store.py
# optional:
./venv/bin/python scripts/_tmp_smoke_hit_max_tokens_gguf.py
```

Offline trio last ran: all exit 0.

## Action Items & Next Steps

1. **Doc alignment (High from Grok):** Update `README.md` Quick Start and `AGENTS.md` §3 / Results Contract so thesis = Phase 2 only; Phase 1 optional/out of scope. Add cell recipe (`a1_pass_rate` + `generation_failure_rate` + `hit_max_tokens_rate` + conditional means) to AGENTS Results Contract / ExperimentDesign if not already clear outside recommendations.
2. **Fix ad-hoc maxed heuristic:** `scripts/run_kvl_beam_csv.py` and `scripts/smoke_kvl_beam_variants.py` should use decoder/`result.hit_max_tokens`, not `steps >= max_new_tokens`.
3. **Optional polish:** print rates on top-level `by_config` in `runs show`; consider putting rates on `summary.overall`; remove duplicate `to_dict` in `result.py`.
4. **Checklist leftovers:** frame Phase 2a “1.0 = prompting + zero bias”; scope KVL claims to Spanish L1 (or sweep); split checklist “logit-bias code DONE” vs “weights re-run OPEN”.
5. **Run formal Phase 2 on ClusterUY** with `--prompts all` (weights / prompting / guided / kvl_beam). Re-run weights on current bias code before citing.
6. **Commit** when user asks (large uncommitted set on `main`).
7. Do **not** cite deprecated `phase2 beam` or trust pre-fix smoke CSVs that recomputed maxed-out via steps.

## Other Notes

- Workspace Python: always `./venv/bin/python` (not system `python3`).
- Formal observation counts (all models × grid × 25 prompts): weights 700, prompting 300, guided 400 (`0,5,10,20`), kvl_beam 400 (`1,4,6,8`). No dedicated `run_phase2_guided.sh` — run guided manually in Singularity per `docs/clusteruy.md`.
- Local `results/runs/` only has smoke guided bundles; no formal Phase 2 thesis bundles committed.
- Grok review agent IDs (this session): code review `b90b5e16-0813-4917-abdd-6365372683ea`; docs `771384e9-a1de-4b6b-91db-a70fd91892bc`; tmp-script design `cc18aadc-07b6-4972-bf52-1b51943ac35c`.
- User previously asked whether Phase 2 is ready for thesis: answer was implementation-ready, results not yet in hand; then pivoted to recommendations update + failure/maxed rates.
