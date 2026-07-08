# Temperature-0 Standardization & top_p Removal — Implementation Plan

## Overview

Standardize **all experiments** on **`temperature=0.0`** (deterministic greedy decoding) and **remove `top_p`** from the codebase entirely. Keep **`top_k=50`** as the sole vocabulary cap applied before token selection in guided and KVL beam paths, and passed through to llama.cpp for standard generation.

**Motivation:** At temperature 0, `top_p` (nucleus sampling) overlaps with `top_k` as a candidate filter without adding experimental value — only API surface and mental overhead. Stochastic decoding (`temperature=0.7`) also conflicted with reproducibility goals and made the Phase 2 beam sweep (`best-of-N` reranking) meaningless at temp 0.

**Status:** Implemented on branch `main` (commit `9baa775` at time of writing).

---

## Design Decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| `temperature` param | **Keep** on `ExperimentConfig`, always `0.0` | Traceability in results/CSVs |
| `top_p` param | **Remove entirely** | Redundant with `top_k` at temp 0; tests already used `top_p=1.0` for guided |
| `top_k` param | **Keep** at `50` | Still defines guided pool input and KVL beam expansion set |
| Phase 2 beam sweep | **Deprecate** (CLI warning, docs) | Best-of-N at temp 0 produces identical candidates; superseded by `kvl_beam` / `guided` |
| Backward compat | **No** | Old runs with `top_p` in CSVs stay as-is |
| Scope | Core + tests + docs + scripts | Includes `scripts/run_kvl_beam_csv.py`, `smoke_kvl_beam_variants.py`, `spike_kvl_beam_eval.py` |
| HTML flow docs | **Not updated** | `docs/guided-decoding-flow.html` still shows `top_p` toy example — follow-up if needed |

---

## Current State (before change)

| Experiment | `temperature` | `top_p` | How params were used |
|------------|---------------|---------|----------------------|
| Phase 1 factorial | 0.7 | 0.95 | llama.cpp sampler |
| Phase 2 weights | 0.7 | 0.95 | llama.cpp sampler |
| Phase 2 prompting | 0.7 | 0.95 | llama.cpp sampler |
| Phase 2 beam | 0.7 | 0.95 | llama.cpp sampler × N independent samples |
| Phase 2 guided | 0.0 | 0.95 | Python logit mask → A1 pool |
| Phase 2 kvl_beam | 0.0 | 0.95 | Python logit mask → branch expansion |

Defaults lived on `ExperimentConfig` (`core/config.py:26-28`). `top_p` was never swept and had no CLI override.

---

## Desired End State

1. Every config factory sets `temperature=0.0`; `ExperimentConfig` default is `0.0`.
2. `top_p` does not exist on `ExperimentConfig`, model APIs, or decode helpers.
3. Guided decoding applies **`top_k` only** before `guided_top_k` pool trim and A1 filter.
4. KVL beam applies **`apply_top_k_mask`** (renamed from `apply_top_k_top_p_mask`) before `branch_factor` selection.
5. `phase2 beam` prints a deprecation warning; docs point to `kvl_beam` / `guided`.
6. All unit tests pass; per-phase smoke checks validate the change.

---

## What We're NOT Doing

- Removing `temperature` from `ExperimentConfig` or results
- Changing `top_k` default or sweeping it
- Redesigning the beam sweep into true beam search
- Updating `thoughts/` historical handoffs/research docs
- Backward-compatible loaders for old CSV columns containing `top_p`
- Updating `docs/guided-decoding-flow.html` (optional follow-up)

---

## Implementation Phases

### Phase 1 — Remove `top_p`

**Config layer**
- Delete `top_p: float = 0.95` from `ExperimentConfig`
- Remove `top_p=0.95` from all config factories:
  - `phase1/configs.py`
  - `phase2/weights.py`, `prompting.py`, `guided.py`, `kvl_beam.py`, `beam.py`

**Model / decode layer**

| File | Change |
|------|--------|
| `models/llamacpp.py` | Drop `top_p=` from all `llm(...)` and `make_llamacpp_eval_fn(...)` calls |
| `models/beam.py` | Remove `top_p` from `generate()` signature and `llm()` call |
| `models/constrained_decoder.py` | Delete `_apply_top_p()` and `_softmax()`; remove `top_p` from `_pool_candidates()` and `decode()` |
| `models/kvl_beam_decoder.py` | Rename `apply_top_k_top_p_mask()` → `apply_top_k_mask()`; delete nucleus branch; remove `top_p` from `make_llamacpp_eval_fn()` |

**Scripts**
- `scripts/run_kvl_beam_csv.py`
- `scripts/smoke_kvl_beam_variants.py`
- `scripts/spike_kvl_beam_eval.py` — drop `top_p` from `llm.sample()` and `llm()` calls

**Tests**
- `tests/test_configs.py` — assert `"top_p" not in field_names`
- `tests/test_constrained_decoder.py` — remove all `top_p=` kwargs
- `tests/test_kvl_beam_integration.py` — remove `top_p`

**Phase 1 smoke test**
```bash
./venv/bin/python -c "
from dataclasses import fields
from slm_experiments.core.config import ExperimentConfig
from slm_experiments.models.kvl_beam_decoder import apply_top_k_mask
assert 'top_p' not in {f.name for f in fields(ExperimentConfig)}
logits = [0.0, 1.0, 2.0, 3.0, 4.0]
masked = apply_top_k_mask(logits, top_k=2)
finite = [i for i, v in enumerate(masked) if v > float('-inf')]
assert finite == [3, 4]
print('Phase 1 smoke: OK')
"
./venv/bin/python -m pytest tests/test_configs.py tests/test_constrained_decoder.py -q
```

---

### Phase 2 — Set `temperature=0.0` everywhere

**Default**
```python
# core/config.py
temperature: float = 0.0   # was 0.7
```

**Config factories updated** (were `0.7`):
- `phase1/configs.py`
- `phase2/weights.py`
- `phase2/prompting.py`
- `phase2/beam.py`

Already at `0.0`: `guided.py`, `kvl_beam.py`, smoke/spike scripts.

**Kept unchanged**
- `ExperimentResult.temperature` — records `0.0` for new runs
- `temperature > 0` branches in `constrained_decoder._pool_candidates()` and `beam.py` logprob logic (dead path, harmless)

**Phase 2 smoke test**
```bash
./venv/bin/python -c "
from slm_experiments.phase1.configs import create_factorial_configs
from slm_experiments.phase2.weights import create_weight_configs
from slm_experiments.phase2.prompting import create_prompting_configs
from slm_experiments.phase2.beam import create_beam_configs
from slm_experiments.phase2.guided import create_guided_configs
from slm_experiments.phase2.kvl_beam import create_kvl_beam_configs
for configs in [
    create_factorial_configs(),
    create_weight_configs([1.0, 1.5]),
    create_prompting_configs([0, 1, 3]),
    create_beam_configs([4, 8]),
    create_guided_configs([5, 10, 20]),
    create_kvl_beam_configs([4, 8]),
]:
    for c in configs:
        assert c.temperature == 0.0, c.experiment_name
print('Phase 2 smoke: OK')
"
./venv/bin/python -m pytest tests/test_phase2_guided.py tests/test_phase2_kvl_beam.py -q
```

---

### Phase 3 — Deprecate Phase 2 beam sweep

**CLI (`cli.py`)**
- Help text: `[DEPRECATED] Sweep beam-search widths — use kvl_beam or guided`
- On `phase2 beam` run: stderr warning pointing to `kvl_beam` / `guided`
- Command still dispatches (no breaking removal)

**Code comments**
- `phase2/beam.py` module docstring
- `models/beam.py` class docstring

**Test**
- `tests/test_cli.py::test_phase2_run_beam_dispatches` — assert `"deprecated" in captured.err.lower()`

**Phase 3 smoke test**
```bash
./venv/bin/python -m pytest tests/test_phase2_beam.py tests/test_cli.py::TestCliPhase2Run::test_phase2_run_beam_dispatches -q
```

---

### Phase 4 — Documentation & final validation

**Docs updated**
| File | Changes |
|------|---------|
| `docs/interventions.md` | Temp 0.0; remove top-p; mark §3 beam as deprecated |
| `docs/guided-decoding.md` | Pipeline is `top_k` → `guided_top_k` → A1 filter; remove temp 0.7 comparison arm |
| `docs/kvl_beamsearch.md` | Expansion filter is `top_k` only; note beam sweep deprecation |

**Final validation**
```bash
./venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_kvl_beam_integration.py \
  --ignore=tests/test_human_eval.py
# Expected: 205 passed
```

---

## Decode Path Summary (after change)

```
Standard generation (Phase 1, weights, prompting):
  llm(..., temperature=0.0, top_k=50)

Guided decoding:
  logits → apply top_k → take guided_top_k pool → A1 filter → argmax/fallback

KVL beam:
  logits → apply_top_k_mask → top branch_factor → rank by KVL mean

Deprecated beam (best-of-N):
  N × llm(..., temperature=0.0, top_k=50)  → identical candidates at temp 0
```

---

## Files Touched

### Source
- `src/slm_experiments/core/config.py`
- `src/slm_experiments/phase1/configs.py`
- `src/slm_experiments/phase2/beam.py`, `weights.py`, `prompting.py`, `guided.py`, `kvl_beam.py`
- `src/slm_experiments/models/llamacpp.py`, `beam.py`, `constrained_decoder.py`, `kvl_beam_decoder.py`
- `src/slm_experiments/cli.py`

### Scripts
- `scripts/run_kvl_beam_csv.py`
- `scripts/smoke_kvl_beam_variants.py`
- `scripts/spike_kvl_beam_eval.py`

### Tests
- `tests/test_configs.py`
- `tests/test_constrained_decoder.py`
- `tests/test_kvl_beam_integration.py`
- `tests/test_cli.py`

### Docs
- `docs/interventions.md`
- `docs/guided-decoding.md`
- `docs/kvl_beamsearch.md`

---

## Re-run Note

Experiments executed **before** this change used `temperature=0.7` and `top_p=0.95`. Results are not comparable to new runs. Re-run affected sweeps on the updated code for publishable claims:

```bash
python -m slm_experiments phase1 --prompts all
python -m slm_experiments phase2 weights --prompts all
python -m slm_experiments phase2 prompting --prompts all
python -m slm_experiments phase2 guided --prompts all
python -m slm_experiments phase2 kvl_beam --prompts all
```

Do **not** re-run `phase2 beam` for new claims — use `kvl_beam` or `guided` instead.

---

## Known Side Effects

- **`top_k=50` can mask EOS/stop tokens** in KVL beam (`apply_top_k_mask`), leaving `finished_pool_size=0` in baseline runs. This predates the `top_p` removal and is unchanged.
- **Beam sweep at temp 0** produces N identical greedy paths; deprecation warning documents this.

---

## Follow-ups (optional)

1. Update `docs/guided-decoding-flow.html` to remove `top_p` step from the visual walkthrough
2. Remove `phase2 beam` CLI command entirely in a future breaking release
3. Add `temperature` assertion to Phase 1 / weights / prompting unit tests (currently only guided/kvl_beam test temp explicitly)
