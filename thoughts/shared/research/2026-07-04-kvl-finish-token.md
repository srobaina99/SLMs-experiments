---
date: 2026-07-04T17:04:58-03:00
researcher: unknown
git_commit: fa3d686f22f15c9ccd03b97919161292db582ed2
branch: main
repository: SLMs-experiments
topic: "How does the KVL experiment treat the finish token?"
tags: [research, codebase, kvl, kvl-beam-decoder, stop-token, eos]
status: complete
last_updated: 2026-07-04
last_updated_by: unknown
---

# Research: How does the KVL experiment treat the finish token?

**Date**: 2026-07-04T17:04:58-03:00
**Researcher**: unknown
**Git Commit**: fa3d686f22f15c9ccd03b97919161292db582ed2
**Branch**: main
**Repository**: SLMs-experiments

## Research Question

How does the KVL experiment treat the finish token?

## Summary

The KVL experiment does not use the term "finish token." Instead, it implements **stop token handling** through a dual-check system in `KvlBeamDecoder`. Candidates that hit a stop condition are marked `finished: bool = True` on `KvlBeamCandidate` and moved into a `finished_pool` for final selection.

Stop token IDs are collected at runtime by `resolve_llamacpp_stop_token_ids()` from llama.cpp's `token_eos()`, `token_eot()`, and Qwen chat special IDs 151643–151648. Stop strings (e.g. `"<|im_end|>"`, `"<|endoftext|>"`) are used for suffix matching but are intentionally **not** tokenized into `stop_token_ids`.

When a stop token ID is emitted during beam expansion:
1. The candidate is marked `finished = True`
2. The stop token is **rolled back** from `token_ids` and `text` (it does not appear in output)
3. Any pending partial word is flushed and KVL-scored before the candidate enters `finished_pool`
4. The stop token itself is **never** KVL-scored

The KVL lookup tables (`data/kvl/kvl_lookup_*.json`) contain English content words only — no stop/EOS token entries.

## Detailed Findings

### Terminology: "finish" vs "stop"

| Concept | Code name | Location |
|---|---|---|
| Hypothesis has terminated | `KvlBeamCandidate.finished: bool` | `kvl_beam_decoder.py:95` |
| Token IDs that trigger termination | `stop_token_ids: frozenset[int]` | `kvl_beam_decoder.py:141` |
| Collection of terminated hypotheses | `finished_pool: list[KvlBeamCandidate]` | `kvl_beam_decoder.py:156` |
| Stop strings for suffix matching | `stop: list[str]` | `kvl_beam_decoder.py:140` |

The word "finish" in KVL lookup JSON files (e.g. `"finish": 0.5597`) refers to the English vocabulary word, not a stop token.

### Stop Token ID Resolution

`resolve_llamacpp_stop_token_ids()` (`kvl_beam_decoder.py:53–82`) builds the stop set:

1. `llm.token_eos()` — always included when available
2. `llm.token_eot()` — always included when available
3. Qwen chat special IDs 151643–151648 — included when within `n_vocab`

For Qwen3, EOS is token ID **151645**, which detokenizes to an empty string (documented at `kvl_beam_decoder.py:21`).

Stop strings passed from model wrappers are **not** tokenized:

```python
# qwen3_llamacpp_wrapper.py:40-41
def _get_stop_tokens(self) -> List[str]:
    return ["<|im_end|>", "<|endoftext|>"]
```

Reason (documented in `kvl_beam_decoder.py:63–66`): BPE splits multi-token stop strings into fragments (e.g. token 91 `"|"` from `"<|im_end|>"`) that would cause false-positive early stops.

Production wiring in `llamacpp.py:127–130, 744–748`:
- `_get_stop_token_ids()` calls `resolve_llamacpp_stop_token_ids(self.llm, self._get_stop_tokens())`
- Both `stop` strings and `stop_token_ids` are passed to `KvlBeamDecoder.decode()`

### Detection During Beam Search

Two independent checks run in `_extend_candidate()` after each token append (`kvl_beam_decoder.py:259–274`):

**Check 1 — Token ID** (`hit_token_stop`):
```python
hit_token_stop = (
    stop_token_ids is not None and token_id in stop_token_ids
)
```

**Check 2 — String suffix** (`hit_string_stop`):
```python
hit_string_stop = self._hits_stop(suffix, stop)
# _hits_stop: any(text.endswith(sequence) for sequence in stop)
```

The token-ID check is the primary guard for Qwen3 EOS because it detokenizes to empty string and suffix checks never fire.

### Branch Expansion: Stop Tokens Forcibly Added

`_branch_token_ids()` (`kvl_beam_decoder.py:310–330`) takes top-K logits plus any stop token IDs that have **finite logits** (not `-inf`):

```python
if (
    0 <= token_id < len(logits)
    and math.isfinite(logits[token_id])
    and logits[token_id] > float("-inf")
):
    expanded.append(token_id)
```

When `apply_top_k_top_p_mask` masks EOS to `-inf` (production uses `top_k=50`), stop tokens may not enter the branch set. Smoke test results show `finished_pool_size=0` for baseline vs `4782` for `always_stop` variant (`results/kvl_beam_smoke/verify_all_params/sweep.csv`).

The experimental `always_branch_stop_tokens` policy in `scripts/smoke_kvl_beam_variants.py:250–268` bypasses the finite-logit guard and always injects stop token IDs into the branch set.

### Behavior When Stop Token Is Emitted

When either check fires (`kvl_beam_decoder.py:264–274`):

1. `child.finished = True`
2. **Token-ID stop only**: full state rollback — `token_ids`, `text`, `pending_word`, `completed_content_words`, `kvl_scores` revert to parent beam state (stop token excised from output)
3. **String stop**: no rollback; stop string remains in text
4. `_flush_candidate_words(child)` runs to KVL-score any trailing partial word
5. Candidate goes to `finished_pool`; it does not continue in `active_beams`

### Finished Pool and Final Selection

Decode loop (`kvl_beam_decoder.py:160–206`):

- Each step: expand active beams → split children into `unfinished` / `finished`
- `finished_pool.extend(finished)` accumulates all terminated hypotheses
- Only `unfinished[:beam_width]` continue as `active_beams`
- Early exit when `not active_beams and finished_pool` (all beams exhausted, at least one finished)
- Final winner: `max(finished_pool + active_beams, key=self._rank_key)` ranked by `(kvl_running_mean, cumulative_logprob)`

Remaining active beams at max_tokens are flushed via `_flush_candidate_words()` before final selection.

The experimental `prefer_finished_epsilon` policy (`smoke_kvl_beam_variants.py:336–360`) prefers the best finished candidate if its KVL mean is within epsilon of the best active candidate. Production `KvlBeamDecoder` does not use this.

### KVL Lookup Interaction

Stop tokens are **not** part of the KVL lookup:

- `scripts/build_kvl_lookup.py` builds lookup from BEA CSV `en_target_word` → `GLMM_score` columns only
- `data/kvl/README.md` describes English content word scores per L1
- KVL scoring in `_extend_candidate()` runs on completed content words **before** the stop check (`kvl_beam_decoder.py:249–257`)
- The stop token itself is never passed to `kvl_lookup.get_score()`

`KvlTokenIndex` (`kvl_token_index.py`) is a diagnostics/collision tool and is not consulted in the beam decode loop.

### A1 Constrained Decoder (Related Pattern)

In the A1 constrained decoder (`a1_token_index.py:111–114`), stop token IDs are registered in the token index pool as `"<stop>"` entries so they can appear in the guided pool. The constrained decoder breaks on stop token ID at `constrained_decoder.py:220–222`. This is a separate decoding path from KVL beam.

## Code References

- `src/slm_experiments/models/kvl_beam_decoder.py:18–33` — Module docstring: dual stop handling design rationale
- `src/slm_experiments/models/kvl_beam_decoder.py:53–82` — `resolve_llamacpp_stop_token_ids()`
- `src/slm_experiments/models/kvl_beam_decoder.py:95` — `KvlBeamCandidate.finished` field
- `src/slm_experiments/models/kvl_beam_decoder.py:156–206` — Decode loop with `finished_pool`
- `src/slm_experiments/models/kvl_beam_decoder.py:259–274` — Stop detection and rollback
- `src/slm_experiments/models/kvl_beam_decoder.py:310–330` — `_branch_token_ids()` with stop injection
- `src/slm_experiments/models/llamacpp.py:127–130, 744–748` — Production wiring of stop tokens
- `src/slm_experiments/models/wrappers/qwen3_llamacpp_wrapper.py:40–41` — Qwen3 stop strings
- `tests/test_kvl_beam_decoder.py:213–304` — Stop token unit tests
- `scripts/smoke_kvl_beam_variants.py:250–268, 336–360` — Experimental stop/finished policies

## Architecture Documentation

```
resolve_llamacpp_stop_token_ids(llm)
    ├── token_eos()
    ├── token_eot()
    └── Qwen IDs 151643–151648
         ↓
LlamaCppWrapper._get_stop_token_ids()
         ↓
KvlBeamDecoder.decode(stop=strings, stop_token_ids=ids)
    ├── _branch_token_ids() — top-K + stop IDs (if finite logit)
    ├── _extend_candidate() per token:
    │   ├── WordTracker + KVL score completed words
    │   ├── hit_token_stop → finished=True, rollback token
    │   └── hit_string_stop → finished=True, keep text
    ├── finished → finished_pool
    └── unfinished → active_beams (top beam_width by KVL mean)
         ↓
max(finished_pool + active_beams) → KvlBeamDecodeResult
```

## Related Research

- `thoughts/shared/plans/2026-06-21-kvl-beam-search.md` — Original KVL beam design plan
- `thoughts/shared/handoffs/general/2026-06-26_22-28-21_kvl-beam-search-smoke-stopping.md` — Smoke test stopping investigation

## Open Questions

- Whether production should adopt `always_branch_stop_tokens` from smoke variants (baseline shows `finished_pool_size=0` in smoke results)
- Whether `prefer_finished_epsilon` selection policy will be promoted from experimental variants to production decoder
