# KVL-Scored Real Beam Search Implementation Plan

## Overview

Implement **KVL-scored token-level beam search** — a fifth inference-time intervention that performs genuine beam search ranked by running mean GLMM learner-vocabulary scores (from `KvlLookup`) instead of post-hoc A1-ratio reranking over independent samples. The experiment answers: *If we steer decoding toward vocabulary Spanish L1 learners are likely to know, does that improve `meets_a1_criteria` more than best-of-N reranking?*

Source design: [`docs/kvl-beam-flow.html`](../../docs/kvl-beam-flow.html), [`docs/kvl_beamsearch.md`](../../docs/kvl_beamsearch.md).

**CLI target:** `python -m slm_experiments phase2 kvl_beam`

## Current State Analysis

### What exists

| Component | Location | Status |
|-----------|----------|--------|
| KVL post-hoc metrics | `src/slm_experiments/evaluation/kvl.py` | Implemented |
| KVL lookup data (es/de/cn) | `data/kvl/kvl_lookup_*.json` | Implemented |
| Pipeline KVL integration | `src/slm_experiments/core/pipeline.py:50-57, 97, 174` | Post-hoc only |
| Phase 2 A1 "beam" sweep | `src/slm_experiments/phase2/beam.py` | Best-of-N + A1-ratio rerank |
| Current beam generator | `src/slm_experiments/models/beam.py` | N independent one-shot samples |
| llama.cpp wrapper | `src/slm_experiments/models/llamacpp.py` | One-shot `llm(prompt, max_tokens=…)` |
| Config `kvl_l1` | `src/slm_experiments/core/config.py:30` | Implemented |
| KVL tests | `tests/test_kvl_metrics.py` | Post-hoc metrics only |

### What's missing

The current `phase2 beam` command is **not real beam search**. It draws N independent full-length samples via `BeamSearchGenerator.generate()` (`beam.py:37-90`) and reranks finished answers by A1 ratio. KVL beam requires a **manual token-by-token decode loop** with W parallel hypotheses pruned by running KVL mean at word boundaries.

No `generate_kvl_beam()`, `KvlBeamDecoder`, `run_kvl_beam()`, `phase2/kvl_beam.py`, or CLI registration exists.

### Key discoveries

- **Highest-risk integration:** llama.cpp per-step eval API — current wrappers have no per-token hook (`llamacpp.py:164-173` uses one-shot generation).
- **Reuse patterns:** Mirror `BeamSweepRunner` (`phase2/beam.py:92-191`) for runner; mirror `create_from_beam_response()` (`result.py:122-161`) for result factory; reuse `_get_stop_tokens()`, `_prepare_beam_scoring_text()`, `_extract_response()` from `LlamaCppBaseWrapper` (`llamacpp.py:96-135`).
- **Scoring parity:** Beam objective must match `compute_kvl_metrics()` — content words only, OOV excluded from mean (`kvl.py:85-106`).
- **Compute cost:** O(steps × beam_width × branch_factor) forward passes vs ~100 for greedy. Default sweep grid is smaller than A1 beam: widths `4, 8` (not `4, 8, 10`).
- **Isolation:** Prompting ON (zero-shot), weighting OFF, temperature `0.0` — same isolation as A1 beam sweep but deterministic expansion.

## Desired End State

After this plan is complete:

1. `python -m slm_experiments phase2 kvl_beam --widths 4,8 --prompts 3` runs end-to-end on at least one GGUF model.
2. Each observation performs token-level beam search: expand top-K logits per beam, update KVL running mean at word boundaries, keep top-W by `(kvl_running_mean ↓, cumulative_logprob ↓)`.
3. Results land in `results/runs/{run_id}/` with `full.csv` (including `kvl_beam_*` metadata), `summary.json` with `by_kvl_beam_width` bucket, and standard readability + post-hoc KVL columns.
4. Unit tests cover decoder ranking logic (mock logits), word-boundary tracking, and sweep runner config parsing — without requiring GGUF in CI.

### Verification

```bash
# Unit tests (no GGUF required)
pytest tests/test_word_tracker.py tests/test_kvl_beam_decoder.py tests/test_phase2_kvl_beam.py -q

# Smoke run (requires GGUF models)
python -m slm_experiments phase2 kvl_beam --models Qwen3 --prompts 1 --widths 4 --no-plot

# Publishable grid
python -m slm_experiments phase2 kvl_beam --prompts all --widths 4,8 --no-plot
```

## What We're NOT Doing

- **Guided decoding** (`phase2 guided`) — separate intervention; share decode-loop utilities only where cheap.
- **Trie-mode word completion** — deferred to v2 (design option in `kvl_beamsearch.md`).
- **Optional lemmatization arm** — deferred to v2.
- **Best-of-N + KVL rerank baseline** — recommended analysis arm but not blocking MVP; add in Phase 6 if time permits.
- **Combining with logit bias, A1 beam, or guided decoding** in the first sweep.
- **KV cache batching across beams** — sequential eval per candidate in v1; optimize later if too slow.
- **Updating `ExperimentDesign.md` or HTML docs** — code-first; doc sync is follow-up.

## Implementation Approach

Build bottom-up: spike llama.cpp eval loop → pure decoder with mock logits → word tracker → wrapper integration → pipeline/result/store → sweep runner + CLI. Follow the thin-runner pattern established by `phase2/beam.py`. Keep KVL scoring logic in the decoder; keep evaluation on the final selected response in the pipeline (unchanged post-hoc path).

**Locked design decisions** (from design docs — no open questions):

| Decision | Choice |
|----------|--------|
| Beam objective | Mean GLMM of looked-up content words |
| Tie-breaker | `cumulative_logprob` (lexicographic, both descending) |
| When to score | Word boundary (whitespace/punctuation in decoded UTF-8) |
| Which words | Content words via `TextEvaluator.extract_content_words()` per completed word |
| OOV policy | Exclude from mean (matches `compute_kvl_metrics`) |
| Expansion | `beam_width ∈ {4, 8}`, `branch_factor=10`, `temperature=0.0` |
| Model logits filter | Apply config `top_k`/`top_p` first, then take top `branch_factor` |
| L1 | `ExperimentConfig.kvl_l1` (default `es`) |
| Intervention mix | `config_prompting=True`, `config_weighting=False`, zero-shot |

---

## Phase 0: llama.cpp Decode Loop Spike

### Overview

Validate that llama-cpp-python supports a manual per-token eval loop on one real GGUF before building the full decoder. This de-risks the highest-integration-risk item identified in the design doc.

### Changes Required

#### 1. Spike script (temporary, not committed to package)

**File:** `scripts/spike_kvl_beam_eval.py` (delete or move to `scripts/` after spike)

```python
"""Spike: manual token loop on one GGUF model."""
# 1. Load Qwen3 via get_model_wrapper("Qwen3")
# 2. Tokenize formatted prompt
# 3. Loop max_tokens times:
#    - llm.eval([token]) or reset + eval(full_prefix) depending on API
#    - Read logits / sample top-5 token IDs
#    - Append chosen token, check stop sequences
# 4. Compare output to one-shot llm(prompt, max_tokens=20) for sanity
```

Research `llama_cpp.Llama` API: `tokenize()`, `detokenize()`, `eval()`, `sample()` / logits access. Document chosen approach in spike script header comment.

#### 2. Document API choice

Add a short "Decode loop API" subsection comment at top of `kvl_beam_decoder.py` (Phase 1) stating which llama.cpp methods the decoder uses.

### Success Criteria

#### Automated Verification:
- [x] Spike script runs without error on Qwen3 GGUF when models are available
- [x] Spike produces non-empty decoded text for a fixed prompt

#### Manual Verification:
- [ ] Manual loop output is qualitatively similar to one-shot generation on same prompt/seed
- [ ] Stop sequences halt decoding correctly
- [ ] API approach documented (reset+full-prefix vs incremental KV)

**Implementation Note:** Do not proceed to Phase 1 until the spike confirms a workable eval pattern. If llama-cpp-python only supports full-context re-eval, accept the slower path for v1.

---

## Phase 1: Core Decoder + Word Tracker (Mock Logits)

### Overview

Implement beam state, word-boundary tracking, and KVL-ranked pruning with **mock logits** — no llama.cpp dependency in unit tests.

### Changes Required

#### 1. Word boundary tracker

**File:** `src/slm_experiments/models/word_tracker.py`

```python
@dataclass
class WordTracker:
    pending_word: str = ""

    def append_token_text(self, token_text: str) -> list[str]:
        """Return list of completed words (may be empty)."""
        ...

    @staticmethod
    def is_content_word(word: str, content_words_set: set[str]) -> bool:
        """Check if cleaned word is a content word."""
        ...
```

Logic:
- Accumulate decoded token text into `pending_word`.
- On boundary (trailing space, punctuation, or end-of-sequence flush): emit completed word, reset buffer.
- Strip to alphanumeric lowercase before lookup (match `compute_kvl_metrics` keying).

#### 2. Beam candidate dataclasses

**File:** `src/slm_experiments/models/kvl_beam_decoder.py`

```python
@dataclass
class KvlBeamCandidate:
    token_ids: list[int]
    text: str
    cumulative_logprob: float
    pending_word: str
    completed_content_words: list[str]
    kvl_scores: list[float]
    finished: bool = False

    def kvl_running_mean(self) -> float | None:
        if not self.kvl_scores:
            return None
        return sum(self.kvl_scores) / len(self.kvl_scores)

@dataclass
class KvlBeamDecodeResult:
    token_ids: list[int]
    text: str
    cumulative_logprob: float
    steps_total: int
    words_scored: int
    running_mean: float | None
    candidates_pruned: int
```

#### 3. KvlBeamDecoder

**File:** `src/slm_experiments/models/kvl_beam_decoder.py`

```python
class KvlBeamDecoder:
    def __init__(
        self,
        *,
        kvl_lookup: KvlLookup,
        l1: str,
        text_evaluator: TextEvaluator,
        beam_width: int = 4,
        branch_factor: int = 10,
    ): ...

    def decode(
        self,
        eval_fn: Callable[[list[int]], tuple[list[float], str]],
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        stop: list[str],
    ) -> KvlBeamDecodeResult:
        """
        eval_fn: given full token prefix, return (logits, decoded_suffix_text).
        Used for tests with mock logits and for llama.cpp in production.
        """
        ...
```

**Per-step algorithm:**

```
candidates = []
for beam in active_beams:
    logits, _ = eval_fn(beam.token_ids)
    top_k_ids = top_k_after_filter(logits, branch_factor, top_k, top_p, temperature=0)
    for token_id in top_k_ids:
        child = extend(beam, token_id, logprob=logits[token_id])
        if word_completed(child):
            if is_content_word(word):
                score = kvl_lookup.get_score(word, l1)
                if score is not None:
                    child.kvl_scores.append(score)
        candidates.append(child)

rank candidates by (kvl_running_mean or -inf, cumulative_logprob)
keep top beam_width unfinished; finished beams go to finished_pool
return best from finished_pool or best active at max_tokens
```

Early steps with zero scored words: rank by `cumulative_logprob` only.

#### 4. Unit tests

**File:** `tests/test_word_tracker.py`

- Multi-token word `"some" + "one"` → single completion at boundary.
- Punctuation boundary: `"friend."` → word `"friend"`.
- Partial word does not emit completion.

**File:** `tests/test_kvl_beam_decoder.py`

- Use fixture lookup from `tests/fixtures/kvl_lookup_es.json`.
- Mock `eval_fn` returning logits that prefer `"fundamentally"` vs `"friend"` paths.
- Assert decoder selects `"friend play"`-like path over hard-word path.
- Assert tie-break: equal KVL mean → higher logprob wins.
- Assert OOV words excluded from mean (beam with OOV inflection not penalized in v1).

### Success Criteria

#### Automated Verification:
- [x] `pytest tests/test_word_tracker.py tests/test_kvl_beam_decoder.py -q` passes
- [x] Decoder test demonstrates KVL-based preference over logprob-only path

#### Manual Verification:
- [ ] Review mock-logits test case matches toy example in `kvl-beam-flow.html` Step 2

---

## Phase 2: llama.cpp Integration

### Overview

Wire `KvlBeamDecoder` to real llama.cpp via `generate_kvl_beam()` on `LlamaCppBaseWrapper`.

### Changes Required

#### 1. Eval adapter for llama.cpp

**File:** `src/slm_experiments/models/kvl_beam_decoder.py` (or `llamacpp.py`)

```python
def make_llamacpp_eval_fn(
    llm,
    *,
    top_k: int,
    top_p: float,
) -> Callable[[list[int]], tuple[list[float], str]]:
    """Reset context, eval prefix, return logits for last position."""
    ...
```

Implementation notes:
- Tokenize prompt once; decoder manages growing prefix.
- On each `eval_fn` call: `llm.reset()`, `llm.eval(token_ids)`, read logits for next token.
- Apply `top_k`/`top_p` masking before selecting branch candidates.
- At `temperature=0.0`, take log-prob argmax within branch set (deterministic).
- Check stop sequences after each appended token via decoded text suffix match.

#### 2. Wrapper method

**File:** `src/slm_experiments/models/llamacpp.py`

```python
def generate_kvl_beam(
    self,
    prompt: str,
    config: ExperimentConfig,
    beam_width: int = 4,
    branch_factor: int = 10,
) -> dict[str, Any]:
    # Format prompt (same as generate_beam: prompting context + chat template)
    # Build KvlBeamDecoder with self.kvl_lookup or injected KvlLookup
    # Run decode with stop=self._get_stop_tokens()
    # Extract response via _prepare_beam_scoring_text()
    # Return dict with response, kvl_beam_* metadata, generation_successful
```

Add `KvlLookup` instance on wrapper (lazy-init) or pass from pipeline.

#### 3. Integration test (optional, marked slow)

**File:** `tests/test_kvl_beam_integration.py`

```python
@pytest.mark.skipif(not gguf_available(), reason="GGUF not present")
def test_generate_kvl_beam_smoke():
    ...
```

### Success Criteria

#### Automated Verification:
- [x] `pytest tests/test_kvl_beam_decoder.py -q` still passes
- [x] Integration test passes when Qwen3 GGUF is available (or skipped cleanly)

#### Manual Verification:
- [ ] `generate_kvl_beam()` on one prompt produces non-empty A1-ish response
- [ ] Response text excludes prompt leakage (reuse beam scoring path)
- [ ] Generation time is acceptable for width=4, branch=10, max_tokens=200 (record baseline)

**Implementation Note:** Pause for manual confirmation on one real model before Phase 3.

---

## Phase 3: Config, Pipeline, Result, RunStore

### Overview

Extend the experiment framework so KVL beam observations flow through the same evaluate → record path as other interventions.

### Changes Required

#### 1. Config fields

**File:** `src/slm_experiments/core/config.py`

```python
config_kvl_beam: bool = False
kvl_beam_width: int = 4
kvl_branch_factor: int = 10
```

#### 2. Result fields + factory

**File:** `src/slm_experiments/core/result.py`

Add optional fields:

| Field | Type |
|-------|------|
| `kvl_beam_width` | `Optional[int]` |
| `kvl_branch_factor` | `Optional[int]` |
| `kvl_beam_steps_total` | `Optional[int]` |
| `kvl_beam_words_scored` | `Optional[int]` |
| `kvl_beam_running_mean` | `Optional[float]` |
| `kvl_beam_logprob_tiebreak` | `Optional[float]` |
| `kvl_beam_candidates_pruned` | `Optional[int]` |

```python
@classmethod
def create_from_kvl_beam_response(cls, ..., **kvl_beam_kwargs) -> "ExperimentResult":
    result = cls.create_from_response(...)
    # set kvl_beam_* fields
    return result
```

#### 3. Pipeline method

**File:** `src/slm_experiments/core/pipeline.py`

```python
@runtime_checkable
class KvlBeamModelWrapper(Protocol):
    def generate_kvl_beam(
        self, prompt: str, config: ExperimentConfig,
        beam_width: int = 4, branch_factor: int = 10,
    ) -> dict[str, Any]: ...

def run_kvl_beam(self, prompt, config, model, ...) -> ExperimentResult:
    # Mirror run_beam(): generate_kvl_beam → clean → evaluate → post-hoc KVL
```

Post-hoc KVL on cleaned response uses existing `_compute_kvl_metrics()` — unchanged.

#### 4. Run store sweep bucket

**File:** `src/slm_experiments/core/run_store.py`

```python
SWEEP_SUMMARY_SECTIONS = {
    ...
    "kvl_beam": ("by_kvl_beam_width", "kvl_beam_width"),
}
```

Ensure `kvl_beam_width` column exists in `full.csv` via `ExperimentResult.to_dict()`.

#### 5. Tests

**File:** `tests/test_pipeline_kvl_beam.py`

- Mock model returning fixed `generate_kvl_beam` dict.
- Assert `run_kvl_beam()` populates KVL + kvl_beam metadata.
- Assert failed generation gets empty defaults.

**File:** `tests/test_run_store.py` (extend)

- Write bundle with `experiment="kvl_beam"`, multiple widths.
- Assert `summary.json` contains `by_kvl_beam_width` with `a1_pass_rate`.

### Success Criteria

#### Automated Verification:
- [x] `pytest tests/test_pipeline_kvl_beam.py tests/test_run_store.py -q` passes
- [x] New result fields serialize to `full.csv` without schema errors

#### Manual Verification:
- [ ] Inspect one mock-run `summary.json` — `by_kvl_beam_width` groups make sense

---

## Phase 4: Sweep Runner + CLI

### Overview

Add the Phase 2 experiment entry point mirroring `BeamSweepRunner`.

### Changes Required

#### 1. Sweep runner

**File:** `src/slm_experiments/phase2/kvl_beam.py`

Copy structure from `phase2/beam.py`:

```python
DEFAULT_KVL_BEAM_WIDTH_GRID = [4, 8]
_KVL_BEAM_WIDTH_PATTERN = re.compile(r"_kvl_beam_w(\d+)$")

def create_kvl_beam_configs(
    beam_widths: list[int],
    branch_factor: int = 10,
    kvl_l1: str = "es",
) -> list[ExperimentConfig]:
    # 4 models × len(widths)
    # config_kvl_beam=True, config_prompting=True, config_weighting=False
    # temperature=0.0, experiment_name=f"{model}_kvl_beam_w{width}"

class KvlBeamSweepRunner:
    def run(
        self,
        widths: str = "4,8",
        branch_factor: int = 10,
        kvl_l1: str = "es",
        prompts=..., models=..., seed=..., no_plot=...,
    ) -> tuple[str, Path]:
        # run_id = make_run_id(2, "kvl_beam")
        # pipeline.run_kvl_beam() for each (config, prompt)
        # run_store.write_bundle(..., experiment="kvl_beam")
```

#### 2. CLI registration

**File:** `src/slm_experiments/cli.py`

```python
p2_kvl_beam = p2_sub.add_parser("kvl_beam", ...)
p2_kvl_beam.add_argument("--widths", default="4,8", ...)
p2_kvl_beam.add_argument("--branch-factor", type=int, default=10, ...)
p2_kvl_beam.add_argument("--kvl-l1", default="es", choices=["es", "de", "cn"], ...)
_add_run_options(p2_kvl_beam)

# main():
if args.sweep == "kvl_beam":
    from slm_experiments.phase2.kvl_beam import KvlBeamSweepRunner
    ...
```

Update `_EPILOG` and phase2 help text.

#### 3. Plot support (if needed)

**File:** `src/slm_experiments/plot.py`

- Add `"kvl_beam"` experiment type grouping by `kvl_beam_width` (mirror beam width grouping).

#### 4. Cluster script (optional)

**File:** `scripts/clusteruy/run_phase2_kvl_beam.sh`

```bash
python -m slm_experiments phase2 kvl_beam --prompts all --widths 4,8 --no-plot
```

#### 5. Tests

**File:** `tests/test_phase2_kvl_beam.py`

- `parse_widths()` reuse or import from beam module.
- `create_kvl_beam_configs()` produces correct count and isolation flags.
- `kvl_beam_width_from_config()` parses experiment names.

### Success Criteria

#### Automated Verification:
- [x] `pytest tests/test_phase2_kvl_beam.py -q` passes
- [x] `python -m slm_experiments phase2 kvl_beam --help` shows new options
- [x] Full unit test suite passes: `pytest tests/ -q`

#### Manual Verification:
- [ ] Smoke run: `python -m slm_experiments phase2 kvl_beam --models Qwen3 --prompts 1 --widths 4 --no-plot`
- [ ] Run bundle contains expected observation count (1 model × 1 width × 1 prompt = 1)
- [ ] `full.csv` has `kvl_beam_running_mean`, post-hoc `kvl_mean_score`, `meets_a1_criteria`
- [ ] `summary.json` has `by_kvl_beam_width` when multiple widths used

**Implementation Note:** Pause for manual smoke-run confirmation before Phase 5.

---

## Phase 5: KVL Token Index + Diagnostics (v1.5)

### Overview

Add optional token index for collision reporting and future trie mode. Not required for MVP word-boundary scoring but validates token↔word mapping assumptions from the design doc.

### Changes Required

#### 1. Token index

**File:** `src/slm_experiments/models/kvl_token_index.py`

```python
@dataclass(frozen=True)
class KvlTokenIndex:
    lemma_to_token_ids: dict[str, tuple[int, ...]]
    id_to_lemmas: dict[int, tuple[str, ...]]  # collision report

    @classmethod
    def build(cls, llm, kvl_lookup: KvlLookup, l1: str) -> "KvlTokenIndex":
        # For each lemma in lookup: tokenize(" " + word) and bare word
        ...
```

Cache on wrapper at model load (same pattern as planned `A1TokenIndex`).

#### 2. Audit test

**File:** `tests/test_kvl_token_index.py`

- Build index from stub tokenizer.
- Report collision count.
- High-frequency KVL words tokenize without error.

#### 3. Optional CLI audit

**File:** `scripts/audit_kvl_token_index.py`

Print collision stats per model for manual inspection.

### Success Criteria

#### Automated Verification:
- [x] `pytest tests/test_kvl_token_index.py -q` passes

#### Manual Verification:
- [ ] Collision report reviewed for Qwen3 — acceptable collision rate for MVP

---

## Phase 6: Baseline Arm + Analysis (Optional Follow-up)

### Overview

Add best-of-N + KVL rerank baseline for empirical comparison (design doc baseline arm #2).

### Changes Required

#### 1. KVL rerank selection method

**File:** `src/slm_experiments/models/beam.py`

Add `select_best_beams_by_kvl()` or extend `select_best_beams()` with `selection_method="kvl_mean"`.

#### 2. Pipeline flag or separate sweep

Either extend `phase2 beam` with `--selection kvl_mean` or add `phase2 kvl_rerank`.

### Success Criteria

#### Automated Verification:
- [ ] Unit test: KVL rerank picks higher `kvl_mean_score` candidate

#### Manual Verification:
- [ ] Run control + KVL rerank + KVL beam on same prompts; compare `meets_a1_criteria` and `a1_pass_rate`

---

## Testing Strategy

### Unit Tests

| Module | Focus |
|--------|-------|
| `word_tracker` | BPE boundaries, punctuation, partial words |
| `kvl_beam_decoder` | Ranking, tie-break, OOV exclusion, early-step logprob fallback |
| `kvl_token_index` | Build, collisions |
| `pipeline` | Mock wrapper integration, failure path |
| `run_store` | `by_kvl_beam_width` summary |
| `phase2/kvl_beam` | Config grid, name parsing |

### Integration Tests

- One GGUF smoke test (skipped without models).
- Compare post-hoc `kvl_mean_score` vs `kvl_beam_running_mean` on same output — should be close but may differ (incremental POS vs full-text extraction).

### Manual Testing Steps

1. Run width=4 on 3 prompts, one model — inspect `full.csv` responses for fluency.
2. Run width=4 vs width=8 — confirm wider beam changes outputs and increases runtime.
3. Compare one prompt across `phase2 beam` (A1 rerank) vs `phase2 kvl_beam` — outputs should differ.
4. Verify stop tokens prevent runaway generation.
5. For publishable claims: `--prompts all --widths 4,8` on all 4 models.

---

## Performance Considerations

| Setting | Approx. forward passes per observation |
|---------|----------------------------------------|
| Greedy (200 tokens) | ~200 |
| Best-of-N (width=4) | ~4 |
| KVL beam (W=4, K=10, 200 tokens) | ~200 × 4 × 10 = **8,000** (worst case) |

Mitigations for v1:
- Smaller width grid: `{4, 8}` only.
- Fixed `branch_factor=10` (don't sweep).
- Sequential eval (no batching) — simple, slow.
- Use `--prompts 3` for dev; `--prompts all` for final runs only.
- Document expected runtime in cluster script comments.

Monitor `response_time_seconds` in `full.csv` per width.

---

## Migration Notes

- **No data migration** — new columns append to `full.csv`.
- **No breaking changes** to existing sweeps.
- **Existing `phase2 beam` unchanged** — KVL beam is a separate command.
- Reuse existing KVL lookup files; no rebuild required.

---

## Experiment Execution Checklist

After all phases complete, run the full experiment:

```bash
# 1. Verify tests
pytest tests/ -q

# 2. Smoke (local)
python -m slm_experiments phase2 kvl_beam --models Qwen3 --prompts 3 --widths 4 --no-plot

# 3. Full sweep (cluster or local — expect hours)
python -m slm_experiments phase2 kvl_beam --prompts all --widths 4,8 --seed 42 --no-plot

# 4. Compare to A1 beam baseline (re-run if needed on fixed code)
python -m slm_experiments phase2 beam --prompts all --widths 4,8,10 --seed 42 --no-plot

# 5. Analysis
python -m slm_experiments runs show <kvl_beam_run_id>
# Compare summary.json: by_kvl_beam_width[*].a1_pass_rate vs by_beam_width[*].a1_pass_rate
# Compare kvl_mean_score, kvl_min_score, response_time_seconds
```

**Primary success metrics:**
- `meets_a1_criteria` / `a1_pass_rate` (same as other sweeps)
- Secondary: `kvl_mean_score`, `kvl_pct_hard_words`, `response_time_seconds`

**Hypothesis:** KVL beam improves `a1_pass_rate` vs control and vs A1 best-of-N beam, at higher compute cost. If not, best-of-N + KVL rerank (Phase 6) is the cheaper alternative.

---

## References

- Visual design: [`docs/kvl-beam-flow.html`](../../docs/kvl-beam-flow.html)
- Detailed design: [`docs/kvl_beamsearch.md`](../../docs/kvl_beamsearch.md)
- Shared decode-loop patterns: [`docs/guided-decoding.md`](../../docs/guided-decoding.md)
- KVL metrics: [`docs/metrics.md`](../../docs/metrics.md), [`src/slm_experiments/evaluation/kvl.py`](../../src/slm_experiments/evaluation/kvl.py)
- Runner pattern: [`src/slm_experiments/phase2/beam.py`](../../src/slm_experiments/phase2/beam.py)
- Current pseudo-beam: [`src/slm_experiments/models/beam.py`](../../src/slm_experiments/models/beam.py)
- Methodological constraints: [`improvements.md`](../../improvements.md)
- KVL data: [`data/kvl/README.md`](../../data/kvl/README.md)
