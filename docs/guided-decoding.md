# Top-K A1-Guided Decoding (Design)

**Status:** Planned — not yet implemented in code.

Design for a fourth inference-time intervention: **top-k A1-constrained greedy decoding**. At each decoding step, inspect the model's top-K token candidates; if any map to the A1 vocabulary, pick the highest-probability one; otherwise take argmax.

This is **not** best-of-N reranking (current `BeamSearchGenerator`) and **not** logit bias. It is step-wise lexical steering with a safe fallback.

**Suggested CLI name:** `phase2 guided`  
**Accurate label:** *top-k A1-constrained greedy decoding*

See also: [interventions.md](interventions.md) (existing interventions), [metrics.md](metrics.md) (A1 pass criteria).

---

## Motivation

| Approach | When it acts | Effect |
|----------|--------------|--------|
| **Logit bias** (Phase 1 weighting) | Every step, soft | Nudges A1 tokens; hard words can still win if the model is very confident |
| **Best-of-N + A1 rerank** (`phase2 beam`) | After full answer | Picks the sample that looks most A1-like post hoc |
| **Guided decoding** (this design) | Every step, hard-ish | Only switches away from argmax when an A1 token is already in the top-K pool |

Guided decoding sits between logit bias and best-of-N: it shapes the whole answer as it is generated, but only when simple vocabulary is already a plausible next step.

---

## End-to-end flow

```
CLI: phase2 guided
    → GuidedSweepRunner (phase2/guided.py)
    → create_guided_configs()
    → LlamaCppBaseWrapper.generate_guided()
        → A1TokenIndex (per model, cached)
        → ConstrainedDecoder.decode()   # token-by-token loop
    → ExperimentPipeline.run_guided()
    → RunStore → summary.json (by_guided_top_k)
```

### One observation

1. `GuidedSweepRunner` loads each model once per model batch (same pattern as `BeamSweepRunner`).
2. For each `(config, prompt)`: `pipeline.run_guided()` → `generate_guided()` → constrained decode → format → evaluate → `meets_a1_criteria`.
3. Results land in the usual run bundle (`full.csv`, `specification.csv`, `summary.json`).

---

## Proposed file layout

```
src/slm_experiments/
├── models/
│   ├── a1_token_index.py       # token↔word maps + optional trie
│   └── constrained_decoder.py  # custom decode loop
├── core/
│   ├── config.py               # + config_guided, guided_top_k, guided_mode
│   ├── pipeline.py             # + run_guided() + protocol
│   └── result.py               # + guided_* metadata fields
├── phase2/
│   └── guided.py               # experiment runner (main entry point)
├── cli.py                      # register: phase2 guided
└── evaluation/                 # unchanged (FK, Fog, Spache, meets_a1_criteria)

tests/
├── test_a1_token_index.py
├── test_constrained_decoder.py
└── test_phase2_guided.py
```

The experiment runner (`phase2/guided.py`) stays thin — same role as `phase2/beam.py`. Decoding logic lives under `models/`.

---

## Phase 2 sweep design

Default grid sweeps **pool size** (how many top logits are considered before the A1 filter):

| Setting | Default | Notes |
|---------|---------|-------|
| `config_weighting` | `False` | Isolates guided decoding from logit bias |
| `config_prompting` | `True` (zero-shot) | Same baseline as beam sweep |
| `config_guided` | `True` | New flag |
| `guided_top_k` | `5, 10, 20` | Swept hyperparameter |
| `guided_mode` | `"flat"` | `"flat"` or `"trie"` (see below) |
| `temperature` | `0.0` (recommended) | Greedy + guided override; optional `0.7` comparison arm |
| Weighting / beam | OFF | Do not combine in first sweep |

**Config naming:** `{model}_guided_k{10}` (parseable like beam's `_beam_w(\d+)`).

**CLI examples (planned):**

```bash
python -m slm_experiments phase2 guided
python -m slm_experiments phase2 guided --top-k-pools 5,10,20 --prompts all
python -m slm_experiments phase2 guided --mode trie --models Qwen3
```

Use `--prompts all` (25 prompts) for publishable claims; default n=3 remains for smoke runs (see `ExperimentDesign.md`).

---

## Module responsibilities

### `phase2/guided.py` — experiment runner

Mirrors `phase2/beam.py`:

- `DEFAULT_TOP_K_POOL_GRID = [5, 10, 20]`
- `parse_top_k_pools(arg) -> list[int]`
- `create_guided_configs(pools) -> list[ExperimentConfig]`
- `guided_top_k_from_config(config) -> int`
- `class GuidedSweepRunner` with `run(top_k_pools, prompts, models, seed, no_plot, ...)`

Does **not** contain tokenization or decode logic — only config grid, model loop, pipeline calls, and run store I/O.

### `models/a1_token_index.py` — token ↔ word mapping

Build once when a model loads; cache on the wrapper.

```python
@dataclass(frozen=True)
class A1TokenIndex:
    mid_sentence_ids: frozenset[int]       # from tokenize(" " + word)
    sentence_start_ids: frozenset[int]     # from tokenize(word) at boundaries
    id_to_words: dict[int, tuple[str, ...]]  # debug / collision report
    trie: A1TokenTrie | None               # optional, for mode="trie"

    @classmethod
    def build(cls, llm, vocab: list[str], *, use_trie: bool) -> A1TokenIndex: ...

    def candidate_set_for_context(self, generated_text: str) -> frozenset[int]:
        """Return mid-sentence or sentence-start ID set based on trailing text."""
```

**Build steps:**

1. For each word in `filtered_starters_vocab.txt`, tokenize `" " + word` and bare `word`.
2. Union all subword token IDs into the appropriate sets.
3. Record collisions where one token ID maps to multiple A1 words.
4. If `use_trie`: store full token ID sequences per word per context.

**Refactor:** `_create_logit_bias()` in `llamacpp.py` should call the same builder so weighting and guided decoding share one tokenization path.

### `models/constrained_decoder.py` — step-wise decode loop

```python
class ConstrainedDecoder:
    def decode(
        self,
        llm,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        stop: list[str],
        guided_pool_size: int,
        index: A1TokenIndex,
        mode: Literal["flat", "trie"],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> ConstrainedDecodeResult:
        """
        Returns token_ids, text, steps_total, steps_a1_chosen,
        steps_fallback_argmax, steps_no_a1_in_pool.
        """
```

#### Per-step algorithm (`mode="flat"`)

```
logits = next_token_logits(context)
candidates = apply model top_k / top_p, then take top guided_pool_size by logit
active_set = index.candidate_set_for_context(decoded_so_far)
a1_hits = [t for t in candidates if t in active_set]   # preserve logit order

if a1_hits:
    chosen = a1_hits[0]
    steps_a1_chosen += 1
else:
    chosen = argmax(logits)
    steps_fallback_argmax += 1

append chosen; check stop sequences
```

#### Per-step algorithm (`mode="trie"`)

If inside a partial A1 word, restrict candidates to trie continuations first. If none match, clear partial state and apply the flat rule.

### `LlamaCppBaseWrapper.generate_guided()`

Thin wrapper (like `generate_beam`):

1. Format prompt with chat template.
2. Load cached `A1TokenIndex`.
3. Run `ConstrainedDecoder.decode()` with `stop=self._get_stop_tokens()`.
4. Extract and clean response.
5. Return generation dict with guided metadata.

### `core/config.py` — new fields

```python
config_guided: bool = False
guided_top_k: int = 10          # pool size at each step (experiment hyperparameter)
guided_mode: str = "flat"       # "flat" | "trie"
```

Note: `guided_top_k` (pool size) is distinct from `ExperimentConfig.top_k` (model sampling cap). Apply model `top_k`/`top_p` first, then take the top `guided_top_k` from survivors, then apply the A1 filter.

### `core/pipeline.py` — `run_guided()`

Mirror `run_beam()`:

```python
class GuidedModelWrapper(Protocol):
    def generate_guided(self, prompt: str, config: ExperimentConfig) -> dict: ...

def run_guided(...) -> ExperimentResult:
    # generate → clean → evaluate → meets_a1_criteria
```

### `core/result.py` + `run_store.py`

**New result fields:**

| Field | Purpose |
|-------|---------|
| `guided_top_k` | Swept hyperparameter |
| `guided_mode` | `flat` or `trie` |
| `guided_steps_a1_chosen` | Steps where A1 token was picked |
| `guided_steps_total` | Total decode steps |
| `guided_intervention_rate` | `steps_a1_chosen / steps_total` |

**Summary bucket** — extend `SWEEP_SUMMARY_SECTIONS`:

```python
"guided": ("by_guided_top_k", "guided_top_k"),
```

---

## Problems to solve

### 1. Tokens ≠ words

| | |
|-|-|
| **Symptom** | BPE subwords; `" cat"` vs `"cat"` tokenize differently; shared prefixes |
| **Solution** | `A1TokenIndex.build()` with mid-sentence and sentence-start contexts; optional trie |
| **Verify** | `test_a1_token_index.py`: every vocab word tokenizes; collision report per model |
| **Owner** | `a1_token_index.py` |

### 2. Multi-token words

| | |
|-|-|
| **Symptom** | `"happy"` may be two tokens; flat set may start but not complete the word |
| **Solution (MVP)** | `mode="flat"` — register all subword IDs; no forced completion |
| **Solution (full)** | `mode="trie"` — after first subword, only allow trie continuations until word completes or fallback |
| **Verify** | Decoder tests with mock logits on multi-token paths |
| **Owner** | `constrained_decoder.py` |

### 3. Shared subword false positives

| | |
|-|-|
| **Symptom** | Token for `"run"` matches `"running"`, `"runner"`, etc. |
| **Solution** | Accept for MVP; trie mode reduces drift; log `guided_intervention_rate` and compare `meets_a1_criteria` vs logit bias |
| **Verify** | Human eval on high-intervention-rate outputs |
| **Owner** | analysis |

### 4. Two different "K" parameters

| | |
|-|-|
| **Symptom** | `ExperimentConfig.top_k=50` (sampling) vs `guided_top_k=10` (pool) |
| **Solution** | Document clearly; apply model `top_k`/`top_p` first, then pool trim, then A1 filter |
| **Verify** | Unit test: pool=3 only considers three candidates |
| **Owner** | `constrained_decoder.py`, config docs |

### 5. llama.cpp step API

| | |
|-|-|
| **Symptom** | Current code uses one-shot `llm(formatted_prompt, ...)` with no per-step hook |
| **Solution** | New decode loop in `ConstrainedDecoder`; do not reuse `generate()` |
| **Options** | (A) manual `eval`+sample loop — recommended; (B) logits processor if available; (C) grammar mask — awkward for top-K-then-filter |
| **Verify** | Integration test on stub; spike on one real GGUF before full sweep |
| **Owner** | `constrained_decoder.py` |

### 6. Temperature / stochasticity

| | |
|-|-|
| **Symptom** | At `temperature=0.7`, behavior is harder to reproduce |
| **Solution** | Default sweep uses `temperature=0.0`; optional `--temperature 0.7` comparison |
| **Verify** | Same prompt + seed → identical output at temp=0 |
| **Owner** | `guided.py` config factory |

### 7. Stop tokens and chat templates

| | |
|-|-|
| **Symptom** | Decoder must halt on model-specific stop sequences |
| **Solution** | Reuse `_get_stop_tokens()`; check stops after each appended token |
| **Verify** | Same patterns as beam stop-token tests (`7a6b105`) |
| **Owner** | `constrained_decoder.py` |

### 8. Comparison to existing interventions

| | |
|-|-|
| **Symptom** | Unclear whether guided beats weighting or best-of-N |
| **Solution** | Keep arms isolated in first sweep; compare `meets_a1_criteria` and `a1_pass_rate` across run bundles with same `--prompts all`, seed, models |
| **Owner** | experiment design |

### 9. Per-model token index

| | |
|-|-|
| **Symptom** | Qwen, Phi3, TinyLlama token IDs differ |
| **Solution** | Build index at model load; cache on wrapper |
| **Verify** | CI on stub tokenizer; optional slow test per real GGUF |
| **Owner** | `LlamaCppBaseWrapper` |

### 10. Summary / CSV schema

| | |
|-|-|
| **Symptom** | New hyperparameter needs sweep bucket and CSV columns |
| **Solution** | Add fields to `full.csv`; register `"guided"` in `SWEEP_SUMMARY_SECTIONS` |
| **Verify** | `test_run_store.py` (mirror beam/weight tests) |
| **Owner** | `run_store.py`, `result.py` |

---

## Implementation order

| PR | Scope | Solves |
|----|-------|--------|
| **1** | `a1_token_index.py` + audit tests (+ optional audit CLI) | #1, #9 |
| **2** | `constrained_decoder.py` on mock logits + stub llm | #2 flat, #4 |
| **3** | `generate_guided()` on one real model, one prompt | #5, #7 |
| **4** | `phase2/guided.py` + pipeline + result + run_store + cli | full experiment |
| **5** | `mode="trie"` + `ExperimentDesign.md` + comparison runs | #2 full, #8 |

---

## Example config

```python
ExperimentConfig(
    model_name="Qwen3",
    config_weighting=False,
    config_prompting=True,
    config_guided=True,
    guided_top_k=10,
    guided_mode="flat",
    temperature=0.0,
    top_k=50,
    top_p=1.0,
    experiment_name="Qwen3_guided_k10",
    description="Guided decode: top-10 pool, A1 preference, zero-shot prompting",
)
```

---

## Success criteria for the experiment

Same framework metrics as other sweeps:

- **`generation_successful`** — non-empty valid output
- **`meets_a1_criteria`** — FK≤5 ∧ Fog≤6 ∧ Spache≤4 on valid generations
- **`summary.json`** → `by_guided_top_k[*].a1_pass_rate`

**New diagnostics:**

- **`guided_intervention_rate`** — fraction of decode steps where an A1 token was chosen over fallback
- If intervention rate ≈ 0, the pool is too narrow or the token index mismatches the tokenizer → run the PR 1 audit

---

## Related files (current codebase)

| Area | Files |
|------|-------|
| Vocabulary | `data/vocabularies/filtered_starters_vocab.txt` |
| Existing logit bias (shared tokenization) | `src/slm_experiments/models/llamacpp.py` (`_create_logit_bias`) |
| Best-of-N baseline | `src/slm_experiments/models/beam.py`, `phase2/beam.py` |
| A1 pass criteria | `src/slm_experiments/evaluation/a1_criteria.py` |
| Sweep summaries | `src/slm_experiments/core/run_store.py` |
| Runner pattern to copy | `src/slm_experiments/phase2/beam.py` |
