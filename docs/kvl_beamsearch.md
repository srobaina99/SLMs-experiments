# KVL Beam Search

**Status:** Implemented — `phase2 kvl_beam` via `KvlBeamSweepRunner` (`phase2/kvl_beam.py`), `generate_kvl_beam()` / `KvlBeamDecoder` under `models/`.

Fifth inference-time intervention: **token-level beam search ranked by KVL/GLMM learner vocabulary scores** instead of model likelihood. At each decoding step, expand multiple partial sequences using the model's top logits, accumulate KVL scores as words complete, and prune to the best-scoring beams.

This is **not** the deprecated `BeamSearchGenerator` (best-of-N + A1-ratio reranking). It is **not** logit bias or guided decoding. It is genuine multi-path search with a custom, learner-grounded objective.

**CLI:** `phase2 kvl_beam`  
**Accurate label:** *KVL-scored beam search* or *learner-vocabulary beam decoding*

**Important — first-finish stopping (intentional):** The decode loop returns the **first** candidate that hits a stop condition with non-empty text (in beam expansion order), not the best KVL-ranked finished hypothesis at the end.

**Why.** Mean KVL over content words does not reward finishing a sentence. If selection waited until `max_tokens` and then took `max(..., key=KVL)`, the “best” hypothesis almost always stretched to the length budget — often incoherent padding — because continuing can keep the running mean competitive while never stopping. First-finish lets a naturally completed answer win. KVL still ranks which **partial** beams survive each prune; first-finish only decides **when** to return.

Empty EOS-only finishes are skipped. If nothing finishes before `max_tokens`, the best surviving active beam is returned. Do **not** interpret width 4 vs 8 as “better final KVL selection among finished hyps”; treat width as more exploration before the first natural stop. Prefer KVL columns in `full.csv` / `summary.json` as primary endpoints for this arm.

See also: [interventions.md](interventions.md), [metrics.md](metrics.md) (KVL evaluation), [guided-decoding.md](guided-decoding.md), [data/kvl/README.md](../data/kvl/README.md).

---

## Motivation

| Approach | When KVL acts | Search strategy |
|----------|---------------|-----------------|
| **KVL metrics only** | After generation | No steering — records `kvl_mean_score` on final text |
| **Best-of-N + A1 rerank** (`phase2 beam`, deprecated) | After full answer | Void at temperature 0 |
| **Logit bias** | Every step, soft | Single path; nudges internal 487-word A1 list |
| **Guided decoding** | Every step, greedy-ish | Single path; pick easy token if in top-K pool |
| **KVL beam search** (this intervention) | During generation | Multiple paths; prune by running KVL aggregate; **first-finish return** |

**Research question:** If we steer decoding toward vocabulary that Spanish L1 learners are likely to know (per `kvl_lookup_es.json`), does that improve the readability proxy (`meets_a1_criteria`) and/or post-hoc KVL metrics more than post-hoc reranking or step-wise A1 filtering?

KVL beam sits between best-of-N and guided decoding: it explores several futures at once, but ranks them by **external learner knowledge** rather than model probability or the internal A1 starter list.

---

## End-to-end flow

```
CLI: phase2 kvl_beam
    → KvlBeamSweepRunner (phase2/kvl_beam.py)
    → create_kvl_beam_configs()
    → LlamaCppBaseWrapper.generate_kvl_beam()
        → KvlLookup (data/kvl/kvl_lookup_{l1}.json)
        → KvlTokenIndex (per model, cached)
        → KvlBeamDecoder.decode()   # token-by-token beam loop
    → ExperimentPipeline.run_kvl_beam()
    → RunStore → summary.json (by_kvl_beam_width + by_model)
```

### One observation

1. `KvlBeamSweepRunner` loads each model once per model batch.
2. For each `(config, prompt)`: format prompt → run KVL beam decode → extract/clean response → evaluate readability + KVL metrics → record `meets_a1_criteria` (proxy).
3. Beam metadata (width, branch factor, running KVL stats, model logprob tie-break) lands in `full.csv`.

### CLI examples

```bash
python -m slm_experiments phase2 kvl_beam
python -m slm_experiments phase2 kvl_beam --widths 1,4,6,8 --prompts all
python -m slm_experiments phase2 kvl_beam --kvl-l1 es --branch-factor 10 --models Qwen3
```

Defaults: `temperature=0.0`, `top_k=50` before branch selection, prompting ON, weighting OFF, `kvl_l1=es`. Default width grid `1,4,6,8` (`1` = greedy in-run baseline). Claims for other L1s require separate sweeps (`de` / `cn`).

---

## Why this is harder than standard beam search

Standard beam search ranks partial sequences by **cumulative log-probability** — a quantity the model provides at every token step. KVL beam must replace that objective with scores from `kvl_lookup_es.json`, which introduces structural mismatches.

### 1. Tokens ≠ words

KVL maps **English lemmas** → GLMM scores. Decoders emit **BPE subword tokens**. A single KVL word may span several tokens; a single token may prefix many words.

| Symptom | Example |
|---------|---------|
| Mid-word tokens | `" fund"` + `"ament"` + `"ally"` before `"fundamentally"` is scorable |
| Context-dependent tokenization | `" cat"` (mid-sentence) vs `"cat"` (sentence start) |
| Shared subword false positives | Token for `"run"` also starts `"running"`, `"runner"` |

**You cannot assign a KVL score to a token.** Scoring must happen at **word boundaries** (or via a trie that tracks in-progress lemmas).

### 2. Content words vs all words

`compute_kvl_metrics()` scores **content words** (NLTK POS: nouns, verbs, adjectives, adverbs). Function words (`the`, `is`, `a`) are excluded from the mean.

During incremental decoding:

- POS tagging on partial text is unreliable (`"The cat is"` — is `"is"` final?).
- Beam ranking must either approximate content-word detection or defer POS until word completion.

### 3. Lemma-only lookup and OOV

`kvl_lookup_es.json` covers ~6.8k lemmas. Inflected forms (`running`, `friends`) are often OOV even when the base lemma exists.

| Word in output | In lookup? | Effect on beam score |
|----------------|------------|----------------------|
| `friend` | Yes | Contributes GLMM (e.g. +2.1) |
| `friends` | Often no | OOV — policy must define penalty or skip |
| `xyznotinlookup` | No | OOV |

A beam that uses easy lemmas the model knows may lose to one that uses rare inflections not in the table — unless you add lemmatization or a fallback policy.

### 4. Incremental score vs final evaluation metric

Beam pruning uses a **running aggregate** (e.g. mean GLMM so far). Final evaluation still computes `kvl_mean_score`, FK, Fog, Spache on the complete cleaned response.

These can diverge:

- Beam optimizes partial means; one hard word at the end tanks `kvl_min_score` but not the running mean.
- Beam ignores function words; readability formulas weight sentence length and structure.

### 5. No native llama.cpp KVL beam API

Current code uses one-shot `llm(prompt, max_tokens=...)`. KVL beam requires a **manual token loop** (`eval` → expand → prune), similar to guided decoding. Each step × each active beam × branch factor is a forward pass — much slower than best-of-N.

### 6. Stochastic vs deterministic search

Phase 2 best-of-N beam sweep (`phase2 beam`) used stochastic sampling at `temperature=0.7` and is **deprecated**. All active experiments use **`temperature=0.0`**. KVL beam and guided decoding apply **`top_k=50`** as the vocabulary cap before branch/pool selection.

---

## Decisions to make

Each row is a design fork. Recommended defaults are marked **(Recommended)** for a first implementation.

### A. Beam objective — what to maximize?

| Option | Formula | Pros | Cons |
|--------|---------|------|------|
| **Mean GLMM of looked-up content words** | `sum(scores) / count` | Aligns with `kvl_mean_score`; intuitive | Late hard words barely affect early pruning |
| **Sum GLMM (no length norm)** | `sum(scores)` | Favors longer answers with many easy words | Length bias; fights `max_new_tokens` |
| **Min GLMM (maximin)** | `min(scores so far)` | Penalizes any hard word immediately | Very conservative; beams collapse to safest words |
| **Mean + hard-word penalty** | `mean - λ × pct_hard` | Matches `kvl_pct_hard_words` concern | Extra hyperparameter λ |
| **Hybrid with logprob** | `α × kvl_mean + (1-α) × logprob` | Keeps outputs fluent | Blurs intervention; α is arbitrary |

**Recommendation:** **Mean GLMM of looked-up content words** at word boundaries, with **model logprob as tie-breaker only** (weight ≈ 0.01–0.05). Primary success metric remains `meets_a1_criteria`; KVL beam objective is a steering proxy.

### B. When to update the KVL score?

| Option | Trigger | Pros | Cons |
|--------|---------|------|------|
| **On whitespace / punctuation** | Word completed in decoded text | Simple; no trie required for MVP | Multi-token words scored only at end |
| **Trie completion** | KVL lemma fully spelled by token sequence | Precise word identity | More complex; must handle failed completions |
| **Every token (token-level proxy)** | Assign score if token ID maps to a KVL word start | Score every step | Noisy; shared-subword errors |

**Recommendation:** **On word boundary (MVP)** — detect completion via trailing space or punctuation in decoded UTF-8 text, strip to alphanumeric lemma, lookup in `KvlLookup`. Add **trie mode** in v2 if false positives are high (reuse pattern from guided decoding).

### C. Which words count toward the beam score?

| Option | Behavior | Aligns with |
|--------|----------|-------------|
| **Content words only (POS on complete word)** | Run lightweight POS on each completed word | `compute_kvl_metrics()` |
| **All looked-up KVL words** | Any word in lookup counts | Simpler; faster |
| **KVL lookup membership only** | ~6.8k words; ignore POS | No POS needed; mismatches evaluation |

**Recommendation:** **Content words only** — reuse `TextEvaluator.extract_content_words` on the decoded prefix (or a single-word POS heuristic) so beam objective matches post-hoc KVL metrics. Fallback: if POS unavailable on one word, count it if it appears in the KVL lookup (reduces OOV noise).

### D. OOV policy

| Option | Beam effect |
|--------|-------------|
| **Ignore (skip in mean)** | Same as current `compute_kvl_metrics` |
| **Penalize with fixed score** (e.g. 0.0 or −1.0) | Pushes beams away from rare/inflected forms |
| **Penalize proportionally** | `oov_count` reduces ranking indirectly |
| **Lemmatize before lookup** | `running` → `run`; needs NLTK/spaCy |

**Recommendation:** **Ignore in mean (MVP)** for parity with evaluation, plus record `kvl_oov_count` in beam metadata. Add optional **`--lemmatize`** arm in Phase 2 if inflection OOV dominates.

### E. Expansion — how to grow beams each step?

| Parameter | Typical range | Role |
|-----------|---------------|------|
| `beam_width` (B) | 4, 8 | Beams kept after pruning |
| `branch_factor` (K) | 5–20 | Top logits expanded per beam per step |
| `temperature` | 0.0 | Greedy expansion from logits |

**Algorithm per step:**

```
candidates = []
for each active beam:
    logits = llm.eval(beam.context)
    top_K = top_k_tokens(logits, branch_factor)   # after model top_k filter
    for token in top_K:
        candidates.append(beam.extend(token))

for each candidate:
    if word_boundary: update kvl_sum, kvl_count

keep top beam_width candidates by kvl_mean (+ logprob tie-break)
```

**Recommendation:** Default **`beam_width=4`, `branch_factor=10`, `temperature=0.0`**. Sweep width like current beam sweep; fix branch factor initially.

### F. Length normalization

| Option | Effect |
|--------|--------|
| **None** | Longer partial sequences with more easy words win |
| **Divide by word count** | Mean GLMM — already normalizes |
| **Length penalty** (transformer-style) | `score / ((5 + len) / 6)^α` | Reduces verbosity bias |

**Recommendation:** **Mean GLMM** (implicit length norm on content words). Apply **`max_new_tokens=200`** same as other interventions. Monitor word count in analysis.

### G. Relationship to model probability

| Option | Description |
|--------|-------------|
| **KVL only** | Pure learner-vocabulary search; may produce fluent nonsense |
| **Logprob tie-breaker** | Same KVL mean → prefer higher model logprob |
| **Constrained expansion** | Only expand tokens in model top-K (K=50) before branch trim |

**Recommendation:** **Constrained expansion within model top-K/top-P**, then **logprob tie-breaker**. Never use logprob as primary rank — that recreates standard beam with KVL as noise.

### H. Stopping / final selection — first-finish vs best finished KVL

| Option | Behavior | Pros | Cons |
|--------|----------|------|------|
| **First-finish (current)** | Return first non-empty stop in expansion order | Completed sentences can win; avoids max-length pad | Width ≠ “better final KVL pick” |
| **Collect finished → max KVL** | Keep going; return best finished (or max-length survivor) | Classic beam “select best hyp” story | Mean KVL does not reward EOS → stretches to `max_new_tokens`, often nonsense |
| **First-finish + length/stop bonus in rank** | Soft preference for shorter / stopped hyps | Could combine both | Extra hyperparameters; not needed if first-finish is policy |

**Recommendation (adopted):** **First-finish.** Mean KVL is a good prune signal for *which* partials to keep, but a poor signal for *when* to stop. Returning on the first natural completion is the length/stop prior that the objective itself lacks. Document claim limits: width sweeps measure exploration before first stop.

### I. L1 selection

| Option | Source |
|--------|--------|
| Fixed `es` | `ExperimentConfig.kvl_l1` (default) |
| Per-run L1 | `--kvl-l1 es|de|cn` |
| Match learner cohort | Experiment design choice |

**Recommendation:** Use **`ExperimentConfig.kvl_l1`** (default `es`); load `data/kvl/kvl_lookup_{l1}.json` via existing `KvlLookup`.

### J. Interaction with other interventions

| Combination | First sweep? |
|-------------|--------------|
| KVL beam + contextual prompting | **Yes** — same as beam sweep (prompting ON) |
| KVL beam + logit bias | **No** — confounds two vocabulary signals |
| KVL beam + best-of-N | **No** — pick one search strategy |
| KVL beam + guided decoding | **No** |

**Recommendation:** Phase 2 KVL beam sweep: **`config_prompting=True`, `config_weighting=False`, zero-shot**, mirroring `phase2/beam.py` isolation.

---

## Proposed solutions (by difficulty)

### Problem 1: Token ↔ word mapping

| | |
|-|-|
| **Symptom** | BPE subwords; context-dependent tokenization |
| **Solution (MVP)** | `KvlTokenIndex.build(llm, kvl_lookup)` — for each lemma in lookup, tokenize `" " + word` and bare `word`; store token sequences (not just ID sets) |
| **Solution (full)** | Trie over token IDs per lemma; track `partial_word_state` on each beam |
| **Verify** | `test_kvl_token_index.py`: every high-frequency KVL word tokenizes; report collisions |
| **Owner** | `models/kvl_token_index.py` |

Reuse the builder pattern planned for `A1TokenIndex` in [guided-decoding.md](guided-decoding.md). Consider a shared `TokenIndexBase` if both land.

### Problem 2: Incremental KVL scoring

| | |
|-|-|
| **Symptom** | GLMM is word-level; most decode steps complete zero words |
| **Solution** | Maintain per-beam `(kvl_sum, kvl_content_count, decoded_text)`; on boundary, extract word, check content-word status, `lookup.get_score(word, l1)`, update sum/count |
| **Ranking key** | `kvl_sum / max(kvl_content_count, 1)`; if count=0, rank by logprob only |
| **Verify** | Unit test: beam prefers `"friend play"` path over `"fundamentally establishment"` using fixture lookup |
| **Owner** | `models/kvl_beam_decoder.py` |

### Problem 3: Content-word detection mid-decode

| | |
|-|-|
| **Symptom** | POS on partial sentences is unstable |
| **Solution (MVP)** | POS-tag each **completed word** in isolation via NLTK; cheap and good enough for nouns/verbs |
| **Solution (alt)** | Count any completed word that exists in KVL lookup (over-counts function words that happen to be in table) |
| **Verify** | Compare incremental content-word set vs final `extract_content_words(cleaned_response)` on golden strings |
| **Owner** | `kvl_beam_decoder.py` + `TextEvaluator` |

### Problem 4: llama.cpp step API

| | |
|-|-|
| **Symptom** | No per-step hook in current `generate_beam()` |
| **Solution** | New decode loop: tokenize prompt → `llm.eval()` in loop → append chosen tokens → check stop sequences |
| **Options** | (A) manual eval loop — **recommended**; (B) logits processor if stable in llama-cpp-python; (C) grammar — poor fit |
| **Verify** | Stub llm with mock logits; integration spike on one GGUF (Qwen3) before sweep |
| **Owner** | `models/kvl_beam_decoder.py` |

Share infrastructure with `ConstrainedDecoder` where possible (stop tokens, prompt tokenization, step counting).

### Problem 5: Inflection / OOV

| | |
|-|-|
| **Symptom** | `friends` OOV while `friend` is in lookup |
| **Solution (MVP)** | Skip OOV in beam mean (matches evaluation) |
| **Solution (v2)** | Optional WordNet lemmatizer before lookup; log `lemmatize_hits` |
| **Verify** | Corpus check: OOV rate on model outputs with vs without lemmatization |
| **Owner** | `evaluation/kvl.py` or decoder |

### Problem 6: Compute cost

| | |
|-|-|
| **Symptom** | O(steps × beam_width × branch_factor) forward passes |
| **Solution** | Document expected slowdown vs best-of-N; default smaller grid (`width=4,8`); ClusterUY job sizing |
| **Mitigation** | Cache eval state per beam (llama.cpp KV cache per sequence — if API supports batching, use it; otherwise sequential) |
| **Verify** | Benchmark: width=4, branch=10, max_tokens=200 on Qwen3 |
| **Owner** | ops / `docs/clusteruy.md` |

### Problem 7: Stop tokens and response extraction

| | |
|-|-|
| **Symptom** | Beam must halt on model stop sequences; scoring must exclude prompt |
| **Solution** | Reuse `_get_stop_tokens()`, `_extract_response()`, `_prepare_beam_scoring_text()` from `llamacpp.py` |
| **Verify** | Same tests as beam fix `7a6b105` — response-only text, no prompt leakage |
| **Owner** | wrapper + decoder |

### Problem 8: Evaluation circularity

| | |
|-|-|
| **Symptom** | Beam optimizes KVL mean; success judged by FK/Fog/Spache |
| **Solution** | Treat KVL beam as **hypothesis**, not ground truth. Report both `kvl_mean_score` and `meets_a1_criteria`. Compare to best-of-N KVL rerank baseline. |
| **Analysis** | Selected vs runner-up beam readability; width=1 degenerates to greedy KVL-aware decode |
| **Owner** | experiment design |

---

## Proposed file layout

```
src/slm_experiments/
├── models/
│   ├── kvl_token_index.py      # KVL lemma → token sequences
│   └── kvl_beam_decoder.py     # beam loop + incremental scoring
├── evaluation/
│   └── kvl.py                  # existing KvlLookup — reuse, do not fork
├── core/
│   ├── config.py               # + config_kvl_beam, kvl_beam_width, kvl_branch_factor
│   ├── pipeline.py             # + run_kvl_beam()
│   └── result.py               # + kvl_beam_* metadata fields
├── phase2/
│   └── kvl_beam.py             # experiment runner
└── cli.py                      # register: phase2 kvl_beam

tests/
├── test_kvl_token_index.py
├── test_kvl_beam_decoder.py
└── test_phase2_kvl_beam.py
```

---

## Phase 2 sweep design

Default grid sweeps **beam width** (branch factor fixed initially). Width `1` is an in-run greedy baseline (same carrier, KVL beam OFF → plain greedy, not a one-wide KVL beam):

| Setting | Default | Notes |
|---------|---------|-------|
| `config_kvl_beam` | `False` at width 1; else `True` | Baseline vs intervention |
| `kvl_beam_width` | `1, 4, 6, 8` | Swept; `1` = greedy baseline |
| `kvl_branch_factor` | `10` | Fixed in v1 |
| `kvl_l1` | `es` | Uses `kvl_lookup_es.json` |
| `config_prompting` | `True` (zero-shot) | Fixed carrier |
| `config_weighting` | `False` | No logit bias |
| `temperature` | `0.0` | Deterministic expansion |
| A1 beam / guided / weighting | OFF | Isolated intervention |

**Config naming:** `{model}_kvl_beam_w4` (parseable like `_beam_w(\d+)`). Baseline: `{model}_kvl_beam_w1`.

**CLI examples:**

```bash
python -m slm_experiments phase2 kvl_beam
python -m slm_experiments phase2 kvl_beam --widths 1,4,6,8 --prompts all
python -m slm_experiments phase2 kvl_beam --kvl-l1 es --branch-factor 10 --models Qwen3
```

Use `--prompts all` (25 prompts) for publishable claims. Report results **per model** (`summary.json` → `by_model`).

---

## Result metadata (new fields)

| Field | Purpose |
|-------|---------|
| `kvl_beam_width` | Swept hyperparameter |
| `kvl_branch_factor` | Expansion width |
| `kvl_beam_steps_total` | Decode steps |
| `kvl_beam_words_scored` | Content words that received GLMM during decode |
| `kvl_beam_running_mean` | Final running mean used to select winning beam |
| `kvl_beam_logprob_tiebreak` | Winner's cumulative logprob |
| `kvl_beam_candidates_pruned` | Total candidates considered (diagnostics) |

Post-hoc KVL columns (`kvl_mean_score`, `kvl_min_score`, etc.) remain computed by the pipeline on the **selected** response — same as today.

---

## Comparison to alternatives

| Method | Search | Objective | Fluency guard |
|--------|--------|-----------|---------------|
| Best-of-N + A1 rerank | N full samples | Internal 487-word A1 ratio | Model sampling |
| Best-of-N + KVL rerank | N full samples | Post-hoc `kvl_mean_score` | Model sampling |
| Logit bias | 1 path | Boost A1 tokens | Model logits |
| Guided decoding | 1 path | A1 in top-K pool | Argmax fallback |
| **KVL beam** | B paths × K branches | Running KVL mean | Logprob tie-break + top-K expansion |

**When KVL beam is worth the complexity:** You believe early word choices matter and post-hoc reranking misses good prefixes that the model would reach only if hard words are pruned mid-generation.

**When to skip it:** Best-of-N + KVL rerank is "good enough," or compute budget is tight — KVL beam is strictly more expensive than both best-of-N and guided decoding.

---

## Implementation roadmap

| Phase | Deliverable | Validates |
|-------|-------------|-----------|
| **0. Spike** | Manual decode loop on Qwen3 stub + fixture lookup | llama.cpp eval API works |
| **1. MVP** | `KvlBeamDecoder` word-boundary scoring, width=4, branch=10 | Problem 2, 4, 7 |
| **2. Index** | `KvlTokenIndex` + collision report | Problem 1 |
| **3. Integration** | `generate_kvl_beam`, `run_kvl_beam`, CLI | End-to-end one observation |
| **4. Sweep** | `phase2/kvl_beam.py`, summary buckets | Phase 2 grid |
| **5. v2** | Trie mode, optional lemmatization, KVL rerank baseline arm | Problems 1, 5, 8 |

### Baseline arms for analysis

Run these on the same prompts, models, and seed:

1. **Control** — prompting only, no beam (current Phase 2 beam config with width=1 or standard generate).
2. **Best-of-N + KVL rerank** — N samples, pick max `kvl_mean_score` (cheap upper bound on "selection helps").
3. **KVL beam** — this design.

Compare `meets_a1_criteria`, `kvl_mean_score`, `kvl_min_score`, generation time.

---

## Key references

| Topic | Location |
|-------|----------|
| KVL lookup data | `data/kvl/kvl_lookup_es.json`, `data/kvl/README.md` |
| KVL evaluation | `src/slm_experiments/evaluation/kvl.py` |
| Current best-of-N "beam" | `src/slm_experiments/models/beam.py` |
| Shared decode-loop design | `docs/guided-decoding.md` |
| Stop-token / scoring fixes | `improvements.md` (#2–4, beam section) |
| Experiment isolation pattern | `src/slm_experiments/phase2/beam.py` |
