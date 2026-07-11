# Experiment Design

Formal specification for the SLM evaluation framework. Phase 1 establishes intervention effects via a factorial design; Phase 2 sweeps hyperparameters on all four models.

## Context

This project evaluates whether inference-time interventions make small language models (SLMs) produce simpler English answers for beginner learners. Prompts are themed around CEFR A1 topics; the primary binary outcome is an **automated readability proxy** (not a CEFR proficiency test). Four models are tested across interventions that aim to simplify vocabulary and sentence structure.

## Models

| Model | Parameters | Backend | Notes |
|-------|-----------|---------|-------|
| Qwen2 | 0.5B | llama.cpp GGUF | ChatML template |
| Qwen3 | 0.6B | llama.cpp GGUF | ChatML; `enable_thinking=False` disables thinking tags |
| TinyLlama | 1.1B | llama.cpp GGUF | Llama2-style template |
| Phi3 | 3.8B | llama.cpp GGUF | Custom Phi-3 template; GPU required |

See [docs/models.md](docs/models.md) for GGUF filenames and deployment details.

## Interventions

Two inference-time interventions plus a control:

### 1. Probability Weighting (Logit Bias)

Boosts A1 vocabulary tokens during decoding. The 487-word list in `data/vocabularies/filtered_starters_vocab.txt` is tokenized in both mid-sentence (`" " + word`) and sentence-start (`word`) forms; the union of those token IDs is mapped to a `logit_bias` dictionary passed to llama.cpp. `weight_factor` is the target probability multiplier; the code applies `log(weight_factor)` as the additive logit bias (`1.0` = no bias). See [docs/interventions.md](docs/interventions.md).

### 2. Contextual Prompting

A context block instructs the model to use simple words, short sentences, and basic vocabulary. In Phase 1 this is a fixed zero-shot prompt; Phase 2 sweeps the number of in-context examples (0, 1, 3 shots).

### 3. Control

No weighting, no contextual prompting.

## Phase 1 — Factorial Experiment

**Design:** 4 models × 4 intervention combinations × N prompts

### Intervention Configurations

| Config | `config_weighting` | `config_prompting` | Label |
|--------|-------------------|-------------------|-------|
| Control | False | False | No intervention |
| Weighting Only | True | False | Logit bias only (`weight_factor=1.5`) |
| Prompting Only | False | True | Context block only |
| Both | True | True | Weighting + prompting |

### Prompts

25 standard prompts grounded in CEFR A1 thematic areas. Default CLI uses 3 prompts; `--prompts all` runs all 25.

| ID | Prompt |
|----|--------|
| P1 | What does the word 'library' mean? |
| P2 | How do I introduce myself in English? |
| P3 | Can you explain what 'breakfast' is? |
| P4 | What is the difference between 'big' and 'large'? |
| P5 | Can you describe what happens in the morning? |
| P6 | What does 'happy' mean? |
| P7 | How do you say goodbye in English? |
| P8 | What is the weather like today? |
| P9 | What is a 'friend'? |
| P10 | What do you do at school? |
| P11 | What is the difference between 'hot' and 'cold'? |
| P12 | Can you describe your family? |
| P13 | What foods do you eat for lunch? |
| P14 | What rooms are in a house? |
| P15 | What animals can be pets? |
| P16 | What do you do on the weekend? |
| P17 | What can people do for fun? |
| P18 | How do people travel to school or work? |
| P19 | What do you do when you feel sick? |
| P20 | How do I ask for something at a shop? |
| P21 | What happens when you go to the doctor? |
| P22 | What can you see in a town? |
| P23 | What does a person do every day? |
| P24 | What do people do at work? |
| P25 | What is the difference between 'this' and 'that'? |

### Generation Defaults

| Parameter | Value |
|-----------|-------|
| System prompt | "You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text" |
| Temperature | 0.0 (deterministic greedy decoding) |
| Top-K | 50 |
| Max tokens | 200 |
| Context window | 2048 (4096 for Phi3) |
| Seed | Configurable via `--seed` (default: 42; library init only — no sampling variance at temp 0) |

`top_p` is **not** used. Source of truth: `src/slm_experiments/core/config.py`.

### Observation Counts

| Prompts | Observations |
|---------|-------------|
| 3 (default, development/smoke only) | 4 × 4 × 3 = 48 |
| 25 (`--prompts all`, formal claims) | 4 × 4 × 25 = 400 |

### Success Criteria (automated readability proxy)

**Generation success** (`generation_successful=True`) means the model produced valid, non-empty output with computable metrics. Failed generations (empty output, thinking-tag artifacts, metric computation errors) are recorded with `generation_successful=False` and excluded from summary metric means.

**Readability proxy pass** (`meets_a1_criteria=True`) is an **automated US readability gate**, not a CEFR A1 communicative assessment. Prompt themes are CEFR-inspired; the binary flag only checks that all three formula thresholds hold on a valid generation:

| Metric | Threshold |
|--------|-----------|
| Flesch-Kincaid Grade | ≤ 5.0 |
| Gunning Fog | ≤ 6.0 |
| Spache Readability | ≤ 4.0 |

SMOG is **not** used — short model outputs cannot satisfy its 30-sentence minimum.

The field name `meets_a1_criteria` is historical. Treat `a1_pass_rate` / `a1_pass_count` in `summary.json` as **proxy pass rate / count**. Human ratings (`human export` / `import`) assess agreement with this proxy; they are not folded into the automatic flag. Metric means still exclude failed generations.

## Phase 2 — Hyperparameter Sweeps

All Phase 2 sweeps run on **all 4 models**. Each sweep varies one hyperparameter while holding other settings at sensible defaults.

### 2a. Weight Factor Sweep

**Intervention:** Weighting + prompting ON ("Both" configuration)

| Weight Factor | Additive Logit Bias `log(w)` | Probability Multiplier |
|---------------|-------------------------------|------------------------|
| 1.0 | 0.0 | 1.0× (no bias) |
| 1.3 | +0.26 | 1.3× |
| 1.5 | +0.41 | 1.5× |
| 2.0 | +0.69 | 2.0× |
| 2.5 | +0.92 | 2.5× |
| 3.0 | +1.10 | 3.0× |
| 4.0 | +1.39 | 4.0× |

Default grid: `1.0,1.3,1.5,2.0,2.5,3.0,4.0`

Beam search disabled (greedy decoding with logit bias).

### 2b. Beam Search Width Sweep — **Deprecated / excluded**

> **Do not cite.** At `temperature=0.0`, best-of-N repeats identical greedy decodes, so beam width cannot change the answer. The CLI hard-fails on `phase2 beam`. Historical code remains in `phase2/beam.py` / `models/beam.py`; use **2d guided** or **2e KVL beam** instead.

Former design (best-of-N + A1-ratio rerank, grids `4,8,10`, prompting ON, weighting OFF) is void under current defaults. Exclude any `results/runs/*_phase2_beam/` tables from thesis claims.

### 2c. Prompting Shot Sweep

**Intervention:** Contextual prompting with varying example counts

| Configuration | Shots | Description |
|---------------|-------|-------------|
| Zero-shot | 0 | Context block only |
| One-shot | 1 | Context + 1 Q/A example |
| Few-shot | 3 | Context + 3 Q/A examples |

Default grid: `0,1,3`

Weighting and beam search disabled. Temperature 0.0, top-k 50.

**Zero-shot context block:**

```
Please respond using simple words that a young non-English speaking student can understand.
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.
```

**One-shot adds:**

```
Question: What is a cat?
Answer: A cat is a small animal. It is soft and likes to play and sleep.
```

**Few-shot adds two more examples** (how-to and listing) before the target question:

```
Question: How do you ask for help in English?
Answer: You can say "Can you help me, please?" People use this when they need help.

Question: What can I find in a park?
Answer: You can find grass, trees, and benches in a park. Children play there.
```

### 2d. Guided Decoding Top-K Sweep

**Intervention:** Top-k A1-constrained greedy decoding (`phase2 guided`). Prompting ON (zero-shot), weighting OFF.

| Setting | Default |
|---------|---------|
| `guided_top_k` grid | `0,5,10,20` (`0` = in-run unconstrained baseline) |
| `guided_mode` | `flat` (optional `trie`) |
| Temperature / top-k | `0.0` / `50` |

**In-run baseline:** pool `0` keeps the same carrier (prompting ON, weighting OFF) but sets `config_guided=False` and uses plain greedy (`pipeline.run()`). Compare `k0` vs `k5`/`k10`/`k20` within the same run bundle.

At each guided step (`guided_top_k` > 0), if any token in the top-`guided_top_k` pool maps to the A1 list, pick the highest-probability A1 token; otherwise take global argmax. See [docs/guided-decoding.md](docs/guided-decoding.md).

### 2e. KVL Beam Width Sweep

**Intervention:** Token-level beam search ranked by Spanish L1 KVL/GLMM scores (`phase2 kvl_beam`). Prompting ON, weighting OFF.

| Setting | Default |
|---------|---------|
| `kvl_beam_width` grid | `1,4,6,8` (`1` = in-run greedy baseline) |
| `kvl_branch_factor` | `10` |
| `kvl_l1` | `es` |
| Temperature / top-k | `0.0` / `50` |

**In-run baseline:** width `1` keeps the same carrier but sets `config_kvl_beam=False` and uses plain greedy (`pipeline.run()`), not a one-wide KVL beam. Compare `w1` vs `w4`/`w6`/`w8` within the same run bundle.

**Stopping rule — first-finish (intentional):** returns the **first** finished non-empty candidate in expansion order (not the best KVL-ranked finished hypothesis). **Rationale:** mean KVL does not reward ending a sentence; without early return, “best KVL” tends to stretch to `max_new_tokens` with incoherent padding. KVL still steers beam pruning; first-finish governs when to stop. Do not interpret width 4 vs 6 vs 8 as “better final KVL selection”; treat KVL columns as primary for this arm. See [docs/kvl_beamsearch.md](docs/kvl_beamsearch.md).

## Output Format

Every run produces a bundle in `results/runs/{run_id}/`:

### specification.csv (paper-compatible)

Reduced columns with European decimal format (`decimal=','`):

- `model`, `config_weighting`, `config_prompting`, `prompt_id`
- `answer`, `time_spent`, `generation_successful`, `meets_a1_criteria`
- `flesch_kincaid_grade`, `gunning_fog`, `spache_readability`
- `word_count`, `difficult_words`

### full.csv

All fields including beam metadata (`beam_width`, `a1_ratio`, `candidates_generated`, etc.), weight factor, shot count, seed, and raw model output.

### summary.json

```json
{
  "overall": { "flesch_kincaid_grade": { "mean": 3.2, "std": 0.5, ... }, ... },
  "by_config": {
    "control": { "count": 12, "a1_pass_rate": 0.25, "flesch_kincaid_grade": { "mean": 3.5, ... } },
    "both": { "count": 84, "a1_pass_rate": 0.40, "flesch_kincaid_grade": { "mean": 2.8, ... } }
  },
  "by_weight_factor": {
    "1.5": { "count": 100, "a1_pass_rate": 0.4, "generation_failure_rate": 0.02, "...": "..." }
  },
  "by_model": {
    "Qwen3": {
      "count": 25,
      "a1_pass_rate": 0.48,
      "by_weight_factor": {
        "1.5": { "count": 25, "a1_pass_rate": 0.48, "flesch_kincaid_grade": { "mean": 2.7, ... } }
      }
    },
    "TinyLlama": { "by_weight_factor": { "1.5": { "...": "..." } } }
  },
  "metadata": { "total_experiments": 700, "sweep_dimension": "weight_factor", ... }
}
```

Pooled sections (`by_config`, `by_weight_factor`, …) remain as overview. **Thesis tables should report `by_model` first** (per-model × config or per-model × sweep key) to avoid Simpson’s paradox; pooled figures are secondary.

Phase 1 runs populate `by_config` and `by_model[*].by_config`. Phase 2 sweeps add a sweep-specific section (`by_weight_factor`, `by_num_shots`, `by_guided_top_k`, `by_kvl_beam_width`) and nest the same keys under `by_model`. All Phase 2 weight runs share the same intervention flags, so pooled `by_config` collapses to a single bucket (typically `both`); use the sweep / `by_model` sections for analysis.

Failed generations excluded from metric means; `generation_failure_rate` and proxy `a1_pass_rate` are computed over all rows.

### manifest.json

Run metadata: phase, experiment type, CLI args, models, prompt count, timestamps, artifact paths.

## Statistical Analysis

The factorial design enables:

- Grouping by intervention flags (`config_weighting`, `config_prompting`)
- Model comparison across the same prompts
- Factorial interaction analysis (additive vs synergistic effects)
- Boxplot and heatmap visualization via `plot --run-id`

Phase 2 sweeps enable within-model hyperparameter optimization curves.

## Research Questions

**Phase 1:**
1. Do weighting and prompting independently reduce text complexity?
2. Is the combined effect additive or synergistic?
3. Which models naturally produce simpler output?

**Phase 2:**
1. What is the optimal weight factor (per model) before fluency degrades under prompting + weighting?
2. How many prompting shots are needed for consistent simplification?
3. Does guided top-k or KVL beam width improve the readability proxy / KVL metrics vs the fixed carrier (per model)?
4. (Excluded) Deprecated best-of-N beam — do not ask thesis questions of `phase2 beam` runs.

## Methodological Notes

| Issue | Handling |
|-------|----------|
| Qwen3 thinking tags | `enable_thinking=False` via Jinja chat template + formatter strip |
| Failed gen → FK=0 | Skip metrics; `generation_successful=False` |
| No reproducibility seed | `--seed` on all CLI runs |
| SMOG on short text | Not included in framework |
