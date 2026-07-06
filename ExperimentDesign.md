# Experiment Design

Formal specification for the SLM evaluation framework. Phase 1 establishes intervention effects via a factorial design; Phase 2 sweeps hyperparameters on all four models.

## Context

This project evaluates whether small language models (SLMs) can produce pedagogically appropriate English for CEFR A1 learners — non-English speakers with little to no prior exposure. Four models are tested across inference-time interventions that aim to simplify vocabulary and sentence structure.

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

Boosts A1 vocabulary tokens during decoding. The 487-word list in `data/vocabularies/filtered_starters_vocab.txt` is tokenized and mapped to a `logit_bias` dictionary passed to llama.cpp. `weight_factor` is the target probability multiplier; the code applies `log(weight_factor)` as the additive logit bias (`1.0` = no bias). See [docs/interventions.md](docs/interventions.md).

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
| Temperature | 0.7 |
| Top-K | 50 |
| Top-P | 0.95 |
| Max tokens | 200 |
| Context window | 2048 (4096 for Phi3) |
| Seed | Configurable via `--seed` (default: 42) |

### Observation Counts

| Prompts | Observations |
|---------|-------------|
| 3 (default) | 4 × 4 × 3 = 48 |
| 25 (`--prompts all`) | 4 × 4 × 25 = 400 |

### Success Criteria

**Generation success** (`generation_successful=True`) means the model produced valid, non-empty output with computable metrics. Failed generations (empty output, thinking-tag artifacts, metric computation errors) are recorded with `generation_successful=False` and excluded from summary metric means.

**A1 pass** (`meets_a1_criteria=True`) means all three readability thresholds are met on a valid generation:

| Metric | Threshold |
|--------|-----------|
| Flesch-Kincaid Grade | ≤ 5.0 |
| Gunning Fog | ≤ 6.0 |
| Spache Readability | ≤ 4.0 |

SMOG is **not** used — short model outputs cannot satisfy its 30-sentence minimum.

`summary.json` reports `a1_pass_rate` and `a1_pass_count` across all observations; metric means still exclude failed generations.

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

### 2b. Beam Search Width Sweep

**Intervention:** Beam search with A1-ratio candidate selection

| Beam Width | Candidates | Selection |
|------------|-----------|-----------|
| 4 | 4 sequences | Highest A1 ratio |
| 8 | 8 sequences | Highest A1 ratio |
| 10 | 10 sequences | Highest A1 ratio |

Default grid: `4,8,10`

- Contextual prompting: enabled (zero-shot)
- Logit bias: disabled
- Temperature: 0.7, Top-P: 0.95, Top-K: 50

**A1 ratio formula:**

```
A1_ratio = (Count of A1 words × 1.5) / Count of content words
```

Content words identified via NLTK POS tagging with heuristic fallback.

### 2c. Prompting Shot Sweep

**Intervention:** Contextual prompting with varying example counts

| Configuration | Shots | Description |
|---------------|-------|-------------|
| Zero-shot | 0 | Context block only |
| One-shot | 1 | Context + 1 Q/A example |
| Few-shot | 3 | Context + 3 Q/A examples |

Default grid: `0,1,3`

Weighting and beam search disabled.

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
    "control": { "count": 12, "flesch_kincaid_grade": { "mean": 3.5, ... } },
    "both": { "count": 84, "flesch_kincaid_grade": { "mean": 2.8, ... } }
  },
  "by_weight_factor": {
    "1.5": { "count": 12, "flesch_kincaid_grade": { "mean": 2.9, ... } }
  },
  "metadata": { "total_experiments": 96, "sweep_dimension": "weight_factor", ... }
}
```

Phase 1 runs populate `by_config` only. Phase 2 sweeps add a sweep-specific section (`by_weight_factor`, `by_beam_width`, or `by_num_shots`) keyed by the swept hyperparameter. All Phase 2 weight runs share the same intervention flags, so `by_config` collapses to a single bucket (typically `both` or `prompting_only`); use the sweep section for per-grid-point analysis.

Failed generations excluded from metric means.

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
1. What is the optimal weight factor before fluency degrades?
2. Does beam width continue to improve A1-ratio selection?
3. How many prompting shots are needed for consistent simplification?

## Methodological Notes

| Issue | Handling |
|-------|----------|
| Qwen3 thinking tags | `enable_thinking=False` via Jinja chat template + formatter strip |
| Failed gen → FK=0 | Skip metrics; `generation_successful=False` |
| No reproducibility seed | `--seed` on all CLI runs |
| SMOG on short text | Not included in framework |
