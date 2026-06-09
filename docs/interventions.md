# Interventions

How each inference-time intervention works in code. All three operate at generation time without model fine-tuning.

## 1. Probability Weighting (Logit Bias)

Increases the probability of A1-level vocabulary tokens during autoregressive decoding.

### Data Flow

```
filtered_starters_vocab.txt (493 words)
    → tokenize each word → token IDs
    → build logit_bias dict: {token_id: weight_factor}
    → pass to llama.cpp at generation time
    → at each step: logit[token_id] += weight_factor (before softmax)
```

### Vocabulary Loading

On model wrapper init, `data/vocabularies/filtered_starters_vocab.txt` is loaded into `target_vocabulary` (493 words, lowercased). Includes punctuation and model stop tokens.

### Building logit_bias

```python
def _create_logit_bias(vocab, weight_factor):
    logit_bias = {}
    for word in vocab:
        tokens = llm.tokenize((" " + word).encode("utf-8"), add_bos=False)
        for token_id in tokens:
            logit_bias[token_id] = weight_factor
    return logit_bias
```

The space prefix (`" " + word`) ensures token IDs match mid-sentence BPE tokenization.

### Mathematical Effect

Adding constant `b` to a token's logit multiplies its probability by `e^b`:

| weight_factor | Additive bias | Probability multiplier |
|---------------|--------------|----------------------|
| 1.0 | +1.0 | 2.72× |
| 1.5 | +1.5 | 4.48× |
| 2.0 | +2.0 | 7.39× |
| 3.0 | +3.0 | 20.09× |
| 4.0 | +4.0 | 54.60× |

**Important:** `weight_factor` is applied **directly** as the logit bias — not `log(weight_factor)`. This produces stronger biasing than a logarithmic transform would.

### Phase 1 Defaults

- `config_weighting=True`, `weight_factor=1.5` for weighted conditions
- Applied at every decoding step, uniformly across all A1 tokens
- Interacts with temperature (0.7), top-k (50), top-p (0.95)

### Phase 2 Weight Sweep

Weighting + prompting ON. Default grid: `1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0`. Beam search disabled.

## 2. Contextual Prompting

Adds a context block instructing the model to simplify its language.

### Phase 1 (Zero-shot)

Appended to the user message:

```
Please respond using simple words that a young non-English speaking student can understand.
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.
```

### Phase 2 Shot Sweep

| Shots | Examples added before target question |
|-------|--------------------------------------|
| 0 | Context block only |
| 1 | + "What is a cat?" / "A cat is a small animal..." |
| 3 | + cat, happy, water examples |

Weighting and beam disabled during prompting sweep.

## 3. Beam Search with A1-Ratio Selection

Generates multiple candidate sequences and selects the one with the highest A1 vocabulary ratio.

### Algorithm

1. Generate `beam_width` candidate sequences (temperature sampling)
2. For each candidate, compute A1 ratio
3. Select candidate with highest ratio
4. Record beam metadata in `full.csv`

### A1 Ratio Formula

```
A1_ratio = (Count of A1 words × 1.5) / Count of content words
```

Content words identified via NLTK POS tagging (`NN`, `VB`, `JJ`, `RB`, etc.) with heuristic fallback for short texts.

### Phase 2 Beam Sweep

Default widths: `4, 8, 10`

| Setting | Value |
|---------|-------|
| Contextual prompting | Enabled (zero-shot) |
| Logit bias | Disabled |
| Temperature | 0.7 |
| Top-P | 0.95 |
| Top-K | 50 |

### Trade-offs

More beams improve A1 vocabulary selection but increase generation time linearly (~76s for width=4 → ~150s for width=8 on Qwen3).

## Intervention Matrix

### Phase 1 Factorial

| Config | Weighting | Prompting | Beam |
|--------|-----------|-----------|------|
| Control | ✗ | ✗ | ✗ |
| Weighting Only | ✓ (1.5) | ✗ | ✗ |
| Prompting Only | ✗ | ✓ | ✗ |
| Both | ✓ (1.5) | ✓ | ✗ |

### Phase 2 Sweeps

| Sweep | Weighting | Prompting | Beam |
|-------|-----------|-----------|------|
| weights | ✓ (varied) | ✓ (zero-shot) | ✗ |
| beam | ✗ | ✓ (zero-shot) | ✓ (varied) |
| prompting | ✗ | ✓ (0/1/3 shots) | ✗ |

## Qwen3 Thinking Tags

Qwen3 may emit `` blocks. Mitigation:
1. Append `/nothink` to user prompts
2. Response formatter strips thinking tags before metric evaluation

## Source Files (v2)

| File | Role |
|------|------|
| `data/vocabularies/filtered_starters_vocab.txt` | A1 vocabulary (493 words) |
| `models/llamacpp.py` | `_create_logit_bias()`, generation with bias |
| `models/beam.py` | Beam search with A1-ratio selection |
| `evaluation/formatter.py` | Response cleaning, thinking-tag strip |
| `phase1/configs.py` | Factorial config factory |
| `phase2/weights.py` | Weight sweep runner |
| `phase2/beam.py` | Beam sweep runner |
| `phase2/prompting.py` | Shot sweep runner |
