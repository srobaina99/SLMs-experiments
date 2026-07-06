# Interventions

How each inference-time intervention works in code. All three operate at generation time without model fine-tuning.

A fourth intervention — **top-k A1-guided decoding** — is designed but not yet implemented. See [guided-decoding.md](guided-decoding.md).

## 1. Probability Weighting (Logit Bias)

Increases the probability of A1-level vocabulary tokens during autoregressive decoding.

### Data Flow

```
filtered_starters_vocab.txt (487 words)
    → tokenize each word → token IDs
    → build logit_bias dict: {token_id: log(weight_factor)}
    → pass to llama.cpp at generation time
    → at each step: logit[token_id] += log(weight_factor) (before softmax)
```

### Vocabulary Loading

On model wrapper init, `data/vocabularies/filtered_starters_vocab.txt` is loaded into `target_vocabulary` (487 words, lowercased). Punctuation-only entries and model stop tokens are skipped at load time.

### Building logit_bias

```python
def _create_logit_bias(vocab, weight_factor):
    bias_value = math.log(weight_factor)
    logit_bias = {}
    for word in vocab:
        tokens = llm.tokenize((" " + word).encode("utf-8"), add_bos=False)
        for token_id in tokens:
            logit_bias[token_id] = bias_value
    return logit_bias
```

The space prefix (`" " + word`) ensures token IDs match mid-sentence BPE tokenization.

### Mathematical Effect

`weight_factor` is the target **probability multiplier** for A1 tokens. llama.cpp applies additive logit bias, so the code sets `bias = log(weight_factor)`:

| weight_factor | Additive bias `log(w)` | Probability multiplier |
|---------------|------------------------|------------------------|
| 1.0 | 0.0 | 1.0× (no bias) |
| 1.3 | +0.26 | 1.3× |
| 1.5 | +0.41 | 1.5× |
| 2.0 | +0.69 | 2.0× |
| 3.0 | +1.10 | 3.0× |
| 4.0 | +1.39 | 4.0× |

**Important:** `weight_factor=1.0` means no biasing. Values above 1.0 increase A1 token probability by that factor (relative to other tokens, holding logits fixed).

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
| 1 | + definition example ("What is a cat?") |
| 3 | + how-to example ("How do you ask for help…?") and listing example ("What can I find in a park?") |

Weighting and beam disabled during prompting sweep.

## 3. Best-of-N Sampling with A1-Ratio Selection

Generates `beam_width` **independent** stochastic samples (not canonical beam decoding) and selects the one with the highest A1 vocabulary ratio. The `BeamSearchGenerator` name is historical; behavior is best-of-N reranking.

### Algorithm

1. Run `beam_width` separate temperature-sampled generations (`echo=False` → response text only)
2. For each candidate response, compute A1 ratio (prompt/context excluded)
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

Qwen3 may emit `<think>` blocks. Mitigation:
1. Render prompts with the GGUF Jinja chat template and `enable_thinking=False` (hard switch: empty `<think>` block in the assistant prefix)
2. Response formatter strips thinking tags before metric evaluation

## Source Files (v2)

| File | Role |
|------|------|
| `data/vocabularies/filtered_starters_vocab.txt` | A1 vocabulary (487 words) |
| `models/llamacpp.py` | `_create_logit_bias()`, generation with bias |
| `models/beam.py` | Beam search with A1-ratio selection |
| `evaluation/formatter.py` | Response cleaning, thinking-tag strip |
| `phase1/configs.py` | Factorial config factory |
| `phase2/weights.py` | Weight sweep runner |
| `phase2/beam.py` | Beam sweep runner |
| `phase2/prompting.py` | Shot sweep runner |
