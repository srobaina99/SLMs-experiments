# Readability Metrics

Reference for all text complexity metrics used in the experiment framework. Metrics are computed by `src/slm_experiments/evaluation/metrics.py` via the `textstat` library.

## Primary Metrics (3)

These form the core of statistical analysis and success criteria.

### Flesch-Kincaid Grade Level

**What it measures:** Reading difficulty based on sentence length and syllables per word.

```
Grade Level = (0.39 × ASL) + (11.8 × ASW) - 15.59
```

- **ASL** = Average Sentence Length (words per sentence)
- **ASW** = Average Syllables per Word

| Range | Interpretation |
|-------|---------------|
| 0–5 | Elementary — **TARGET for A1** |
| 6–8 | Middle school |
| 9+ | High school and above |

**A1 threshold:** ≤ 5.0

### Gunning Fog Index

**What it measures:** Years of formal education needed, emphasizing polysyllabic (3+ syllable) words.

```
Fog Index = 0.4 × [(Words/Sentences) + 100 × (Complex Words/Words)]
```

| Score | Interpretation |
|-------|---------------|
| 6 | Sixth grade — **TARGET for A1** |
| 8+ | Above beginner level |

**A1 threshold:** ≤ 6.0

### Spache Readability

**What it measures:** Difficulty for primary-grade materials (grades 1–4) using a familiar-word list.

```
Spache Score = (0.141 × ASL) + (0.086 × % Unfamiliar Words) + 0.839
```

| Score | Interpretation |
|-------|---------------|
| 1–2 | 1st–2nd grade — ideal for A1 |
| 3–4 | 3rd–4th grade — **TARGET for A1** |
| 5+ | Above primary level |

**A1 threshold:** ≤ 4.0

## Secondary Statistics (2)

Not used for success criteria but included in output for interpretation.

| Metric | Purpose |
|--------|---------|
| **Word count** | Verbosity / cognitive load indicator (target: 30–60 words) |
| **Difficult words** | Count of words not in Dale-Chall easy list with 3+ syllables |

## KVL / GLMM Vocabulary Difficulty (secondary)

External, learner-grounded vocabulary metrics from the [BEA 2026 Shared Task](https://github.com/britishcouncil/bea2026st) (British Council Knowledge-based Vocabulary Lists). Implemented in `src/slm_experiments/evaluation/kvl.py`.

**What it measures:** Whether content words in a model response are likely known by real L2 learners (by L1), using GLMM difficulty estimates from vocabulary knowledge tests. This complements — but does not replace — readability formulas or the internal 487-word A1 list used for logit bias and beam reranking.

| Dimension | Metric |
|-----------|--------|
| Sentence structure | Flesch-Kincaid |
| Polysyllabic complexity | Gunning Fog |
| Primary-grade vocabulary | Spache |
| **Learner-known vocabulary (external)** | **KVL / GLMM** |
| Internal A1 word overlap | Beam A1 ratio (487-word list) |

**License:** Source data is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only. See `data/kvl/README.md` for provenance and rebuild instructions.

**Default L1:** Spanish (`es`), configurable via `ExperimentConfig.kvl_l1`. Lookups also available for German (`de`) and Mandarin (`cn`).

### Columns

Computed on **content words** from `cleaned_response` (same NLTK POS extraction as beam A1-ratio scoring):

| Column | Type | Definition |
|--------|------|------------|
| `kvl_l1` | str | L1 code used for lookup (`es`, `de`, or `cn`) |
| `kvl_content_word_count` | int | Number of content words extracted |
| `kvl_lookup_count` | int | Content words found in KVL lookup |
| `kvl_oov_count` | int | Content words not found in KVL lookup (`kvl_content_word_count − kvl_lookup_count`) |
| `kvl_lookup_coverage` | float | `kvl_lookup_count / kvl_content_word_count` (0 if none) |
| `kvl_mean_score` | float \| None | Mean GLMM score over looked-up words; None if none found |
| `kvl_min_score` | float \| None | Min GLMM (hardest looked-up word); None if none found |
| `kvl_pct_hard_words` | float \| None | Fraction of looked-up words with GLMM < −1.0; None if none found |

**Interpretation:** Higher `kvl_mean_score` = easier vocabulary for that L1. Lower `kvl_min_score` = at least one very hard word present. GLMM range ≈ −6 to +5.

### Limitations

- **Lemma-only lookup (v1):** Keys are lowercased surface forms; no POS disambiguation. Inflected forms not in the ~6.8k-word lookup are OOV.
- **Productive knowledge proxy:** Scores reflect estimated receptive knowledge from learner tests, not production frequency.
- **No pass/fail criterion:** KVL metrics are recorded for analysis only; `meets_a1_criteria` is unchanged.

Failed generations receive zero counts and `None` aggregates, and are excluded from `summary.json` KVL means (same as readability metrics).

## Success Criteria (automated readability proxy)

**Generation success** (`generation_successful=True`) means valid, non-empty output with computable metrics.

**Readability proxy pass** (`meets_a1_criteria=True`) means **all three** primary US readability thresholds are met simultaneously on a valid generation. This is **not** a CEFR A1 communicative assessment — prompt themes are CEFR-inspired, but the flag only encodes formula thresholds. The field name is historical; report it as a proxy in the thesis.

| Metric | Threshold |
|--------|-----------|
| Flesch-Kincaid Grade | ≤ 5.0 |
| Gunning Fog | ≤ 6.0 |
| Spache Readability | ≤ 4.0 |

The three formulas are near-collinear; treat the conjunction as a single proxy gate. Optional human ratings (`human export` / `import`) measure agreement with the automatic flag.

## SMOG — Not Used

SMOG Index requires a minimum of 30 sentences. Typical model outputs are 1–3 sentences (~30–60 words), causing `textstat.smog_index()` to return 0.0 for nearly all responses. SMOG cannot discriminate between conditions and is excluded from the framework.

## Failed Generations

When generation fails (empty output, unparseable response, thinking-tag artifacts), metrics are **not computed**. The observation is recorded with `generation_successful=False` and `meets_a1_criteria=False`, and excluded from `summary.json` metric means. Proxy pass rate (`a1_pass_rate`) is still computed over all observations.

## Metric Selection Rationale

| Dimension | Covered by |
|-----------|-----------|
| Sentence structure | Flesch-Kincaid |
| Polysyllabic complexity | Gunning Fog |
| Vocabulary difficulty (primary-grade) | Spache |
| Learner-known vocabulary (external) | KVL / GLMM |

These three metrics provide complementary US-readability coverage aimed at very easy text without the SMOG short-text problem. They are a proxy for beginner-friendly output, not a CEFR level certificate.

## Discarded Metrics

| Metric | Reason |
|--------|--------|
| Flesch Reading Ease | Redundant with Flesch-Kincaid (same inputs, inverse scale) |
| Dale-Chall | Redundant with Spache; poor A1 discrimination |
| ARI, Coleman-Liau | Redundant character-based alternatives |
| Linsear Write | Designed for technical writing, not conversation |
| McAlpine EFLAW | Designed for spoken/auditory content |
| Reading time | Linear transform of word count |

## References

- Flesch, R. (1948). *Journal of Applied Psychology*, 32(3), 221–233.
- Kincaid, J.P., et al. (1975). *Research Branch Report 8-75*.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- Spache, G. (1953). *The Elementary School Journal*, 53(7), 410–413.
