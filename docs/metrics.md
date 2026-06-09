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

## Success Criteria

A generation is successful when **all three** primary thresholds are met simultaneously:

| Metric | Threshold |
|--------|-----------|
| Flesch-Kincaid Grade | ≤ 5.0 |
| Gunning Fog | ≤ 6.0 |
| Spache Readability | ≤ 4.0 |

## SMOG — Not Used

SMOG Index requires a minimum of 30 sentences. Typical model outputs are 1–3 sentences (~30–60 words), causing `textstat.smog_index()` to return 0.0 for nearly all responses. SMOG cannot discriminate between conditions and is excluded from the framework.

## Failed Generations

When generation fails (empty output, unparseable response, thinking-tag artifacts), metrics are **not computed**. The observation is recorded with `generation_successful=False` and excluded from `summary.json` metric aggregates.

## Metric Selection Rationale

| Dimension | Covered by |
|-----------|-----------|
| Sentence structure | Flesch-Kincaid |
| Polysyllabic complexity | Gunning Fog |
| Vocabulary difficulty (primary-grade) | Spache |

These three metrics provide non-redundant coverage calibrated for A1 learners without the SMOG short-text problem.

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
