# Readability Metrics

Reference for all text complexity metrics used in the experiment framework. Metrics are computed by `src/slm_experiments/evaluation/metrics.py` via the `textstat` library.

## Evaluation protocol

Locked endpoint hierarchy and reporting rules live in [ExperimentDesign.md — Evaluation Protocol (locked)](../ExperimentDesign.md#evaluation-protocol-locked). Summary:

| Role | Endpoint |
|------|----------|
| **Primary** | CEFR-SP document-level A1 (`meets_a1_criteria`) |
| **Guardrail** | Human answer-adequacy (human subset only) |
| **Secondary** | TSAR ensemble ordinal (assessment-only; never an A1 gate) + Spanish KVL v2 |
| **Legacy / descriptive** | FK / Fog / Spache (this section) |

Every endpoint is **always reported beside** `generation_failure_rate` and `hit_max_tokens_rate` — never conditional-only. Full construct, baseline, bootstrap, and multiplicity rules are in ExperimentDesign only.

**Quality-preservation scope:** claims stay **limited to the human sample** until LLM-judge results are imported (`assess judge-import`) and shown to agree acceptably with human consensus. The judge seam is provider-neutral (no API adapter).

## Descriptive readability metrics (legacy)

FK / Fog / Spache remain recorded for analysis as **legacy / descriptive** scores under the locked endpoint hierarchy. They **do not** decide `meets_a1_criteria` (that gate is CEFR-SP). Cutoffs below are historical proxy context only — not gates.

### Flesch-Kincaid Grade Level

**What it measures:** Reading difficulty based on sentence length and syllables per word.

```
Grade Level = (0.39 × ASL) + (11.8 × ASW) - 15.59
```

- **ASL** = Average Sentence Length (words per sentence)
- **ASW** = Average Syllables per Word

| Range | Interpretation |
|-------|---------------|
| 0–5 | Elementary — historical beginner target (descriptive) |
| 6–8 | Middle school |
| 9+ | High school and above |

**Historical proxy cutoff (not a gate):** ≤ 5.0

### Gunning Fog Index

**What it measures:** Years of formal education needed, emphasizing polysyllabic (3+ syllable) words.

```
Fog Index = 0.4 × [(Words/Sentences) + 100 × (Complex Words/Words)]
```

| Score | Interpretation |
|-------|---------------|
| 6 | Sixth grade — historical beginner target (descriptive) |
| 8+ | Above beginner level |

**Historical proxy cutoff (not a gate):** ≤ 6.0

### Spache Readability

**What it measures:** Difficulty for primary-grade materials (grades 1–4) using a familiar-word list.

```
Spache Score = (0.141 × ASL) + (0.086 × % Unfamiliar Words) + 0.839
```

| Score | Interpretation |
|-------|---------------|
| 1–2 | 1st–2nd grade — ideal for beginners (descriptive) |
| 3–4 | 3rd–4th grade — historical beginner target (descriptive) |
| 5+ | Above primary level |

**Historical proxy cutoff (not a gate):** ≤ 4.0

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
- **No pass/fail criterion:** KVL metrics are recorded for analysis only; the A1 gate is CEFR-SP.

Failed generations receive zero counts and `None` aggregates, and are excluded from `summary.json` KVL means (same as readability metrics).

### KVL v2 (assessment-only)

Occurrence-level, lemmatized KVL path used by `assess build` (`evaluation/assessment/kvl_v2.py` → `compute_kvl_v2_metrics` in `evaluation/kvl.py`). **Diagnostic only — never an A1 gate.** Does not change KVL beam decoding. Install marker: `pip install -e ".[kvl-v2]"` (empty extra; NLTK WordNet is already a core dep).

**Semantics:** lemmatizes **English** answer content-word tokens (POS-aware `nltk.WordNetLemmatizer`, with `simplemma`/`identity` fallbacks) and looks them up in the **Spanish-L1** KVL table by default (`kvl_v2_l1=es`). This is not a Spanish lemmatizer.

Unlike v1 (unique surface-form set), v2 scores **every** content-word occurrence. Aggregates are over looked-up tokens only — **never interpret `kvl_v2_mean_score` without also reporting `kvl_v2_lookup_coverage`.** Hard-token threshold is the same as v1 (`GLMM < −1.0`). Lower-tail is the 10th-percentile nearest-rank score.

| Column | Type | Definition |
|--------|------|------------|
| `kvl_v2_l1` | str | L1 code used for lookup (default `es`) |
| `kvl_v2_token_count` | int | Content-word occurrence count after lemmatization |
| `kvl_v2_lookup_count` | int | Occurrences found in the KVL table |
| `kvl_v2_oov_count` | int | `token_count − lookup_count` |
| `kvl_v2_lookup_coverage` | float | `lookup_count / token_count` (0 if none) |
| `kvl_v2_mean_score` | float \| None | Mean GLMM over looked-up occurrences |
| `kvl_v2_hard_token_share` | float \| None | Fraction of looked-up occurrences with GLMM < −1.0 |
| `kvl_v2_lower_tail_score` | float \| None | 10th-percentile GLMM among looked-up occurrences |
| `kvl_v2_status` | str | `ok` / `missing` / `error` |
| `kvl_v2_error` | str \| None | Error detail when status is `error` |

v1 columns remain on generation runs for old-run compatibility.

## CEFR-SP Sentence Difficulty (primary A1 gate)

Official Arase et al. contrastive sentence CEFR scorer ([EMNLP 2022](https://aclanthology.org/2022.emnlp-main.416/); Zenodo ckpt [10.5281/zenodo.7234096](https://doi.org/10.5281/zenodo.7234096), CC BY 4.0). Implemented in `src/slm_experiments/evaluation/cefr_sp.py` with vendored Lightning modules under `evaluation/cefr_sp_vendor/`.

**Default ON.** Disable with `ExperimentConfig.enable_cefr_sp=False` or CLI `--no-enable-cefr-sp`. Requires optional extras and the downloaded checkpoint:

```bash
./venv/bin/pip install -e ".[cefr-sp]"
./venv/bin/python scripts/download_cefr_sp_ckpt.py
```

**What it measures:** Predicted CEFR level (A1–C2) of each sentence in `cleaned_response`, then document-level aggregates. **`meets_a1_criteria` is True iff CEFR-SP is enabled and `cefr_sp_level == "A1"`** on a valid generation.

### Columns

| Column | Type | Definition |
|--------|------|------------|
| `cefr_sp_enabled` | bool | Whether CEFR-SP was enabled for this observation (not “scoring succeeded”) |
| `cefr_sp_sentence_count` | int | Sentences after NLTK `sent_tokenize` |
| `cefr_sp_level` | str \| None | Discrete A1–C2 from mean hard-label ordinal (nearest) |
| `cefr_sp_level_ordinal` | float \| None | Mean of per-sentence argmax ordinals (0=A1 … 5=C2) |
| `cefr_sp_max_level_ordinal` | int \| None | Max sentence ordinal |
| `cefr_sp_pct_a1` | float \| None | Fraction of sentences predicted A1 |
| `cefr_sp_adjacency` | float \| None | Fraction of sentences with level ≤ A2 (project aggregate; not a paper metric) |
| `cefr_sp_expected_level` | float \| None | Mean of per-sentence ∑ᵢ i·pᵢ over the 6 class probs |

Document-level fields are **project aggregates** over sentence scores. The Arase et al. model itself is sentence-level only.

**Tokenization:** Matches training — whitespace-split words with `is_split_into_words=True`. See `data/cefr_sp/README.md`.

### Limitations

- Heavy (~1.2GB ckpt + BERT); torch/transformers/Lightning import only when scoring is enabled and text is non-empty.
- With `--no-enable-cefr-sp`, `meets_a1_criteria` is always False (gate requires a CEFR-SP score).
- Disabled path and failed generations emit the empty schema (`None` aggregates) without importing torch. Missing ckpt / load errors raise (fail loud) when enabled.

## TSAR ModernBERT ensemble (assessment-only secondary)

Pinned three-model **confidence-max** ensemble from the TSAR 2025 shared-task evaluators (`evaluation/assessment/cefr_tsar.py`). Runs only on the assessment path (`assess build`); never on generation runs and **never** as an A1 gate.

```bash
./venv/bin/pip install -e ".[cefr-tsar]"
```

**What it measures:** Discrete CEFR label (A1–C2) and ordinal 1–6 for each scorable `cleaned_response`, plus disagreement with the source-run CEFR-SP label. Aggregation keeps the member prediction with the highest softmax score (not majority vote).

**A1-blind caveat:** the ensemble has known weak A1 discrimination (ensemble A1 F1 = 0.00 on the TSAR test split). Use it only as an **ordinal / directional cross-check**. If an A1-class TSAR signal is ever needed, that is the `doc_en` / `TRAIN_DOC_EN` member alone — not this ensemble.

### Pinned models (`TSAR_MODELS`)

| Key | Hugging Face model id | Revision SHA |
|-----|----------------------|--------------|
| `doc_en` | `AbdullahBarayan/ModernBERT-base-doc_en-Cefr` | `3c29f5fbcdc753e99bb437ff9303df983486915b` |
| `doc_sent_en` | `AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr` | `b00d1d4780e46f6410ea8a8509649044dee18298` |
| `reference_alllang` | `AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2` | `83337437aa82277e96b293665dc3186088a4a839` |

SHAs are the single source of truth in code; cluster image prefetch parses the same tuple (see [docs/clusteruy.md](clusteruy.md)).

### Columns

| Column | Type | Definition |
|--------|------|------------|
| `cefr_tsar_{key}_label` | str \| None | Per-member discrete label (`doc_en`, `doc_sent_en`, `reference_alllang`) |
| `cefr_tsar_{key}_confidence` | float \| None | Per-member softmax score for that label |
| `cefr_tsar_ensemble_label` | str \| None | Confidence-max member label (A1–C2) |
| `cefr_tsar_ensemble_ordinal` | int \| None | 1=A1 … 6=C2 |
| `cefr_tsar_ensemble_confidence` | float \| None | Softmax score of the winning member |
| `cefr_tsar_status` | str | `ok` / `missing` / `error` |
| `cefr_tsar_error` | str \| None | Error detail when status is `error` |
| `cefr_tsar_disagrees_with_cefr_sp` | bool \| None | True when both discrete labels exist and differ; None if either missing |

Assessment cluster packaging (eval image, offline Hub pins, `run_assess.sh`) lives in [docs/clusteruy.md](clusteruy.md) — do not duplicate it here.

## Success Criteria (CEFR-SP A1 gate)

**Generation success** (`generation_successful=True`) means valid, non-empty output with computable metrics.

**A1 pass** (`meets_a1_criteria=True`) means a valid generation whose CEFR-SP document level is **A1** (`cefr_sp_level == "A1"`). US readability formulas (FK / Fog / Spache) are still recorded for analysis but **do not** decide the binary gate.

| Gate input | Rule |
|------------|------|
| `generation_successful` | Must be True |
| `cefr_sp_enabled` | Must be True |
| `cefr_sp_level` | Must equal `"A1"` |

Optional human ratings (`human export` / `import`, or the three-rater study) measure agreement with the automatic flag.

## LLM-judge seam (provider-neutral)

Assessment path only. Export `judge_input.jsonl` over assessment-bundle items, score externally against `beginner_suitability_judge_rubric_v0` (same four 1–4 dimensions as the human study), then import validated `judge_scores.csv` (`judge_scores_v0` schema).

```bash
python -m slm_experiments assess judge-export --assessment-run-id <id>
python -m slm_experiments assess judge-import --assessment-run-id <id> --scores judge_scores.csv
```

**No API / provider adapter** is implemented. Quality-preservation claims stay **human-sample-limited** until imported judge scores agree acceptably with human consensus.

## SMOG — Not Used

SMOG Index requires a minimum of 30 sentences. Typical model outputs are 1–3 sentences (~30–60 words), causing `textstat.smog_index()` to return 0.0 for nearly all responses. SMOG cannot discriminate between conditions and is excluded from the framework.

## Failed Generations and Maxed-Out Outputs

When generation fails (empty output, unparseable response, thinking-tag artifacts, timeout), metrics are **not computed**. The observation is recorded with `generation_successful=False` and `meets_a1_criteria=False`, and excluded from `summary.json` metric means. Pass rate (`a1_pass_rate`) is still computed over all observations.

Separately, `hit_max_tokens=True` marks outputs that exhausted the `max_new_tokens` budget without a natural stop. These can still be `generation_successful=True` (non-empty text) while padding toward the length limit. Summaries report both `generation_failure_rate` and `hit_max_tokens_rate` in every bucket; thesis tables should print both beside conditional means.

## Metric Selection Rationale

| Dimension | Covered by |
|-----------|-----------|
| **Primary A1 gate** | **CEFR-SP (`cefr_sp_level == "A1"`)** |
| Secondary CEFR cross-check (assessment) | TSAR ModernBERT ensemble ordinal / disagreement |
| Learner-known vocabulary v2 (assessment) | KVL v2 (English lemmas → Spanish-L1 table) |
| Learner-known vocabulary v1 (in-pipeline) | KVL / GLMM (surface-form) |
| Sentence structure (descriptive) | Flesch-Kincaid |
| Polysyllabic complexity (descriptive) | Gunning Fog |
| Vocabulary difficulty (descriptive) | Spache |

FK / Fog / Spache remain legacy / descriptive metrics (not the locked Secondary tier — that is TSAR + KVL v2). The binary experiment outcome is CEFR-SP document level A1.


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

Classic formulas (legacy / descriptive only):


- Flesch, R. (1948). *Journal of Applied Psychology*, 32(3), 221–233.
- Kincaid, J.P., et al. (1975). *Research Branch Report 8-75*.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- Spache, G. (1953). *The Elementary School Journal*, 53(7), 410–413.

For proposed neural / CEFR / simplicity upgrades and a prioritized reading list, see [`docs/cites-to-include.md`](cites-to-include.md). KVL citations live in `data/kvl/README.md`.
