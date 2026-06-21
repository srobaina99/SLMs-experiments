# KVL / GLMM Vocabulary Lookup

Compact word difficulty lookups derived from the [BEA 2026 Shared Task](https://github.com/britishcouncil/bea2026st) (British Council Knowledge-based Vocabulary Lists).

## License

Source data is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only. See the [BEA repo README](https://github.com/britishcouncil/bea2026st) for attribution details.

## Files

| File | Description |
|------|-------------|
| `kvl_lookup_es.json` | Spanish L1 learners — mean GLMM score per English word |
| `kvl_lookup_de.json` | German L1 learners |
| `kvl_lookup_cn.json` | Mandarin L1 learners |

Each JSON maps lowercased `en_target_word` → mean `GLMM_score` across train/dev/test splits (~6,847 words per L1).

**Interpretation:** Lower GLMM score = harder word for that L1. Typical range ≈ −6 to +5.

## Default L1: Spanish (`es`)

`ExperimentConfig.kvl_l1` defaults to `es` because Spanish-speaking learners represent the largest ESL context in the KVL study and the most common L2 English learner group in many classroom settings. German (`de`) and Mandarin (`cn`) lookups are available for future experiments.

## Limitations (v1)

- **Lemma-only lookup:** Keys are lowercased surface forms with no POS disambiguation. Inflected forms not in the lookup are treated as OOV.
- **Coverage:** ~6.8k content words from the KVL test set; words outside this set are excluded from mean/min/hard-word aggregates but still count toward content-word totals.
- **Productive knowledge proxy:** GLMM scores reflect estimated learner knowledge from vocabulary tests, not production frequency in model outputs.

## Rebuild

Clone the BEA repo (or point to an existing checkout) and run:

```bash
python3 scripts/build_kvl_lookup.py --source-dir /path/to/bea2026st/data
```

Without `--source-dir`, the script clones to `/tmp/bea2026st` automatically.

Source CSV columns used: `en_target_word`, `GLMM_score`. When a word appears in multiple items for the same L1, scores are averaged.

## References

- Skidmore et al. (2025). BEA 2026 Shared Task baseline. [DOI:10.18653/v1/2025.bea-1.12](https://doi.org/10.18653/v1/2025.bea-1.12)
- Schmitt et al. (2024). Knowledge-based vocabulary lists. [British Council Research Insight](https://www.britishcouncil.org/research-insight/knowledge-based-vocabulary-lists)
