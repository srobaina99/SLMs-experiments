# Human Evaluation

Three-rater blinded study for beginner-suitability ratings against assessment-bundle items. Complements the locked evaluation protocol in [ExperimentDesign.md](../ExperimentDesign.md#evaluation-protocol-locked). Metric definitions stay in [docs/metrics.md](metrics.md); cluster packaging stays in [docs/clusteruy.md](clusteruy.md).

**Construct:** a correct, coherent English answer that a Spanish-L1 CEFR A1 learner can understand without help.

## What this documents

| Path | Role |
|------|------|
| **Three-rater study** (`human study-export` / `study-import`) | Blinded ordinal ratings + consensus + percent-agreement reliability |
| **Legacy single-rater** (`human export` / `import`) | Smoke / historical only — not blind; mutates generation `full.csv` |
| **LLM-judge seam** | Provider-neutral export/import only — see `AGENTS.md` §10 / `docs/metrics.md` |

The human guardrail (adequacy non-inferiority, automatic-vs-human validation) is **human-sample-limited**. Quality-preservation claims stay limited to that sample until imported judge scores agree acceptably with human consensus.

## CLI

```bash
source venv/bin/activate

# From an assessment bundle (kind: assessment)
python -m slm_experiments human study-export --assessment-run-id <id> \
  [--sample 100] [--calibration 10] [--seed 42]

# After raters fill blinded sheets (long-format ratings CSV)
python -m slm_experiments human study-import --assessment-run-id <id> --ratings <csv>

# Legacy (smoke only)
python -m slm_experiments human export --run-id <generation_id> [--sample 60]
python -m slm_experiments human import --run-id <generation_id> --tags <csv>
```

## Study artifacts (`{assessment}/study/`)

| Artifact | Audience | Contents |
|----------|----------|----------|
| `source_key.csv` | Private (not for raters) | `item_id` → model / family / arm / stratum / inclusion weights / source ids |
| `analysis_items.csv` | Analysts | Shared analysis sample with weights |
| `calibration_items.csv` | Pilot only | Calibration draw **excluded** from analysis |
| `rater_sheets/rater_{id}.csv` | Raters | Blind columns only: `item_id`, `prompt`, `answer`, empty 1–4 dims + notes |
| `beginner_suitability_rubric_v0.md` | Raters | Copied Spanish-anchor rubric |
| `manifest.json` | Analysts | Sample sizes, seed, strata axes, rater ids |
| `ratings.csv` | After import | Long-format validated ratings |
| `consensus.csv` | After import | Per-item medians + `human_suitable` |
| `reliability.json` | After import | Percent-agreement reliability report |

Modules: `human/study_export.py`, `study_import.py`, `reliability.py`, `rubric.py`. Rubric file: `human/rubrics/beginner_suitability_rubric_v0.md`.

**Never** write ratings back into generation runs. Export requires `manifest.kind == "assessment"`.

## Blinding and sample

- **Blind sheet columns:** `item_id`, `prompt`, `answer` (= assessment `cleaned_response`). Never `model`, `config`, `experiment_id`, CEFR, KVL, or other scorer fields.
- **Answer text:** rate the same string the assessment scorers see (`cleaned_response`), not raw `response`.
- **Default analysis n = 100** unique items after an **excluded calibration pilot** (default 10).
- **Stratification axes** on the assessment item frame: family, model, arm (baseline/intervention), CEFR band, TSAR↔CEFR-SP disagreement, truncation.
- One shared item set; **independent per-rater shuffle**; exactly one rating per `(item_id, rater_id)` (validated on import).

### Inclusion-weight caveat

`inclusion_weight` on analysis items is **conditional on the post-calibration frame** — approximately \(P(\text{select} \mid \text{not in calibration})\), not the unconditional inclusion probability against the original pool. Automatic-vs-human validation (`analysis.py`) records the same caveat and uses the weights as stored. Do not treat weighted estimates as clean Horvitz–Thompson against the pre-calibration pool.

## Rubric (`beginner_suitability_rubric_v0`)

Spanish anchors for human raters; English column keys in data. Bump `rubric_version` on any edit to the markdown file.

| Dimension | Column key | Role |
|-----------|------------|------|
| Overall A1 suitability | `overall_suitability` | Enters binary gate |
| Vocabulary accessibility | `vocabulary_accessibility` | Diagnostic only |
| Syntax accessibility | `syntax_accessibility` | Diagnostic only |
| Answer adequacy | `answer_adequacy` | Enters binary gate |

Scale: integers **1–4** only. If unsure between two anchors, choose the lower.

### Binary human-suitable

Consensus uses the **unweighted median** across raters per dimension.

\[
\texttt{human\_suitable} \iff m(\texttt{overall\_suitability}) \ge 3 \;\textbf{and}\; m(\texttt{answer\_adequacy}) \ge 3
\]

Vocabulary and syntax medians are diagnostic only — they do not enter this gate. Adequacy non-inferiority (margin **0.5** on the 1–4 scale, human subset only) is specified in ExperimentDesign’s locked protocol, not here.

## Reliability (shipped)

Reliability is **mean pairwise exact and adjacent (±1) percent agreement per dimension**, plus consensus medians.

| Statistic | Definition |
|-----------|------------|
| Exact agreement | Mean over rater pairs of the fraction of items with identical scores |
| Adjacent agreement | Mean over rater pairs of the fraction with \|diff\| ≤ 1 |
| Consensus | Per-item median across raters |
| `human_suitable_rate` | Mean of binary `human_suitable` over items with both medians |

**Krippendorff's α is not computed.** An earlier design draft specified ordinal α; the shipped study reports mean pairwise exact and adjacent (±1) percent agreement instead, and deliberately omits chance-corrected coefficients. `reliability.json` records `"krippendorff_alpha": "not computed (dropped by design)"`. Automatic-vs-human validation likewise uses percent agreement / Kendall τ-b / Spearman ρ — **no** Cohen's κ or other chance-corrected agreement coefficient.

## Downstream validation

After import, `assess analyze` (also auto at the end of `assess build`) joins study consensus into `analysis/validation.csv` and `analysis/adequacy_noninferiority.csv`. CEFR-SP is the scorer under validation; TSAR is reported as a diagnostic only. Details: ExperimentDesign Evaluation Protocol + `evaluation/assessment/analysis.py`.

## References in-repo

- Rubric draft (plan artifact, reliability reconciled): `thoughts/plans/human-study-rubric-draft-v0.md`
- Plan Decision 4: `thoughts/plans/evaluation-stack-plan.md`
- Shipped rubric: `src/slm_experiments/human/rubrics/beginner_suitability_rubric_v0.md`
