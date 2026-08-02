# Which CEFR scorer should be PRIMARY? — Gold-standard grounding + applicability to this thesis

**Ticket:** [RESEARCH: which CEFR scorer (CEFR-SP, TSAR, or both) to take as the PRIMARY measure](https://github.com/srobaina99/SLMs-experiments/issues/11) · **Map:** [Wayfinder: revise the beginner-suitability evaluation-stack plan](https://github.com/srobaina99/SLMs-experiments/issues/2)
**Date:** 2026-07-19 · **Type:** AFK research · Feeds the primary-endpoint decision in [#7](https://github.com/srobaina99/SLMs-experiments/issues/7); confirms/flags [#5](https://github.com/srobaina99/SLMs-experiments/issues/5) for synthesis in [#10](https://github.com/srobaina99/SLMs-experiments/issues/10)

**Locked construct:** *a correct, coherent English answer that a Spanish-L1 CEFR A1 learner can understand without help.* This ticket concerns only the **English-difficulty scorer** that backs the binary A1 gate (`meets_a1_criteria`). Spanish-L1 familiarity stays with KVL; answer adequacy stays with the human/judge layer.

**Primary sources this session:** TSAR 2025 findings paper — [aclanthology.org/2025.tsar-1.8](https://aclanthology.org/2025.tsar-1.8/) (Tables 1, 8, 6, and the annotator-agreement table read from the PDF); TSAR official evaluation repo [tsar-workshop/tsar-2025-shared-task](https://github.com/tsar-workshop/tsar-2025-shared-task); Arase et al. EMNLP 2022 CEFR-SP; prior repo notes `2026-07-14_cefr-l2-grading-sota-2026.md`, `2026-07-14_cefr-sp-sota-status.md`, `2026-07-19_cefr-sp-vs-tsar-ensemble.md`.

---

## 1. Recommendation (one paragraph)

**Keep Arase CEFR-SP as the PRIMARY A1 gate — this *confirms* #5's "both", now on gold-standard evidence rather than integration convenience.** The decisive fact is on the A1 class, which *is* this thesis's primary endpoint: the CEFR-SP classifier is explicitly rebalanced for A1 scarcity and reports **A1 F1 ≈ 0.78** on its own expert-annotated test set, whereas the TSAR 2025 **confidence-based ensemble** — the exact configuration published and imported as the "TSAR evaluator" — reports **A1 F1 = 0.00 on the test set and 0.40 on validation** (Findings Tables 1 and 8). TSAR's headline weighted-F1 of 0.89 / AdjAcc 0.99 / RMSE 0.34 is real but is carried entirely by the B1–C2 classes that dominate its Cambridge-Exams/ELG document test set; it buys nothing for a binary "is this A1?" decision and, as configured, is actively A1-blind. **New, sharper instruction for #10 / the assessment-bundle spec:** if TSAR is kept as the post-hoc cross-check (still recommended), it must be used as an *ordinal / simplification-direction* diagnostic only — **never** as an A1 classifier — and if any A1 discrimination is wanted from the TSAR family, use the single `TRAIN_DOC_EN` model (A1 F1 = 0.80), *not* the confidence ensemble.

---

## 2. Gold-standard grounding — what each scorer is validated against, and how well

Neither scorer *is* the gold standard; both are automatic classifiers judged against expert-annotated CEFR corpora. The question is which corpus, and how well each agrees — **on A1 specifically**, not just in aggregate.

| | **Arase CEFR-SP** (live gate) | **TSAR 2025 ModernBERT confidence ensemble** (plan candidate) |
|---|---|---|
| **Ground-truth corpus** | CEFR-SP: 17,676 English sentences, 2 English-education professionals (Pearson r ≈ 0.73–0.75 vs a senior expert), 6-level CEFR. A1 is the smallest class (n≈771 total). Register: Newsela/Wiki/SCoRE, 5–30 words. | UniversalCEFR English subsets. **Test/validation = Cambridge-Exams + ELG-CEFR-EN document-level** splits (stratified, 15%/15%). Training mixes CEFR-SP + ReadMe++(EN) + multilingual reference data. |
| **Task type** | Type 2A ARA, **sentence-level** → project aggregates sentences to a document mean-ordinal → nearest level. | Type 2A ARA, **document-level** (its strongest granularity); ensemble picks the label of the highest-confidence member model. |
| **Aggregate agreement w/ ground truth** | Macro-F1 **84.5% ± 0.7**, QWK **0.628** on the CEFR-SP test set (still the highest published figure on that set as of 2026). | Weighted-F1 **0.89**, AdjAcc **0.99**, RMSE **0.34** on the Cambridge/ELG test set. |
| **A1-class agreement (the endpoint that matters here)** | **A1 F1 ≈ 0.78** with loss-weighting (the released config); **0.00** without — the model is *deliberately* rebalanced so A1 works. | **A1 F1 = 0.00 (test) / 0.40 (val)** for the adopted confidence ensemble. Best single member `TRAIN_DOC_EN` = **0.80**; `TRAIN_DOC_SENT_EN` = 0.00; `REFERENCE_ALLLANG` = 0.50; majority-vote = 0.50. |
| **What the headline number hides** | 84.5% is in-domain (same register/protocol as train); generalization to short chatty answers is unknown and likely lower. A1 tail is statistically fragile (test n≈111 A1). | 0.89 weighted-F1 is **weighted by class frequency** on a corpus where A1 documents are near-absent, so the aggregate is essentially a B1–C2 score. AdjAcc 0.99 counts an A1→A2 error as "correct". |

**Ground-truth reliability ceiling (both are chasing a noisy target).** TSAR's own inter-annotator numbers on their reference simplifications are modest: exact-level Acc ≈ 0.61–0.65, Spearman ρ ≈ 0.37–0.68, RMSE ≈ 0.59–0.62 (AdjAcc = 1.00). CEFR-SP's annotators agree at r ≈ 0.73–0.75. So *exact* CEFR level is genuinely hard even for experts — which is exactly why a **binary A1** decision (the thesis endpoint) is more defensible than a 6-way level, and why adjacency-based metrics (AdjAcc, RMSE) flatter a scorer that we would be using in its least-flattered regime (the A1 boundary).

**Grounding verdict:** On the metric this thesis actually reports — the A1 decision — CEFR-SP is validated at ≈0.78 and engineered for it; the TSAR *ensemble* is validated at 0.00–0.40 and its strong aggregate comes from classes the thesis does not gate on. "Best against ground truth" therefore favors **CEFR-SP for the A1 gate**, even though TSAR looks stronger on the whole-scale headline.

---

## 3. Applicability to *this* thesis — construct validity here, regardless of accuracy elsewhere

The construct is a **binary A1 judgment** on **short (1–3 sentence, often ~10-word) generated English QA answers** meant for a Spanish-L1 A1 reader.

1. **Domain/length OOD for both.** CEFR-SP trains on 5–30-word news/Wiki/SCoRE sentences (A1 mean ≈ 7.7 words); TSAR trains/tests on Cambridge/ELG *documents*. Neither was built for ultra-short conversational LLM output. This is a shared limitation, not a differentiator — and it is why the human study (#9) and judge layer exist as the real quality check. Do **not** let either scorer's headline stand in as validation of the construct.

2. **Granularity fit favors CEFR-SP for short answers.** A "document-level" label is TSAR's advantage, but for a 1–3-sentence answer the document ≈ a sentence or two, so the doc-level edge largely evaporates. CEFR-SP's sentence-level scoring + mean-ordinal aggregation degrades more gracefully to very short inputs than a doc-trained ensemble does.

3. **The A1 class is the whole ballgame, and TSAR-as-configured cannot see it.** For a binary A1 gate you need A1 precision/recall, not aggregate F1. The adopted TSAR ensemble's A1 F1 = 0.00 (test) is disqualifying *as a primary A1 classifier here* independent of its accuracy on B1–C2 elsewhere. CEFR-SP's A1 F1 ≈ 0.78 is imperfect but usable and honestly caveated.

4. **Neither encodes the Spanish-L1 side** — that is KVL + humans, unchanged. Correctness/coherence is the judge/human layer, unchanged. So the scorer choice is narrowly about English A1 difficulty, and nothing about TSAR closes the L1 or adequacy gap.

**Applicability verdict:** For a binary A1 gate on short answers, CEFR-SP is the more construct-valid *primary*. TSAR is construct-valid only as a *secondary ordinal/direction* signal, and only if its A1-blind ensemble output is not mistaken for an A1 label.

---

## 4. Recommendation and conditions

**Primary scorer: Arase CEFR-SP.** `meets_a1_criteria == (cefr_sp_level == "A1")` stays as-is. This **confirms #5** ("both", CEFR-SP primary) — it does **not** reopen it — but replaces the earlier "keep it because it's already wired / no worse on the A1 tail" rationale with a stronger one: *on gold-standard A1 agreement the CEFR-SP classifier is materially better than the TSAR ensemble, which is A1-blind as published.*

**Secondary (assessment-only) scorer: TSAR ModernBERT ensemble — kept, with two hard conditions that #10 and the assessment-bundle spec must encode:**

- **C1 — Never an A1 gate.** The `cefr_tsar_*` fields feed ordinal / disagreement diagnostics and simplification-direction analysis only. Do not derive any A1 pass/fail from the confidence ensemble; its A1 F1 = 0.00 on the test split.
- **C2 — Pin the exact config and name the A1 caveat in the plan.** Pin the three HF model SHAs (`AbdullahBarayan/ModernBERT-base-doc_en-Cefr`, `-doc_sent_en-Cefr`, `-reference_AllLang2-Cefr2`) and the confidence-max aggregation. If, later, any A1-level TSAR signal is genuinely wanted, use the single `TRAIN_DOC_EN` model (A1 F1 = 0.80) — documented as a deliberate deviation from the official ensemble.

**One-word answer if #7 wants one:** **both**, CEFR-SP primary — same as #5, now evidence-backed.

---

## 5. Divergence / flags for synthesis (#10)

- **No divergence from #5's fork** (both / CEFR-SP primary). This ticket *strengthens* it.
- **Correction to prior note `2026-07-19_cefr-sp-vs-tsar-ensemble.md`:** that note cited TSAR "A1 F1 ≈ 0.50". The primary source is more damning: the *adopted confidence ensemble* is **A1 F1 = 0.00 (test) / 0.40 (val)**; 0.50 is the `REFERENCE_ALLLANG` single-model / majority-vote figure, not the deployed evaluator. #10 should carry the 0.00/0.40 number and the "A1-blind ensemble" framing.
- **Plan spec impact:** wherever the plan describes the TSAR scorer, it must (a) label it assessment-only, (b) state the A1-F1=0.00 ensemble caveat, (c) pin SHAs + aggregation. This dovetails with the execution-model decision (#6, in-process behind extras) and the assessment-bundle design (#8).
- **Endpoint-hierarchy input for #7:** the primary A1 endpoint is CEFR-SP's binary A1; TSAR contributes an ordinal cross-check and human-agreement comparison, not a competing gate. Any multiplicity correction should treat CEFR-SP A1 as the confirmatory endpoint and TSAR/readability as descriptive/secondary.

---

## 6. Evidence table (primary-source numbers, TSAR Findings Tables 1 & 8)

| Setup | A1 | A2 | B1 | B2 | C1 | C2 | Weighted-F1 | AdjAcc | RMSE |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN_DOC_EN (test) | **0.80** | 0.90 | 0.84 | 0.84 | 0.74 | 0.83 | 0.83 | 0.97 | 0.50 |
| TRAIN_DOC_SENT_EN (test) | 0.00 | 0.83 | 0.90 | 0.94 | 0.85 | 0.86 | 0.86 | 0.99 | 0.38 |
| REFERENCE_ALLLANG (test) | 0.50 | 0.86 | 0.89 | 0.97 | 0.88 | 0.88 | 0.89 | 1.00 | 0.32 |
| MAJORITY VOTE (test) | 0.50 | 0.87 | 0.92 | 0.95 | 0.85 | 0.86 | 0.89 | 0.99 | 0.35 |
| **CONFIDENCE-BASED (test, adopted)** | **0.00** | 0.89 | 0.94 | 0.94 | 0.87 | 0.89 | **0.89** | 0.99 | **0.34** |
| CONFIDENCE-BASED (validation) | **0.40** | 0.88 | 0.87 | 0.84 | 0.86 | 0.98 | 0.87 | 0.99 | 0.39 |

CEFR-SP (Arase 2022, own test set): macro-F1 **0.845**, QWK **0.628**; **A1 F1 = 0.78** with loss-weighting, **0.00** without.

---

## 7. Pointers

- Live gate: `src/slm_experiments/evaluation/a1_criteria.py`, `evaluation/cefr_sp.py`, `docs/metrics.md` § CEFR-SP
- TSAR Findings (Tables 1/8): [aclanthology.org/2025.tsar-1.8](https://aclanthology.org/2025.tsar-1.8/) · eval repo: [github.com/tsar-workshop/tsar-2025-shared-task](https://github.com/tsar-workshop/tsar-2025-shared-task)
- CEFR-SP: Arase, Uchida & Kajiwara (2022) [aclanthology.org/2022.emnlp-main.416](https://aclanthology.org/2022.emnlp-main.416/) · repo [github.com/yukiar/CEFR-SP](https://github.com/yukiar/CEFR-SP)
- UniversalCEFR (source of TSAR train/test splits): Imperial et al. (2025) [aclanthology.org/2025.emnlp-main.491](https://aclanthology.org/2025.emnlp-main.491/)
- Prior repo notes: `thoughts/shared/research/2026-07-19_cefr-sp-vs-tsar-ensemble.md` (#4), `2026-07-14_cefr-l2-grading-sota-2026.md`, `2026-07-14_cefr-sp-sota-status.md`
