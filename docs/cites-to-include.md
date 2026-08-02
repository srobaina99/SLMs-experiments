# Cites to Include — Text Difficulty / Readability

Bibliography and reading list for text-difficulty / CEFR evaluation. The live primary A1 gate is **CEFR-SP** (`meets_a1_criteria` when `cefr_sp_level == "A1"`); FK / Fog / Spache are legacy descriptive diagnostics, not the gate. Full locked protocol (construct, endpoint hierarchy, paired CEFR-SP ordinal delta, in-run baselines, adequacy margin 0.5, 10k percentile bootstrap, no multiplicity correction, human-sample quality scope): [ExperimentDesign.md — Evaluation Protocol (locked)](../ExperimentDesign.md#evaluation-protocol-locked). See also `docs/metrics.md` (current metrics) and `data/kvl/README.md` (KVL citations).

**Thesis context:** we score *short model-generated beginner English answers*, not long reading passages. Prefer sentence-level / short-text work over document-level CEFR alone.

**2026 SOTA (short sentences) — one-line map:** gold *resource* = CEFR-SP; closest *setting* to our outputs = Ace-CEFR; practical open *classifier* = TSAR 2025 ModernBERT (doc+sent); framing that encoders beat zero-shot LLMs and sentence ≪ document = Matsumura & Arase 2026 / UniversalCEFR. Full notes: `thoughts/shared/research/2026-07-14_cefr-l2-grading-sota-2026.md`.

---

## Suggested metric stack (relative to the locked protocol)

Framing relative to the locked protocol: **CEFR-SP is already live** as the primary in-pipeline A1 gate; FK/Fog/Spache stay legacy/descriptive; the **TSAR ModernBERT ensemble** is the shipped assessment-only ordinal cross-check (never an A1 gate); **KVL v2** is the shipped assessment-only occurrence-level Spanish-L1 vocabulary secondary. Candidate additions beyond that contract:

| Priority | Metric | Role | Primary cite |
|----------|--------|------|--------------|
| Keep (live primary) | CEFR-SP document A1 | Primary gate → `meets_a1_criteria` / `cefr_sp_*` | Arase et al. 2022; already in `docs/metrics.md` |
| Keep (legacy) | FK / Fog / Spache | Legacy / descriptive readability | Already in `docs/metrics.md` |
| Keep (in-pipeline secondary) | KVL / GLMM v1 | Learner-L1 vocab difficulty (surface-form) | `data/kvl/README.md` |
| Keep (assessment secondary) | KVL v2 | Occurrence-level lemmatized Spanish-L1 KVL | `docs/metrics.md` |
| Keep (assessment secondary) | TSAR ModernBERT ensemble | Ordinal / disagreement cross-check — **never an A1 gate** | Alva-Manchego et al. 2025 (TSAR evaluator) |
| 1 | Short conversational CEFR (Ace-CEFR–style) | Domain-matched check for ultra-short / chatty answers | Kogan et al. 2025 |
| 2 | SLE / ε-SLE | Continuous simplicity (reference-free; *not* CEFR labels) | Cripwell et al. 2023 |
| 3 | EVP / EFLLex A1 coverage % | Interpretable vocab profile (alongside KVL) | Capel 2010/2012; Dürlich & François 2018 |
| 4 (optional) | GPT-4 continuous difficulty | Human-aligned check on a subsample | Trott 2024 |
| 5 (optional) | MeaningBERT | Meaning preservation vs “simple but wrong” | Beauchemin et al. 2023 |

Wire path for in-pipeline metrics: `TextEvaluator` → `pipeline.py` → `result.py` → `run_store.py`. Assessment-only scorers (TSAR, KVL v2) write into the assessment bundle via `assess build`.

---

## Reading list (priority order)

Read these first — they map directly to the stack above. Items marked **(short-text SOTA)** are the 2025–2026 frontier for *sentence / short* CEFR grading.

### 1. Arase, Uchida & Kajiwara (2022) — CEFR-SP *(must read · short-text SOTA resource)*

Arase, Y., Uchida, S., & Kajiwara, T. (2022). CEFR-Based Sentence Difficulty Annotation and Assessment. In *Proceedings of EMNLP 2022*, pp. 6206–6219.  
https://aclanthology.org/2022.emnlp-main.416/ · DOI: [10.18653/v1/2022.emnlp-main.416](https://doi.org/10.18653/v1/2022.emnlp-main.416)  
Resources: https://github.com/yukiar/CEFR-SP · pretrained: https://zenodo.org/records/7234096

**Why:** De facto English *sentence-level* CEFR gold (~17k expert labels; ~84.5% macro-F1 in-domain). Still the resource later SOTA work trains/evaluates on. Use for annotation criteria, A1/C2 imbalance, and adjacency reporting.

### 2. Kogan et al. (2025) — Ace-CEFR *(must read · short-text SOTA setting)*

Kogan, D., Schumacher, M., Nguyen, S., Suzuki, M., Smith, M., Bellows, C. S., & Bernstein, J. (2025). Ace-CEFR: A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications. arXiv:2506.14046.  
https://arxiv.org/abs/2506.14046 · OpenReview: https://openreview.net/forum?id=e1bM1YofLh

**Why:** Only purpose-built dataset for *short conversational / LLM-like* English difficulty (~12-word passages). Fine-tuned BERT can beat human expert consistency (MSE ≈ 0.37 vs human ≈ 0.75). Closest published setting to our short generated answers; CEFR-SP alone is more news/Wiki/textbook-like.

### 3. Alva-Manchego et al. (2025) — TSAR 2025 findings + ModernBERT CEFR evaluator *(must read · short-text SOTA classifier)*

Alva-Manchego, F., Stodden, R., Imperial, J. M., Barayan, A., North, K., & Tayyar Madabushi, H. (2025). Findings of the TSAR 2025 Shared Task on Readability-Controlled Text Simplification. In *Proceedings of TSAR 2025*, pp. 116–130.  
https://aclanthology.org/2025.tsar-1.8/ · DOI: [10.18653/v1/2025.tsar-1.8](https://doi.org/10.18653/v1/2025.tsar-1.8)  
Evaluator (open): https://huggingface.co/AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr  
Task repo: https://github.com/tsar-workshop/tsar-2025-shared-task

**Why:** Current practical open English CEFR classifier for shared-task evaluation (ModernBERT fine-tuned on CEFR-SP + ReadMe++ + exam passages). Prefer this (or CEFR-SP–trained encoders) over zero-shot LLM judges for categorical CEFR. Note: A1 remains the weakest class; ultra-short answers may still be OOD.

### 4. Matsumura & Arase (2026) — multi-granular CEFR benchmark *(must read · short-text SOTA framing)*

Matsumura, E., & Arase, Y. (2026). A Benchmark Study of Multi-Granular CEFR Level Assessment. In *Proceedings of the Annual Meeting of the Association for Natural Language Processing (ANLP)*, Japan.  
https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf

**Why:** Mid-2026 evidence that for CEFR *classification*, fine-tuned encoders match or beat decoder-only LLMs, and **sentence-level is much harder** than document-level (~0.55–0.59 vs ~0.90 macro-F1 on UniversalCEFR English). Cite when claiming why we use a sentence CEFR model, not passage tools or prompted GPT alone. (Workshop/ANLP venue — not ACL peer review.)

### 5. Imperial et al. (2025) — UniversalCEFR *(must read · short-text SOTA benchmark)*

Imperial, J. M., Barayan, A., Stodden, R., Wilkens, R., et al. (2025). UniversalCEFR: Enabling Open Multilingual Research on Language Proficiency Assessment. In *Proceedings of EMNLP 2025*, pp. 9703–9755.  
https://aclanthology.org/2025.emnlp-main.491/ · DOI: [10.18653/v1/2025.emnlp-main.491](https://doi.org/10.18653/v1/2025.emnlp-main.491) · Site: https://universalcefr.github.io/

**Why:** Largest open CEFR compilation (~505k texts; includes CEFR-SP). Shows fine-tuned encoders beat prompted LLMs for CEFR labels. Cite for SOTA framing; still prefer CEFR-SP / Ace-CEFR / ModernBERT for *our* short outputs.

### 6. Naous et al. (2024) — ReadMe++ *(recommended · short-text SOTA / multi-domain)*

Naous, T., Ryan, M. J., Lavrouk, A., Chandra, M., & Xu, W. (2024). ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment. In *Proceedings of EMNLP 2024*, pp. 12230–12266.  
https://aclanthology.org/2024.emnlp-main.682/

**Why:** Multi-domain *sentence* CEFR readability (incl. English); training source for the TSAR ModernBERT evaluator alongside CEFR-SP. Useful when arguing cross-domain fragility of single-corpus classifiers.

### 7. Cripwell, Legrand & Gardent (2023) — SLE *(must read)*

Cripwell, L., Legrand, J., & Gardent, C. (2023). Simplicity Level Estimate (SLE): A Learned Reference-Less Metric for Sentence Simplification. In *Proceedings of EMNLP 2023*, pp. 12053–12059.  
https://aclanthology.org/2023.emnlp-main.739/ · DOI: [10.18653/v1/2023.emnlp-main.739](https://doi.org/10.18653/v1/2023.emnlp-main.739)  
Model: https://huggingface.co/liamcripwell/sle-base · Code: https://github.com/liamcripwell/sle

**Why:** Reference-free continuous *simplicity* (Newsela-trained), not CEFR labels. ε-SLE is the right framing for “how close to beginner simplicity?” without gold simplified references — complementary to CEFR classifiers.

### 8. Capel (2010, 2012) — English Vocabulary Profile *(must read)*

Capel, A. (2010). A1–B2 vocabulary: Insights and issues arising from the English Profile Wordlists project. *English Profile Journal, 1*(1), 1–12.  
https://doi.org/10.1017/S2041536210000048

Capel, A. (2012). Completing the English Vocabulary Profile: C1 and C2 vocabulary. *English Profile Journal, 3*, 1–14.  
https://doi.org/10.1017/S2041536212000013

**Why:** Defines CEFR-tagged vocabulary from learner evidence. Needed to interpret / upgrade the 487-word A1 list and any EVP-based coverage metric (e.g. Text Inspector). Note: EVP is largely production evidence, not pure reading difficulty.

### 9. Dürlich & François (2018) — EFLLex *(must read for vocab coverage)*

Dürlich, L., & François, T. (2018). EFLLex: A Graded Lexical Resource for Learners of English as a Foreign Language. In *Proceedings of LREC 2018*.  
https://aclanthology.org/L18-1140/

**Why:** Textbook-derived CEFR word frequencies. Practical for “% content words ≤ A1” — interpretable secondary metric used in recent ESL simplification work (e.g. Li et al. 2025).

### 10. Rooein, Röttger, Shaitarova & Hovy (2024) — Beyond Flesch-Kincaid *(must read for framing)*

Rooein, D., Röttger, P., Shaitarova, A., & Hovy, D. (2024). Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts. In *Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024)*, pp. 51–67.  
https://aclanthology.org/2024.bea-1.5/ · arXiv: [2405.09482](https://arxiv.org/abs/2405.09482)

**Why:** Documents that static formulas (including FK) are crude/brittle for educational text difficulty; motivates neural / prompt-based metrics as upgrades while keeping formulas as baselines.

### 11. Trott (2024) — GPT-4 readability *(optional upgrade)*

Trott, S. (2024). Measuring and Modifying the Readability of English Texts with GPT-4. In *Proceedings of the Third Workshop on Text Simplification, Accessibility and Readability (TSAR 2024)*, pp. 130–144.  
https://aclanthology.org/2024.tsar-1.13/

**Why:** GPT-4 Turbo zero-shot readability correlates with human judgments at **r ≈ 0.76** on CLEAR-style data — better than classic formulas. Good subsample validator, not a replacement for a local CEFR classifier.

### 12. Beauchemin, Saggion & Khoury (2023) — MeaningBERT *(optional)*

Beauchemin, D., Saggion, H., & Khoury, R. (2023). MeaningBERT: Assessing meaning preservation between sentences. *Frontiers in Artificial Intelligence, 6*, 1223924.  
https://doi.org/10.3389/frai.2023.1223924 · HF: https://huggingface.co/davebulaval/MeaningBERT

**Why:** Separates simplicity from meaning preservation (also used in TSAR 2025 eval). Useful if interventions make outputs “simple but off-topic / wrong” (pairs with human `response_appropriateness`).

### 13. Li, Arase & Crespi (2025) — ESL-aligned simplification rewards *(recommended)*

Li, X., Arase, Y., & Crespi, N. (2025). Aligning Sentence Simplification with ESL Learner’s Proficiency for Language Acquisition. In *Proceedings of NAACL 2025*.  
https://aclanthology.org/2025.naacl-long.21/

**Why:** Uses CEFR-SP + EFLLex as evaluation/reward signals for ESL-targeted simplification — closest methodological cousin to guided decoding / KVL beam steered toward beginner English.

### 14. Liu, Jin & Lee (2025) — sentence ARA survey/comparison *(optional · methods)*

Liu, F., Jin, T., & Lee, J. S. Y. (2025). Automatic readability assessment for sentences: neural, hybrid and large language models. *Language Resources and Evaluation*.  
https://doi.org/10.1007/s10579-024-09800-5

**Why:** Compares neural / hybrid / LLM approaches for *sentence* readability; supports “encoder/hybrid > naive LLM” narrative alongside Matsumura & Arase.

---

## Supporting cites (lit review / construct validity)

### Classic formulas (already used)

- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology, 32*(3), 221–233.
- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of New Readability Formulas…* Research Branch Report 8-75.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- Spache, G. (1953). A new readability formula for primary-grade reading materials. *The Elementary School Journal, 53*(7), 410–413.

### L2 / multi-feature readability

- Crossley, S. A., Greenfield, J., & McNamara, D. S. (2008). Assessing text readability using cognitively based indices. *TESOL Quarterly, 42*(3), 475–493. https://doi.org/10.1002/j.1545-7249.2008.tb00142.x  
  *(Often cited with Crossley & McNamara L2 Coh-Metrix work — why formulas alone are weak for EFL.)*
- Crossley, S. A., Skalicky, S., & Dascalu, M. (2019/2017 line). Moving beyond classic readability formulas… *Journal of Research in Reading*. https://doi.org/10.1111/1467-9817.12283
- Crossley, S. A., et al. (2023). A large-scaled corpus for assessing text readability (CLEAR). *Behavior Research Methods, 55*, 967–986. https://doi.org/10.3758/s13428-022-01802-x

### Lexical sophistication toolkits (if extending `TextEvaluator`)

- Kyle, K., & Crossley, S. A. (2015). Automatically assessing lexical sophistication… *TESOL Quarterly, 49*(4), 757–786. https://doi.org/10.1002/tesq.194
- Kyle, K., Crossley, S. A., & Berger, C. (2018). The tool for the automatic analysis of lexical sophistication (TAALES): Version 2.0. *Behavior Research Methods, 50*, 1030–1046. https://doi.org/10.3758/s13428-017-0924-4

### Word-level CEFR / EVP + LLMs

- Bannò, S., Knill, K., & Gales, M. (2025). Exploiting the English Vocabulary Profile for L2 word-level vocabulary assessment with LLMs. In *Proceedings of BEA 2025*. https://aclanthology.org/2025.bea-1.45/

### Simplification metrics (only if you have references)

- Maddela, M., et al. (2023). LENS: A Learnable Evaluation Metric for Text Simplification. In *Proceedings of ACL 2023*. https://aclanthology.org/2023.acl-long.905/  
  *(Strong learned metric, but needs references — usually N/A for our free-form answers.)*

### Official CEFR framing

- Council of Europe. (2020). *Common European Framework of Reference for Languages… Companion volume*. https://rm.coe.int/common-european-framework-of-reference-for-languages-learning-teaching/16809ea0d4  
  *(Remind readers: CEFR describes learner can-do ability; text leveling is an inference.)*

---

## Minimal reading set (if short on time)

**Short-sentence CEFR SOTA path:**
1. Arase et al. (2022) — CEFR-SP (resource)  
2. Kogan et al. (2025) — Ace-CEFR (closest setting)  
3. Alva-Manchego et al. (2025) — TSAR / ModernBERT evaluator (practical classifier)  
4. Matsumura & Arase (2026) — encoders vs LLMs; sentence ≪ document  

**Plus (stack / framing):**
5. Cripwell et al. (2023) — SLE (simplicity, not CEFR)  
6. Capel (2010) — EVP  
7. Rooein et al. (2024) — Beyond Flesch-Kincaid  
8. Trott (2024) — only if using GPT-4 as a secondary score  

---

## Notes for thesis writing

- Report **adjacency / ±1 CEFR** and/or continuous SLE, not only exact A1 match — human CEFR agreement is imperfect; A1 is the weakest class in TSAR-style evaluators.
- Keep FK/Fog/Spache as **legacy / descriptive** diagnostics — they are **not** the A1 gate (that is CEFR-SP); see the locked protocol in `ExperimentDesign.md`.
- Do **not** drop KVL — it is learner-L1 grounded; neural CEFR/SLE are complementary text-side scores.
- Prompted LLMs alone are weak CEFR *labelers* (UniversalCEFR; Matsumura & Arase 2026); prefer fine-tuned encoders (CEFR-SP checkpoint or TSAR ModernBERT) for categorical CEFR.
- Prefer **sentence / short-text** cites (CEFR-SP, Ace-CEFR, ReadMe++, TSAR evaluator) over document-only CEFR tools when justifying the metric choice for our outputs.
