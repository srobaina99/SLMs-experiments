# CEFR-SP: SOTA Status Assessment (July 2026)

**Query:** How gold-standard / state-of-the-art are the corpus and classifier from Arase, Uchida & Kajiwara (2022)?  
**Context:** Evaluating short beginner-English answers from small LMs; considering CEFR-SP as a readability proxy upgrade over Flesch-Kincaid.

---

## 1. What Exactly Is CEFR-SP?

### 1a. The Corpus (CEFR-SP)

- **17,676 unique English sentences** drawn from three sources: Newsela-Auto (news), Wiki-Auto (Wikipedia), and SCoRE (EFL classroom sentences).
- Annotated by **two English-education professionals** (selected from eight after a calibration trial; Pearson r = 0.75 and 0.73 vs. a senior expert).
- Labels: 6-point CEFR scale (A1–C2). A sentence may carry two labels (if annotators differed by ≤1 level, both are accepted). Final dataset: **27,841 labels** for the 17,676 sentences.
- Corpus distribution is skewed: **A1 is the smallest class** (n=771 train), C2 is scarcer still (n=100 train). B1–B2 dominate.
- Sentence length filter: **5–30 words**; only standalone, context-free sentences (no named entities, no coreference dependence, first sentence of paragraphs).
- Available at <https://github.com/yukiar/CEFR-SP> (CC BY-NC-SA 4.0 for Wiki/SCoRE portions; Newsela portion requires separate access).
- A companion journal article with expanded linguistic analysis: Uchida, Arase & Kajiwara (2024), *ITL-International Journal of Applied Linguistics*, 175(1):103–126. DOI: [10.1075/itl.22018.uch](https://doi.org/10.1075/itl.22018.uch).

**Sources:** Arase et al. (2022), pp. 6206–6219. <https://aclanthology.org/2022.emnlp-main.416>

### 1b. The Assessment Model

- Architecture: **BERT-Base (cased) + metric-based prototype classification**.
  - Mean-pool BERT token embeddings (layer 11) → sentence embedding.
  - K=3 *prototypes* per CEFR level (KJ = 18 prototype vectors total), initialized from mean sentence embeddings + orthogonalization.
  - Prediction = CEFR level with highest cosine similarity to prototypes.
- **Loss weighting** (α = 0.2) rescales cross-entropy by inverse class frequency to handle A1/C2 scarcity.
- Trained 12 times with random seeds; best/worst discarded; mean reported.
- **Result: macro-F1 = 84.5% ± 0.7%, quadratic weighted κ = 0.628 ± 0.010** on the CEFR-SP test set.

---

## 2. Claims at Publication (EMNLP 2022)

### Baselines compared

| Model | Macro-F1 |
|---|---|
| BoW SVM (no loss-weighting) | 41.2% |
| BoW SVM (with loss-weighting) | 52.3% |
| kNN (frozen BERT embeddings) | 38.8% |
| BERT fine-tuned (no loss-weighting) | 71.7% |
| BERT fine-tuned (with loss-weighting) | 82.5% |
| **Proposed (prototypes + loss-weighting + init)** | **84.5%** |

The paper claimed SOTA for sentence-level CEFR classification. The comparison baselines were internal (no existing sentence-level CEFR classifier existed prior to this paper — this was the **first** English sentence-level CEFR corpus).

### Caveats explicitly acknowledged

1. **A1 scarcity**: With loss-weighting OFF, the proposed model scores 0% on A1 (misclassified to adjacent levels). With loss-weighting ON: A1 F1 = 78.0% but confidence interval is relatively wide.
2. **No lexical analysis only**: The paper confirms sentence-level CEFR correlates with lexical level but is NOT reducible to it — grammar, length, and discourse cues all contribute.
3. **Newsela portion restricted**: Full 84.5% figure uses all three subcorpora; subsequent work with only the publicly available portions (Wiki-Auto + SCoRE) gets lower scores.
4. **Context-free assumption**: The model was designed for standalone sentences; long-form text or very short fragments may be outside the training distribution.

---

## 3. Post-2022: Follow-on Work (2023–2026)

### 3a. Papers citing CEFR-SP as a benchmark or standard resource

**Citation count: ~67 on Google Scholar (July 2026)**. One DOI aggregator reports 14, but the Google Scholar citing-papers page shows "About 67 results." The true count is in the 50–70 range across venues.

Selected key citers:

| Paper | Venue | How cited |
|---|---|---|
| Liu & Lee (2023). "Hybrid models for sentence readability assessment." | BEA workshop (ACL 2023) | Benchmark corpus; hybrid feature model achieves 0.752 macro-F1 on public subcorpora only (vs. 0.845 on all 3) — not a fair comparison |
| Naous et al. (2024). "ReadMe++: Benchmarking multilingual LMs for multi-domain readability assessment." | EMNLP 2024 | Acknowledges CEFR-SP; introduces ReadMe++ as broader multilingual alternative with rank-and-rate annotation |
| Liu, Jin & Lee (2025). "Automatic readability assessment for sentences: neural, hybrid, LLMs." | Language Resources and Evaluation (Springer, 2025) | CEFR-SP benchmark; hybrid model achieves "competitive results" on public subcorpora; encoder/hybrid > LLMs |
| Barayan et al. (2025). "Analysing zero-shot readability-controlled sentence simplification." | EMNLP 2025 / arXiv 2409.20246 | Uses CEFR-SP as evaluation set; GPT-4-Turbo and Llama-3-8B **score lower** than Arase et al. (2022) model on zero-shot CEFR assessment |
| Li, Arase & Crespi (2025). "Aligning sentence simplification with ESL learner's proficiency for language acquisition." | NAACL 2025 | Trains CEFR-SP reward model for RL-based simplification |
| Imperial et al. (2025). "UniversalCEFR." | EMNLP 2025 | Includes CEFR-SP as one of 26 corpora in mega-multilingual compilation |
| Matsumura & Arase (2026). "A Benchmark Study of Multi-Granular CEFR Level Assessment." | NLP 2026 (ANLP, March 2026) | Benchmarks encoders/LLMs on UniversalCEFR English subset; all models cluster at 0.54–0.59 macro-F1 on sentences (different test set) |
| Uchida & Negishi (2025). "Assigning CEFR-J levels to English learners' writing (CWLA)." | Research Methods in Applied Linguistics (Elsevier) | Uses CEFR-SP as foundation |
| Alva-Manchego et al. (2025). "Findings of TSAR 2025 shared task on readability-controlled text simplification." | TSAR workshop 2025 | Uses CEFR-SP as target-level annotation standard |

**Sources:**
- Barayan et al. 2025: <https://doi.org/10.48550/arxiv.2409.20246>
- Naous et al. 2024: <https://aclanthology.org/2024.emnlp-main.682>
- Liu & Lee 2023: <https://aclanthology.org/2023.bea-1.10>
- Imperial et al. 2025 (UniversalCEFR): <https://aclanthology.org/2025.emnlp-main.491>
- Matsumura & Arase 2026: <https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf>

### 3b. Papers claiming to *outperform* the CEFR-SP assessment model

**No paper has cleanly outperformed the 84.5% macro-F1 on the full CEFR-SP test set** as of July 2026. Reasons:

1. The Newsela-Auto subcorpus requires a license agreement; papers that test on only the public (Wiki-Auto + SCoRE) subcorpora cannot be compared fairly.
2. Barayan et al. (2025) explicitly tested GPT-4-Turbo, Llama-3-8B, and OpenChat-3.5 on the public CEFR-SP test portions and found they all score **lower** than Arase et al. (2022).
3. Liu et al.'s (2025) hybrid model gets 0.752 on public subcorpora. This is below 0.845, but again on fewer data.
4. The 2026 UniversalCEFR benchmark (Matsumura & Arase, 2026): best models get 0.57–0.59 macro-F1 on a cross-corpus English sentence test set — but this is a different test set drawn from 26 merged corpora, not CEFR-SP's own test split.

**Key finding: on its own test set, the CEFR-SP classifier remains the highest published figure.**

### 3c. Competing sentence-level CEFR/difficulty assessors

| System | Data | Scope | Status |
|---|---|---|---|
| **ReadMe++ / readabert-en** (Naous et al., 2024) | 9,757 sentences, 5 languages, rank-and-rate | Multilingual; English subset publicly available on HuggingFace (`tareknaous/readabert-en`) | Smaller English corpus; Krippendorff κ = 0.67–0.78; EMNLP 2024 |
| **Ace-CEFR** (Anon, OpenReview) | 890 short conversational passages (avg. 12 words) | Short, conversational English only; A1–C2 | Explicitly designed for LLM-generated short text; preprint; not yet peer-reviewed |
| **CVLA / TextInspector** | Feature-based (CEFR-J vocab lookup) | Passage-level; vocabulary analysis | Not sentence-level classifiers in the same sense; used as baselines |
| **LLM prompting (GPT-4 etc.)** | Zero-shot | General | Consistently below CEFR-SP model on CEFR-SP test set per Barayan et al. (2025) |
| **UniversalCEFR fine-tuned encoders** | 505,807 texts, 13 languages | Multilingual mega-benchmark | Best sentence-level macro-F1 ~0.59 (on a heterogeneous test set) |
| **dksysd/cefr-classifier** (HuggingFace) | DeBERTa-v3-large, HF community model | English sentences | Unofficial community model; Nov 2025 |

---

## 4. Is CEFR-SP Considered a Gold Standard?

**Short answer: For English sentence-level CEFR, yes — it is the de facto gold standard, though not the only resource.**

Evidence:
- It was the **first** professionally annotated English sentence-level CEFR corpus. Competing datasets (ReadMe++, Ace-CEFR) explicitly position themselves as complements or extensions for different use cases, not replacements.
- UniversalCEFR (EMNLP 2025) includes CEFR-SP as one of only 26 collected corpora globally across 13 languages; for English reference sentences at sentence level, it is one of only two listed (the other is `deplain-apa-sent`, which is German sentences with CEFR, not English).
- Li, Arase & Crespi (NAACL 2025) use CEFR-SP to train a reward model for ESL sentence simplification — a direct endorsement of corpus quality.
- TSAR 2025 shared task adopts CEFR-SP as the annotation standard.

**Limitations of gold-standard claim:**
- Only **two annotators** with Pearson r = 0.75/0.73 to the senior expert. Inter-annotator agreement is reasonable but not exceptionally high.
- A1 and C2 levels are **scarce** (771 and 248 sentences in the full corpus respectively). Any conclusion about the tails is statistically fragile.
- Annotation reflects **comprehension difficulty** (what level of learner can read this), not production difficulty. This distinction matters.

---

## 5. Practical Limitations for Scoring Short Generated Beginner Answers

This is the core use-case question. Several limitations apply:

### 5a. Domain / register mismatch
CEFR-SP sentences come from **news articles, Wikipedia, and an EFL textbook corpus (SCoRE)**. SLM-generated beginner answers to conversational questions are a different register — shorter, more colloquial, possibly telegraphic. No fine-tuning or domain adaptation is described in the original paper.

### 5b. Length distribution mismatch
CEFR-SP training sentences are 5–30 words. Average A1 sentence: **7.7 words**. If your generated answers are 3–6 words (typical for minimalist beginner responses), they may fall outside the training distribution for the model. The Ace-CEFR paper (which explicitly targets avg. 12-word conversational passages) notes CEFR-SP is "not representative of conversations" and "single-sentence, complete-thought" only.

### 5c. A1 class scarcity — the most critical issue for your use case
If you are scoring beginner-English answers, you want the **A1 class to be reliable**. The model achieves A1 F1 = 78.0% ± 1.3%, and this only works *with* loss weighting. The training set has only **535 A1 examples**, and the test set has **111 A1 examples**. For a binary "is this A1?" classifier (which is essentially what you need), the model's A1 recall and precision behavior should be validated on your actual data distribution, not assumed from the paper's test set.

The paper's own Table 6 shows: without loss weighting, the model scores **0% F1 on A1** (misclassifies all to A2). With loss weighting, it recovers to 78%. This sensitivity to hyperparameters (α = 0.2) means the pretrained model from the GitHub repo is likely fine, but must be used exactly as released.

### 5d. Comprehension vs. production difficulty
The corpus annotates sentences by the level of reader who can *understand* them. Your use case is evaluating sentences *generated* by a model trying to produce A1-level text. A generated sentence might use A1 vocabulary but unusual syntax — the model has no explicit training signal for this distinction.

### 5e. Calibration for threshold-based decisions
Using the CEFR-SP classifier as a binary readability proxy (e.g., "predicted A1 = passes") is reasonable as a research tool, but the classifier's output probabilities are not calibrated for your domain. A1 false-positive rate (A2 misclassified as A1) could be non-trivial.

---

## 6. Citation Impact and Citation Mode

- **Google Scholar:** ~67 citing papers (July 2026) — approximately 3 years post-publication, this is a reasonable but not exceptional count for a niche NLP-edu paper.
- **Semantic Scholar API:** ~14 (likely an undercount due to PDF parsing gaps; Google Scholar is broader).
- **GitHub stars:** 60 stars (as of fetch date).

**How it is cited (bifurcated):**
1. **As a corpus** (majority): Papers cite CEFR-SP to describe using the dataset for training, evaluation, or as a benchmark. Examples: Liu & Lee (2023), Liu et al. (2025), Barayan et al. (2025), Li et al. (NAACL 2025), UniversalCEFR (2025), TSAR 2025.
2. **As a classifier/model** (minority): Papers compare against or build on the assessment model itself. The model is mostly used as a baseline or black-box evaluator, not as a foundation to improve.

The corpus-versus-model split suggests the community sees CEFR-SP primarily as a **data resource** and the assessment model as a **baseline that holds its own** rather than a springboard.

---

## 7. Plain-English Verdict

### Is the corpus gold-standard?

**Yes, within its niche.** CEFR-SP is the most cited English sentence-level CEFR corpus, has been canonized into UniversalCEFR, and continues to be used in 2025–2026 shared tasks. Its professional annotation (though only two annotators) is significantly more rigorous than document-level heuristics or crowd-sourced ratings.

**However**, for your use case of scoring short generated beginner answers, it has important caveats:
- It was built for reading-comprehension difficulty, not production difficulty.
- Short (<5 word) generated outputs may be out-of-distribution.
- A1 examples are sparse; model behavior at the tail of the distribution should not be over-trusted.

### Is the classifier still SOTA?

**Yes, on its own test set.** No published paper has cleanly beaten 84.5% macro-F1 on the full CEFR-SP test set. LLMs (GPT-4, Llama-3) underperform it on zero-shot CEFR classification. The closest competitor (Liu et al., 2025 hybrid model) is evaluated on a subset and cannot be directly compared.

On the *broader* cross-corpus benchmark (UniversalCEFR 2026), sentence-level models cluster around 0.54–0.59 — but this is a harder, more heterogeneous task.

**One important qualifier:** the 84.5% is on CEFR-SP's *own* test set, which shares the same domain and annotation protocol as training data. Generalization to new domains (your use case) is unknown and likely lower.

### What should a thesis claim carefully?

1. **If using the corpus as a reference:** You may cite CEFR-SP as the leading professionally annotated English sentence-level CEFR corpus. You should note the A1 sparsity and that it covers reading-level difficulty, not production difficulty.

2. **If using the classifier as a proxy:** You may cite it as the best-published sentence-level CEFR classifier (macro-F1 84.5% on its own test set). You should **not** claim it validates A1-level outputs from your models without an explicit caveat about domain mismatch and A1-class reliability. A safer claim: "we use the CEFR-SP classifier as an automated readability proxy, acknowledging that it was trained on longer news/Wikipedia sentences and may not generalize to short conversational outputs."

3. **Do NOT claim:** "CEFR-SP scores confirm our model produces A1 English" — this over-interprets a proxy that was not validated for this domain or output format.

4. **Framing recommendation:** Position your evaluation as *pilot/proxy*, cite CEFR-SP as the best available automated tool for this purpose, and note ReadMe++ (readabert-en) and Ace-CEFR as alternative assessors worth comparing if time permits.

---

## Key References

| Reference | URL / DOI |
|---|---|
| Arase, Uchida & Kajiwara (2022). CEFR-Based Sentence Difficulty Annotation and Assessment. EMNLP 2022. | <https://aclanthology.org/2022.emnlp-main.416> — DOI: 10.18653/v1/2022.emnlp-main.416 |
| Uchida, Arase & Kajiwara (2024). Profiling English sentences based on CEFR levels. ITL-International Journal of Applied Linguistics, 175(1):103–126. | DOI: 10.1075/itl.22018.uch |
| Naous et al. (2024). ReadMe++. EMNLP 2024. | <https://aclanthology.org/2024.emnlp-main.682> |
| Liu & Lee (2023). Hybrid models for sentence readability assessment. BEA 2023. | <https://aclanthology.org/2023.bea-1.10> |
| Liu, Jin & Lee (2025). Automatic readability assessment for sentences. Language Resources and Evaluation. | <https://link.springer.com/article/10.1007/s10579-024-09800-5> |
| Barayan et al. (2025). Analysing zero-shot readability-controlled sentence simplification. EMNLP 2025. | <https://doi.org/10.48550/arxiv.2409.20246> |
| Li, Arase & Crespi (2025). Aligning sentence simplification with ESL learner's proficiency. NAACL 2025. | <https://aclanthology.org/2025.naacl-long.21> |
| Imperial et al. (2025). UniversalCEFR. EMNLP 2025. | <https://aclanthology.org/2025.emnlp-main.491> |
| Matsumura & Arase (2026). A Benchmark Study of Multi-Granular CEFR Level Assessment. NLP 2026. | <https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf> |
| Ace-CEFR (Anon, preprint). Automated Evaluation of the Linguistic Difficulty of Short Texts for LLM Applications. | <https://openreview.net/pdf?id=e1bM1YofLh> |
| CEFR-SP GitHub repository | <https://github.com/yukiar/CEFR-SP> |
| UniversalCEFR HuggingFace | <https://huggingface.co/UniversalCEFR> |
| readabert-en (ReadMe++ English classifier) | <https://huggingface.co/tareknaous/readabert-en> |
