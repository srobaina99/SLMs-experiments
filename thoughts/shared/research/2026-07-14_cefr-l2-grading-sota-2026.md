# CEFR Grading / CEFR-Level Assessment for L2 English — State of the Art (mid-2026)

**Compiled:** 2026-07-14  
**For:** SLMs-experiments thesis — short beginner-English answers from small LMs  
**Scope:** Primary sources; 2024–2026 for "current SOTA"; earlier classics when they still define practice

---

## 1. Executive Verdict — SOTA Map by Task Type (mid-2026)

| Task | Gold Standard Dataset | SOTA Method | Best Reported Metric |
|------|----------------------|-------------|---------------------|
| **Sentence-level difficulty** (reading) | CEFR-SP (Arase et al., EMNLP 2022) | Fine-tuned encoder (DeBERTa/RoBERTa) + hybrid linguistic features | 84.5% macro-F1 on CEFR-SP (Arase 2022); ~0.55–0.59 macro-F1 on UniversalCEFR (Matsumura & Arase 2026) |
| **Document/passage-level difficulty** (reading) | Cambridge Exams subset (Xia et al., 2016), ELG-CEFR-EN | Fine-tuned ModernBERT or RoBERTa-base | Up to 0.90 macro-F1 (Matsumura & Arase 2026) |
| **Multi-domain sentence** (cross-domain) | ReadMe++ (Naous et al., EMNLP 2024) | Fine-tuned XLM-R/monolingual BERT | Strong domain generalization vs. zero-shot LLMs |
| **Multilingual unified** | UniversalCEFR (Imperial et al., EMNLP 2025) | Fine-tuned XLM-R (English: ~64% doc, ~53-62% sentence) | Weighted F1 ~0.64–0.71 (doc); ~0.53–0.62 (sentence) |
| **Learner production** (essays/writing) | EFCAMDAT, CLC-FCE, MERLIN | Fine-tuned BERT/DeBERTa or Longformer; encoder >> generative | QWK 0.798 Longformer (ASAP 2.0, Aimecon 2025) |
| **Short conversational text / LLM output** | Ace-CEFR (Kogan et al., 2025) | Fine-tuned BERT-3L (MSE 0.37, beats human raters) | MSE 0.37 vs. human expert MSE 0.75 |
| **CEFR-controlled simplification** (shared task) | TSAR 2025 dataset | Multi-LLM ensembles + CEFR-as-judge (GPT-5 ensembles win) | Best: RMSE 0.000 CEFR, MeaningBERT ≥ 0.85 (EhiMeNLP, 2025) |

**Top-line takeaways:**

1. **Fine-tuned encoders (BERT/RoBERTa/ModernBERT) dominate** over zero-shot or even fine-tuned generative LLMs for CEFR *classification*. Model size and decoder-only vs. encoder-only architecture have minimal effect once fine-tuned (Matsumura & Arase 2026).
2. **Document-level CEFR is substantially easier** than sentence-level (0.90 vs. 0.57 macro-F1 on the same models/dataset; Matsumura & Arase 2026). Sentence-level remains hard due to context sparsity.
3. **The critical Arase et al. (2022) CEFR-SP** baseline (84.5% macro-F1) was achieved with domain-specific training data; newer benchmarks on UniversalCEFR are harder because they are cross-domain.
4. **For short LLM-generated text** (the thesis use case): Ace-CEFR (Kogan 2025) is the *only* purpose-built dataset and model for conversational/short text CEFR grading. Fine-tuned BERT outperforms both LLMs and human experts on its test set.
5. **LLM-as-judge / prompt-based CEFR** is competitive only with complex multi-step pipelines (TSAR 2025 top systems). Single-shot zero-shot prompting consistently underperforms fine-tuned classifiers.
6. **Alignment drift** is a real phenomenon: LLMs constrained to a CEFR level via system prompting drift toward unconstrained complexity over multi-turn dialogue (Almasi et al., BEA 2025).

---

## 2. Taxonomy of CEFR-Grading Tasks

It is essential to keep these distinct — many papers conflate them, which creates misleading comparisons.

### 2A. Automated Readability Assessment (ARA) — Text Difficulty for L2 Readers
**What it measures:** How difficult a *text* is for a reader at a given CEFR level to understand.  
**Input:** Any text (native, simplified, pedagogical).  
**Label source:** Human raters assess the *text*, not the author.  
**Subtypes:**
- Sentence-level ARA (hardest — most context-sparse)
- Paragraph/discourse-level ARA
- Document-level ARA (most reliable signals; longer texts)

### 2B. Automated Essay Scoring (AES) / Learner Proficiency Assessment
**What it measures:** The CEFR proficiency *of the learner/writer/speaker* as evidenced by their production.  
**Input:** Learner-produced text (essays, writing tasks, transcripts).  
**Label source:** Examiners assign a holistic CEFR level to the learner's *performance*.  
**Key difference from 2A:** A learner's essay may contain A1-level text that reveals B2-level understanding through structure/errors; the "level" reflects the writer, not the text surface.

### 2C. CEFR-Controlled Generation Evaluation
**What it measures:** Whether LLM-generated text *meets* a target CEFR level.  
**Input:** Short generated passages, conversational turns, chatbot responses.  
**Label source:** Expert annotators or trained classifiers rate the *generated text's difficulty*.  
**Closest to the thesis use case.**

### 2D. Learner Proficiency from Speaking Transcripts
Not covered in depth here; SOTA is separate from text-based approaches and involves prosodic features.

**Note on the thesis:** The SLMs-experiments pipeline generates short responses from small LMs and evaluates them as text-difficulty artifacts — this is **Type 2C / ARA**, not learner proficiency (2B). The readability proxy (FK/Fog/Spache + KVL) is an approximation for Type 2C CEFR grading.

---

## 3. Leading Resources (Datasets) with URLs/DOIs

### 3.1 Sentence-Level CEFR (Type 2A)

| Dataset | Size | Language | Annotation | License | URL/DOI |
|---------|------|----------|------------|---------|---------|
| **CEFR-SP** | 17k sentences | English | Expert professionals | CC BY-NC-SA 4.0 | [aclanthology.org/2022.emnlp-main.416](https://aclanthology.org/2022.emnlp-main.416/) |
| **ReadMe++** | 9,757 sentences | 5 languages (EN, AR, FR, HI, RU) | 112 domains, CEFR 1-6 | CC BY-NC-SA | [aclanthology.org/2024.emnlp-main.682](https://aclanthology.org/2024.emnlp-main.682/) |
| **Ace-CEFR** | 890 short passages | English (conversational) | Expert linguists, QWK=0.89 | Public | [arxiv.org/abs/2506.14046](https://arxiv.org/abs/2506.14046) |

**CEFR-SP** is the de facto gold standard for English sentence-level readability. 17k sentences annotated by English-education professionals on the 6-level CEFR scale. **Macro-F1 of 84.5%** was the original paper's best result. Referenced by almost all subsequent work.

**ReadMe++** (EMNLP 2024) extends to multilingual, multi-domain. 9,757 human-annotated sentences across 112 sources. Includes English. Models fine-tuned on ReadMe++ show superior cross-domain and cross-lingual transfer. Has released pre-trained classifiers per language.

**Ace-CEFR** (Kogan et al., arXiv 2025 / ACL ARR submission) is purpose-built for **short conversational texts** (avg. 12 words). 890 passages from diverse sources including LLM-generated text. Near-uniform CEFR distribution. Annotated on productive difficulty (what level of learner would produce this). **The only dataset directly analogous to short LLM outputs.**

### 3.2 Document/Passage-Level CEFR (Type 2A)

| Dataset | Size | Coverage | URL/DOI |
|---------|------|----------|---------|
| **Cambridge Exams** | 331 docs | A2-C2 | [aclanthology.org/W16-0502](https://aclanthology.org/W16-0502/) |
| **ELG-CEFR-EN** | ~500 docs | A1-C2 | Breukker (2022) via UniversalCEFR |
| **OneStopEnglish** | 567 docs (3 levels) | 3 grades | [aclanthology.org/W18-0535](https://aclanthology.org/W18-0535/) |

### 3.3 Learner Production / AES Corpora (Type 2B)

| Dataset | Size | Coverage | Notes |
|---------|------|----------|-------|
| **EFCAMDAT** | ~1.18M scripts | A1-C1 (16 levels) | Largest open-access L2 English corpus |
| **CLC-FCE** | ~1,244 scripts | B1-B2 | Cambridge Learner Corpus FCE subset |
| **MERLIN** | ~2,300 texts | A2-B2 | German, Italian, Czech (not English) |
| **ICLE-500** | 500 docs | A1-C2+ | 28 L1s; see Thwaites et al. (2024) |
| **ASAP-AES / ASAP 2.0** | Kaggle | Holistic | Popular AES benchmark; Longformer SOTA QWK=0.798 |

### 3.4 Unified Multilingual Benchmark

**UniversalCEFR** (Imperial et al., EMNLP 2025) — 505,807 CEFR-labeled texts across 13 languages, multiple granularities. Aggregates CEFR-SP, ReadMe++, Cambridge Exams, EFCAMDAT, MERLIN, and others. **The largest open CEFR benchmark as of 2026.** Provides standardized splits. Used by Matsumura & Arase (2026) for their benchmark study.

> **GitHub/Hugging Face:** [universalcefr.github.io](https://universalcefr.github.io) / [huggingface.co/UniversalCEFR](https://huggingface.co/UniversalCEFR)  
> **DOI:** [10.48550/arxiv.2506.01419](https://doi.org/10.48550/arxiv.2506.01419) (arXiv preprint); ACL: [aclanthology.org/2025.emnlp-main.491](https://aclanthology.org/2025.emnlp-main.491/)

### 3.5 Lexical/Vocabulary Resources

| Resource | Coverage | Use in CEFR research |
|----------|----------|---------------------|
| **English Vocabulary Profile (EVP)** | A1-C2 word senses | Commercial; Text Inspector uses it |
| **EFLLex** (Dürlich & François, LREC 2018) | 15,280 lemmas, A1-C1 | Open; frequency-based; textbook receptive vocab |
| **CEFR-J Wordlist** | Japanese CEFR-J levels | Used in CVLA tool |

**EFLLex** ([cental.uclouvain.be/cefrlex/efllex](https://cental.uclouvain.be/cefrlex/efllex/)) is widely used in academic CEFR research for lexical profiling. CC BY-NC-SA 4.0.

---

## 4. Leading Methods/Models with Numbers and Fair-Comparison Caveats

### 4.1 Sentence-Level CEFR Classification

#### Matsumura & Arase (ANLP 2026, not peer-reviewed)
**Benchmark study** using UniversalCEFR English data. Compared encoders, decoders, LLM2Vec.

| Model | Sentence Macro-F1 | Document Macro-F1 |
|-------|-------------------|-------------------|
| BERT-base | 0.56 | 0.86 |
| RoBERTa-base | 0.55 | **0.90** |
| RoBERTa-large | 0.57 | 0.87 |
| LLaMA-3.1-8B-Instruct | **0.59** | 0.77 |
| Qwen2.5-7B-Instruct | 0.58 | 0.79 |
| Mistral-7B-Instruct-v0.2 | 0.56 | 0.76 |
| LLM2Vec-8B | 0.50 | 0.85 |

**Key finding:** Larger models and decoder-only architectures offer *no consistent advantage* over base encoders for CEFR classification. LoRA fine-tuned 7-8B LLMs achieve ~same as BERT-base. Prompting (zero-shot) consistently hurts smaller decoders.

**⚠️ Caveat:** These results are on UniversalCEFR's cross-domain English subset (multiple sources, some harder/more diverse than CEFR-SP). The ~0.57 macro-F1 for sentence-level is lower than the CEFR-SP original 84.5% because the test data is harder/broader.

#### Arase et al. (EMNLP 2022) — CEFR-SP original
- Best model: supervised BERT-based sentence assessor with class-rebalancing technique
- **Macro-F1: 84.5%** on CEFR-SP in-domain test set
- Note: this is in-domain, single-source; not directly comparable to UniversalCEFR numbers.

#### Liu & Lee (Language Resources and Evaluation, 2024)
- **Hybrid models** (linguistic features + neural) outperform pure neural and LLMs on sentence-level ARA
- Best hybrid model surpasses previous SOTA on Wall Street Journal by ~15% absolute
- LLMs (few-shot) competitive but not SOTA for sentence-level on CEFR-SP

#### UniversalCEFR Benchmarks (Imperial et al., EMNLP 2025)
- Random Forest + features: English sentence ~57%, doc ~64-71%
- Fine-tuned XLM-R: English sentence ~53-62%, doc ~64-71%
- Gemma-1-7B prompting: much weaker (overestimates lower levels on short texts)

### 4.2 Short Conversational Text — Ace-CEFR

| Model | MSE (1-6 scale, ↓) | 90% CI | Latency | Production-ready? |
|-------|---------------------|---------|---------|-------------------|
| Linear regression | 0.81 | [0.71, 0.91] | ~50µs CPU | Yes |
| PaLM 2-L (few-shot, 3×) | 0.48 | [0.43, 0.54] | ~3s API | No (latency) |
| BERT-3L (fine-tuned) | 0.37 | [0.32, 0.41] | ~100ms CPU / ~10ms TPU | Yes |
| BERT+LLM ensemble | **0.33** | — | — | Maybe |
| Human expert | 0.75 | [0.67, 0.84] | Minutes | N/A |

**Key finding:** A fine-tuned BERT model with 3 layers (45.7M params) beats human expert inter-rater agreement. Used LLM-labeled pre-training data (10k examples from PaLM) + human fine-tuning.

### 4.3 Learner Production / AES

- **Encoder models remain SOTA** for AES scoring tasks (QWK metric).
- Longformer achieves QWK 0.798 on ASAP 2.0 (vs. human-human 0.745; Aimecon 2025).
- Fine-tuned Latxa (Basque LM) surpasses GPT-5 and Claude Sonnet 4.5 on Basque CEFR AES (BEA 2025).
- Fine-tuned LLaMA-3-8B achieves weighted F1 0.769 on CEFR classification task (vs. zero-shot ~30-50%; Rooein et al. approach; arXiv 2512.06483).
- **GPT-4 for L2 short essays (Trott 2023):** With 1 calibration example per category, GPT-4 reaches near-conventional AWE model performance for holistic scoring. However, bias varies by L1.

### 4.4 LLM-as-Judge / Prompt-Based CEFR

| Setting | Performance vs. fine-tuned |
|---------|---------------------------|
| Zero-shot LLMs (GPT-4, Claude, LLaMA) | Systematically worse than fine-tuned BERT-size on classification (Bucher & Martini 2024 across general classification; Askadinov et al. 2024 on CEFR) |
| Few-shot prompting | Closes gap somewhat; GPT-4 ~near-human on essays with 1 example |
| Fine-tuned decoder-only (LLaMA-3-8B) | ~= base encoders, not better (Matsumura & Arase 2026) |
| Hybrid (prompt-based features + static) | Better than static alone (Rooein et al., BEA 2024) |

**Critical nuance:** For *classification* (6-class CEFR), fine-tuned encoders win. For *generation control* (making an LLM produce text at a target level), LLMs with complex prompting pipelines dominate (TSAR 2025). These are different tasks.

---

## 5. Shared Tasks / Benchmarks that Define the Frontier (2024-2026)

### 5.1 TSAR 2025 Shared Task — Readability-Controlled Text Simplification
**First edition; 20 teams, 48 submissions. EMNLP 2025.**

- **Task:** Simplify English B2/C1 paragraphs to A2 or B1 target CEFR level.
- **Evaluation:** ModernBERT-based CEFR classifier (confidence-based ensemble, weighted F1 ~0.89, AdjAcc ~0.99) + MeaningBERT.
- **Winner:** EhiMeNLP — multi-LLM ensemble (GPT-5, GPT-4.1, o3, GPT-OSS-20B, Qwen3-32B, Llama 3.3-70B-Instruct) + CEFR-as-judge. Achieved RMSE=0.000 (perfect CEFR compliance) and MeaningBERT ~0.845–0.902.
- **Key observation:** Top systems use 4+ LLMs with iterative refinement. Single-LLM approaches lag significantly. Evaluation metrics are beginning to saturate — new metrics needed.
- **CEFR evaluator model released** (ModernBERT-base fine-tuned, publicly available on HuggingFace under `AbdullahBarayan/ModernBERT-base-*`).

**Reference:** Alva-Manchego et al. (2025). *Findings of the TSAR 2025 Shared Task on Readability-Controlled Text Simplification.* TSAR @ EMNLP 2025. [aclanthology.org/2025.tsar-1.8](https://aclanthology.org/2025.tsar-1.8/)

### 5.2 BEA 2025 Shared Task — Pedagogical Ability of AI Tutors
- **Not CEFR-grading specific.** Focus on math dialogues, tutor quality.
- Relevant adjacent work *presented at BEA 2025*: alignment drift (Almasi et al.), Arabic CEFR AES (BEA 2025).
- **Reference:** [aclanthology.org/volumes/2025.bea-1/](https://aclanthology.org/volumes/2025.bea-1/)

### 5.3 Matsumura & Arase 2026 Benchmark
- **Not a shared task** — a systematic benchmark paper using UniversalCEFR.
- Defines the current fair multi-architecture comparison protocol for sentence/document CEFR.
- Published at NLP-Japan Annual Meeting 2026 (March 2026), not peer-reviewed in ACL sense.
- **Reference:** [anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf](https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf)

---

## 6. What Is NOT Settled (Open Problems)

1. **Sentence-level CEFR is unsolved.** Even the best models achieve ~0.55–0.59 macro-F1 on cross-domain data (UniversalCEFR English), well below classroom-use thresholds. The high 84.5% on CEFR-SP is in-domain.

2. **The text-difficulty vs. learner-proficiency conflation** is pervasive and rarely addressed cleanly. UniversalCEFR is the first to systematically separate "reference texts" from "learner texts" across 4 languages. Most prior benchmarks mix them.

3. **Short generated text grading** is almost entirely addressed by Ace-CEFR (2025). The dataset is small (890 passages) and English-only. No shared task has directly benchmarked this.

4. **Alignment drift in generation** is documented but not solved. Zero-shot prompting for CEFR compliance is brittle, especially over multi-turn dialogues (Almasi et al., BEA 2025). A1 compliance degrades to ~70% by turn 9.

5. **Cross-domain generalization** for CEFR classifiers remains poor. Models trained on CEFR-SP fail on ReadMe++ domains; ReadMe++ fine-tuning helps but UniversalCEFR numbers are lower than single-corpus results.

6. **Metric saturation for CEFR-controlled simplification**: TSAR 2025 found that top LLM ensembles are beginning to saturate the automatic evaluation metrics (CEFR RMSE, MeaningBERT). Human evaluation is still necessary.

7. **CEFR A1 is the hardest to classify** in both directions — texts that should be A1 are rare in training data, and the A1 class is consistently the weakest per-class F1 across systems. TSAR 2025's CEFR evaluator scores A1: F1=0.50 even with the best training setup.

8. **Ordinal structure of CEFR is underexploited.** Current models treat CEFR as nominal 6-class; confusion is highest between adjacent levels. Regression and ranking approaches are flagged as future work (Matsumura & Arase 2026).

---

## 7. Recommendations for Short Generated Beginner Answers (Thesis-Actionable)

**The thesis evaluates short beginner-English answers from 4 SLMs using FK/Fog/Spache + KVL as a readability proxy.** Here is what 2025–2026 SOTA says about this choice and what could be added:

**Your current proxy is reasonable but demonstrably crude:** Trott & Rivière (TSAR 2024) show GPT-4 Turbo zero-shot achieves r=0.76 with human readability judgments on general English, vs. r≈0.5 for traditional formulas — but this requires API calls. Rooein et al. (BEA 2024) confirm that static metrics like FK are "crude and brittle" especially for short texts where syllable/word-count signals are unreliable. The KVL lexical coverage component partially compensates by adding semantic vocabulary difficulty.

**Best available classifier for your use case:** The TSAR 2025 CEFR evaluator (ModernBERT-base, publicly available as `AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr` on HuggingFace) is a strong, open, English sentence+document CEFR classifier fine-tuned on CEFR-SP + ReadMe++ + Cambridge Exams data. It is production-ready and could serve as a **secondary validation signal** alongside your FK/KVL proxy — at inference time on a sample of your runs.

**Do not use it as your primary metric** in the thesis without caveating: (a) it was not designed for ultra-short ~10-word LLM outputs specifically; (b) Ace-CEFR shows that sentence-level models struggle on short conversational text. For your generated answers, apply it but report its 90% CI or adjacent-level accuracy alongside.

**If you want to validate your A1 binary proxy:** Run the TSAR 2025 evaluator on a random sample of your `meets_a1_criteria=True` and `False` predictions and report agreement. A1 is the hardest class — expect adjacent-level errors. This is a legitimate thesis validation contribution.

**Do not claim your FK/Fog/Spache proxy "measures CEFR"** — per the literature, it measures surface readability, which *correlates with but does not equal* CEFR level, especially for short text. The Rooein et al. (2024) paper is your primary citation for why this matters and what the limitations are.

---

## 8. Bibliography with Primary-Source Links

### Core Sentence-Level CEFR

1. **Arase, Y., Uchida, S., & Kajiwara, T. (2022).** CEFR-Based Sentence Difficulty Annotation and Assessment. *EMNLP 2022*, pp. 6206–6219.  
   [https://aclanthology.org/2022.emnlp-main.416/](https://aclanthology.org/2022.emnlp-main.416/)  
   DOI: 10.18653/v1/2022.emnlp-main.416

2. **Uchida, S., Arase, Y., & Kajiwara, T. (2024).** Profiling English sentences based on CEFR levels. *ITL - International Journal of Applied Linguistics*, 175(1), 103–126.  
   [https://doi.org/10.1075/itl.22018.uch](https://doi.org/10.1075/itl.22018.uch)

3. **Liu, Y., & Lee, J. (2024).** Automatic readability assessment for sentences: neural, hybrid and large language models. *Language Resources and Evaluation*.  
   [https://doi.org/10.1007/s10579-024-09800-5](https://doi.org/10.1007/s10579-024-09800-5)

### Multi-Corpus / Multilingual Benchmarks

4. **Imperial, J. M., Barayan, A., Stodden, R., et al. (2025).** UniversalCEFR: Enabling Open Multilingual Research on Language Proficiency Assessment. *EMNLP 2025*, pp. 9703–9755.  
   [https://aclanthology.org/2025.emnlp-main.491/](https://aclanthology.org/2025.emnlp-main.491/)  
   arXiv: [https://arxiv.org/abs/2506.01419](https://arxiv.org/abs/2506.01419)

5. **Naous, T., Ryan, M. J., Lavrouk, A., Chandra, M., & Xu, W. (2024).** ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment. *EMNLP 2024*, pp. 12230–12266.  
   [https://aclanthology.org/2024.emnlp-main.682/](https://aclanthology.org/2024.emnlp-main.682/)

6. **Matsumura, E., & Arase, Y. (2026).** A Benchmark Study of Multi-Granular CEFR Level Assessment. *NLP Annual Meeting Japan 2026* (not peer-reviewed).  
   [https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf](https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/P6-18.pdf)

### Short / Conversational Text CEFR

7. **Kogan, D., Schumacher, M., Nguyen, S., Suzuki, M., Smith, M., Bellows, C. S., & Bernstein, J. (2025).** Ace-CEFR — A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications. *arXiv preprint arXiv:2506.14046*.  
   [https://arxiv.org/abs/2506.14046](https://arxiv.org/abs/2506.14046)  
   OpenReview (ACL ARR): [https://openreview.net/forum?id=e1bM1YofLh](https://openreview.net/forum?id=e1bM1YofLh)

### Readability Proxies and Beyond FK

8. **Rooein, D., Röttger, P., Shaitarova, A., & Hovy, D. (2024).** Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts. *BEA 2024*, pp. 54–67.  
   [https://aclanthology.org/2024.bea-1.5/](https://aclanthology.org/2024.bea-1.5/)  
   DOI: 10.18653/v1/2024.bea-1.5

9. **Trott, S., & Rivière, P. D. (2024).** Measuring and Modifying the Readability of English Texts with GPT-4. *TSAR 2024*.  
   [https://aclanthology.org/2024.tsar-1.13/](https://aclanthology.org/2024.tsar-1.13/)  
   arXiv: [https://arxiv.org/abs/2410.14028](https://arxiv.org/abs/2410.14028)

### LLM vs. Fine-Tuned Classifiers

10. **Bucher, M., & Martini, S. (2024).** Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification. *arXiv:2406.08660*.  
    [https://arxiv.org/abs/2406.08660](https://arxiv.org/abs/2406.08660)

11. **Askadinov, A., et al. (2024).** Assessing how accurately large language models encode and apply the common European framework of reference for languages. *Computers and Education: AI*.  
    [https://doi.org/10.1016/j.caeai.2024.100353](https://doi.org/10.1016/j.caeai.2024.100353)

12. **Imperial, J. M., & Tayyar Madabushi, H. (2023).** Flesch or Fumble? Evaluating Readability Standard Alignment of Instruction-Tuned Language Models. *GEM 2023*, pp. 205–223.  
    [https://aclanthology.org/2023.gem-1.18/](https://aclanthology.org/2023.gem-1.18/)

### LLM for Learner Production / AES

13. **Trott, S. (2023, presented at BEA 2023).** Rating Short L2 Essays on the CEFR Scale with GPT-4. *BEA 2023*, pp. 354–362.  
    [https://aclanthology.org/2023.bea-1.49/](https://aclanthology.org/2023.bea-1.49/)

14. **North, K., et al. (2025).** Long context Automated Essay Scoring with Language Models. *AIMECON 2025*.  
    [https://aclanthology.org/2025.aimecon-main.5/](https://aclanthology.org/2025.aimecon-main.5/)

### Alignment Drift in LLM Generation

15. **Almasi, M., et al. (2025).** Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring. *BEA 2025*.  
    [https://aclanthology.org/2025.bea-1.6/](https://aclanthology.org/2025.bea-1.6/)  
    arXiv: [https://arxiv.org/abs/2505.08351](https://arxiv.org/abs/2505.08351)

### TSAR 2025 Shared Task

16. **Alva-Manchego, F., Stodden, R., Imperial, J. M., Barayan, A., North, K., & Tayyar Madabushi, H. (2025).** Findings of the TSAR 2025 Shared Task on Readability-Controlled Text Simplification. *TSAR @ EMNLP 2025*, pp. 116–130.  
    [https://aclanthology.org/2025.tsar-1.8/](https://aclanthology.org/2025.tsar-1.8/)

17. **Miyata, R., et al. (EhiMeNLP) (2025).** EhiMeNLP at TSAR 2025 Shared Task: Candidate Generation via Iterative Simplification and Reranking by Readability and Semantic Similarity. *TSAR @ EMNLP 2025*.  
    [https://aclanthology.org/2025.tsar-1.18/](https://aclanthology.org/2025.tsar-1.18/)

### SLE (Cripwell 2023) — Simplicity Metric, not CEFR corpus

18. **Cripwell, L., Legrand, J., & Gardent, C. (2023).** Simplicity Level Estimate (SLE): A Learned Reference-Less Metric for Sentence Simplification. *EMNLP 2023*, pp. 12053–12059.  
    [https://aclanthology.org/2023.emnlp-main.739/](https://aclanthology.org/2023.emnlp-main.739/)  
    **Note:** SLE is a simplicity *evaluation metric* (trained on Newsela reading levels), not a CEFR-SP corpus. It predicts relative simplicity gain and is useful for evaluating simplification systems but does not produce CEFR labels.

### Vocabulary Resources

19. **Dürlich, L., & François, T. (2018).** EFLLex: A Graded Lexical Resource for Learners of English as a Foreign Language. *LREC 2018*.  
    [https://aclanthology.org/L18-1140/](https://aclanthology.org/L18-1140/)  
    Tool: [https://cental.uclouvain.be/cefrlex/efllex/](https://cental.uclouvain.be/cefrlex/efllex/)

20. **Uchida, S., & Negishi, M. (2025).** Estimating the CEFR-J level of English reading passages: Development and accuracy of CVLA3. *English Corpus Studies (ECS)*, 32.  
    [https://cvla.langedu.jp/](https://cvla.langedu.jp/)  
    Tool: CVLA 3.1 (as of March 2026)

### Key Learner Corpora

21. **Geertzen, J., Alexopoulou, T., & Korhonen, A. (2013).** Automatic linguistic annotation of large scale L2 databases: EFCAMDAT.  
    Dataset hosted at: [https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html)

---

## Appendix: Commercial/Educational Tools — Academic Standing

| Tool | Developer | CEFR basis | Academic validity | Use in NLP research |
|------|-----------|------------|-------------------|---------------------|
| **Text Inspector** | Stephen Bax / CRELLA, Uni Bedfordshire | EVP + BNC + COCA + 200+ metrics | Won BritCouncil ELTons 2017; some applied linguistics validation | Used in applied linguistics; not an NLP benchmark |
| **CVLA 3.1** | Uchida & Negishi (Kyushu/Tokyo Univ) | CEFR-J Wordlist + 8 features | 65.74% exact accuracy (108-text test, 2025 paper); 99.07% adj-level | Used in Japanese EFL research; some NLP studies |

**Academic standing:** Text Inspector and CVLA are useful practical tools but are **not considered SOTA in NLP research**. Their accuracy (~53-66%) on CEFR classification is substantially below fine-tuned transformer classifiers (84.5% CEFR-SP, ~0.90 doc-level). They are more often cited in applied linguistics than in ACL/EMNLP papers. Neither provides an open test set that NLP researchers use as a benchmark.

---

*Research compiled 2026-07-14. All URLs verified at time of writing. arXiv papers may not be peer-reviewed.*
