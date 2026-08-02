# Arase CEFR-SP vs TSAR 2025 ModernBERT Ensemble — Scorer Comparison

**Ticket:** [Compare Arase CEFR-SP vs TSAR ModernBERT ensemble for the A1/English construct](https://github.com/srobaina99/SLMs-experiments/issues/4) · **Map:** [Wayfinder: revise the beginner-suitability evaluation-stack plan](https://github.com/srobaina99/SLMs-experiments/issues/2)  
**Date:** 2026-07-19 · **Type:** AFK research (decision feeds [DECIDE: primary CEFR scorer](https://github.com/srobaina99/SLMs-experiments/issues/5))  
**Sources:** existing notes `2026-07-14_cefr-l2-grading-sota-2026.md`, `2026-07-14_cefr-sp-sota-status.md`; live code (`evaluation/cefr_sp.py`, `evaluation/a1_criteria.py`, `docs/metrics.md`); plan `thoughts/plans/evaluation-stack-plan.md` (TSAR section); audit `evaluation-stack-plan-stale-claims-audit.md`.

**Locked construct:** *a correct, coherent English answer that a Spanish-L1 CEFR A1 learner can understand without help.*  
This comparison is about the **English difficulty scorer** only. Spanish-L1 familiarity stays with KVL; answer adequacy stays with the human/judge layer.

---

## 1. Recommendation (one paragraph)

**Keep Arase CEFR-SP as the primary in-pipeline A1 gate.** Prefer the **"both"** option over a hard migration: if the revised stack needs a second automatic CEFR signal, add the pinned TSAR three-model ModernBERT ensemble as a **post-hoc ordinal cross-check** in the assessment-bundle path — not as a replacement for `meets_a1_criteria`. A full swap to TSAR-as-primary is not justified on construct fit or A1-class reliability, and it burns integration budget the thesis does not need to spend. Ace-CEFR / ReadMe++ are noted only as out-of-ticket alternatives (not candidates here).

---

## 2. Side-by-side

| Dimension | Arase CEFR-SP (live) | TSAR 2025 ModernBERT ensemble (plan candidate) |
|-----------|----------------------|--------------------------------------------------|
| **What it is** | Contrastive BERT-base sentence CEFR classifier (Arase et al., EMNLP 2022); project aggregates sentences → document level | Confidence-based ensemble of 3 ModernBERT-base HF checkpoints (doc; doc+sent; multilingual ref) used as TSAR 2025 official CEFR evaluator |
| **Construct fit (English difficulty)** | Type 2A ARA — *reader* comprehension difficulty on standalone English sentences. Matches "understand without help" better than learner-proficiency (AES). Document label is a **project mean-ordinal → nearest level**, not a native doc model. | Also Type 2A ARA. Includes a dedicated **doc-level** model and a sent+doc model; aggregation is confidence-max across models. Closer to "document-level" wording in the construct, but still general English readability — not Spanish-L1-specific. |
| **Fit to short SLM answers** | Train sentences 5–30 words (A1 mean ~7.7); news/Wiki/SCoRE register. Ultra-short or colloquial answers can be OOD. A1 is scarce in train (F1 ~78% with loss weighting; 0% without). | Trained on CEFR-SP + ReadMe++ + Cambridge-style data; used to score B2/C1→A2/B1 *paragraph* simplifications. Still **not** purpose-built for ~10-word conversational LLM answers. **A1 class remains the weakest** (reported A1 F1 ≈ 0.50 even in strong setups). |
| **Reported quality** | Macro-F1 **84.5%** ±0.7 on CEFR-SP own test; QWK 0.628. Still unbeaten *on that test set*. Cross-domain UniversalCEFR sentence numbers (~0.55–0.59) are a different, harder task. | Confidence ensemble: weighted F1 ≈ **0.89**, AdjAcc ≈ **0.99**, RMSE ≈ **0.34** (TSAR findings / plan). Not comparable 1:1 to Arase's in-domain 84.5% (different data, different aggregation). High AdjAcc softens but does not erase A1 weakness for a binary A1 gate. |
| **Reproducibility / pinning** | Local Zenodo ckpt `data/cefr_sp/level_estimator.ckpt` (~1.2GB, not in git); vendored Lightning modules; known load quirks (hparams path override, drop `position_ids`). Revision = file on disk + download script. | Three HF IDs; **upstream does not pin revisions**. Plan already lists tip SHAs to pin. Repro = pin those three SHAs + transformers/torch versions in the eval env. Tip drift is a real risk if unpinned. |
| **Deps + hardware** | Optional `[cefr-sp]`: `torch`, `transformers>=4.36`, `pytorch-lightning` 2.x. **One** BERT-base. CPU is the documented default (`cefr_sp_device=cpu`). Lightning is only needed because of checkpoint format. | Plan: `transformers>=4.55`, `torch` — **no Lightning**. **Three** ModernBERT forwards per item (or cache per model). Higher VRAM/RAM and latency; GPU preferred for batch assessment; CPU feasible but slower. |
| **Integration given live wiring** | **Already done.** `CefrSpScorer` → `compute_cefr_sp_metrics` → pipeline → `meets_a1_criteria` (`cefr_sp_level == "A1"`). CLI default ON. Columns and docs already contract on CEFR-SP. | **Greenfield.** New scorer module, new CSV columns, confidence aggregation, device auto-detect, likely assessment-bundle / `venv-eval` path (ticket on execution model). Gate rename or dual fields if it becomes primary. Touches protocol, run store, docs. |

---

## 3. Trade-offs by decision option

### A — Keep Arase CEFR-SP as primary (status quo for the gate)

- **Pros:** Zero scorer-migration cost; thesis claims stay continuous with runs already gated on CEFR-SP; single model; documented columns; construct-adequate (English ARA → A1 binary).
- **Cons:** Domain/length mismatch for short answers remains; A1-class fragility remains; no native document model; Lightning dep stays.
- **When this is enough:** Thesis needs a stable primary gate now; human validation (not a second encoder) is the planned quality check.

### B — Migrate to TSAR ModernBERT ensemble as primary (demote/remove Arase)

- **Pros:** Newer shared-task official evaluator; broader training mix; native doc checkpoint; no Lightning; pinable HF SHAs; headline metrics look stronger at aggregate level.
- **Cons:** Large integration + re-documentation cost; **A1 F1 ≈ 0.50** undercuts the binary gate the thesis cares about; still OOD for ultra-short chatty answers; 3× inference cost; breaks continuity with existing CEFR-SP-gated run bundles unless re-scored; "document-level" gain is modest for 1–3 sentence answers.
- **When this would win:** Human study shows systematic CEFR-SP failure on this domain *and* TSAR agrees better with raters — evidence we do not have yet.

### C — Both (recommended if a second automatic signal is wanted)

- **Pros:** Keeps live gate and historical continuity; TSAR becomes an ordinal / disagreement diagnostic in the assessment layer; disagreement flags feed human sampling; matches research-note advice ("secondary validation, not primary without caveats").
- **Cons:** Two scorers to maintain, pin, and explain; extra compute on the assessment path; risk of reader confusion unless endpoint hierarchy is explicit (CEFR-SP = gate; TSAR = cross-check).
- **Shape:** `meets_a1_criteria` stays CEFR-SP; assessment bundle writes `cefr_tsar_*` (per-model + ensemble ordinal/label/confidence/disagreement); analysis reports agreement with humans for **both**.

---

## 4. What does *not* decide this

- **Spanish-L1 side of the construct** — neither scorer encodes L1; KVL (and humans) do.
- **Answer correctness / coherence** — neither scorer; adequacy / judge / human layers do.
- **FK/Fog/Spache** — already descriptive; not in the fork.
- **Ace-CEFR / readabert-en** — closer to short conversational text in spirit, but **out of this ticket's fork**; mention only if #5 wants a third candidate later.

---

## 5. Explicit recommendation for #5

| Commit in revised plan | Recommendation |
|------------------------|----------------|
| Primary A1 gate | **Arase CEFR-SP** (keep live wiring and `meets_a1_criteria` semantics) |
| TSAR ensemble | **Candidate secondary** in the assessment bundle (post-hoc), not the in-pipeline gate — *unless* #5 explicitly chooses a hard swap after weighing integration cost |
| Default if #5 wants one word | **"both" with CEFR-SP primary**, not "TSAR-only" |

**Rationale in one line:** For a binary A1 gate on short English answers, Arase is already paid-for and no worse on the A1 tail than TSAR's reported A1 weakness; TSAR's aggregate F1 / AdjAcc do not buy a cleaner A1 decision, only a costly second stack — useful as a cross-check, not as a replacement.

---

## 6. Pointers

- Live gate: `src/slm_experiments/evaluation/a1_criteria.py`, `evaluation/cefr_sp.py`, `docs/metrics.md` § CEFR-SP  
- Plan's TSAR spec (candidate): `thoughts/plans/evaluation-stack-plan.md` § "The TSAR CEFR scorer"  
- Prior research: `thoughts/shared/research/2026-07-14_cefr-l2-grading-sota-2026.md` §§ 5.1, 7; `2026-07-14_cefr-sp-sota-status.md` §§ 5–7  
- Stale framing already corrected: S2 in `thoughts/plans/evaluation-stack-plan-stale-claims-audit.md`  
- Upstream TSAR: [aclanthology.org/2025.tsar-1.8](https://aclanthology.org/2025.tsar-1.8/), [tsar-2025-shared-task evaluation script](https://github.com/tsar-workshop/tsar-2025-shared-task/blob/main/code/tsar2025_evaluation_script.py)
