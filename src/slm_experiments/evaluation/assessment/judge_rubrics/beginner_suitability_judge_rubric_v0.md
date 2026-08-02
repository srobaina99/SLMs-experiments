# Judge rubric — beginner_suitability_judge_rubric_v0

**Construct:** a correct, coherent English answer that a Spanish-L1 CEFR A1 learner can understand without help.

**Aligned human rubric:** `beginner_suitability_rubric_v0` (same four dimensions and 1–4 scale).

**Column / JSON keys:** `overall_suitability`, `vocabulary_accessibility`, `syntax_accessibility`, `answer_adequacy`.

**Scale:** integers **1–4** only. If unsure between two anchors, choose the lower.

**Binary suitable (for agreement checks with humans):** overall ≥ 3 **and** adequacy ≥ 3. Vocabulary and syntax are diagnostic only.

This file is the **provider-neutral** instruction surface for an external LLM judge. The repo ships **no API adapter**; produce `judge_scores.csv` externally and import via `assess judge-import`.

Bump `rubric_version` on any future edit to this file.

---

## Task

You are given a learner prompt and an English model answer. Score the answer on each dimension below. Do **not** invent CEFR labels (A1/A2…), FK/Fog/Spache grades, model names, or intervention configs.

### Overall suitability (`overall_suitability`)

| Score | Anchor |
|------:|--------|
| **4** | An A1 learner would understand the **whole** answer alone. Easy English end to end. |
| **3** | Almost all is understandable alone. May stumble on a **short** stretch, but the main message lands. |
| **2** | Understood **in part**. Would need help with several words/phrases, or would miss an important idea. |
| **1** | **Not** understandable alone. Hard or confusing; without substantial help the message is lost. |

### Vocabulary accessibility (`vocabulary_accessibility`)

| Score | Anchor |
|------:|--------|
| **4** | Almost all words are easy. Hardly any difficult vocabulary. |
| **3** | Mostly easy. **Some** hard words, but they do not block the main message. |
| **2** | **Several** hard or technical words. A beginner would need help. |
| **1** | **Many** hard, technical, or unclear words. |

### Syntax accessibility (`syntax_accessibility`)

| Score | Anchor |
|------:|--------|
| **4** | Short, simple sentences. Easy to read straight through. |
| **3** | Generally simple. Occasional longer sentence, but an attentive beginner can follow. |
| **2** | Long or tangled sentences. A beginner would get lost or slow down heavily. |
| **1** | Structure is hard almost throughout: very long or very tangled. |

### Answer adequacy (`answer_adequacy`)

| Score | Anchor |
|------:|--------|
| **4** | Answers the question clearly, correctly, and sufficiently. No serious errors or evasion. |
| **3** | Mostly answers well. Minor missing detail, slight vagueness, or a small error that does not change the idea. |
| **2** | Weak or incomplete: misses something important, or mixes useful content with a clear error. |
| **1** | Not usable as an answer: off-topic, nonsensical, empty, or clearly wrong. |

### Output contract

Emit one row per `item_id` matching schema `judge_scores_v0` (see `judge_scores_v0.json`): required integer scores for all four dimensions; optional `notes` / `judge_id` / `rubric_version`. No other columns.

### Quality-preservation scope

Quality-preservation claims stay **limited to the human sample** until judge results are imported and shown to agree acceptably with human consensus.
