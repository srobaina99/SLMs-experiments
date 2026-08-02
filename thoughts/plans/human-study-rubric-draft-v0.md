# Human study design validation + rubric draft v0

**Ticket:** [VALIDATE + draft: three-rater human study, rubric, reliability](https://github.com/srobaina99/SLMs-experiments/issues/9)
**Map:** [Wayfinder: revise the beginner-suitability evaluation-stack plan](https://github.com/srobaina99/SLMs-experiments/issues/2)
**Status:** **approved** — plan artifact; shipped by [#16](https://github.com/srobaina99/SLMs-experiments/issues/16). Reliability reconciled to shipped code in [#21](https://github.com/srobaina99/SLMs-experiments/issues/21) (see §2 / §6).
**Construct (locked):** *a correct, coherent English answer that a Spanish-L1 CEFR A1 learner can understand without help.*
**`rubric_version`:** `beginner_suitability_rubric_v0` (Spanish anchors in §3; bumps on future edits).

---

## 1. Validation against the current single-rater flow

Checked against `human/export.py`, `human/import.py` (loaded via `importlib` because the module name is reserved), and `cli.py` `human {export,import}`.

| Plan claim | Current code | Verdict |
|------------|--------------|---------|
| Replace single-rater round trip with study-level long-format | One CSV per run; tags merge onto that run’s `full.csv` by `experiment_id` | **Compatible intent** — new study modules, not an in-place patch of export/import |
| Avoid reserved `import` module name | `human/import.py` + `import_module("slm_experiments.human.import")` | **Confirmed hazard** — `study_import.py` (or similar) is required |
| Blinded items: opaque id + prompt + answer only | Export columns include `model`, `config`, `experiment_id`, `prompt_id`; **no prompt text** | **Incompatible** — current sheet is not blind and cannot support adequacy without looking up prompts |
| Answer text for rating | Export sets `answer = response` (raw), not `cleaned_response` | **Correction** — study items must use the same text the assessment bundle scores (`cleaned_response`), matching [#8](https://github.com/srobaina99/SLMs-experiments/issues/8) dedup |
| Write path | Import mutates source-run `full.csv` | **Incompatible with assessment-bundle rule** — ratings land in the assessment / study bundle; never write back to generation runs |
| Stratified sample | Stratifies by Phase-1-shaped `config` = weighting×prompting label; default n=60 | **Must redesign** — stratify on assessment `item_map` axes (family, model, baseline/intervention, CEFR band/disagreement, truncation); default **100** unique items after excluded calibration pilot |
| Three raters, independent random order | Single sheet, fixed order | **New** — one shared item set; per-rater shuffle; private source key separate from rater sheets |
| Four 1–4 ratings + notes | `response_appropriateness` (float), `vocabulary_level` (string), `notes` | **Schema replace** — not a rename; old columns are not the criterion |
| Exactly one rating per `(item_id, rater_id)` | One row per `experiment_id` (single rater implied) | **New validation rule** on long-format import |
| Exact + adjacent (±1) percent agreement + consensus medians | Shipped in `reliability.py` | **Shipped** — see §6 for the Krippendorff deviation |
| Human suitability from median overall + adequacy thresholds | No binary human suitability flag | **New** — defined in §3 below |

### Plan corrections to fold into synthesis

1. **Rater sheet columns (blind):** `item_id`, `prompt`, `answer` (= cleaned), then empty rating columns; never `model` / `config` / `experiment_id` / CEFR / KVL.
2. **Private key** (study-bundle only): `item_id → source experiment_id(s), model, config, …` — not distributed to raters.
3. **Text identity:** rate `cleaned_response`; keep alignment with assessment `items.csv` / `(prompt_id, cleaned_response)` dedup.
4. **Storage:** long-format ratings + reliability live under the assessment/study bundle; retire “merge tags into generation `full.csv`” as the criterion path (legacy single-rater may remain for smoke tests until deleted — out of this ticket’s decision scope unless you want it gone).
5. **Prompt text** must be materialised on the sheet (join from `STANDARD_PROMPTS` / stored prompt at export time) — `prompt_id` alone is insufficient for adequacy.

---

## 2. Study protocol sketch (for plan; not implemented here)

- **Raters:** 3; same item set; independently seeded random order.
- **Calibration:** small pilot set **excluded** from the analysis sample; default analysis n = **100** unique items with inclusion weights from stratified sampling. **Correction (post-ship nits #16/#20):** earlier draft caveat (conditional on post-calibration) superseded — weights are unconditional Horvitz–Thompson against the original pool (\(\pi_i = n_s / N_s\)); see `docs/human-eval.md` / Decision 4 in `evaluation-stack-plan.md`.
- **Ratings per item:** four ordinal 1–4 dimensions (§3) + optional free-text notes.
- **Consensus:** per item, per dimension = **median** across raters.
- **Reliability (shipped):** mean pairwise **exact + adjacent (±1) percent agreement** per dimension. **Not** Krippendorff's α — see §6.
- **Binary human suitability** (criterion used in validation / guardrail analyses): see §3.5.

---

## 3. Rúbrica borrador v0 — (español, lenguaje simple)

**Idioma de la rúbrica:** español (para evaluadores hispanohablantes). Las claves de columna en datos siguen en inglés (`overall_suitability`, etc.).

**Instrucciones al evaluador:** Imagina a un adulto hispanohablante que recién empieza inglés (**nivel A1**). Lee la respuesta **sin diccionario y sin ayuda**. No premies que “suene inteligente”. Una respuesta corta está bien si es clara y responde la pregunta.

**Escala (las cuatro dimensiones):** solo enteros **1–4**. Elige la ancla que mejor encaje; si dudas entre dos, elige la **más baja**.

### 3.1 Adecuación general al A1 (`overall_suitability`)

Juicio global de si un principiante **entiende la respuesta sin ayuda** (la calidad de la respuesta se puntúa aparte).

| Puntos | Ancla |
|------:|--------|
| **4** | Un estudiante A1 entendería **toda** la respuesta solo. El inglés es fácil de punta a punta. |
| **3** | Casi toda se entiende solo. Puede trabarse en **un tramo corto**, pero el mensaje principal llega. |
| **2** | Se entiende **en parte**. Necesitaría ayuda con varias palabras o frases, o se perdería una idea importante. |
| **1** | **No** se entiende solo. El inglés es difícil o confuso; sin mucha ayuda no captaría el mensaje. |

### 3.2 Facilidad del vocabulario (`vocabulary_accessibility`)

Sobre el **vocabulario**.

| Puntos | Ancla |
|------:|--------|
| **4** | Casi todas las palabras son fáciles. Casi no hay palabras difíciles. |
| **3** | La mayoría son fáciles. Hay **algunas** difíciles, pero no impiden el mensaje principal. |
| **2** | Hay **varias** palabras difíciles o técnicas. Un principiante necesitaría ayuda. |
| **1** | Hay **muchas** palabras difíciles, técnicas o poco claras. |

### 3.3 Facilidad de la estructura (`syntax_accessibility`)

Sobre las **oraciones** (largas, enredadas, etc.)

| Puntos | Ancla |
|------:|--------|
| **4** | Oraciones cortas y simples. Fácil de leer de corrido. |
| **3** | En general simple. A veces una oración un poco más larga, pero un principiante atento aún sigue. |
| **2** | Hay oraciones largas o enredadas. Un principiante se perdería o iría muy lento. |
| **1** | La estructura es difícil casi siempre: muy larga o muy enredada. |

### 3.4 Calidad de la respuesta (`answer_adequacy`)

Sobre la **correctitud de la respuesta** — independiente de la dificultad.

| Puntos | Ancla |
|------:|--------|
| **4** | Responde la pregunta de forma clara, correcta y suficiente. Sin errores graves ni evasivas. |
| **3** | Responde casi bien. Falta un detalle menor, es un poco vaga, o tiene un error chico que no cambia la idea. |
| **2** | Respuesta débil o incompleta: falta algo importante de la pregunta, o mezcla algo útil con un error claro. |
| **1** | No sirve como respuesta: fuera de tema, sin sentido, vacía o claramente incorrecta. |

### 3.5 Idoneidad humana binaria (umbrales)

Sea \(m_s\) = mediana de `overall_suitability`, \(m_a\) = mediana de `answer_adequacy` entre los tres evaluadores.

**Regla (ya aprobada):** el ítem es **human-suitable** si y solo si

\[
m_s \ge 3 \quad \textbf{y} \quad m_a \ge 3
\]

Vocabulario y estructura son **solo diagnósticos** — no entran en esta puerta.

El margen de no-inferioridad de adecuación entre intervención y baseline queda para [DECIDE: protocol lock — endpoint hierarchy, statistical margins, multiplicity](https://github.com/srobaina99/SLMs-experiments/issues/7).

### 3.6 Qué **no** debe hacer el evaluador

- No uses etiquetas CEFR (A1/A2…), ni FK/Fog/Spache, ni nombres de modelos.
- No “repartas” puntajes para forzar variedad.
- Las notas son opcionales; **nunca** reemplazan el número en el análisis.

---

## 4. Decisions (this ticket)

1. Four dimensions — **approved**.
2. Binary rule \(m_s \ge 3 \land m_a \ge 3\) — **approved**.
3. Plan corrections in §1 — **approved**.
4. Default n — **100** unique items after excluded calibration pilot.
5. Anchor wording — **approved** as Spanish `beginner_suitability_rubric_v0` (§3, human-edited).

---

## 5. Out of this ticket

- Implementing `study_export` / `study_import` / `reliability.py` → done in #16
- Recruiting real raters
- Approving statistical margins / endpoint hierarchy (#7)
- Choosing the automatic CEFR primary scorer (#5)

---

## 6. Reliability reconciliation (shipped vs draft)

**Original draft (§1 / §2):** ordinal Krippendorff's α per dimension + consensus medians.

**Shipped (`human/reliability.py`, issue #16):** mean pairwise **exact** and **adjacent (±1) percent agreement** per dimension + consensus medians. `reliability.json` explicitly records `"krippendorff_alpha": "not computed (dropped by design)"`. Automatic-vs-human validation (#20) likewise omits chance-corrected coefficients (no Cohen's κ either).

**Why the deviation is visible:** #16 listed this as an intentional deviation from the draft; #21 reconciles Decision 4 and this draft to the shipped approach rather than silently deleting α. Operational docs: `docs/human-eval.md`.
