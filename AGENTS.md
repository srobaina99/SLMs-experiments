# AGENTS.md

## 1. Mission

Evaluate whether inference-time interventions make 4 SLMs produce simpler English answers for beginner learners.
Primary binary outcome is **CEFR-SP document level A1** (`meets_a1_criteria` when `cefr_sp_level == "A1"`). FK/Fog/Spache remain descriptive only.
Two research phases, one shared pipeline, run-centric results.

## 2. Documentation Map

| Document | Purpose | Read when |
|----------|---------|-----------|
| **AGENTS.md** (this file) | Structure, CLI, code rules | Always start here |
| [ExperimentDesign.md](ExperimentDesign.md) | Formal experiment spec, phases, success criteria | Designing or changing experiments |
| [docs/metrics.md](docs/metrics.md) | Readability metrics, CEFR-SP / TSAR / KVL definitions | Working on `evaluation/` |
| [docs/human-eval.md](docs/human-eval.md) | Three-rater blinded study, rubric, reliability | Human study or validation |
| [docs/models.md](docs/models.md) | GGUF files, templates, GPU setup | Adding/fixing model wrappers |
| [docs/interventions.md](docs/interventions.md) | Weighting, prompting, guided, KVL, deprecated beam | Changing intervention logic |
| [docs/guided-decoding.md](docs/guided-decoding.md) | Top-k A1-guided decoding (`phase2 guided`) | Changing guided decode |
| [docs/kvl_beamsearch.md](docs/kvl_beamsearch.md) | KVL-scored beam (`phase2 kvl_beam`) | Changing KVL beam |
| [docs/clusteruy.md](docs/clusteruy.md) | ClusterUY SSH, Singularity, Phase 2 + assess jobs | Running experiments on the cluster |
| [docs/experiment-setup-recommendations.md](docs/experiment-setup-recommendations.md) | Thesis-defensibility checklist | Before publishing claims |
| [README.md](README.md) | Human setup quickstart | Environment issues |

## 3. Research Phases

→ Full design in [ExperimentDesign.md](ExperimentDesign.md)

**Thesis path = Phase 2 only.** Formal claims use `--prompts all` (25 prompts); CLI default `n=3` is a smoke-test guardrail. Phase 1 code remains but is optional / out of thesis scope.

**Phase 2 — Hyperparameter sweeps (all 4 models):**
- `weights` — logit bias grid with prompting ON
- `prompting` — zero/one/few-shot contextual prompting
- `guided` — A1-constrained greedy top-k pool sweep
- `kvl_beam` — KVL-scored beam width sweep
- `beam` — **deprecated / hard-fails** (void at temperature 0)

**Phase 1 — Factorial (optional):** 4 models × 4 interventions × N prompts — not run/cited for the master thesis.

Shared generation defaults: `temperature=0.0`, `top_k=50`, max 200 new tokens; **no `top_p`**.

## 4. Directory Map

```
SLMs-experiments/
├── AGENTS.md                 # This file — agent entry point
├── README.md                 # Human quickstart
├── ExperimentDesign.md       # Formal experiment specification
├── scripts/clusteruy/        # SLURM batch scripts (smoke test + Phase 2 sweeps)
├── requirements.txt          # Runtime dependencies (7 packages)
├── requirements-dev.txt      # pytest
├── pytest.ini
├── docs/                     # Detailed reference docs
├── data/vocabularies/        # A1 vocabulary (487 words)
├── data/kvl/                 # KVL lookup tables (es/de/cn)
├── data/cefr_sp/             # CEFR-SP ckpt (download; not in git) + README
├── models/gguf/              # GGUF model files (not in git)
├── results/runs/{run_id}/    # Run bundles (manifest, CSVs, summary, plots)
├── src/slm_experiments/      # Package source
│   ├── cli.py                # Single CLI entry point
│   ├── core/                 # Pipeline, config, result, run_store, prompts
│   ├── models/               # Base, llamacpp, guided/KVL decoders, wrappers
│   ├── evaluation/           # Metrics, CEFR-SP, assessment/ (TSAR, KVL v2, analysis, judge)
│   ├── phase1/               # Factorial experiment
│   ├── phase2/               # Weight, prompting, guided, kvl_beam (+ deprecated beam)
│   ├── human/                # Legacy export/import + three-rater study
│   └── plot.py               # Boxplot generation from run bundles
└── tests/                    # pytest suite (mocked pipeline)
```

## 5. How to Run

Always activate the virtualenv and run from the repo root (`pip install -e .` once after setup):

```bash
source venv/bin/activate

# Phase 2 (thesis path)
python -m slm_experiments phase2 weights   [--weights 1.0,1.5,2.0,4.0] [--prompts N|all] [--models all]
python -m slm_experiments phase2 prompting [--shots 0,1,3]              [--prompts N|all] [--models all]
python -m slm_experiments phase2 guided    [--top-k-pools 5,10,20]      [--prompts N|all] [--models all]
python -m slm_experiments phase2 kvl_beam  [--widths 4,8]               [--prompts N|all] [--models all]

# Phase 1 (optional / out of thesis scope)
python -m slm_experiments phase1 [--prompts N|all] [--models all|Qwen3,...] [--seed 42] [--no-plot]

# Post-run
python -m slm_experiments plot --run-id <id>
python -m slm_experiments runs list
python -m slm_experiments runs show <id>

# Human eval
python -m slm_experiments human export --run-id <id> [--sample 60]
python -m slm_experiments human import --run-id <id> --tags <csv>
python -m slm_experiments human study-export --assessment-run-id <id>
python -m slm_experiments human study-import --assessment-run-id <id> --ratings <csv>

# Assessment + LLM-judge seam (no API adapter)
python -m slm_experiments assess build --source-run-ids <id> [<id> ...]
python -m slm_experiments assess analyze --assessment-run-id <id>   # re-run; also auto at end of build
python -m slm_experiments assess judge-export --assessment-run-id <id>
python -m slm_experiments assess judge-import --assessment-run-id <id> --scores judge_scores.csv
```

Auto-plot after each run by default; `--no-plot` to skip. Formal thesis claims: always Phase 2 with `--prompts all`.
`assess build` runs paired-delta analysis automatically (default 10,000 bootstrap resamples; seed in `manifest.analysis`). See [ExperimentDesign.md](ExperimentDesign.md) / Decision 2 for the protocol.

## 6. Results Contract

Every run → `results/runs/{run_id}/`:

| Artifact | Contents |
|----------|----------|
| `manifest.json` | Run metadata, CLI args, observation counts; assessment sets `kind: "assessment"` |
| `specification.csv` | Reduced columns, European decimals (paper-compatible) — generation runs |
| `full.csv` | All fields including guided / KVL metadata — generation runs |
| `summary.json` | Aggregates: overall, by_config, sweep sections, **by_model** |
| `plots/` | Boxplots (after `plot --run-id`) |
| `human_review.csv` | After legacy human export/import |
| `items.csv` / `item_map.csv` / `scores.csv` (assessment) | Deduped texts, source links (`cefr_sp_level_ordinal`, `meets_a1_criteria`), `cefr_tsar_*` + `kvl_v2_*` |
| `study/` (assessment) | Three-rater blinded study (see [docs/human-eval.md](docs/human-eval.md)) |
| `judge/` (assessment) | `judge_input.jsonl`, rubric, schema, optional `judge_scores.csv` |
| `analysis/` (assessment) | `paired_deltas.csv`, `analysis.json`, validation artifacts (auto after `assess build`; 10k resamples; seed in `manifest.analysis`) |

Run ID (generation): `{YYYYMMDD_HHMMSS}_{phase}_{experiment}`  
Run ID (assessment): `{YYYYMMDD_HHMMSS}_assessment_beginner_suitability`

Failed generations are recorded but excluded from metric means in `summary.json`. Rates (`a1_pass_rate`, `generation_failure_rate`, `hit_max_tokens_rate`) use all rows as denominator.

**Thesis table cell recipe** (every reported cell): print `a1_pass_rate`, `generation_failure_rate`, `hit_max_tokens_rate`, then conditional means (readability / KVL as appropriate). Stratify by model first.

## 7. Code Architecture

```
CLI → Phase runner → Pipeline (generate → format → evaluate → record) → RunStore
```

**Edit rules:**
- `cli.py` dispatches only — no experiment logic
- Phase runners (`phase1/runner.py`, `phase2/*.py`) orchestrate configs and call the pipeline
- `core/pipeline.py` owns the generate→format→evaluate→record loop
- `core/run_store.py` owns manifest, CSV, and summary writing
- Model wrappers extend `models/llamacpp.py` — one file per model in `models/wrappers/`

## 8. Models

→ Details in [docs/models.md](docs/models.md)

| Model | Parameters | GGUF File | GPU |
|-------|-----------|-----------|-----|
| Qwen2 | 0.5B | `qwen2.5-0.5b-instruct-q4_0.gguf` | No (`n_gpu_layers=0`) |
| Qwen3 | 0.6B | `Qwen3-0.6B-Q4_0.gguf` | No |
| TinyLlama | 1.1B | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | No |
| Phi3 | 3.8B | `Phi-3-mini-4k-instruct-q4.gguf` | Yes (`n_gpu_layers=-1`) |

## 9. Interventions

→ Details in [docs/interventions.md](docs/interventions.md)

| Intervention | Mechanism | Where |
|--------------|-----------|-------|
| Weighting | Logit bias on A1 vocabulary tokens (mid ∪ start IDs) | Phase 1 + Phase 2 weights |
| Prompting | System/context block for simplification | Phase 1 + Phase 2 prompting |
| Guided | Top-k A1-constrained greedy | Phase 2 guided |
| KVL beam | Token-level beam ranked by KVL/GLMM; **first-finish** stop (avoids max-length pad) | Phase 2 kvl_beam |
| Beam (deprecated) | Best-of-N + A1 ratio — hard-fails at CLI | Excluded |

## 10. Human Evaluation

```bash
python -m slm_experiments human export --run-id <id> [--sample 60]
python -m slm_experiments human import --run-id <id> --tags <csv>
python -m slm_experiments human study-export --assessment-run-id <id>
python -m slm_experiments human study-import --assessment-run-id <id> --ratings <csv>
```

**Three-rater study** (criterion path): blinded sheets under `{assessment}/study/`; four ordinal 1–4 dimensions; consensus = unweighted median; `human_suitable` iff median overall ≥ 3 **and** median adequacy ≥ 3; reliability = mean pairwise exact + adjacent (±1) percent agreement (**no Krippendorff's α**). Full protocol: [docs/human-eval.md](docs/human-eval.md).

Legacy `human export` / `import` samples rows from `specification.csv` into `human_review.csv` — smoke / historical only (not blind).

**LLM-judge seam** (provider-neutral; no API adapter):

```bash
python -m slm_experiments assess judge-export --assessment-run-id <id>
python -m slm_experiments assess judge-import --assessment-run-id <id> --scores judge_scores.csv
```

Quality-preservation claims stay **human-sample-limited** until imported judge scores agree acceptably with human consensus.

## 11. Plotting

```bash
python -m slm_experiments plot --run-id <id>
```

Reads `manifest.json` + `specification.csv` → writes boxplots to `runs/{id}/plots/`.

## 12. Testing

```bash
pip install -r requirements-dev.txt && pytest
```

No GGUF required for most tests (mocked pipeline).

## 13. Do NOT

- Import from old thesis `Codigo/`
- Add torch/transformers to **core** deps (`requirements.txt`) — only via optional extras (`[cefr-sp]`, `[cefr-tsar]`; `[kvl-v2]` is an empty install marker)
- Multiple CLI scripts (one `cli.py`)
- Results outside `results/runs/{run_id}/`
- SMOG metric
- Edit `SLMs-master-thesis/paper/`
- Duplicate ExperimentDesign content into AGENTS.md — link instead
- Cite deprecated `phase2 beam` runs in thesis claims
- Commit `data/cefr_sp/*.ckpt` (~1.2GB)
- Implement an LLM-judge API / provider adapter (seam only: export + import)
- Use the TSAR ensemble as an A1 gate (assessment-only ordinal / disagreement cross-check)

## 14. Relationship to Thesis Repo

Clean replacement for `Tesis/Codigo/`. Thesis repo frozen. Spec CSV format preserved for comparison with published results.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `srobaina99/SLMs-experiments` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.
