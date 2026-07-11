# AGENT.md

## 1. Mission

Evaluate whether inference-time interventions make 4 SLMs produce simpler English answers for beginner learners.
Primary binary outcome is an **automated readability proxy** (`meets_a1_criteria`), not a CEFR proficiency test.
Two research phases, one shared pipeline, run-centric results.

## 2. Documentation Map

| Document | Purpose | Read when |
|----------|---------|-----------|
| **AGENT.md** (this file) | Structure, CLI, code rules | Always start here |
| [ExperimentDesign.md](ExperimentDesign.md) | Formal experiment spec, phases, success criteria | Designing or changing experiments |
| [docs/metrics.md](docs/metrics.md) | Readability metrics, proxy thresholds | Working on `evaluation/` |
| [docs/models.md](docs/models.md) | GGUF files, templates, GPU setup | Adding/fixing model wrappers |
| [docs/interventions.md](docs/interventions.md) | Weighting, prompting, guided, KVL, deprecated beam | Changing intervention logic |
| [docs/guided-decoding.md](docs/guided-decoding.md) | Top-k A1-guided decoding (`phase2 guided`) | Changing guided decode |
| [docs/kvl_beamsearch.md](docs/kvl_beamsearch.md) | KVL-scored beam (`phase2 kvl_beam`) | Changing KVL beam |
| [docs/clusteruy.md](docs/clusteruy.md) | ClusterUY SSH, Singularity, Phase 2 batch jobs | Running experiments on the cluster |
| [docs/experiment-setup-recommendations.md](docs/experiment-setup-recommendations.md) | Thesis-defensibility checklist | Before publishing claims |
| [README.md](README.md) | Human setup quickstart | Environment issues |

## 3. Research Phases

→ Full design in [ExperimentDesign.md](ExperimentDesign.md)

**Phase 1 — Factorial:** 4 models × 4 interventions × N prompts (default 3 smoke; `--prompts all` for 25 formal)

**Phase 2 — Hyperparameter sweeps (all 4 models):**
- `weights` — logit bias grid with prompting ON
- `prompting` — zero/one/few-shot contextual prompting
- `guided` — A1-constrained greedy top-k pool sweep
- `kvl_beam` — KVL-scored beam width sweep
- `beam` — **deprecated / hard-fails** (void at temperature 0)

Shared generation defaults: `temperature=0.0`, `top_k=50`, max 200 new tokens; **no `top_p`**.

## 4. Directory Map

```
SLMs-experiments/
├── AGENT.md                  # This file — agent entry point
├── README.md                 # Human quickstart
├── ExperimentDesign.md       # Formal experiment specification
├── scripts/clusteruy/        # SLURM batch scripts (smoke test + Phase 2 sweeps)
├── requirements.txt          # Runtime dependencies (7 packages)
├── requirements-dev.txt      # pytest
├── pytest.ini
├── docs/                     # Detailed reference docs
├── data/vocabularies/        # A1 vocabulary (487 words)
├── data/kvl/                 # KVL lookup tables (es/de/cn)
├── models/gguf/              # GGUF model files (not in git)
├── results/runs/{run_id}/    # Run bundles (manifest, CSVs, summary, plots)
├── src/slm_experiments/      # Package source
│   ├── cli.py                # Single CLI entry point
│   ├── core/                 # Pipeline, config, result, run_store, prompts
│   ├── models/               # Base, llamacpp, guided/KVL decoders, wrappers
│   ├── evaluation/           # Metrics and response formatter
│   ├── phase1/               # Factorial experiment
│   ├── phase2/               # Weight, prompting, guided, kvl_beam (+ deprecated beam)
│   ├── human/                # Export/import for human review
│   └── plot.py               # Boxplot generation from run bundles
└── tests/                    # pytest suite (mocked pipeline)
```

## 5. How to Run

Always activate the virtualenv and run from the repo root (`pip install -e .` once after setup):

```bash
source venv/bin/activate

# Phase 1
python -m slm_experiments phase1 [--prompts N|all] [--models all|Qwen3,...] [--seed 42] [--no-plot]

# Phase 2
python -m slm_experiments phase2 weights   [--weights 1.0,1.5,2.0,4.0] [--prompts N|all] [--models all]
python -m slm_experiments phase2 prompting [--shots 0,1,3]              [--prompts N|all] [--models all]
python -m slm_experiments phase2 guided    [--top-k-pools 5,10,20]      [--prompts N|all] [--models all]
python -m slm_experiments phase2 kvl_beam  [--widths 4,8]               [--prompts N|all] [--models all]

# Post-run
python -m slm_experiments plot --run-id <id>
python -m slm_experiments runs list
python -m slm_experiments runs show <id>

# Human eval
python -m slm_experiments human export --run-id <id> [--sample 60]
python -m slm_experiments human import --run-id <id> --tags <csv>
```

Auto-plot after each run by default; `--no-plot` to skip. Formal claims: always `--prompts all`.

## 6. Results Contract

Every run → `results/runs/{run_id}/`:

| Artifact | Contents |
|----------|----------|
| `manifest.json` | Run metadata, CLI args, observation counts |
| `specification.csv` | Reduced columns, European decimals (paper-compatible) |
| `full.csv` | All fields including guided / KVL metadata |
| `summary.json` | Aggregates: overall, by_config, sweep sections, **by_model** |
| `plots/` | Boxplots (after `plot --run-id`) |
| `human_review.csv` | After human export/import |

Run ID: `{YYYYMMDD_HHMMSS}_{phase}_{experiment}`

Failed generations are recorded but excluded from metric aggregates in `summary.json`. Thesis tables: stratify by model first.

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
```

Export samples rows from `specification.csv` into `human_review.csv`. Import merges reviewer tags back into the run bundle. Use for agreement with the readability proxy.

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
- Add torch/transformers
- Multiple CLI scripts (one `cli.py`)
- Results outside `results/runs/{run_id}/`
- SMOG metric
- Edit `SLMs-master-thesis/paper/`
- Duplicate ExperimentDesign content into AGENT.md — link instead
- Cite deprecated `phase2 beam` runs in thesis claims

## 14. Relationship to Thesis Repo

Clean replacement for `Tesis/Codigo/`. Thesis repo frozen. Spec CSV format preserved for comparison with published results.
