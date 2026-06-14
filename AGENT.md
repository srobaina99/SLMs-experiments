# AGENT.md

## 1. Mission

Evaluate whether inference-time interventions make 4 SLMs produce CEFR A1 English.
Two research phases, one shared pipeline, run-centric results.

## 2. Documentation Map

| Document | Purpose | Read when |
|----------|---------|-----------|
| **AGENT.md** (this file) | Structure, CLI, code rules | Always start here |
| [ExperimentDesign.md](ExperimentDesign.md) | Formal experiment spec, phases, success criteria | Designing or changing experiments |
| [docs/metrics.md](docs/metrics.md) | Readability metrics, A1 thresholds | Working on `evaluation/` |
| [docs/models.md](docs/models.md) | GGUF files, templates, GPU setup | Adding/fixing model wrappers |
| [docs/interventions.md](docs/interventions.md) | Weighting, prompting, beam mechanics | Changing intervention logic |
| [docs/clusteruy.md](docs/clusteruy.md) | ClusterUY SSH, Singularity, Phase 2 batch jobs | Running experiments on the cluster |
| [README.md](README.md) | Human setup quickstart | Environment issues |

## 3. Research Phases

→ Full design in [ExperimentDesign.md](ExperimentDesign.md)

**Phase 1 — Factorial:** 4 models × 4 interventions × N prompts (default 3, `--prompts all` for 25)

**Phase 2 — Hyperparameter sweeps (all 4 models):**
- `weights` — logit bias grid with prompting ON
- `beam` — beam width sweep with A1-ratio selection
- `prompting` — zero/one/few-shot contextual prompting

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
├── models/gguf/              # GGUF model files (not in git)
├── results/runs/{run_id}/    # Run bundles (manifest, CSVs, summary, plots)
├── src/slm_experiments/      # Package source
│   ├── cli.py                # Single CLI entry point
│   ├── core/                 # Pipeline, config, result, run_store, prompts
│   ├── models/               # Base, llamacpp, beam, per-model wrappers
│   ├── evaluation/           # Metrics and response formatter
│   ├── phase1/               # Factorial experiment
│   ├── phase2/               # Weight, beam, prompting sweeps
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
python -m slm_experiments phase2 beam      [--widths 4,8,10]           [--prompts N|all] [--models all]
python -m slm_experiments phase2 prompting [--shots 0,1,3]              [--prompts N|all] [--models all]

# Post-run
python -m slm_experiments plot --run-id <id>
python -m slm_experiments runs list
python -m slm_experiments runs show <id>

# Human eval
python -m slm_experiments human export --run-id <id> [--sample 60]
python -m slm_experiments human import --run-id <id> --tags <csv>
```

Auto-plot after each run by default; `--no-plot` to skip.

## 6. Results Contract

Every run → `results/runs/{run_id}/`:

| Artifact | Contents |
|----------|----------|
| `manifest.json` | Run metadata, CLI args, observation counts |
| `specification.csv` | Reduced columns, European decimals (paper-compatible) |
| `full.csv` | All fields including beam metadata |
| `summary.json` | Aggregated stats (overall + by_config; Phase 2 adds by_weight_factor / by_beam_width / by_num_shots) |
| `plots/` | Boxplots (after `plot --run-id`) |
| `human_review.csv` | After human export/import |

Run ID: `{YYYYMMDD_HHMMSS}_{phase}_{experiment}`

Failed generations are recorded but excluded from metric aggregates in `summary.json`.

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

| Intervention | Mechanism | Phase 1 config flags |
|--------------|-----------|---------------------|
| Weighting | Logit bias on A1 vocabulary tokens | `config_weighting=True`, `weight_factor=1.5` |
| Prompting | System/context block for simplification | `config_prompting=True` |
| Beam | Generate N candidates, select highest A1 ratio | Phase 2 only |

## 10. Human Evaluation

```bash
python -m slm_experiments human export --run-id <id> [--sample 60]
python -m slm_experiments human import --run-id <id> --tags <csv>
```

Export samples rows from `specification.csv` into `human_review.csv`. Import merges reviewer tags back into the run bundle.

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

## 14. Relationship to Thesis Repo

Clean replacement for `Tesis/Codigo/`. Thesis repo frozen. Spec CSV format preserved for comparison with published results.
