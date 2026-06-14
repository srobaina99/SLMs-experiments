# SLM Experiments

Evaluate whether inference-time interventions make small language models (0.5B–3.8B) produce CEFR A1-level English for beginner learners.

## Quick Start

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies (editable install adds src/ to Python path)
pip install -e .
pip install -r requirements-dev.txt

# 3. GGUF models — auto-resolved from sibling thesis repo by default:
#    ../SLMs-master-thesis/Tesis/Codigo/models/gguf/
#    Override with SLM_GGUF_DIR or copy files into models/gguf/

# 4. Run Phase 1 factorial experiment (default: 3 prompts, 4 models)
python -m slm_experiments phase1

# Full experiment (25 prompts, 400 observations)
python -m slm_experiments phase1 --prompts all
```

## CLI Overview

Run `python -m slm_experiments --help` for a quick-start guide with examples.

```bash
# Phase 1 — 2×2 factorial (4 models × 4 interventions)
python -m slm_experiments phase1 [--prompts N|all] [--models all|Qwen3,...] [--seed 42]

# Phase 2 — hyperparameter sweeps (all 4 models)
python -m slm_experiments phase2 weights   [--weights 1.0,1.5,2.0,4.0]
python -m slm_experiments phase2 beam      [--widths 4,8,10]
python -m slm_experiments phase2 prompting [--shots 0,1,3]

# Post-run utilities
python -m slm_experiments plot --run-id <id>
python -m slm_experiments runs list
python -m slm_experiments runs show <id>

# Human evaluation
python -m slm_experiments human export --run-id <id> [--sample 60]
python -m slm_experiments human import --run-id <id> --tags <csv>
```

## Results

Every run writes a self-contained bundle to `results/runs/{run_id}/`:

| File | Description |
|------|-------------|
| `manifest.json` | Run metadata, CLI args, observation counts |
| `specification.csv` | Reduced columns, European decimals (paper-compatible) |
| `full.csv` | All fields including beam metadata |
| `summary.json` | Aggregated stats (overall + by_config; Phase 2 adds sweep sections) |
| `plots/` | Boxplots (after `plot --run-id`) |

Run ID format: `{YYYYMMDD_HHMMSS}_{phase}_{experiment}`

## ClusterUY (HPC)

Run Phase 2 sweeps on Uruguay's national cluster via Singularity (not Docker on-cluster):

```bash
# On cluster login node
cd ~/SLMs-experiments
sbatch scripts/clusteruy/smoke_test.sh          # quick Phi3 check
sbatch scripts/clusteruy/run_phase2_weights.sh  # full sweep
```

Full workflow (SSH, image pull, results download): [docs/clusteruy.md](docs/clusteruy.md)

Dockerfile lives in the sibling thesis repo:
`../SLMs-master-thesis/Tesis/Codigo/scripts/clusteruy/Dockerfile`

## Documentation

| Document | Purpose |
|----------|---------|
| [AGENT.md](AGENT.md) | Agent entry point — structure, CLI, rules |
| [ExperimentDesign.md](ExperimentDesign.md) | Formal experiment specification |
| [docs/clusteruy.md](docs/clusteruy.md) | ClusterUY SSH, Singularity, batch jobs |
| [docs/metrics.md](docs/metrics.md) | Readability metrics and A1 thresholds |
| [docs/models.md](docs/models.md) | GGUF files, chat templates, GPU setup |
| [docs/interventions.md](docs/interventions.md) | Weighting, prompting, beam mechanics |

## Testing

```bash
pytest
```

Most tests mock the pipeline — no GGUF files required.

## Relationship to Thesis Repo

This repo is a clean replacement for `Tesis/Codigo/` in the [SLMs-master-thesis](https://github.com/) repository. The thesis paper is frozen; this repo carries the experiment framework forward with run-centric results and a single CLI entry point.
