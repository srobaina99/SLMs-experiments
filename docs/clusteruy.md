# ClusterUY — Agent Connection Guide

How an agent (or human) connects to [ClusterUY](https://www.cluster.uy/) and runs
Phase 2 experiments for this repo.

**Official docs:** https://www.cluster.uy/ayuda/

**Dockerfile location:** `../SLMs-master-thesis/Tesis/Codigo/scripts/clusteruy/Dockerfile`

---

## Agent prompt (copy-paste to start a cluster session)

```
You are helping run SLM Phase 2 experiments on ClusterUY (Uruguay national HPC cluster).

CONTEXT
- Repo: SLMs-experiments (this repo). CLI: python -m slm_experiments phase2 {weights|beam|prompting}
- ClusterUY does NOT run Docker. Workflow: build image with Docker locally → push to Docker Hub →
  pull as Singularity .sif on cluster → run with singularity exec --nv
- GPU: Tesla P100 (sm_60). Phi-3 needs GPU; other models run on CPU inside the container.
- Login node is for file management and job submission ONLY. Never run experiments there.
  Always use sbatch (batch) or interactivo -gpun (short smoke test).
- Package install: use PYTHONPATH=/workspace/src (pip install -e . fails in the lean image).

CONNECTION
  ssh santiago.robaina@login.cluster.uy
No VPN. SSH key auth only (password login is not used).

VERIFY YOU ARE ON THE CLUSTER:
  hostname          # e.g. login.datos.cluster.uy
  whoami            # santiago.robaina
  quota -gvs        # home dir quota (300 GB)
  squeue -u $USER   # your job queue

EXPECTED HOME LAYOUT (may already exist from prior work):
  ~/slm-thesis.sif                              # Singularity image (~1.5 GB lean build)
  ~/SLMs-experiments/                           # this repo
  ~/SLMs-master-thesis/Tesis/Codigo/models/gguf/  # GGUF models (~3.7 GB)

PHASE 2 COMMANDS (inside Singularity on a compute node):
  export SLM_GGUF_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo/models/gguf"
  cd ~/SLMs-experiments

  singularity exec --nv --bind $(pwd):/workspace ~/slm-thesis.sif \
    bash -c 'export PYTHONPATH=/workspace/src && cd /workspace && \
      python -m slm_experiments phase2 weights --prompts all --no-plot'

SBATCH SCRIPTS (submit from login node):
  cd ~/SLMs-experiments
  sbatch scripts/clusteruy/smoke_test.sh
  sbatch scripts/clusteruy/run_phase2_weights.sh
  sbatch scripts/clusteruy/run_phase2_beam.sh
  sbatch scripts/clusteruy/run_phase2_prompting.sh

DOWNLOAD RESULTS (from user's local machine, port 10022):
  rsync -avz -e "ssh -p 10022" \
    santiago.robaina@cluster.uy:~/SLMs-experiments/results/runs/ \
    ./results/runs/

RULES
- Do not run long commands on the login node (processes get killed).
- Use sbatch for pulls, downloads, and full experiments.
- Always pass --no-plot on cluster; plot locally after rsync.
- Do not modify the .sif image on cluster; rebuild Docker image locally if deps change.
- Ask the user before submitting jobs that consume GPU quota.
```

---

## Connection details

### SSH

| Item | Value |
|------|-------|
| Host | `login.cluster.uy` |
| Username | `santiago.robaina` |
| Port (SSH) | 22 (default) |
| Auth | SSH public key (registered at signup) |
| VPN | Not required |
| File transfer port | **10022** (preferred for scp/rsync — avoids login-node bandwidth limits) |

```bash
ssh santiago.robaina@login.cluster.uy

# With explicit key
ssh -i ~/.ssh/id_ed25519 santiago.robaina@login.cluster.uy

# File transfer from local machine
rsync -avz -e "ssh -p 10022" \
  santiago.robaina@cluster.uy:~/SLMs-experiments/results/runs/ ./results/runs/
```

Ref: [Cómo conectarse](https://www.cluster.uy/ayuda/como_conectarse/)

### Prerequisites (human must set up once)

1. [ClusterUY account](https://www.cluster.uy/registro/) with SSH public key submitted
2. UdelaR students: written endorsement from a faculty supervisor
3. SSH key loaded locally (`ssh-add ~/.ssh/id_ed25519` if needed)
4. Docker image built and pushed from local machine (see below)
5. Repos cloned on the cluster

---

## Docker vs Singularity (important)

ClusterUY runs **CentOS 7** — too old for modern Python/CUDA wheels. The workaround:

| Step | Where | Tool |
|------|-------|------|
| Build container | Local Mac/Linux | **Docker** |
| Publish | Docker Hub | `docker push` |
| Pull on cluster | ClusterUY compute node | **Singularity** (`singularity pull docker://...`) |
| Run experiments | ClusterUY compute node | **Singularity** (`singularity exec --nv`) |

There is no `docker` command on ClusterUY. Never instruct the agent to run `docker` on the cluster.

### Build image locally (one-time or when deps change)

```bash
cd ../SLMs-master-thesis/Tesis/Codigo
docker login
docker build --platform=linux/amd64 -t srobaina99/slm-thesis:latest scripts/clusteruy/
docker push srobaina99/slm-thesis:latest
```

The Dockerfile compiles `llama-cpp-python` for P100 (`sm_60`). Prebuilt CUDA wheels
target `sm_70+` and crash on ClusterUY. The lean image is ~1.5 GB as a `.sif` file.

### Pull Singularity image on cluster

**Use sbatch** — the login node kills long-running processes:

```bash
cd ~/SLMs-master-thesis
sbatch Tesis/Codigo/scripts/clusteruy/pull_image.sh srobaina99/slm-thesis:latest
```

Ref: [Singularity on ClusterUY](https://www.cluster.uy/ayuda/singularity/)

---

## First-time cluster setup

Run on the **login node** (short commands only):

```bash
# 1. Clone repos
git clone https://github.com/srobaina99/SLMs-experiments.git ~/SLMs-experiments
git clone https://github.com/srobaina99/SLMs-master-thesis.git ~/SLMs-master-thesis

# 2. Download GGUF models (submit as batch job — ~3.7 GB)
cd ~/SLMs-master-thesis
sbatch Tesis/Codigo/scripts/clusteruy/download_models.sh

# 3. Pull Singularity image (submit as batch job)
sbatch Tesis/Codigo/scripts/clusteruy/pull_image.sh srobaina99/slm-thesis:latest
```

Monitor:

```bash
squeue -u $USER --long
tail -f smoke_<jobid>.out
```

---

## Running experiments

### Why `PYTHONPATH` instead of `pip install`

The lean Singularity image lacks a PEP 660 build backend. Inside the container:

```bash
export PYTHONPATH=/workspace/src
python -m slm_experiments ...
```

Do **not** use `pip install -e .` — it fails with a missing `build_editable` hook.

### Smoke test (verified 2026-06-09)

Submit from the login node:

```bash
cd ~/SLMs-experiments
sbatch scripts/clusteruy/smoke_test.sh
```

Expected result: run bundle `results/runs/*_phase2_weights/` with 7 observations
(Phi3, 1 prompt, full weight grid), all successful. Job completed in ~49s on node24 (P100).

For an interactive session instead:

```bash
interactivo -gpun
export SLM_GGUF_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo/models/gguf"
cd ~/SLMs-experiments
singularity exec --nv --bind $(pwd):/workspace ~/slm-thesis.sif \
  bash -c 'export PYTHONPATH=/workspace/src && cd /workspace && \
    python -m slm_experiments phase2 weights --models Phi3 --prompts 1 --no-plot'
```

Ref: [Cómo ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)

### Phase 2 batch jobs

| Script | Sweep | Observations (`--prompts all`) | Time limit |
|--------|-------|-------------------------------|------------|
| `scripts/clusteruy/run_phase2_weights.sh` | 7 weight factors | 700 | 12 h |
| `scripts/clusteruy/run_phase2_beam.sh` | 3 beam widths | 300 | 24 h |
| `scripts/clusteruy/run_phase2_prompting.sh` | 3 shot counts | 300 | 12 h |

```bash
cd ~/SLMs-experiments
sbatch scripts/clusteruy/run_phase2_weights.sh
sbatch scripts/clusteruy/run_phase2_beam.sh
sbatch scripts/clusteruy/run_phase2_prompting.sh
```

Edit `--mail-user` in each script before submitting.

### SLURM defaults

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--partition` | `normal` | Guaranteed resources |
| `--qos` | `gpu` | Required for GPU jobs |
| `--gres` | `gpu:1` | Phi-3 needs one GPU |
| `--cpus-per-task` | 8 | Data loading + text evaluation |
| `--mem` | 32768 | Model loading headroom |
| `--tmp` | `50G` | Scratch space |

### Monitor and cancel

```bash
squeue -u $USER --long
tail -f phase2_weights_<jobid>.out
scancel <jobid>
seff <jobid>
```

---

## Download results and plot locally

Results are in `~/SLMs-experiments/results/runs/{run_id}/` on the cluster.

```bash
rsync -avz -e "ssh -p 10022" \
  santiago.robaina@cluster.uy:~/SLMs-experiments/results/runs/ \
  ./results/runs/

python -m slm_experiments plot --run-id <run_id>
```

Ref: [Tips y buenas prácticas](https://www.cluster.uy/ayuda/tips/)

---

## Operational rules (learned from prior runs)

| Rule | Why |
|------|-----|
| Never run experiments on the login node | Login cgroup kills long processes |
| Always submit pulls/downloads via `sbatch` | SSH drops kill interactive sessions; no tmux/screen |
| Use `$SLURM_SUBMIT_DIR` in sbatch scripts | `$0` points to SLURM spool copy, not the repo |
| Use `PYTHONPATH=/workspace/src` in container | `pip install -e .` fails in lean image |
| Pre-download NLTK data in the Docker image | Compute nodes may be air-gapped |
| Use `--no-plot` on cluster | No display; matplotlib not in lean image |
| Delete stale `~/singularity_tmp/` before re-pull | Failed pulls can fill the 300 GB home quota |
| No cluster backups | Download results promptly |

---

## Verify SSH before asking the agent to connect

```bash
ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes santiago.robaina@login.cluster.uy "hostname && whoami"
```

| Result | Meaning | Fix |
|--------|---------|-----|
| `login.datos.cluster.uy` + `santiago.robaina` | Ready | Agent can connect |
| `Permission denied (publickey,...)` | Key not authorized | Register `~/.ssh/id_ed25519.pub` with ClusterUY |
| `Connection timed out` | Network issue | Check internet; no VPN needed |

---

## Related files

| Location | Contents |
|----------|----------|
| `scripts/clusteruy/` | sbatch scripts for smoke test and Phase 2 sweeps |
| `../SLMs-master-thesis/Tesis/Codigo/scripts/clusteruy/` | Dockerfile, pull/download scripts |
| [ExperimentDesign.md](../ExperimentDesign.md) | Phase 2 sweep specifications |
| [docs/models.md](models.md) | GGUF filenames, GPU requirements |
