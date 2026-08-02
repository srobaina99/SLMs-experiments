# ClusterUY — Agent Connection Guide

How an agent (or human) connects to [ClusterUY](https://www.cluster.uy/) and runs
Phase 2 experiments for this repo.

**Official docs:** https://www.cluster.uy/ayuda/

**Dockerfile location (generation, lean):** `../SLMs-master-thesis/Tesis/Codigo/scripts/clusteruy/Dockerfile`
**Dockerfile location (assessment, eval):** `scripts/clusteruy/Dockerfile.eval` (this repo)

---

## Agent prompt (copy-paste to start a cluster session)

```
You are helping run SLM Phase 2 experiments on ClusterUY (Uruguay national HPC cluster).

CONTEXT
- Repo: SLMs-experiments (this repo). CLI: python -m slm_experiments phase2 {weights|prompting|guided|kvl_beam}
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
  ~/slm-thesis.sif                              # Singularity image (~1.5 GB lean generation)
  ~/slm-thesis-eval.sif                         # Assessment image (torch + TSAR pins; heavier)
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
  sbatch scripts/clusteruy/run_phase2_prompting.sh
  sbatch scripts/clusteruy/run_phase2_kvl_beam.sh
  # run_phase2_beam.sh is deprecated (hard-fails) — use kvl_beam or guided
  # guided: python -m slm_experiments phase2 guided --prompts all --no-plot

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

### Eval image (`slm-thesis-eval`) — assessment path

The **generation** image (`slm-thesis` / `~/slm-thesis.sif`) stays lean and **torch-free**.
Assessment (`assess build`: TSAR ModernBERT ensemble + KVL v2) needs a **separate heavier
image** with torch, `transformers>=4.55`, pinned ModernBERT SHAs under `HF_HOME`, and
NLTK corpora under `NLTK_DATA`.

| Item | Value |
|------|-------|
| Dockerfile | `scripts/clusteruy/Dockerfile.eval` (this repo; build context = repo root) |
| Docker Hub tag | `srobaina99/slm-thesis-eval:latest` |
| Singularity path | `~/slm-thesis-eval.sif` |
| Base image | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (`linux/amd64`) |
| Torch pin | `torch==2.4.1` from `https://download.pytorch.org/whl/cu121` (includes `sm_60`) |
| Transformers pin | `transformers==4.55.4` (ModernBERT; stay on 4.x — do not float to 5.x) |
| `HF_HOME` (baked) | `/opt/hf_home` |
| `NLTK_DATA` (baked) | `/usr/share/nltk_data` |
| CEFR-SP `.ckpt` | **Bind-mount** (not baked; ~1.2 GB, gitignored). Optional for `assess` — labels are joined from the source run `full.csv` |

The eval image does **not** bake `data/` (KVL lookups, vocabularies, CEFR-SP ckpt). Every `assess`
invocation must bind the repo to `/workspace` so `PYTHONPATH=/workspace/src` and
`data/kvl/` resolve from the host checkout (same bind used by generation jobs).

**P100 / attention:** Torch is pinned to a CUDA 12.1 wheel that ships `sm_60` (build asserts
`'sm_60' in (torch.cuda.get_arch_list() or torch._C._cuda_getArchFlags().split())` —
`get_arch_list()` alone is `[]` on this driver-less image). TSAR uses
`resolve_attn_implementation` — never `flash_attention_2`; `eager` on sm_60 (P100), `sdpa`
otherwise. Do not install flash-attn.

**Offline / air-gapped:** compute nodes may have no Hub access. The image stages the three
pinned ModernBERT revisions at build time (`scripts/clusteruy/prefetch_tsar_models.py`
AST-parses `TSAR_MODELS` from `cefr_tsar.py` — single SHA source of truth, no package
import side effects). Runtime must set `TRANSFORMERS_OFFLINE=1` (and `HF_HUB_OFFLINE=1`)
with `HF_HOME=/opt/hf_home`. NLTK WordNet + tagger corpora are baked so KVL v2 does not
attempt `nltk.download` on the node.

#### Local build + smoke (Mac/Linux)

```bash
# From SLMs-experiments repo root
./scripts/clusteruy/build_eval_image.sh
# equivalent:
# docker build --platform=linux/amd64 \
#   -f scripts/clusteruy/Dockerfile.eval \
#   -t srobaina99/slm-thesis-eval:latest .

./scripts/clusteruy/smoke_assess_local.sh
```

Smoke stages the committed fixture
`scripts/clusteruy/fixtures/20260101_000000_phase2_weights_smoke/` into
`results/runs/`, runs `assess build` in-container with `--network=none`,
`TRANSFORMERS_OFFLINE=1`, `HF_HOME=/opt/hf_home`, `PYTHONPATH=/workspace/src`,
`--no-plot`, then **asserts** every `scores.csv` row has `cefr_tsar_status=ok` and
`kvl_v2_status=ok` (exit 0 alone is not enough — scorer failures become error rows).

Device auto-detect inside the linux/amd64 container is typically **CPU** on Apple Silicon
(QEMU; no CUDA/MPS in that VM). On ClusterUY with `singularity exec --nv`, the expectation
is `detect_device() -> 'cuda'` once the P100-capable torch wheel is present — **not yet
verified on-cluster** (local smoke only proves offline amd64 CPU). Host Mac `assess`
(outside Docker) can use MPS when torch is installed locally.

Optional CEFR-SP ckpt bind for local Docker (only if you have the file):

```bash
docker run --rm --platform=linux/amd64 --network=none \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/data/cefr_sp/level_estimator.ckpt:/workspace/data/cefr_sp/level_estimator.ckpt:ro" \
  -e PYTHONPATH=/workspace/src \
  -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/opt/hf_home -e NLTK_DATA=/usr/share/nltk_data \
  srobaina99/slm-thesis-eval:latest \
  bash -c 'cd /workspace && python -m slm_experiments assess build \
    --source-run-ids 20260101_000000_phase2_weights_smoke --no-plot'
```

#### Observed smoke output (rebuilt P100-safe image, 2026-08-01, Docker Desktop 4.80 / engine 29.6.1, macOS arm64)

Build:

```bash
./scripts/clusteruy/build_eval_image.sh
# docker build --platform=linux/amd64 \
#   -f scripts/clusteruy/Dockerfile.eval \
#   -t srobaina99/slm-thesis-eval:latest .
# real 260.01s  →  srobaina99/slm-thesis-eval:latest (content 6.22GB)
# build-time assert printed:
#   torch 2.4.1+cu121 cuda 12.1 arch_list ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
#   post-deps torch 2.4.1+cu121 cuda 12.1 arch_list ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

Smoke (`--network=none`, offline HF/NLTK):

```text
$ ./scripts/clusteruy/smoke_assess_local.sh
Staged fixture → .../results/runs/20260101_000000_phase2_weights_smoke
Running in-container assess (network=none, TRANSFORMERS_OFFLINE=1)…
Device set to use cpu
Device set to use cpu
Device set to use cpu
[... NVIDIA CUDA banner elided ...]
WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
Assessment bundle complete: 20260801_232448_assessment_beginner_suitability
Output: /workspace/results/runs/20260801_232448_assessment_beginner_suitability
Asserting scorer statuses in .../results/runs/20260801_232448_assessment_beginner_suitability/scores.csv …
OK: 2/2 rows with cefr_tsar_status=ok and kvl_v2_status=ok
assessment_run_id=20260801_232448_assessment_beginner_suitability
smoke_assess_local: SUCCESS (run_id=20260801_232448_assessment_beginner_suitability)
real 33.78s
```

In-container torch / device probe (same image, `--network=none`), via:

```bash
python -c "import torch; flags=torch._C._cuda_getArchFlags() or ''; arches=torch.cuda.get_arch_list() or flags.split(); print(...)"
```

```text
torch 2.4.1+cu121
cuda 12.1
get_arch_list() []
_cuda_getArchFlags() 'sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90'   # raw str from the API
effective arch_list ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']  # after .split()
sm_60 True
transformers 4.55.4
detect_device() 'cpu'
resolve_attn_implementation() 'sdpa'
```

Note: under this driver-less runtime image `torch.cuda.get_arch_list()` returns `[]`; the
Dockerfile assert (and the probe above) therefore falls back to
`torch._C._cuda_getArchFlags()`, which is where `sm_60` is visible for `2.4.1+cu121`.

`scores.csv`: 2/2 rows `cefr_tsar_status=ok` and `kvl_v2_status=ok`. Platform under test was
`linux/amd64` via QEMU on Apple Silicon — proves the offline amd64 image + P100-capable
wheel. On-cluster CUDA (`singularity exec --nv` → `detect_device() == 'cuda'`) remains an
expectation pending ClusterUY verification.

#### Docker → Hub → `.sif` (eval image)

```bash
# Local
docker login
./scripts/clusteruy/build_eval_image.sh
docker push srobaina99/slm-thesis-eval:latest

# ClusterUY login node — submit pull as batch (login kills long pulls)
cd ~/SLMs-experiments
sbatch scripts/clusteruy/pull_eval_image.sh srobaina99/slm-thesis-eval:latest
# → writes ~/slm-thesis-eval.sif
```

#### Cluster `assess` sbatch

```bash
cd ~/SLMs-experiments
sbatch scripts/clusteruy/run_assess.sh <source_run_id> [<source_run_id> ...]
# or: SOURCE_RUN_IDS="id1 id2" sbatch scripts/clusteruy/run_assess.sh
```

Uses `singularity exec --nv`, `PYTHONPATH=/workspace/src`, `TRANSFORMERS_OFFLINE=1`,
`HF_HOME=/opt/hf_home`, `NLTK_DATA=/usr/share/nltk_data`, `--no-plot`. If
`data/cefr_sp/level_estimator.ckpt` (or `$CEFR_SP_CKPT`) exists on the host, it is
bind-mounted read-only.

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
| `scripts/clusteruy/run_phase2_prompting.sh` | 3 shot counts | 300 | 12 h |
| `scripts/clusteruy/run_phase2_kvl_beam.sh` | 2 KVL beam widths | 200 | (see script) |
| `scripts/clusteruy/run_phase2_beam.sh` | **Deprecated** | — | exits 1 |
| `scripts/clusteruy/run_assess.sh` | Assessment (TSAR + KVL v2) | source-run dependent | 4 h |

```bash
cd ~/SLMs-experiments
sbatch scripts/clusteruy/run_phase2_weights.sh
sbatch scripts/clusteruy/run_phase2_prompting.sh
sbatch scripts/clusteruy/run_phase2_kvl_beam.sh
# guided (no dedicated sbatch yet):
# singularity … python -m slm_experiments phase2 guided --prompts all --no-plot
# assessment (requires ~/slm-thesis-eval.sif):
sbatch scripts/clusteruy/run_assess.sh <source_run_id>
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
| Generation vs eval images | Lean `~/slm-thesis.sif` for Phase 2 gen; `~/slm-thesis-eval.sif` for `assess` (torch + HF pins) |
| `TRANSFORMERS_OFFLINE=1` + baked `HF_HOME` on assess | No Hub fetch on air-gapped nodes; pins staged at image build |
| Never flash-attn on ClusterUY | P100 is sm_60; TSAR forces eager/sdpa only |
| CEFR-SP `.ckpt` bind-mount for eval image | Not baked (~1.2 GB); optional for `assess` (joins source labels) |

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
| `scripts/clusteruy/` | sbatch scripts for smoke test, Phase 2 sweeps, and `assess` |
| `scripts/clusteruy/Dockerfile.eval` | `slm-thesis-eval` recipe (torch + TSAR pins + NLTK) |
| `scripts/clusteruy/build_eval_image.sh` | Local `docker build --platform=linux/amd64` |
| `scripts/clusteruy/smoke_assess_local.sh` | Offline in-container assess + status asserts |
| `scripts/clusteruy/run_assess.sh` | ClusterUY `assess build` sbatch |
| `scripts/clusteruy/pull_eval_image.sh` | Singularity pull → `~/slm-thesis-eval.sif` |
| `scripts/clusteruy/prefetch_tsar_models.py` | Build-time HF + NLTK prefetch (AST-parses `TSAR_MODELS`) |
| `scripts/clusteruy/fixtures/` | Committed tiny generation bundle for assess smoke |
| `../SLMs-master-thesis/Tesis/Codigo/scripts/clusteruy/` | Lean generation Dockerfile, pull/download scripts |
| [ExperimentDesign.md](../ExperimentDesign.md) | Phase 2 sweep specifications |
| [docs/models.md](models.md) | GGUF filenames, GPU requirements |
