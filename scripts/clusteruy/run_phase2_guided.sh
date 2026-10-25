#!/bin/bash
#SBATCH --job-name=phase2_guided
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --tmp=50G
#SBATCH --output=phase2_guided_%j.out
#SBATCH --error=phase2_guided_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srobaina99@gmail.com

# ============================================================
# Phase 2 guided top-k pool sweep (4 models × 4 pools × 25 prompts;
# default pools 0,5,10,20; 0 = unconstrained in-run baseline).
# Submit: sbatch scripts/clusteruy/run_phase2_guided.sh
# Requires data/cefr_sp/level_estimator.ckpt (or $CEFR_SP_CKPT) on the host.
# See docs/clusteruy.md
# ============================================================

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/SLMs-experiments}"
SIF_IMAGE="$HOME/slm-thesis.sif"
export SLM_GGUF_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo/models/gguf"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST at $(date)"
cd "$PROJECT_DIR"
nvidia-smi

# CEFR-SP is the A1 gate during generation. Require the ckpt on the host
# (bind-mounted; not baked into the image).
CKPT_HOST="${CEFR_SP_CKPT:-$PROJECT_DIR/data/cefr_sp/level_estimator.ckpt}"
if [[ ! -f "$CKPT_HOST" ]]; then
  echo "ERROR: CEFR-SP checkpoint missing: $CKPT_HOST" >&2
  echo "Thesis-valid Phase 2 jobs require the ckpt (scored during generation)." >&2
  echo "Download locally: ./venv/bin/python scripts/download_cefr_sp_ckpt.py" >&2
  echo "Then rsync/scp to the cluster (file-transfer port 10022), or set CEFR_SP_CKPT." >&2
  exit 1
fi
CEFR_BIND=(--bind "$CKPT_HOST":/workspace/data/cefr_sp/level_estimator.ckpt:ro)
echo "Binding CEFR-SP ckpt: $CKPT_HOST"

singularity exec --nv \
  --bind "$PROJECT_DIR":/workspace \
  "${CEFR_BIND[@]}" \
  "$SIF_IMAGE" \
  bash -c 'export PYTHONPATH=/workspace/src \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/opt/hf_home \
    NLTK_DATA=/usr/share/nltk_data \
    && cd /workspace \
    && python -m slm_experiments phase2 guided --prompts all --no-plot'

echo "Done at $(date)"
