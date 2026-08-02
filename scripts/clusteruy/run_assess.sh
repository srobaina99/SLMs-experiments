#!/bin/bash
#SBATCH --job-name=assess
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --tmp=50G
#SBATCH --output=assess_%j.out
#SBATCH --error=assess_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srobaina99@gmail.com

# ============================================================
# Assessment build on ClusterUY (TSAR + KVL v2) using slm-thesis-eval.
# Submit: sbatch scripts/clusteruy/run_assess.sh [source_run_id ...]
# Or:     SOURCE_RUN_IDS="id1 id2" sbatch scripts/clusteruy/run_assess.sh
#
# Requires ~/slm-thesis-eval.sif (see docs/clusteruy.md eval image section).
# Offline: TRANSFORMERS_OFFLINE=1, HF_HOME=/opt/hf_home (baked pins).
# P100-safe: never flash-attn (eager on sm_60).
# CEFR-SP ckpt optional for assess (labels joined from source full.csv).
#   If needed: bind -B $HOME/level_estimator.ckpt:/workspace/data/cefr_sp/level_estimator.ckpt:ro
# See docs/clusteruy.md
# ============================================================

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/SLMs-experiments}"
SIF_IMAGE="$HOME/slm-thesis-eval.sif"

# Source generation run ids: CLI args win, else SOURCE_RUN_IDS env, else error.
if [[ $# -gt 0 ]]; then
  SOURCE_IDS=("$@")
elif [[ -n "${SOURCE_RUN_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  SOURCE_IDS=($SOURCE_RUN_IDS)
else
  echo "ERROR: pass source run ids as args or set SOURCE_RUN_IDS" >&2
  echo "  sbatch scripts/clusteruy/run_assess.sh <run_id> [<run_id> ...]" >&2
  exit 1
fi

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST at $(date)"
echo "Source run ids: ${SOURCE_IDS[*]}"
cd "$PROJECT_DIR"
nvidia-smi

# Optional CEFR-SP ckpt bind (only if present on the host).
CEFR_BIND=()
CKPT_HOST="${CEFR_SP_CKPT:-$PROJECT_DIR/data/cefr_sp/level_estimator.ckpt}"
if [[ -f "$CKPT_HOST" ]]; then
  CEFR_BIND=(--bind "$CKPT_HOST":/workspace/data/cefr_sp/level_estimator.ckpt:ro)
  echo "Binding CEFR-SP ckpt: $CKPT_HOST"
fi

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
    && python -m slm_experiments assess build \
         --source-run-ids "$@" --no-plot' \
  _ "${SOURCE_IDS[@]}"

echo "Done at $(date)"
