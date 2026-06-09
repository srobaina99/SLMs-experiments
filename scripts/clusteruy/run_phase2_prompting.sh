#!/bin/bash
#SBATCH --job-name=phase2_prompting
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --tmp=50G
#SBATCH --output=phase2_prompting_%j.out
#SBATCH --error=phase2_prompting_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srobaina99@gmail.com

# ============================================================
# Phase 2 prompting shot sweep (4 models × 3 shots × 25 prompts).
# Submit: sbatch scripts/clusteruy/run_phase2_prompting.sh
# See docs/clusteruy.md
# ============================================================

set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/SLMs-experiments}"
SIF_IMAGE="$HOME/slm-thesis.sif"
export SLM_GGUF_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo/models/gguf"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST at $(date)"
cd "$PROJECT_DIR"
nvidia-smi

singularity exec --nv \
  --bind "$PROJECT_DIR":/workspace \
  "$SIF_IMAGE" \
  bash -c 'export PYTHONPATH=/workspace/src && cd /workspace && \
    python -m slm_experiments phase2 prompting --prompts all --no-plot'

echo "Done at $(date)"
