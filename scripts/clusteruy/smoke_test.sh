#!/bin/bash
#SBATCH --job-name=smoke_phi3
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srobaina99@gmail.com

# ============================================================
# Smoke test: Phi3 + 1 prompt + full weight grid on ClusterUY.
# Submit: sbatch scripts/clusteruy/smoke_test.sh
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
    python -m slm_experiments phase2 weights --models Phi3 --prompts 1 --no-plot'

echo "Done at $(date)"
