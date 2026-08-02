#!/bin/bash
#SBATCH --job-name=pull_eval_image
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --time=06:00:00
#SBATCH --output=pull_eval_image_%j.out
#SBATCH --error=pull_eval_image_%j.err

# ============================================================
# Pull the slm-thesis-eval Singularity image on a compute node.
# Login node kills long-running pulls; use sbatch.
#
# Submit: sbatch scripts/clusteruy/pull_eval_image.sh [dockerhub_user/image:tag]
# Default: srobaina99/slm-thesis-eval:latest → ~/slm-thesis-eval.sif
#
# Mirrors thesis-repo pull_image.sh but writes the eval .sif name.
# ============================================================

IMAGE_REF="${1:-srobaina99/slm-thesis-eval:latest}"
SIF_PATH="$HOME/slm-thesis-eval.sif"

mkdir -p "$HOME/singularity_tmp"
export SINGULARITY_TMPDIR="$HOME/singularity_tmp"
export TMPDIR="$HOME/singularity_tmp"

echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST  start: $(date)"
echo "Pulling docker://$IMAGE_REF -> $SIF_PATH"

rm -f "$SIF_PATH"
singularity pull --name "$SIF_PATH" "docker://$IMAGE_REF"

echo "Done: $(date)"
ls -lh "$SIF_PATH"
