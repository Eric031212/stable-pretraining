#!/usr/bin/env bash
#SBATCH --job-name=ssley-smoke
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=ssley-smoke-%j.out
#SBATCH --error=ssley-smoke-%j.err
set -euo pipefail
: "${SPT_REPO:?Set SPT_REPO}"
: "${SPT_PYTHON:?Set SPT_PYTHON}"
: "${SPT_DATA:?Set SPT_DATA}"
: "${SPT_OUTPUT:?Set SPT_OUTPUT}"
cd "$SPT_REPO"
export PYTHONUNBUFFERED=1
nvidia-smi
srun --ntasks=1 "$SPT_PYTHON" -u benchmarks/cifar10/ssley-paper.py \
  --data-dir "$SPT_DATA" --output "$SPT_OUTPUT/smoke-$SLURM_JOB_ID" \
  --accelerator gpu --workers 2 --threads 2 --max-steps 10 --checkpoint-every-steps 5
