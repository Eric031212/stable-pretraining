#!/usr/bin/env bash
#SBATCH --job-name=ssley-cifar10
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=ssley-%j.out
#SBATCH --error=ssley-%j.err

set -euo pipefail
: "${SPT_REPO:?Set SPT_REPO to the absolute stable-pretraining checkout path}"
: "${SPT_PYTHON:?Set SPT_PYTHON to the server virtualenv python}"
: "${SPT_DATA:?Set SPT_DATA to the prepared CIFAR-10 directory}"
: "${SPT_OUTPUT:?Set SPT_OUTPUT to a persistent experiment directory}"
cd "$SPT_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
# Do not overwrite Slurm's CUDA_VISIBLE_DEVICES or run GPU work on the gateway.
nvidia-smi
"$SPT_PYTHON" -c 'import torch; assert torch.cuda.is_available(); x=torch.randn(16,16,device="cuda"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (x@x).mean().item())'

resume_args=()
# Manager stores periodic saves under runs/, while clean exits also save at the root.
resume_checkpoint=$("$SPT_PYTHON" -c 'import os; from pathlib import Path; root=Path(os.environ["SPT_OUTPUT"]); paths=list(root.glob("checkpoints/last.ckpt"))+list(root.glob("runs/**/checkpoints/last.ckpt")); print(max(paths,key=lambda p:p.stat().st_mtime).resolve() if paths else "")')
if [[ -n "$resume_checkpoint" ]]; then
    resume_args=(--resume "$resume_checkpoint")
fi
if [[ -n "${SPT_STOP_AFTER_EPOCHS:-}" ]]; then
    resume_args+=(--stop-after-epochs "$SPT_STOP_AFTER_EPOCHS")
fi
# Stop before the Slurm wall limit to flush a resumable checkpoint.
srun --ntasks=1 "$SPT_PYTHON" -u benchmarks/cifar10/ssley-paper.py \
    --config "${SPT_CONFIG:-benchmarks/cifar10/server/ssley-paper.json}" \
    --data-dir "$SPT_DATA" --output "$SPT_OUTPUT" \
    --accelerator gpu --workers 4 --threads 4 \
    --max-time 00:11:45:00 "${resume_args[@]}"
