#!/usr/bin/env bash
# Run with bash. Uses the site's existing allocation; never selects GPUs itself.
set -euo pipefail
spider_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$spider_root"
source "$spider_root/.venvs/nnunet/bin/activate"
export PYTHONPATH="$spider_root${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_raw="$spider_root/Models/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="$spider_root/Models/nnUNet/nnUNet_preprocessed"
# New root: never reuse copied single-GPU or Windows results.
export nnUNet_results="${SPIDER_DDP_RESULTS_ROOT:-$spider_root/Models/nnUNet/nnUNet_results_ddp7}"
export nnUNet_extTrainer="$spider_root/training/nnunet/trainers"
export nnUNet_wandb_enabled=0
export nnUNet_def_n_proc="${nnUNet_def_n_proc:-1}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export nnUNet_compile="${nnUNet_compile:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-BeyondDice-SPIDER}"
export PYTHONUNBUFFERED=1
mode="${1:-smoke}"
action="${2:-fresh}"
if [[ "$mode" != smoke && "$mode" != full ]]; then
    echo 'Usage: bash scripts/run_spider_ddp7.sh smoke|full fresh|resume|validate' >&2
    exit 2
fi
case "$action" in
    fresh) extra=() ;;
    resume) extra=(--c) ;;
    validate) extra=(--val) ;;
    *) echo 'Action must be fresh, resume or validate' >&2; exit 2 ;;
esac
if [[ "$mode" == smoke ]]; then
    trainer=nnUNetTrainerSPIDERInstancesDDP7_5epochs
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_NAME="${WANDB_NAME:-SPIDER-instances-DDP7-smoke}"
else
    trainer=nnUNetTrainerSPIDERInstancesDDP7
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_NAME="${WANDB_NAME:-SPIDER-instances-DDP7-full}"
fi
# Require an explicit allocation mask. Do not treat visible GPUs as permission.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the seven GPU IDs assigned to this job}"
python - <<'PY'
import os
import torch
from importlib.metadata import version
from training.nnunet.trainers.nnUNetTrainerSPIDERInstancesDDP7 import (
    nnUNetTrainerSPIDERInstancesDDP7, nnUNetTrainerSPIDERInstancesDDP7_5epochs)
if version('nnunetv2') != '2.8.1':
    raise SystemExit('Requires nnunetv2==2.8.1')
if not torch.distributed.is_nccl_available():
    raise SystemExit('NCCL backend unavailable')
if not torch.cuda.is_available() or torch.cuda.device_count() != 7:
    raise SystemExit('This experiment requires exactly seven visible CUDA GPUs on one node')
print('Trainer imports passed; visible GPUs:', torch.cuda.device_count())
print('Results root:', os.environ['nnUNet_results'])
print('W&B mode:', os.environ['WANDB_MODE'])
PY
python scripts/prepare_spider_ddp7.py
exec nnUNetv2_train 12 3d_fullres_ddp7 0 \
    -p nnUNetPlansSPIDERDDP7 -tr "$trainer" -num_gpus 7 \
    --val_best "${extra[@]}"
