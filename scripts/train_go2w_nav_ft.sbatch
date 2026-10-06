#!/usr/bin/env bash
# Fine-tune the KITTI-360 R2Flow (1-RF) on the simulated navigation scans, on bowser.
# See docs/FINETUNING.md for the dataset and the choice of flags.
#
#   sbatch scripts/train_go2w_nav_ft.sbatch                 # submit
#   bash scripts/train_go2w_nav_ft.sbatch                   # same (submits itself)
#   sbatch scripts/train_go2w_nav_ft.sbatch --lr 1e-5       # extra args go to train.py
#
# Logs: logs/slurm/<job name>-<job id>.out, tensorboard under --output_dir.

#SBATCH --job-name=r2flow-go2w-nav-ft
#SBATCH --partition=debug
#SBATCH --nodelist=bowser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

# Slurm opens --output before the script runs, so its directory must already exist.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    cd "$(dirname "$0")/.."
    mkdir -p logs/slurm
    exec sbatch "$0" "$@"
fi

cd "${SLURM_SUBMIT_DIR}"

# Batch shells don't get the `conda` function; load its hook explicitly.
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate r2sflow

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
echo "job ${SLURM_JOB_ID} on $(hostname): GPUs ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS}), ${SLURM_CPUS_PER_TASK} CPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH=(accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" --num_machines 1)
else
    LAUNCH=(accelerate launch --num_processes 1)
fi

# --batch_size is global (train.py uses split_batches), so it is shared across GPUs.
"${LAUNCH[@]}" train.py \
    --dataset go2w_nav_sim --projection spherical-64x1024 --resolution 64 1024 \
    --min_depth 0.5 --max_depth 30.0 --train_reflectance False \
    --init_weights r2flow-kitti360-1rf \
    --lr 2e-5 --num_images_training 400000 --num_images_lr_warmup 8000 \
    --num_workers $(( SLURM_CPUS_PER_TASK / NUM_GPUS )) \
    --output_dir logs/r2flow-go2w-nav-ft \
    "$@"
