#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --chdir="/home/dknigge/platonic_policy_learning"
#SBATCH --output=./runs/slurm/eval_policy_%A.out
#SBATCH --error=./runs/slurm/eval_policy_%A.err
#SBATCH --job-name='ppl-eval'

# -----------------------------------------------------------------------------
# Cluster environment setup: load modules required for headless rendering.
module purge
module load 2024
module load Xvfb/21.1.14-GCCcore-13.3.0

# Launch a virtual framebuffer so CoppeliaSim can open an OpenGL context.
Xvfb :99 -screen 0 1280x1024x24 &
export XVFB_PID=$!
trap 'kill ${XVFB_PID}' EXIT

source /home/dknigge/.bashrc
mamba activate ppl

export PYTHONPATH=/home/dknigge/platonic_policy_learning:$PYTHONPATH
export DISPLAY=:99
export QT_QPA_PLATFORM=xcb

# Fail fast when required CLI arguments are missing; these environment
# variables should be provided via --export or sbatch --export=ALL,FOO=...
: "${CHECKPOINT:?Set CHECKPOINT to the checkpoint file}" 
: "${CACHE_PATH:?Set CACHE_PATH to the temporal cache file}" 
: "${TASK_NAME:?Set TASK_NAME to the RLBench task name}" 

# Optional overrides with sensible defaults.
: "${NORMALIZATION_STATS:=}" 
: "${DATASET_ROOT:=}" 
: "${EPISODES:=5}"
: "${MAX_STEPS:=200}"
: "${POINTCLOUD_MAX_POINTS:=}"
: "${DETERMINISTIC:=}" 

CMD=(
  python eval.py \
    --checkpoint "${CHECKPOINT}" \
    --cache-path "${CACHE_PATH}" \
    --task "${TASK_NAME}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}"
)

if [[ -n "${NORMALIZATION_STATS}" ]]; then
  CMD+=(--normalization-stats "${NORMALIZATION_STATS}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${POINTCLOUD_MAX_POINTS}" ]]; then
  CMD+=(--pointcloud-max-points "${POINTCLOUD_MAX_POINTS}")
fi
if [[ -n "${DETERMINISTIC}" ]]; then
  CMD+=(--deterministic)
fi

set -x
"${CMD[@]}"
set +x

wait ${XVFB_PID}
