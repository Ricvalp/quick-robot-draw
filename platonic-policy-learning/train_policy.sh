#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --chdir="/home/dknigge/platonic_policy_learning"
#SBATCH --output=./runs/slurm/mnist_diffusion_%A.out
#SBATCH --error=./runs/slurm/mnist_diffusion_%A.err
#SBATCH --job-name='mnist-diffusion'

source /home/dknigge/.bashrc
mamba activate ppl

export PYTHONPATH=/home/dknigge/platonic_policy_learning:$PYTHONPATH
echo "Running on node: $SLURM_NODELIST"

python train.py --policy=platonic