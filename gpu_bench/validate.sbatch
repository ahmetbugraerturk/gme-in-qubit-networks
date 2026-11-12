#!/bin/bash
#SBATCH --job-name=qnet_gmn_validate
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=01:00:00
#SBATCH --output=gpu_bench/slurm_logs/%x_%j.out
#SBATCH --error=gpu_bench/slurm_logs/%x_%j.err
# Validate GPU ADMM GMN vs CLARABEL near GMN=0. Submit from repo root: sbatch gpu_bench/validate.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/gpu_bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
/home/aerturk23/.conda/envs/qutip-gpu/bin/python gmn_validate.py 2>&1 | grep -v -i warn
