#!/bin/bash
#SBATCH --job-name=qnet_stuck_scan
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
# Do any settings help the non-converging states? Submit from repo root: sbatch gpu_bench/stuck_scan.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/gpu_bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
/home/aerturk23/.conda/envs/qutip-gpu/bin/python -u stuck_scan.py 2>&1 | grep --line-buffered -v -i warn
