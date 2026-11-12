#!/bin/bash
#SBATCH --job-name=qnet_stages
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
# Stage-by-stage: ours vs QuTiP alternatives. Submit from repo root: sbatch gpu_bench/stages.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/gpu_bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
/home/aerturk23/.conda/envs/qutip-gpu/bin/python stages.py 2>&1 | grep -v -i warn
