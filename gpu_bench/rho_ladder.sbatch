#!/bin/bash
#SBATCH --job-name=qnet_rho_ladder
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=02:00:00
#SBATCH --output=gpu_bench/slurm_logs/%x_%j.out
#SBATCH --error=gpu_bench/slurm_logs/%x_%j.err
# Effect of rho and alpha on convergence and decision quality. Submit from repo root: sbatch gpu_bench/rho_ladder.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/gpu_bench
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
/home/aerturk23/.conda/envs/qutip-gpu/bin/python -u rho_ladder.py 2>&1 | grep --line-buffered -v -i warn
