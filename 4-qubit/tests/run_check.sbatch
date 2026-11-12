#!/bin/bash
#SBATCH --job-name=q4_gpu_check
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=00:30:00
#SBATCH --output=4-qubit/slurm_logs/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/%x_%j.err
# Correctness check + example run. Submit from repo root: sbatch 4-qubit/run_check.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u tests/check_vs_atlas.py 2>&1 | grep --line-buffered -v -i warn
echo "== example GPU run: K4, ex|+>, with GMN"
$PY -u preparation/run_gpu.py --n 4 --graph-index 5 --gmn 2>&1 | grep --line-buffered -v -i warn
