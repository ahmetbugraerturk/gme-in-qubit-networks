#!/bin/bash
#SBATCH --job-name=q4_setupB
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=48:00:00
#SBATCH --output=4-qubit/slurm_logs/setup_B_five_vertex/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/setup_B_five_vertex/%x_%j.err
# Setup B of run_setups.py (see its docstring): ONE node, ONE task, all pairs in turn, restartable (existing result files are skipped).  Submit from the project root.
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u tests/test_setups.py 2>&1 | grep --line-buffered -v -i warn
$PY -u scripts/run_setups.py --setup B 2>&1 | grep --line-buffered -v -i warn
