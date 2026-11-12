#!/bin/bash
#SBATCH --job-name=q4_second
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=4-qubit/slurm_logs/second_solver/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/second_solver/%x_%j.err
# Plan Sec. 4.7: independent second solver (CVXPY + CLARABEL, CPU) on random near-threshold GMN samples of results/contact. 
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=4
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
$PY -u scripts/run_second_solver.py --n3 400 --n4 60 2>&1 | grep --line-buffered -v -i warn
