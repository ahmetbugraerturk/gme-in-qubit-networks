#!/bin/bash
#SBATCH --job-name=q4_stoprule
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=00:40:00
#SBATCH --output=4-qubit/slurm_logs/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/%x_%j.err
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/aerturk23/.conda/envs/qutip-gpu/bin/python -u tests/test_stopping_rule.py 2>&1 | grep --line-buffered -v -i warn
