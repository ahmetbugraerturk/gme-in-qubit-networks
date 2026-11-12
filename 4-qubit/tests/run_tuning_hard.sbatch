#!/bin/bash
#SBATCH --job-name=q4_tune
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=01:30:00
#SBATCH --output=4-qubit/slurm_logs/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/%x_%j.err
cd /scratch/aerturk23/quest/4-qubit
/home/aerturk23/.conda/envs/qutip-gpu/bin/python -u tests/test_tuning_hard.py 2>&1 | grep --line-buffered -v -i warn
