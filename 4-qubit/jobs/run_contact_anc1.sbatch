#!/bin/bash
#SBATCH --job-name=q4_contact1
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=10:00:00
#SBATCH --array=0-10
#SBATCH --output=4-qubit/slurm_logs/ancilla1_zz_xx/%x_%A_%a.out
#SBATCH --error=4-qubit/slurm_logs/ancilla1_zz_xx/%x_%A_%a.err
# Coherent ancilla |1> (field -g Z_v, pure network state): collisions, continuous (window Jt<=20) and long continuous (Jt<=80), one array task per (graph, site).
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u scripts/run_contact.py --anc 1 --task $SLURM_ARRAY_TASK_ID 2>&1 | grep --line-buffered -v -i warn
$PY -u scripts/run_contact.py --anc 1 --no-coll --no-free --cont-suffix long --gmn-window 80 --gmn-dt 0.2 --task $SLURM_ARRAY_TASK_ID 2>&1 | grep --line-buffered -v -i warn
