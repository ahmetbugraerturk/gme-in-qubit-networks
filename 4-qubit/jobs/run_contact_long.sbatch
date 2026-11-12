#!/bin/bash
#SBATCH --job-name=q4_contactlong
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=06:00:00
#SBATCH --array=0-10
#SBATCH --output=4-qubit/slurm_logs/ancilla+_zz_xx/%x_%A_%a.out
#SBATCH --error=4-qubit/slurm_logs/ancilla+_zz_xx/%x_%A_%a.err
# Contact (ancilla) study, one array task per (graph, contact site): 11 tasks.  Restartable (existing result files are skipped).  Submit from the project root: sbatch 4-qubit/run_contact.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u scripts/run_contact.py --anc + --no-coll --cont-suffix long --gmn-window 80 --gmn-dt 0.2 --task $SLURM_ARRAY_TASK_ID 2>&1 | grep --line-buffered -v -i warn
