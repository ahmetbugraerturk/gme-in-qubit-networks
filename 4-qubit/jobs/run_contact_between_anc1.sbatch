#!/bin/bash
#SBATCH --job-name=q4_betw1
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=48:00:00
#SBATCH --output=4-qubit/slurm_logs/ancilla1_zz_xx/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/ancilla1_zz_xx/%x_%j.err
# GMN of the |+> collision model between collisions (g/J = 20, step 0.1 in Jt): |1> variant.
# Submit from the project root: sbatch 4-qubit/run_contact_between.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u scripts/run_contact_between_anc1.py 2>&1 | grep --line-buffered -v -i warn
