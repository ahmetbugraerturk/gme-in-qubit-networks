#!/bin/bash
#SBATCH --job-name=qnet_entangle
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Production run for the entanglement scan (n=2..5). Resumable: run_task()
# skips any (n, graph_index, interaction, init) whose results/*.npz already
# exists, so re-submitting this script after a failure/timeout just picks up
# where it left off. No GPU needed; this is a pure-CPU, multiprocessing job
# (qnet.gmn SDP solves + exact diagonalization), parallelized across
# SLURM_CPUS_PER_TASK via run.py.

set -eo pipefail
mkdir -p slurm_logs

# Call the "qutip" conda env's python directly by absolute path rather than
# `conda activate`: activation inside sbatch scripts is fragile (it depends
# on exactly what CONDA_*/PATH state got inherited from the submitting
# shell), and this sidesteps that entirely -- the interpreter resolves its
# own site-packages from its own prefix regardless of inherited shell state.
PY=/home/aerturk23/.conda/envs/qutip/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /scratch/aerturk23/quest/atlas

echo "=== $(date) : starting production run, n<=5, ${SLURM_CPUS_PER_TASK} workers ==="
"$PY" run.py --all --n-max 5

echo "=== $(date) : generating report ==="
"$PY" make_report.py --n-max 5

echo "=== $(date) : done ==="
