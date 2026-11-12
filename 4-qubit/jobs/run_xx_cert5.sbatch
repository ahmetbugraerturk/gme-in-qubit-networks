#!/bin/bash
#SBATCH --job-name=q4_xx_cert5
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=06:00:00
#SBATCH --output=4-qubit/slurm_logs/no_ancilla/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/no_ancilla/%x_%j.err
# Study with tuned ADMM (penalty/over-relaxation per size) and relative stop 1e-4 -> witness certificates -> atlas comparison -> headline.
# Results: results/xx_vacuum_cert5 (earlier results untouched).  Submit from the project root.
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export XX_REL_GAP=1e-4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XX_RESULTS=xx_vacuum_cert5
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u tests/check_xx_fast.py 2>&1 | grep --line-buffered -v -i warn
$PY -u scripts/run_xx.py --out results/xx_vacuum_cert5 2>&1 | grep --line-buffered -v -i warn
$PY -I -u scripts/witness_certs.py --results results/xx_vacuum_cert5 2>&1 | grep -v -i warn
$PY -u tests/compare_atlas.py 2>&1 | grep --line-buffered -v -i warn || true
$PY -I -u tests/headline_positives.py 2>&1 | grep -v "atlas GS\|atlas SDP"
