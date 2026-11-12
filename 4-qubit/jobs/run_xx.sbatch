#!/bin/bash
#SBATCH --job-name=q4_xx_vacuum
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
# XX network from |0000>: fast gate, the study (certified GMN bracket, relative stop 1e-5, tuned ADMM per size), witness certificates, atlas comparison, headline. Submit from the project root: sbatch 4-qubit/run_xx.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XX_RESULTS=xx_vacuum
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u tests/check_xx_fast.py 2>&1 | grep --line-buffered -v -i warn
$PY -u scripts/run_xx.py --out results/xx_vacuum 2>&1 | grep --line-buffered -v -i warn
$PY -I -u scripts/witness_certs.py --results results/xx_vacuum 2>&1 | grep -v -i warn
$PY -I -u scripts/plan_extras.py --results results/xx_vacuum 2>&1 | grep -v -i warn
$PY -I -u tests/test_extras.py 2>&1 | grep -v -i warn || true
$PY -u tests/compare_atlas.py 2>&1 | grep --line-buffered -v -i warn || true
$PY -I -u tests/headline_positives.py 2>&1 | grep -v "atlas GS\|atlas SDP"
