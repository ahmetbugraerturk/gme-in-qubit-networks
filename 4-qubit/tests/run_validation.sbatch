#!/bin/bash
#SBATCH --job-name=q4_validation
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=02:00:00
#SBATCH --output=4-qubit/slurm_logs/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/%x_%j.err
# Validation suite on the reference run: plan table, GMN symmetry, SCS cross-check against the certified bracket, GPU stages against the phase-1 pipeline. Submit from the project root.
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
$PY -u tests/check_xx.py 2>&1 | grep --line-buffered -v -i warn
$PY -u tests/check_vs_atlas.py 2>&1 | grep --line-buffered -v -i warn
$PY -u tests/test_delta_bound.py 2>&1 | grep --line-buffered -v -i warn || true
$PY -u tests/test_extras.py 2>&1 | grep --line-buffered -v -i warn || true
