#!/bin/bash
#SBATCH --job-name=q4_cmp_cert
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=4-qubit/slurm_logs/%x_%j.out
#SBATCH --error=4-qubit/slurm_logs/%x_%j.err
# Atlas comparison of the certified rerun (results/xx_vacuum_cert); starts after the study job finished OK:
#   sbatch --dependency=afterok:<study jobid> 4-qubit/tests/run_compare_cert.sbatch      (from the project root)
cd /scratch/aerturk23/quest/4-qubit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XX_RESULTS=${XX_RESULTS:-xx_vacuum_cert}
/home/aerturk23/.conda/envs/qutip-gpu/bin/python -u tests/compare_atlas.py 2>&1 | grep --line-buffered -v -i warn
/home/aerturk23/.conda/envs/qutip-gpu/bin/python -I -u tests/headline_positives.py 2>&1 | grep -v "atlas GS\|atlas SDP" 
