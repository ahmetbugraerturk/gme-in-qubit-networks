#!/bin/bash
#SBATCH --job-name=qnet_always_gmn
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# "Always GMN" experiment: zz|init=plus> and ex|init=plus> only, across n=2..5.
# Writes to results_always_gmn/ (NOT results/) and REPORT_always_gmn.md /
# summary_*_always_gmn.csv / figures_always_gmn/ -- deliberately separate
# from the main production sweep (which already has zz_plus/ex_plus computed
# with the default sampled pipeline) so this experiment never overwrites or
# mixes into that existing, already-reviewed report.
# Note: zz|init=plus> is NOT stationary (it's the Hadamard-transform of
# xx|init=0>, which has genuine oscillatory dynamics), unlike zz|init=0>
# (stationary -- Hadamard-transform of the trivially-stationary xx|init=plus>
# per section 1), so this combo gets real zz entanglement data rather than an
# all-PPT trivial case. Adds ~26k grid points of real SDP work across
# n=3..5, small next to ex|plus>'s ~840k, so the overall cost estimate below
# is essentially unchanged.
# Unlike the default production pipeline (PPT -> negativity-sum upper bound
# -> Guhne-Seevinck witness -> SDP on a cost-controlled sample), this skips
# the GS witness AND the SDP subsampling: every single remaining (non-PPT,
# non-bound-zero) grid point gets an exact SDP solve. Real cost estimate
# (measured, not guessed): ~866k SDP solves, ~207 CPU-hours serial -- with 40
# cores, expect several hours of wall time even in the best case; the bound
# pre-filter was checked empirically across all 21 n=5 graphs and does NOT
# meaningfully cut this (still ~99.98% need the SDP), so don't expect this
# run to finish fast. The time limit above is a safety margin, not an
# estimate.
#
# Resumable at TWO granularities:
#   1. Per task-file: run_task() skips any results/{n}/g{idx}_{interaction}_{init}.npz
#      that already exists.
#   2. Per triple (within an in-progress task): each triple's full SDP sweep
#      is checkpointed to results/{n}/.ckpt_g{idx}_{interaction}_{init}/triple_{m}.npz
#      as soon as it finishes, and reloaded instead of recomputed on restart.
#      This matters because here a SINGLE triple can itself take tens of
#      minutes to hours of SDP time -- if this job times out or is
#      preempted, at most the one in-flight triple's progress is lost, not
#      the whole task. Just re-submit this script again to continue.

set -eo pipefail
mkdir -p slurm_logs

PY=/home/aerturk23/.conda/envs/qutip/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /scratch/aerturk23/quest

echo "=== $(date) : starting always-GMN run (zz|+>, ex|+>), n<=5, ${SLURM_CPUS_PER_TASK} workers ==="
"$PY" run.py --all --n-max 5 --always-gmn --combo zz:plus --combo ex:plus --results-dir results_always_gmn

echo "=== $(date) : generating report ==="
"$PY" make_report.py --n-max 5 --results-dir results_always_gmn --suffix _always_gmn

echo "=== $(date) : done ==="
