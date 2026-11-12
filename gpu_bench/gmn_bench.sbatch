#!/bin/bash
#SBATCH --job-name=qnet_gmn_bench
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=01:00:00
#SBATCH --output=gpu_bench/slurm_logs/%x_%j.out
#SBATCH --error=gpu_bench/slurm_logs/%x_%j.err
# GMN SDP: cvxpy+SCS (CPU) vs batched ADMM (GPU) for 3-,4-,5-qubit sub-states of n=5 K5.
# Submit from repo root: sbatch gpu_bench/gmn_bench.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/gpu_bench
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
C="--n 5 --graph 20 --no-cpu --tol 1e-7 --rho-admm 0.03"
echo "== triples (8x8, 3 cuts)";  $PY gmn_bench.py $C --nq 3 --nbatch 2000 --nscs 20 --max-iter 20000 2>&1 | grep -v -i warn | tail -4
echo "== quads (16x16, 7 cuts)";  $PY gmn_bench.py $C --nq 4 --nbatch 500  --nscs 2  --max-iter 10000 2>&1 | grep -v -i warn | tail -4
echo "== quints (32x32, 15 cuts)"; $PY gmn_bench.py $C --nq 5 --nbatch 100  --nscs 0  --max-iter 5000  2>&1 | grep -v -i warn | tail -4
