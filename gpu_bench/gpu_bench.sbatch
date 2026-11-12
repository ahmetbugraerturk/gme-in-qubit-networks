#!/bin/bash
#SBATCH --job-name=qnet_gpu_bench
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=00:45:00
#SBATCH --output=gpu_bench/slurm_logs/%x_%j.out
#SBATCH --error=gpu_bench/slurm_logs/%x_%j.err
# GPU vs CPU backend comparison. Submit from repo root: sbatch gpu_bench/gpu_bench.sbatch
set -eo pipefail
cd /scratch/aerturk23/quest/gpu_bench
PY=/home/aerturk23/.conda/envs/qutip-gpu/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; hostname
CASES=3:1:ex:plus:100,4:5:ex:plus:100,5:20:ex:plus:100
OUT=results_$SLURM_JOB_ID.jsonl; : > $OUT
for b in numpy_spectral cupy_spectral; do $PY bench.py --backend $b --cases $CASES --repeats 5 2>/dev/null >> $OUT; done
for b in qutip_cpu qutip_cuquantum qutip_jax; do $PY bench.py --backend $b --cases $CASES --repeats 1 2>/dev/null >> $OUT; done
echo "wrote $OUT"
