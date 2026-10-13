# qnet: entanglement scan of small qubit networks (n = 2..5)

Exact simulation of XX/ZZ/EX-coupled qubit networks on every connected graph
for n = 2..5, computing pairwise concurrence, genuine tripartite entanglement
(GMN, via SDP), and whole-system entanglement (C_GME, Meyer-Wallach Q).

## Setup

Uses the existing `qutip` conda environment (numpy, scipy, networkx,
matplotlib, qutip, cvxpy, clarabel, scs, pytest):

```bash
source /etc/profile.d/lmod.sh
module load anaconda3/2025.06
conda activate qutip
```

## Tests

Run the correctness test suite (section 6 of the spec) before any production
run:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest qnet/tests -q
```

## Usage

```bash
# (re)download graph data and build graphs.json
python run.py --build-graphs --n-max 5

# one task
python run.py --n 5 --graph-index 3 --interaction ex --init plus

# everything for n<=5 (checkpointed/resumable, multiprocessing)
python run.py --all --n-max 5

# generate summary_*.csv, figures/, REPORT.md from whatever is in results/
python make_report.py --n-max 5
```

For HPC batch runs see `submit_production.sbatch` (full sweep) and
`submit_always_gmn.sbatch` (exact-SDP variant, no sampling/shortcuts —
much more expensive, see the comments in that file for the cost estimate).

## Layout

- `qnet/` — the package: `graphs.py`, `hamiltonians.py`, `evolve.py`,
  `measures.py`, `gs.py`, `bound.py`, `gmn.py`, `pipeline.py` (simulation),
  `report.py` (CSV/figures/REPORT.md), `tests/`
- `run.py` — CLI entry point
- `make_report.py` — regenerates reports from `results/`
- `data/graphs/` — cached graph6 files (McKay's collection; downloader also
  falls back to `networkx.graph_atlas_g()` offline)
- `results*/`, `figures*/`, `summary_*.csv`, `REPORT*.md` — generated
  outputs, not tracked in git (see `.gitignore`)
- `latex_report/` — a LaTeX "entanglement atlas" built from one production
  run's results (not tracked; rebuild with `latex_report/build.sh`)
