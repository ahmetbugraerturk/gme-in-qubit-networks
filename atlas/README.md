# qnet: entanglement scan of small qubit networks (n = 2..5)

Exact simulation of XX/ZZ/EX-coupled qubit networks on every connected graph
for n = 2..5, computing pairwise concurrence, genuine tripartite entanglement
(GMN, via SDP), and whole-system entanglement (C_GME, Meyer-Wallach Q).

## Setup

Uses the `qutip` conda environment (numpy, scipy, networkx,
matplotlib, qutip, cvxpy, clarabel, scs, pytest):

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

For HPC batch runs see `jobs/submit_production.sbatch` (full sweep) and
`jobs/submit_always_gmn.sbatch` (exact-SDP variant, no sampling/shortcuts —
much more expensive, see the comments in that file for the cost estimate).

## Implementation

### `qnet/hamiltonians.py` — operators & Hamiltonian
- `SP`, `SM`, `X`, `Z`, `PLUS`, `HADAMARD` — built explicitly from kets (not
  QuTiP's `sigmap`/`sigmam`, which use the opposite convention)
- `two_qubit_term(interaction)` — the 4x4 two-qubit operator for `xx`/`zz`/`ex`
- `build_hamiltonian(G, interaction, w0)` — sums `h_ij` over every edge of
  graph `G`, embedded into the full n-qubit space via `qt.expand_operator`
- `initial_state(n, init)` — `|0...0>` or `|+...+>` as a QuTiP ket
- `hadamard_all(n)` — `H^⊗n`, used for the xx<->zz symmetry mapping
- `adjacency_matrix(G)` — plain numpy adjacency matrix

### `qnet/evolve.py` — exact time evolution
- `diagonalize(H)` — one-time dense eigendecomposition of H
- `make_time_grid(evals, t_max)` — time grid, `dt = min(0.1/dE_max, 0.02)`
- `evolve_batch(evals, evecs, psi0, times)` — vectorized `psi(t) = V e^{-iEt} V+ psi0`
  for the whole grid at once
- `evolve_single(...)` — same, at one arbitrary `t` (used to refine maxima)
- `partial_trace_pure_batch(psi_t, n, keep)` — vectorized reduced density
  matrix rho_keep(t) for a qubit subset, via reshape/transpose, batched over time
- `hadamard_transform_batch` / `local_unitary_tensor` / `apply_local_unitary_batch`
  — applies a local unitary directly to an already-reduced rho(t), used to
  derive zz results from xx results without re-simulating
- `enforce_physical(rho)` — Hermitize + clip negative eigenvalues + renormalize
- `cross_check_sesolve`, `cross_check_ptrace` — checks against QuTiP's own
  solver/`ptrace`

### `qnet/measures.py` — entanglement quantities
- `concurrence_single`/`concurrence_batch` — Wootters concurrence of a 2-qubit rho
- `partial_transpose`, `min_eig_partial_transpose_batch`, `negativity_batch`
  — PPT/negativity primitives
- `c_gme_pure_single`/`c_gme_pure_batch` — whole-system C_GME = min over
  bipartitions of `sqrt(2(1-Tr rho_S^2))`
- `meyer_wallach_Q_batch` — Meyer-Wallach Q from single-qubit purities

### `qnet/graphs.py` — graph data
- `download_graphs`/`load_graphs` — fetch/parse graph6 files, verify counts
  against OEIS A001349
- `all_automorphisms`, `edge_orbits` — Aut(G) and edge-orbit computation
  (symmetric edges give identical concurrence curves)
- `induced_connected_triples(G)` — all connected 3-qubit subsets

### `qnet/gs.py` — Guhne-Seevinck witness
- `ghz_margins`/`w_margins` (+ `_batch` variants) — the two margin formulas
  from arXiv:0905.1349
- `scan_all_bases`/`scan_all_bases_batch` — scans all 27 local bases
  (I/H/HS+)^3, returns the certifying max margin

### `qnet/bound.py` — free upper bound on GMN
- `partial_transpose_batch`, `negativity_sum_batch` — per-cut negativity N_M(rho)
- `gmn_upper_bound_batch(rho_t, n_sub)` — U(t) = min_M N_M(rho(t)), an exact
  upper bound on GMN (generic in subsystem size)

### `qnet/gmn.py` — GMN SDP + classification cascade
- `_GmnProblem` / `solve_gmn(rho)` — the Jungnitsch-Moroder-Guhne SDP (cvxpy,
  DPP-cached, SCS primary/CLARABEL fallback)
- `select_sdp_times` — sampling policy (uniform subsample + refine around
  proxy maxima) for cost control
- `analyze_triple(rho_t, times, config)` — the full per-triple cascade: PPT
  -> bound -> (optional) GS -> SDP on remaining points; returns labeled arrays
- `save_triple_analysis`/`load_triple_analysis` — per-triple checkpoint
  (atomic write) for resumability

### `qnet/pipeline.py` — orchestration
- `_time_grid_and_evolution` — builds H, evolves psi(t) for one
  (graph, interaction, init)
- `_refine_pair_max` — `scipy.optimize.minimize_scalar` refinement of a
  concurrence peak
- `compute_pair_results`/`derive_zz_pair_results` — per-edge-orbit
  concurrence curves (direct, or Hadamard-derived from xx)
- `compute_triple_results`/`derive_zz_triple_results`/`_analyze_triples` —
  runs `gmn.analyze_triple` per triple, with optional per-triple checkpointing
- `compute_whole_results` — whole-system C_GME(t)/Q(t)
- `check_ex_convergence` — re-evolves on `[0,200]` to flag any orbit whose
  max concurrence hadn't converged on `[0,100]`
- `run_task(n, idx, interaction, init, config)` — top-level entry point
  tying all of the above together for one task

## Layout

- `qnet/` — the package: `graphs.py`, `hamiltonians.py`, `evolve.py`,
  `measures.py`, `gs.py`, `bound.py`, `gmn.py`, `pipeline.py` (simulation),
  `report.py` (CSV/figures/REPORT.md), `tests/`
- `qnet_qutip/` — QuTiP-native benchmark version of the hot path
- `run.py` — CLI entry point
- `make_report.py` — regenerates reports from `results/`
- `jobs/` — SLURM scripts (submit from this folder: `sbatch jobs/<name>.sbatch`)
- `data/graphs/`, `graphs.json` — cached graph6 files and the derived graph list
- `results*/`, `figures*/`, `summary_*.csv`, `REPORT*.md`, `latex_report/`,
  `slurm_logs/` — phase 1 outputs (untracked); `latex_report/` is the atlas
- Sibling folders at the repo root: `../4-qubit/` (next task) and
  `../gpu_bench/` (GPU backend benchmark, imports `qnet` from here)

## QuTiP: where used, where not

| | used for |
|---|---|
| QuTiP | Hamiltonian/state construction (`Qobj`, `tensor`, `expand_operator`) |
| own numpy/cvxpy | time evolution, partial trace, concurrence (vectorized over the whole time grid); GMN SDP, GS witness, negativity bound, C_GME, Meyer-Wallach Q (no QuTiP equivalent) |

`qnet_qutip/` is a QuTiP-native version of the hot path (`sesolve` + looped
`Qobj.ptrace`/`qt.concurrence`) built to benchmark against `qnet/evolve.py` +
`qnet/measures.py`. Both agree to ~1e-8. Result (`qnet_qutip/benchmark.py`,
n=2 and n=3):

| case | T | numpy | qutip-native | ratio |
|---|---|---|---|---|
| n=2 xx\|0> | 80 | 0.005s | 0.026s | 5.2x |
| n=2 ex\|+> | 5001 | 0.225s | 1.217s | 5.4x |
| n=3 xx\|0> | 80 | 0.009s | 0.051s | 5.9x |
| n=3 ex\|+> | 5001 | 0.446s | 2.985s | 6.7x |

QuTiP-native is consistently 5-7x slower, and the gap grows with system
size — so the vectorized numpy pipeline stays.
