# 4-qubit: XX networks from |0000>

Entanglement of four qubits coupled by H = sum_{(i,j) in E} X_i X_j on every connected four-vertex graph
(`atlas/data/graphs/graph4c.g6`), starting from |0000>, for Jt in [0, pi/2].
Pairs (concurrence), triples (GMN), the whole state (Q, C_GME^2, GMN), all cuts and partitions (negativity),
and fidelities with Bell / GHZ / W targets. Equivalent subsets share one computed curve.

## Layout
| path | what |
|---|---|
| `qgpu/` | the library (NumPy/CuPy generic): `core.py` stages, `xxvac.py` the XX study, `contact.py`, `contact_analysis.py`, `gmn_admm.py` batched GMN solver with certified bound, `store.py` save/load |
| `plot_xx.py` | plotting module (also imported by the artifact and report builders): `--style light|dark|paper|mono` |
| `scripts/` | run and analysis scripts: `run_xx.py`, `run_contact*.py`, `run_setups.py`, `run_pairs_table.py`, `run_second_solver.py`, `witness_certs.py`, `plan_extras.py`, `contact_collfine.py`, `contact_summary.py`, `plot_contact.py` (run from the project folder: `python scripts/<name>.py`) |
| `jobs/` | Slurm batch files (`sbatch 4-qubit/jobs/<name>.sbatch` from the folder above the project); each logs to `slurm_logs/<study>/` |
| `slurm_logs/` | job output by study: `no_ancilla/`, `ancilla+_zz_xx/`, `ancilla1_zz_xx/`, `ancilla0.25_zz_xx/`, `setup_A_exchange_xx/`, `setup_B_five_vertex/`, `second_solver/` |
| `results/` | saved results (`xx_vacuum/`, `contact/`, `setII/` = setup A, `five/` = setup B) |
| `artifact/` | interactive pages (builders, templates, Node tests) |
| `report/` | reports; see `report/README.md` (`tex/`, `tables/`, `figs/`, `data/`, `scripts/`, `pdf/`, `build/`) |
| `tests/` | `test_*.py`, `check_*.py`, `compare_atlas.py`, `results/` (machine-readable) |
| `figures/`, `preparation/` | generated figures; generic GPU example (not part of the study) |

Environment: conda `qutip-gpu`. GPU work only on compute nodes (`srun`/`sbatch`, one V100), never on the login node.

## Things to remember
* GMN: `U` (certified upper bound) is rigorous; the primal estimate `L` is not. Overshoots `L - U` up to the solver tolerance (1e-7) are numerical noise and ignored; larger ones mean the ADMM run has not converged: the loader (`store.py`) clips `L` to `U`, marks the sample unconverged and never classes it positive. Convergence depends on the graph (good on K4/cycle/star, poor on path/paw/diamond). A certified lower bound exists as an option (`rigorous_lower=True`); the study has NOT been rerun with it. See `report/TODO.md` and the report.
* Concurrence has ~1e-8 square-root noise; use tolerances accordingly.
* Fidelities are not mirror symmetric (all other curves are symmetric about Jt = pi/4); real Bell states give exactly 1/2 for pairs, use the phase-optimised targets.
* Atlas comparison: `tests/compare_atlas.py` (same graph file and labels as the atlas run).

## Results folders (2026-11-11)
* `results/xx_vacuum` = reference run (certified GMN bracket, relative stop 1e-5, tuned ADMM `xxvac.TUNED`); convergence is reported at both 1e-5 and 1e-4 (`report/conv_tolerances.py`, explorer toggle).
* `results/xx_vacuum_first` = first run (uncertified L, absolute stop 1e-7); `xx_vacuum_cert5` = same as the reference with stop 1e-4; `xx_vacuum_cert`, `_cert2`, `_cert3` = intermediate bounds. `run_xx.sbatch` reproduces the reference run (study, `witness_certs.py`, atlas comparison, headline); `XX_REL_GAP` overrides the stop.

## Contact (ancilla) study (2026-11-12)
`qgpu/contact.py` (collision map, continuous retained ancilla, 5-qubit checks), `qgpu/contact_analysis.py` (all toolbox quantities for series of mixed states), `run_contact.py` / `run_contact.sbatch` (array over the 11 contact sites; ancilla |+> by default), `tests/test_contact.py`, `contact_summary.py`, `plot_contact.py`, results `results/contact/<graph>/<site>/p_g<g>_t0<0|pi_4>_{coll_tau<tau>,cont}.npz`, report `report/contact_report.pdf`.

Contact study files: `run_contact_anc1.sbatch` (ancilla |1>), `run_contact_long.sbatch` (long continuous, |+>), `contact_collfine.py` (concurrence between collisions), `artifact/build_contact.py` + `artifact/contact.template.html` (explorer page), `report/analyze_contact.py` (contact vs free vs coherent).

Setups A and B (2026-11-13): `run_setups.py`, `run_setups_A.sbatch`, `run_setups_B.sbatch`, `tests/test_setups.py`; results `results/setII/` (EX network, XX contact) and `results/five/` (five-vertex graphs, one vertex as ancilla).
