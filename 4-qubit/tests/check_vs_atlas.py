#!/usr/bin/env python3
"""Correctness check: qgpu (GPU) vs atlas/qnet (NumPy, the phase-1 reference) on 4-qubit K4, ex|+>."""
import itertools, os, sys
import numpy as np, networkx as nx
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE)   # HERE = 4-qubit/tests
sys.path.insert(0, os.path.join(os.path.dirname(Q4), "atlas")); sys.path.insert(0, Q4)
from qnet import graphs as G, hamiltonians as Hm, evolve as E, measures as M, gmn as GM
from qgpu import core as C, gmn_admm as A
import cupy as cp
n, gi = 4, 5
Gph = G.load_graphs(n)[gi]; edges = [tuple(sorted(e)) for e in Gph.edges()]
Hmat = Hm.build_hamiltonian(Gph, "ex"); psi0 = Hm.initial_state(n, "plus").full().ravel()
ev, V = E.diagonalize(Hmat); times = E.make_time_grid(ev, 100.0)
ref = E.evolve_batch(ev, V, psi0, times); g = C.evolve(cp, Hmat.full(), psi0, times)
err = {"evolution": np.abs(C.to_host(g) - ref).max()}
err["concurrence"] = max(np.abs(C.to_host(C.concurrence(cp, C.ptrace_pure(cp, g, n, e))) - M.concurrence_batch(E.partial_trace_pure_batch(ref, n, e))).max() for e in edges)
err["meyer_wallach"] = np.abs(C.to_host(C.meyer_wallach(cp, g, n)) - M.meyer_wallach_Q_batch(ref, n)).max()
sub = (0, 1, 2); r_ref = E.partial_trace_pure_batch(ref, n, sub); r_g = C.ptrace_pure(cp, g, n, sub)
err["ppt_mineig"] = max(np.abs(C.to_host(C.min_eig_pt(cp, r_g, 3, [q])) - M.min_eig_partial_transpose_batch(r_ref, (2, 2, 2), [q])).max() for q in range(3))
# GMN vs cvxpy+SCS on a few non-PPT time points
me = np.stack([M.min_eig_partial_transpose_batch(r_ref, (2, 2, 2), [q]) for q in range(3)], 1); idx = np.nonzero(~(me >= -1e-10).any(1))[0][:: max(1, 20000 // 8)][:8]
res = A.solve_batch(cp, cp.asarray(r_ref[idx]), nq=3, tol_gap=1e-7)
scs = np.array([GM.solve_gmn(r_ref[i])["gmn"] for i in idx]); L, U = C.to_host(res["gmn"]), C.to_host(res["gmn_upper"])
err["gmn_vs_scs"] = np.abs(L - scs).max()
for k, v in err.items(): print(f"{k:14s} max|diff| = {v:.2e}")
print(f"GMN certificate: max(U-L)={np.max(U-L):.1e}, scs within [L-1e-6, U+1e-6]: {bool(np.all((scs>=L-1e-6)&(scs<=U+1e-6)))}")
ok = all(v < 5e-7 for v in err.values()); print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
