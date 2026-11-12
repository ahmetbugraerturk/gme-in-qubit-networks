#!/usr/bin/env python3
"""Do any penalty / relaxation / adaptive settings help the ADMM states that fail to converge?
Finds the non-converged states of K4 ex|+> (3-qubit subset and 4-qubit whole state) with the baseline settings,
then re-solves just those under a grid of settings; also reports whether the adaptive penalty actually moved."""
import json, os, sys, time
import numpy as np, cupy as cp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "atlas")); sys.path.insert(0, HERE)
import gmn_admm as A
from qnet import graphs as G, hamiltonians as Hm, evolve as E, measures as M
sync = cp.cuda.Stream.null.synchronize
n = 4; Gph = G.load_graphs(n)[5]; Hmat = Hm.build_hamiltonian(Gph, "ex"); psi0 = Hm.initial_state(n, "plus").full().ravel()
ev, V = E.diagonalize(Hmat); times = E.make_time_grid(ev, 100.0); psi_t = E.evolve_batch(ev, V, psi0, times)
def nonppt(rho, nq):
    me = np.stack([M.min_eig_partial_transpose_batch(rho, (2,) * nq, list(c)) for c in A.bipartition_cuts(nq)], 1)
    return ~(me >= -1e-10).any(1)
r3 = E.partial_trace_pure_batch(psi_t, n, (0, 1, 2)); r3 = r3[nonppt(r3, 3)]; r3 = r3[:: max(1, len(r3) // 400)][:400]
r4 = np.einsum("it,jt->tij", psi_t, psi_t.conj()); r4 = r4[nonppt(r4, 4)]; r4 = r4[:: max(1, len(r4) // 150)][:150]
for name, nq, rhos in [("K4 triple", 3, r3), ("K4 whole", 4, r4)]:
    base = A.solve_batch(cp, cp.asarray(rhos), nq=nq, max_iter=20000)
    bad = ~cp.asnumpy(base["converged"]); sub = cp.asarray(rhos[bad])
    print(f"\n===== {name}: {int(bad.sum())} non-converged of {len(rhos)}  (their time-grid positions: {np.nonzero(bad)[0][:12].tolist()}...)", flush=True)
    gm = cp.asnumpy(base["gmn"])[bad]; print(f"  GMN of the stuck states: min {gm.min():.3g} median {np.median(gm):.3g} max {gm.max():.3g}", flush=True)
    def go(label, **kw):
        t0 = time.perf_counter(); r = A.solve_batch(cp, sub, nq=nq, max_iter=40000, **kw); sync(); t = time.perf_counter() - t0
        h = {k: cp.asnumpy(v) for k, v in r.items()}
        print(f"  {label:38s} conv {h['converged'].mean():5.1%}  iters mean {h['iters'].mean():7.0f}  median gap {np.median(h['gap']):.2e}  "
              f"pen[min,med,max]=[{h['pen'].min():.1e},{np.median(h['pen']):.1e},{h['pen'].max():.1e}]  {t:.1f}s", flush=True)
    go("baseline rho=0.03 alpha=1.5 (40k iters)")
    for rho in (0.003, 0.01, 0.1, 0.3, 1.0): go(f"rho={rho}", rho_admm=rho)
    for al in (1.0, 1.8): go(f"alpha={al}", alpha=al)
    go("adaptive mu=10 tau=2", adaptive_rho=True)
    go("adaptive mu=2  tau=2", adaptive_rho=True, mu=2.0)
    go("adaptive mu=2  tau=2 start rho=0.3", adaptive_rho=True, mu=2.0, rho_admm=0.3)
