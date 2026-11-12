#!/usr/bin/env python3
"""Does the stopping rule explain the 'unconverged' samples?  On a subsample of the real states (xx|0000>, K4 and path, whole state and triple abc, 41 times in [0, pi/4]),
compare  (A) the rule used in the study,  (B) strict gap |U - L| <= tol,  (C) the rigorous bracket [gmn_lower, gmn_upper].  GPU, one V100."""
import math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
import cupy as cp
from qgpu import core as C, xxvac as X, gmn_admm as A, store
g6 = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); sync = cp.cuda.Stream.null.synchronize
print(f"{'case':26s} {'rule':22s} {'converged':>10s} {'iters mean':>10s} {'median U-L':>11s} {'L>U':>5s} {'certified positive (L>1e-6)':>28s} {'time':>7s}")
for g in g6:
    nm = X.graph_name(g)
    if nm not in ("K4", "path"): continue
    th = np.linspace(0, math.pi / 4, 1001)[::25][1:]; psi = X.evolution(cp, sorted(g.edges()), 4, th)
    cases = {"whole state": (psi.T[:, :, None] * cp.conj(psi.T)[:, None, :], 4), "triple abc": (C.ptrace_pure(cp, psi, 4, [0, 1, 2]), 3)}
    for cname, (rho, nq) in cases.items():
        for rule, kw in (("A: study rule", {}), ("B: strict |U-L|", dict(strict_gap=True)), ("C: rigorous bracket", dict(rigorous_lower=True))):
            A.solve_batch(cp, rho[:2], nq=nq, max_iter=100, **kw); sync(); t0 = time.perf_counter()
            r = A.solve_batch(cp, rho, nq=nq, **kw); sync(); dt = time.perf_counter() - t0; h = {k: cp.asnumpy(v) for k, v in r.items()}
            lo = h["gmn_lower"] if rule.startswith("C") else h["gmn"]; w = h["gmn_upper"] - lo
            print(f"{nm + ' ' + cname:26s} {rule:22s} {100 * h['converged'].mean():9.0f}% {h['iters'].mean():10.0f} {np.median(w):11.1e} {int((h['gmn'] > h['gmn_upper'] * (1 + 1e-9) + 1e-12).sum()):5d} {int((lo > 1e-6).sum()):>10d} of {len(lo):d}   {dt:6.1f}s", flush=True)
