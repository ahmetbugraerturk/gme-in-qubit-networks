#!/usr/bin/env python3
"""Does over-relaxation / a larger or adaptive penalty speed up the certified ADMM on the hard graphs (paw, diamond whole state; paw triple)?
Fixed random-free subset: 120 evenly spaced times in (0, pi/4); relative stop 1e-5 (floor 1e-7) on the certified bracket, 20000 iterations max.
Reports per setting: fraction stopped, median / 90th-percentile iterations, mean relative gap at the end, wall time.  GPU (CuPy).  Usage: python tests/test_tuning_hard.py"""
import itertools, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
import cupy as cp
from qgpu import core as C, xxvac as X, gmn_admm as A, store
CASES = [("diamond", "whole"), ("paw", "whole"), ("paw", "acd")]
SETS = [dict(rho_admm=0.01, alpha=1.5), dict(rho_admm=0.03, alpha=1.5), dict(rho_admm=0.03, alpha=1.8), dict(rho_admm=0.1, alpha=1.0), dict(rho_admm=0.1, alpha=1.5), dict(rho_admm=0.3, alpha=1.5)]   # round 2 (round 1: 0.003/0.01/0.03 x alpha 1/1.5, adaptive variants)
t = np.linspace(0.02, math.pi / 4, 120)
for name, sub in CASES:
    R = store.load(os.path.join(Q4, "results", "xx_vacuum_cert3", name)); edges = [tuple(map(int, e)) for e in R.edges]
    psi = X.evolution(np, edges, 4, t)
    if sub == "whole": rho = psi[:, :, None] * np.conj(psi)[:, None, :] if psi.shape[0] == len(t) else psi.T[:, :, None] * np.conj(psi.T)[:, None, :]; nq = 4
    else: rho = C.ptrace_pure(np, psi, 4, [ord(c) - 97 for c in sub]); nq = 3
    rho_g = cp.asarray(rho); print(f"\n== {name} {sub} ({nq} qubits, {len(t)} states)", flush=True)
    for kw in SETS:
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        r = A.solve_batch(cp, rho_g, nq=nq, max_iter=20000, strict_gap=True, rigorous_lower=True, rel_gap=1e-5, **kw); cp.cuda.Stream.null.synchronize(); dt = time.perf_counter() - t0
        it = cp.asnumpy(r["iters"]); cv = cp.asnumpy(r["converged"]); U = cp.asnumpy(r["gmn_upper"]); L = cp.asnumpy(r["gmn_lower"]); rg = (U - L) / np.maximum(U, 1e-12)
        print(f"  {str(kw):62s} stopped {100*cv.mean():5.1f}%  iters median {np.median(it):7.0f} p90 {np.percentile(it, 90):7.0f}  median rel gap {np.median(rg):8.1e}  max L>U {float(np.max(L-U)):8.1e}  {dt:6.1f}s", flush=True)
