#!/usr/bin/env python3
"""GMN SDP throughput: cvxpy+SCS (CPU, what production uses) vs batched ADMM
on numpy (CPU) and cupy (GPU). Realistic inputs: n=4/5 triples of ex|+> that
survive the PPT + negativity-bound filters (i.e. the ones that need an SDP)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qnet import graphs as G, hamiltonians as H, evolve as E, gmn as GM, bound as BD, measures as MS
import gmn_admm as A

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5); ap.add_argument("--graph", type=int, default=20)
ap.add_argument("--nbatch", type=int, default=2000); ap.add_argument("--nscs", type=int, default=100)
ap.add_argument("--rho-admm", type=float, default=1.0); ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--max-iter", type=int, default=20000)
ap.add_argument("--nq", type=int, default=3, help="qubits per sub-state (3=triples, 4=quads)")
ap.add_argument("--no-cpu", action="store_true"); ap.add_argument("--alpha", type=float, default=1.5)
a = ap.parse_args()

Gph = G.load_graphs(a.n)[a.graph]
Hm = H.build_hamiltonian(Gph, "ex"); psi0 = H.initial_state(a.n, "plus")
ev, V = E.diagonalize(Hm); times = E.make_time_grid(ev, 100.0)
psi_t = E.evolve_batch(ev, V, psi0.full().ravel(), times)
import itertools, networkx as nx, cvxpy as cp
subsets = [s for s in itertools.combinations(range(a.n), a.nq) if nx.is_connected(Gph.subgraph(s))]
d = 2 ** a.nq; DIMS = (2,) * a.nq
cuts = A.bipartition_cuts(a.nq)
rhos = []
for trip in subsets:
    r = E.partial_trace_pure_batch(psi_t, a.n, trip)
    me = np.stack([MS.min_eig_partial_transpose_batch(r, DIMS, list(c)) for c in cuts], 1)
    ub, _ = BD.gmn_upper_bound_batch(r, n_sub=a.nq)
    rhos.append(r[~(me >= -1e-10).any(1) & (ub >= 1e-6)])
rhos = np.concatenate(rhos)
print(f"n={a.n} g{a.graph}: {len(rhos)} grid points need an SDP (of {len(times)*len(subsets)})", flush=True)
rng = np.random.default_rng(0)
batch = rhos[rng.choice(len(rhos), min(a.nbatch, len(rhos)), replace=False)]
out = dict(nq=a.nq, cuts=len(cuts), rho_admm=a.rho_admm, alpha=a.alpha, n=a.n, graph=a.graph, n_need_sdp=int(len(rhos)), nbatch=len(batch))

# --- CPU baseline: cvxpy + SCS, one at a time (current production path)
sub = batch[: a.nscs]
class _Prob:
    def __init__(s):
        s.rho = cp.Parameter((d, d), hermitian=True); s.W = cp.Variable((d, d), hermitian=True); cons = []
        for c in cuts:
            P = cp.Variable((d, d), hermitian=True); Q = cp.Variable((d, d), hermitian=True)
            cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]
            QT = Q
            for q in c: QT = cp.partial_transpose(QT, dims=DIMS, axis=q)
            cons.append(s.W == P + QT)
        s.p = cp.Problem(cp.Minimize(cp.real(cp.trace(s.W @ s.rho))), cons)
    def __call__(s, r):
        s.rho.value = 0.5 * (r + r.conj().T); v = s.p.solve(solver=cp.SCS, eps=1e-7, max_iters=50000)
        return {"gmn": max(0.0, -float(v))}
_pr = _Prob(); _solve = _pr
if len(sub): _solve(sub[0])  # warm cache
t0 = time.perf_counter(); ref = np.array([_solve(r)["gmn"] for r in sub]); t = time.perf_counter() - t0
out["scs_per_solve_s"] = t / max(len(sub), 1)
print(f"cvxpy+SCS : {t/max(len(sub),1)*1e3:8.2f} ms/solve  ({len(sub)} solves)", flush=True)

def run(name, xp, sync):
    rb = xp.asarray(batch)
    kw = dict(nq=a.nq, rho_admm=a.rho_admm, alpha=a.alpha, tol_gap=a.tol)
    A.solve_batch(xp, rb[:8], max_iter=200, **kw); sync()   # warm-up
    t0 = time.perf_counter()
    res = A.solve_batch(xp, rb, max_iter=a.max_iter, **kw); sync()
    t = time.perf_counter() - t0
    g, U, it, ok = [np.asarray(res[k].get() if hasattr(res[k], "get") else res[k]) for k in ("gmn", "gmn_upper", "iters", "converged")]
    err = np.abs(g[: len(sub)] - ref) if len(sub) else np.zeros(1)
    d = dict(per_solve_ms=t / len(batch) * 1e3, total_s=t, iters_mean=float(it.mean()), iters_max=int(it.max()),
             converged_frac=float(ok.mean()), max_gap=float(np.max(U - g)), median_gap=float(np.median(U - g)),
             max_err_vs_scs=float(err.max()), mean_err_vs_scs=float(err.mean()),
             sign_mismatch=int(((g[: len(sub)] > 1e-6) != (ref > 1e-6)).sum()))
    out[name] = d
    print(f"{name:10s}: {d['per_solve_ms']:8.3f} ms/solve  batch={len(batch)} total={t:.2f}s iters mean/max={d['iters_mean']:.0f}/{d['iters_max']} "
          f"converged={d['converged_frac']:.1%} max gap U-L={d['max_gap']:.1e} max|err vs SCS|={d['max_err_vs_scs']:.2e} "
          f"sign-mismatch={d['sign_mismatch']}", flush=True)

import cupy as cp
run("cupy_gpu", cp, cp.cuda.Stream.null.synchronize)
if not a.no_cpu: run("numpy_cpu", np, lambda: None)
print(json.dumps(out))
