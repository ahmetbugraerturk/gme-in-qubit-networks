#!/usr/bin/env python3
"""Validate the GPU ADMM GMN solver against CLARABEL (tight, interior point) at and around
GMN = 0, where wrong decisions would matter.  Test sets:
  (1) 3-qubit GHZ + white noise  rho = (1-p)|GHZ><GHZ| + p I/8 : GMN crosses 0 at a known p
  (2) 3-qubit W + white noise                                  : same
  (3) 4-qubit GHZ + white noise (16x16, 7 cuts)
  (4) real states: ex|+> triples with the smallest negativity bound (closest to GMN=0)
Checks: |ADMM - CLARABEL|, bracket (CLARABEL within [L-eps, U+eps]), decision at GMN_TOL=1e-6."""
import itertools, json, os, sys, time
import numpy as np, cvxpy as cp, networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp_gpu
import gmn_admm as A
from qnet import graphs as G, hamiltonians as H, evolve as E, bound as BD, measures as MS

TOL = 1e-6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_results_v2.jsonl")
open(OUT, "w").close()

class Clarabel:
    def __init__(s, nq):
        d = 2 ** nq; dims = (2,) * nq; s.cuts = A.bipartition_cuts(nq)
        s.rho = cp.Parameter((d, d), hermitian=True); s.W = cp.Variable((d, d), hermitian=True); cons = []
        for c in s.cuts:
            P = cp.Variable((d, d), hermitian=True); Q = cp.Variable((d, d), hermitian=True)
            cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]
            QT = Q
            for q in c: QT = cp.partial_transpose(QT, dims=dims, axis=q)
            cons.append(s.W == P + QT)
        s.p = cp.Problem(cp.Minimize(cp.real(cp.trace(s.W @ s.rho))), cons)
    def __call__(s, r):
        s.rho.value = 0.5 * (r + r.conj().T)
        v = s.p.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        return max(0.0, -float(v))

def ghz(nq):
    v = np.zeros(2 ** nq, complex); v[0] = v[-1] = 1 / np.sqrt(2); return np.outer(v, v.conj())
def wst(nq):
    v = np.zeros(2 ** nq, complex)
    for q in range(nq): v[1 << q] = 1 / np.sqrt(nq)
    return np.outer(v, v.conj())

def compare(name, nq, rhos, n_clar=None, rho_admm=0.03, max_iter=20000):
    rhos = np.asarray(rhos)
    t0 = time.perf_counter()
    res = A.solve_batch(cp_gpu, cp_gpu.asarray(rhos), nq=nq, rho_admm=rho_admm, max_iter=max_iter, tol_gap=1e-8)
    L, U, it, ok = [cp_gpu.asnumpy(res[k]) for k in ("gmn", "gmn_upper", "iters", "converged")]
    tg = time.perf_counter() - t0
    idx = np.arange(len(rhos)) if n_clar is None else np.unique(np.linspace(0, len(rhos) - 1, n_clar).astype(int))
    print(f"[{name}] ADMM on {len(rhos)} states: {tg:.1f}s, converged {ok.mean():.0%}, iters mean/max {it.mean():.0f}/{it.max()}; "
          f"now CLARABEL on {len(idx)} states", flush=True)
    cl = Clarabel(nq); ref = []
    for j, i in enumerate(idx):
        t1 = time.perf_counter(); ref.append(cl(rhos[i]))
        print(f"  [{name}] clarabel {j+1}/{len(idx)}  {time.perf_counter()-t1:6.1f}s  ref={ref[-1]:.3e}  admm L={L[i]:.3e} U={U[i]:.3e}", flush=True)
    ref = np.array(ref)
    err = np.abs(L[idx] - ref)
    bracket_viol = np.maximum(ref - (U[idx] + 1e-7), 0).max()        # reference above the dual upper bound would be a bug
    dec_mis = int(((L[idx] > TOL) != (ref > TOL)).sum())
    # "certified zero" decisions: U <= TOL must imply CLARABEL <= TOL (+ solver noise)
    cert0_bad = int(((U[idx] <= TOL) & (ref > 2 * TOL)).sum())
    row = dict(set=name, nq=nq, n=len(rhos), n_vs_clarabel=len(idx), max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
               max_U_minus_L=float((U - L).max()), converged=f"{ok.mean():.0%}", iters_mean=float(it.mean()), iters_max=int(it.max()),
               ref_above_U=float(bracket_viol), decision_mismatch=dec_mis, certified_zero_wrong=cert0_bad,
               n_ref_zero=int((ref <= TOL).sum()), gpu_s=tg)
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
    return row

rows = []
ps = np.linspace(0, 0.9, 46)
for name, st, nq in [("GHZ3+noise", ghz, 3), ("W3+noise", wst, 3), ("GHZ4+noise", ghz, 4)]:
    d = 2 ** nq
    rows.append(compare(name, nq, [(1 - p) * st(nq) + p * np.eye(d) / d for p in ps], n_clar=None if nq == 3 else 6))

# real states closest to GMN=0 (smallest negativity bound that still needs an SDP)
for n, gi, nq in [(4, 5, 3), (5, 20, 3), (5, 20, 4)]:
    Gph = G.load_graphs(n)[gi]
    psi_t = E.evolve_batch(*E.diagonalize(H.build_hamiltonian(Gph, "ex")), H.initial_state(n, "plus").full().ravel(),
                           E.make_time_grid(E.diagonalize(H.build_hamiltonian(Gph, "ex"))[0], 100.0))
    cuts = A.bipartition_cuts(nq); pool = []
    for s in itertools.combinations(range(n), nq):
        if not nx.is_connected(Gph.subgraph(s)): continue
        r = E.partial_trace_pure_batch(psi_t, n, s)
        me = np.stack([MS.min_eig_partial_transpose_batch(r, (2,) * nq, list(c)) for c in cuts], 1)
        ub, _ = BD.gmn_upper_bound_batch(r, n_sub=nq)
        m = ~(me >= -1e-10).any(1) & (ub >= 1e-6)
        pool += [(u, x) for u, x in zip(ub[m], r[m])]
    pool.sort(key=lambda t: t[0]); sel = [x for _, x in pool[:60]]
    print(f"real n={n} g{gi} nq={nq}: smallest bound values {[f'{u:.1e}' for u,_ in pool[:3]]}", flush=True)
    rows.append(compare(f"real n{n}g{gi} nq{nq} (smallest bound)", nq, sel, n_clar=5 if nq == 4 else 20))
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_results_v2.json"), "w"), indent=1)
