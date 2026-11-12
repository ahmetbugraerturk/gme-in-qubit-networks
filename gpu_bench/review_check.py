#!/usr/bin/env python3
"""Check the points of an external code review of gmn_admm.py by measurement.
 (1) W_u == 0 identically -> dropping it must reproduce v1 exactly.
 (2) dual-bound estimator: v1's u-average + repair vs the x-step multiplier ("y") vs u_P only.
 (3) which stopping condition blocks convergence (primal residual vs duality gap); affine violation of clipped z.
 (4) compaction vs masking, check-interval schedule (sync cost).
 (5) per-problem adaptive penalty.
Data: K4 ex|+>: a 3-qubit subset, the 4-qubit whole state (hard cases: 97% / 73% converged before) + W3-noise sweep (CLARABEL ref)."""
import itertools, json, os, sys, time
import numpy as np, cupy as cp, cvxpy as cvx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "atlas")); sys.path.insert(0, HERE)
import gmn_admm as A2, gmn_admm_v1 as A1
from qnet import graphs as G, hamiltonians as Hm, evolve as E

sync = cp.cuda.Stream.null.synchronize
n = 4; Gph = G.load_graphs(n)[5]; Hmat = Hm.build_hamiltonian(Gph, "ex"); psi0 = Hm.initial_state(n, "plus").full().ravel()
ev, V = E.diagonalize(Hmat); times = E.make_time_grid(ev, 100.0); psi_t = E.evolve_batch(ev, V, psi0, times)

def nonppt(rho, nq):
    from qnet import measures as M
    cuts = A2.bipartition_cuts(nq)
    me = np.stack([M.min_eig_partial_transpose_batch(rho, (2,) * nq, list(c)) for c in cuts], 1)
    return ~(me >= -1e-10).any(1)
r3 = E.partial_trace_pure_batch(psi_t, n, (0, 1, 2)); r3 = r3[nonppt(r3, 3)]
r4 = np.einsum("it,jt->tij", psi_t, psi_t.conj()); r4 = r4[nonppt(r4, 4)]
sets = {"K4 triple (8x8)": (3, r3[:: max(1, len(r3) // 400)][:400]), "K4 whole (16x16)": (4, r4[:: max(1, len(r4) // 150)][:150])}
w = np.zeros(8, complex)
for q in range(3): w[1 << q] = 1 / np.sqrt(3)
Wst = np.outer(w, w.conj()); sets["W3+noise sweep"] = (3, np.array([(1 - p) * Wst + p * np.eye(8) / 8 for p in np.linspace(0, 0.9, 46)]))

def clarabel_refs(rhos, nq):
    d = 2 ** nq; rho = cvx.Parameter((d, d), hermitian=True); Wv = cvx.Variable((d, d), hermitian=True); cons = []
    for c in A2.bipartition_cuts(nq):
        P = cvx.Variable((d, d), hermitian=True); Q = cvx.Variable((d, d), hermitian=True)
        cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]
        QT = Q
        for q in c: QT = cvx.partial_transpose(QT, dims=(2,) * nq, axis=q)
        cons.append(Wv == P + QT)
    pr = cvx.Problem(cvx.Minimize(cvx.real(cvx.trace(Wv @ rho))), cons); out = []
    for r in rhos:
        rho.value = 0.5 * (r + r.conj().T); out.append(max(0, -pr.solve(solver=cvx.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)))
    return np.array(out)
refs = {"K4 triple (8x8)": clarabel_refs(sets["K4 triple (8x8)"][1][:20], 3), "W3+noise sweep": clarabel_refs(sets["W3+noise sweep"][1], 3)}
print("reference solves done", flush=True)

def run(mod, name, setname, **kw):
    nq, rhos = sets[setname]; rb = cp.asarray(rhos)
    mod.solve_batch(cp, rb[:4], nq=nq, max_iter=100, **kw); sync()
    t0 = time.perf_counter(); r = mod.solve_batch(cp, rb, nq=nq, max_iter=20000, **kw); sync(); t = time.perf_counter() - t0
    h = {k: cp.asnumpy(v) for k, v in r.items()}
    row = dict(cfg=name, t=round(t, 2), it_mean=float(h["iters"].mean()), it_max=int(h["iters"].max()), conv=float(h["converged"].mean()))
    bad = ~h["converged"]
    if "pres" in h:
        row.update(n_bad=int(bad.sum()), pres_ok_gap_bad=int(((h["pres"] <= 1e-8) & (h["gap"] > 1e-7)).sum()),
                   gap_ok_pres_bad=int(((h["gap"] <= 1e-7) & (h["pres"] > 1e-8)).sum()), max_gap=float(np.nanmax(h["gap"])),
                   med_gap_bad=float(np.nanmedian(h["gap"][bad])) if bad.any() else 0.0, max_aff=float(np.nanmax(h["aff"])),
                   aff_over_pres_max=float(np.nanmax(h["aff"] / np.maximum(h["pres"], 1e-300))))
    if setname in refs:
        k = len(refs[setname]); row["max_err_vs_clarabel"] = float(np.abs(h["gmn"][:k] - refs[setname]).max()); row["min_U_minus_ref"] = float((h["gmn_upper"][:k] - refs[setname]).min())
    print(json.dumps({"set": setname, **{k: (float("%.3g" % v) if isinstance(v, float) else v) for k, v in row.items()}}), flush=True)
    return h

for s in sets:
    print(f"\n===== {s}: {len(sets[s][1])} states", flush=True)
    h1 = run(A1, "v1 baseline", s)
    h2 = run(A2, "v2 u_avg (W_u dropped)", s, dual="u_avg")
    if "pres" in h1 or True:
        same = np.array_equal(cp.asnumpy(A2.solve_batch(cp, cp.asarray(sets[s][1][:30]), nq=sets[s][0], dual="u_avg")["gmn"]), cp.asnumpy(A1.solve_batch(cp, cp.asarray(sets[s][1][:30]), nq=sets[s][0])["gmn"]))
        print(f"  (1) v2(u_avg) reproduces v1 gmn bit-for-bit on 30 states: {same}", flush=True)
    run(A2, "v2 u_P", s, dual="u_P")
    run(A2, "v2 y (fixed rho)", s, dual="y")
    run(A2, "v2 u_avg + adaptive rho", s, dual="u_avg", adaptive_rho=True)
    run(A2, "v2 y + adaptive rho", s, dual="y", adaptive_rho=True)
    run(A2, "v2 y, masking (no compaction)", s, dual="y", compact=False)
    for c in (25, 50, 200, 500):
        run(A2, f"v2 y, check={c}", s, dual="y", check=c)
