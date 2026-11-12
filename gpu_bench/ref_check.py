#!/usr/bin/env python3
"""Effect of penalty rho and relaxation alpha on convergence, cost and decision quality, on ALL states of five hard sets:
  K4 triple (400), K4 whole 4-qubit (150) and the three 'smallest negativity bound' real-state sets of the validation.
Decision quality: classify with the certificate  zero: U<=1e-6 | positive: L>1e-6 | ambiguous: otherwise,
and (where a CLARABEL reference exists) count wrong decisions."""
import itertools, json, os, sys, time
import numpy as np, cupy as cp, cvxpy as cvx, networkx as nx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "atlas")); sys.path.insert(0, HERE)
import gmn_admm as A
from qnet import graphs as G, hamiltonians as Hm, evolve as E, measures as M, bound as BD
sync = cp.cuda.Stream.null.synchronize; TOL = 1e-6

def traj(n, gi):
    Gph = G.load_graphs(n)[gi]; H = Hm.build_hamiltonian(Gph, "ex"); p0 = Hm.initial_state(n, "plus").full().ravel()
    ev, V = E.diagonalize(H); return Gph, E.evolve_batch(ev, V, p0, E.make_time_grid(ev, 100.0))
def nonppt(rho, nq):
    me = np.stack([M.min_eig_partial_transpose_batch(rho, (2,) * nq, list(c)) for c in A.bipartition_cuts(nq)], 1)
    return ~(me >= -1e-10).any(1), me
def clarabel(rhos, nq):
    d = 2 ** nq; rho = cvx.Parameter((d, d), hermitian=True); W = cvx.Variable((d, d), hermitian=True); cons = []
    for c in A.bipartition_cuts(nq):
        P = cvx.Variable((d, d), hermitian=True); Q = cvx.Variable((d, d), hermitian=True)
        cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]; QT = Q
        for q in c: QT = cvx.partial_transpose(QT, dims=(2,) * nq, axis=q)
        cons.append(W == P + QT)
    pr = cvx.Problem(cvx.Minimize(cvx.real(cvx.trace(W @ rho))), cons); out = []
    for r in rhos:
        rho.value = 0.5 * (r + r.conj().T); out.append(max(0, -pr.solve(solver=cvx.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)))
    return np.array(out)

sets = {}; refs = {}
G4, psi4 = traj(4, 5)
r3 = E.partial_trace_pure_batch(psi4, 4, (0, 1, 2)); r3 = r3[nonppt(r3, 3)[0]]; sets["K4 triple"] = (3, r3[:: max(1, len(r3) // 400)][:400])
r4 = np.einsum("it,jt->tij", psi4, psi4.conj()); r4 = r4[nonppt(r4, 4)[0]]; sets["K4 whole"] = (4, r4[:: max(1, len(r4) // 150)][:150])
for n, gi, nq in [(4, 5, 3), (5, 20, 3), (5, 20, 4)]:
    Gph, psi = traj(n, gi); pool = []
    for s in itertools.combinations(range(n), nq):
        if not nx.is_connected(Gph.subgraph(s)): continue
        r = E.partial_trace_pure_batch(psi, n, s); ok, _ = nonppt(r, nq); ub, _ = BD.gmn_upper_bound_batch(r, n_sub=nq)
        m = ok & (ub >= 1e-6); pool += [(u, x) for u, x in zip(ub[m], r[m])]
    pool.sort(key=lambda t: t[0]); sets[f"near-zero n{n}g{gi} nq{nq}"] = (nq, np.array([x for _, x in pool[:60]]))

def build(nq):
    d = 2 ** nq; rho = cvx.Parameter((d, d), hermitian=True); W = cvx.Variable((d, d), hermitian=True); cons = []
    for c in A.bipartition_cuts(nq):
        P = cvx.Variable((d, d), hermitian=True); Q = cvx.Variable((d, d), hermitian=True)
        cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]; QT = Q
        for q in c: QT = cvx.partial_transpose(QT, dims=(2,) * nq, axis=q)
        cons.append(W == P + QT)
    return rho, cvx.Problem(cvx.Minimize(cvx.real(cvx.trace(W @ rho))), cons)

for name in ("near-zero n4g5 nq3", "near-zero n5g20 nq3"):
    nq, rhos = sets[name]; rb = cp.asarray(rhos)
    r = A.solve_batch(cp, rb, nq=nq, max_iter=40000, rho_admm=0.003, alpha=1.0); L = cp.asnumpy(r["gmn"]); U = cp.asnumpy(r["gmn_upper"])
    r0 = A.solve_batch(cp, rb, nq=nq, max_iter=40000, rho_admm=0.03, alpha=1.5); L0 = cp.asnumpy(r0["gmn"]); U0 = cp.asnumpy(r0["gmn_upper"])
    # the states where candidate L overshoots the CLARABEL reference the most
    idx = np.unique(np.linspace(0, len(rhos) - 1, 20).astype(int)); worst = idx[np.argsort(-(L[idx] - np.array([0]*len(idx))))[:0]]
    pick = [4, 11, 19, 30, 45]
    rho, prob = build(nq)
    print(f"\n===== {name}", flush=True)
    for j in pick:
        rho.value = 0.5 * (rhos[j] + rhos[j].conj().T); res = {}
        for lab, kw in [("CLARABEL 1e-10 (as used)", dict(solver=cvx.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)),
                        ("CLARABEL 1e-13", dict(solver=cvx.CLARABEL, tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13, max_iter=500)),
                        ("SCS 1e-10", dict(solver=cvx.SCS, eps=1e-10, max_iters=300000))]:
            try:
                v = prob.solve(**kw); res[lab] = (max(0.0, -float(v)), prob.status)
            except Exception as e:
                res[lab] = (float("nan"), type(e).__name__)
        print(f"  state {j:2d}: ADMM old L={L0[j]:.4e} U={U0[j]:.4e} | ADMM cand L={L[j]:.4e} U={U[j]:.4e}", flush=True)
        for lab, (v, st) in res.items(): print(f"           {lab:26s} GMN={v:.4e}  status={st}", flush=True)
