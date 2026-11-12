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
for k in ("near-zero n4g5 nq3", "near-zero n5g20 nq3"):
    nq, rh = sets[k]; idx = np.unique(np.linspace(0, len(rh) - 1, 20).astype(int)); refs[k] = (idx, clarabel(rh[idx], nq))
refs["near-zero n5g20 nq4"] = (np.array([0, 14, 29, 44, 59]), np.array([3.393e-08, 1.040e-06, 5.608e-06, 1.375e-05, 1.578e-05]))   # CLARABEL values logged by the validation run
print("data + references ready", flush=True)

for name, (nq, rhos) in sets.items():
    print(f"\n===== {name}: {len(rhos)} states", flush=True); rb = cp.asarray(rhos)
    for alpha in (1.5, 1.0):
        for rho in (0.03, 0.01, 0.003, 0.001):
            A.solve_batch(cp, rb[:4], nq=nq, max_iter=100, rho_admm=rho, alpha=alpha); sync()
            t0 = time.perf_counter(); r = A.solve_batch(cp, rb, nq=nq, max_iter=40000, rho_admm=rho, alpha=alpha); sync(); t = time.perf_counter() - t0
            h = {k: cp.asnumpy(v) for k, v in r.items()}; L, U = h["gmn"], h["gmn_upper"]
            zero = U <= TOL; pos = (L > TOL) & ~zero; amb = ~zero & ~pos
            row = dict(set=name, alpha=alpha, rho=rho, t=round(t, 1), conv=round(float(h["converged"].mean()), 3), it_mean=int(h["iters"].mean()),
                       certified_zero=int(zero.sum()), positive=int(pos.sum()), ambiguous=int(amb.sum()), n=len(L))
            if name in refs:
                idx, ref = refs[name]
                row.update(wrong_zero=int((zero[idx] & (ref > TOL)).sum()), wrong_pos=int((pos[idx] & (ref <= TOL)).sum()),
                           bracket_ok=bool(np.all((L[idx] <= ref + 1e-9) & (ref <= U[idx] + 1e-9))), max_err_L=float(np.abs(L[idx] - ref).max()))
            print(json.dumps({k: (float("%.3g" % v) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
