#!/usr/bin/env python3
"""Plan Sec. 4.7 step 1: re-solve near-threshold GMN samples with an independent solver (CVXPY + CLARABEL, interior point) and compare with the certified ADMM bracket [L, U].
Samples: every state of results/contact whose triple / whole-state classification is NEAR THRESHOLD (class 2: U > 1e-6 and L <= 1e-6) and whose density matrix is stored (collision runs and continuous runs).
A random subsample (--n3 triples, --n4 whole states, seed fixed) is solved.  Output: results/second_solver.json and report/second_solver.tex.
Statuses: 'inside' = L - tol <= value <= U + tol;  'resolves zero' = value <= 1e-6;  'resolves positive' = value > 1e-6."""
import argparse, glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "atlas"))
import warnings; warnings.filterwarnings("ignore")
import cvxpy as cvx
from qgpu import contact as K
ap = argparse.ArgumentParser(); ap.add_argument("--n3", type=int, default=400); ap.add_argument("--n4", type=int, default=60); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--list", action="store_true"); a = ap.parse_args()
def cuts(n):
    import itertools
    out = []
    for k in range(1, n // 2 + 1):
        for c in itertools.combinations(range(n), k):
            if 2 * k == n and 0 not in c: continue
            out.append(c)
    return out
class Prob:
    def __init__(self, n):
        d = 2 ** n; self.n = n; self.rp = cvx.Parameter((d, d), hermitian=True); W = cvx.Variable((d, d), hermitian=True); cons = []
        for c in cuts(n):
            P = cvx.Variable((d, d), hermitian=True); Q = cvx.Variable((d, d), hermitian=True); cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]; QT = Q
            for q in c: QT = cvx.partial_transpose(QT, dims=(2,) * n, axis=q)
            cons.append(W == P + QT)
        self.pr = cvx.Problem(cvx.Minimize(cvx.real(cvx.trace(W @ self.rp))), cons)
    def solve(self, rho):
        self.rp.value = 0.5 * (rho + rho.conj().T)
        v = self.pr.solve(solver=cvx.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10); return max(0.0, -float(v)), self.pr.status
def states(f, z):
    if "_coll_tau" in f:
        r = z["rho"]; r = r[:, :, None] * r.conj()[:, None, :] if r.ndim == 2 else r; return r, np.arange(len(r))
    if f.endswith("_cont.npz") or f.endswith("_contlong.npz"):
        di = z["deep_idx"]; tr = z["traj"].astype(complex); w = z["weights"]; r = sum(wi * (tr[s][:, di].T[:, :, None] * np.conj(tr[s][:, di].T)[:, None, :]) for s, wi in enumerate(w)); return r, np.arange(len(di))
    return None, None
cand3 = []; cand4 = []
for f in sorted(glob.glob(os.path.join(HERE, "results/contact/*/*/*.npz"))):
    if "collgmn" in f or "collfine" in f: continue
    z = np.load(f)
    keys3 = [k for k in z.files if k.startswith("deep/tri/") and k.endswith("/cls")]; has4 = "deep/gmn4/cls" in z.files
    near3 = [(k, np.nonzero(z[k] == 2)[0]) for k in keys3]; near3 = [(k, i) for k, i in near3 if len(i)]; n4 = np.nonzero(z["deep/gmn4/cls"] == 2)[0] if has4 else []
    if not near3 and not len(n4): continue
    for k, idx in near3:
        rep = k.split("/")[2]
        for i in idx: cand3.append((f, rep, int(i)))
    for i in n4: cand4.append((f, "whole", int(i)))
print(f"near-threshold samples with stored states: triples {len(cand3)}, whole {len(cand4)}", flush=True)
if a.list: sys.exit(0)
rng = np.random.default_rng(a.seed); pick3 = [cand3[i] for i in rng.permutation(len(cand3))[:a.n3]]; pick4 = [cand4[i] for i in rng.permutation(len(cand4))[:a.n4]]
P3, P4 = Prob(3), Prob(4); rows = []; t0 = time.time()
cache = {}
for kind, pick in ((3, pick3), (4, pick4)):
    for f, rep, i in pick:
        if f not in cache: cache.clear(); z = np.load(f); cache[f] = (z, states(f, z))
        z, (rho, _) = cache[f]
        if rho is None: continue
        if kind == 3: sub = [ "abcd".index(c) for c in rep ]; r3 = K.ptrace_rho(np, rho[i][None], 4, sub)[0]; L = float(z[f"deep/tri/{rep}/L"][i]); U = float(z[f"deep/tri/{rep}/U"][i]); val, stt = P3.solve(r3)
        else: L = float(z["deep/gmn4/L"][i]); U = float(z["deep/gmn4/U"][i]); val, stt = P4.solve(rho[i])
        rows.append(dict(file=os.path.relpath(f, HERE), rep=rep, i=i, L=L, U=U, second=val, status=stt, inside=bool(L - 1e-6 <= val <= U + 1e-6), zero=bool(val <= 1e-6)))
        if len(rows) % 20 == 0: print(f"  {len(rows)} solved, {time.time() - t0:.0f}s", flush=True)
json.dump(rows, open(os.path.join(HERE, "results", "second_solver.json"), "w"))
for kind, name in ((3, "triples"), (4, "whole")):
    sub = [r for r in rows if (r["rep"] == "whole") == (kind == 4)]
    if sub: print(f"{name}: {len(sub)} solved, inside bracket {sum(r['inside'] for r in sub)}, value <= 1e-6 {sum(r['zero'] for r in sub)}, max |value - [L,U]| outside {max(max(r['L'] - r['second'], r['second'] - r['U'], 0) for r in sub):.2e}")
print("DONE")
