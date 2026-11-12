#!/usr/bin/env python3
"""Proposal test: a RIGOROUS lower bound on GMN from any ADMM iterate, by shrinking towards the strictly feasible point W = I.
Take boxed P_M, Q_M (0 <= . <= I), W_M = P_M + Q_M^{T_M}, W~ = mean_M W_M, E_M = W_M - W~, eps = max_M ||E_M||_op.
For lambda >= 2 eps / (1 + 2 eps):  W_l = (1-l) W~ + l I  is feasible, with P'_M = (1-l)(P_M - E_M) + l/2 I, Q'_M = (1-l) Q_M + l/2 I,
hence  GMN >= -(1-l) Re Tr(W~ rho) - l  (a certified lower bound).   This script runs ADMM for a fixed number of iterations and reports the
bracket [L_rig, U] against the primal estimate L and the atlas value, on K4 triples (xx|0000>)."""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, gmn_admm as A, store
xp = np; nq = 3; d = 8; cuts = A.bipartition_cuts(nq); K = len(cuts)

def admm(rho, iters, pen=0.003, alpha=1.0):
    B = rho.shape[0]; rs = rho; z0 = lambda: np.zeros((K, B, d, d), complex)
    Wz = np.zeros((B, d, d), complex); Pz, Pu, Qz, Qu = z0(), z0(), z0(), z0()
    for _ in range(iters):
        vW = Wz - rs / pen; vP = Pz - Pu; vQ = Qz - Qu
        r = vP + A._pt_stack(xp, vQ, cuts, nq) - vW[None]; y = 0.5 * (r - r.sum(0)[None] / (2.0 + K))
        xW = vW + y.sum(0); xP = vP - y; xQ = vQ - A._pt_stack(xp, y, cuts, nq)
        hW = alpha * xW + (1 - alpha) * Wz; hP = alpha * xP + (1 - alpha) * Pz; hQ = alpha * xQ + (1 - alpha) * Qz
        proj = A._box(xp, np.concatenate([hP + Pu, hQ + Qu]).reshape(2 * K * B, d, d)).reshape(2 * K, B, d, d); nP, nQ = proj[:K], proj[K:]
        Pu = Pu + hP - nP; Qu = Qu + hQ - nQ; Wz, Pz, Qz = hW, nP, nQ
    return xW, Pz, Qz, pen * y

def rigorous_lower(rho, P, Q):
    WM = P + A._pt_stack(xp, Q, cuts, nq); Wt = WM.mean(0); E = WM - Wt[None]
    eps = np.max(np.abs(np.linalg.eigvalsh(0.5 * (E + np.conj(np.swapaxes(E, -1, -2))))), axis=(0, 2)); lam = 2 * eps / (1 + 2 * eps)
    tr = np.real(np.einsum("bij,bji->b", Wt, rho)); return np.maximum(0.0, -(1 - lam) * tr - lam), eps, lam

R = store.load(os.path.join(Q4, "results", "xx_vacuum", "K4")); g = R.triple_gmn("abc"); idx = [100, 200, 300, 400, 500, 700, 900]
psi = X.evolution(np, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4, R.t[idx]); rho = C.ptrace_pure(np, psi, 4, [0, 1, 2])
print(f"K4 triple abc, ADMM (rho=0.003, alpha=1.0), fixed iterations; U = certified upper bound from the dual")
print(f"{'iters':>6s} {'Jt/pi':>7s} {'L (primal est.)':>16s} {'U (dual)':>10s} {'eps':>9s} {'L_rig (new)':>12s} {'U - L_rig':>10s}  L_rig<=U? L_rig<=L?")
ok = True
for iters in (2000, 10000):
    xW, P, Q, Lam = admm(rho, iters); cutsN = None
    Lest = np.maximum(0, -np.real(np.einsum("bij,bji->b", xW, rho))); U = A.dual_upper_bound(xp, rho, Lam, cuts, nq); Lr, eps, lam = rigorous_lower(rho, P, Q)
    for i in range(len(idx)):
        c1 = Lr[i] <= U[i] + 1e-9; c2 = Lr[i] <= Lest[i] + 1e-9; ok &= bool(c1)
        print(f"{iters:6d} {R.t[idx[i]]/math.pi:7.3f} {Lest[i]:16.6f} {U[i]:10.6f} {eps[i]:9.1e} {Lr[i]:12.6f} {U[i]-Lr[i]:10.1e}  {c1!s:9s} {c2!s}")
print("rigorous lower bound never above the certified upper bound:", ok)
sys.exit(0 if ok else 1)
