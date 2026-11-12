#!/usr/bin/env python3
"""Test of the tightened certified lower bound: eps (Weyl, ||E||) versus delta (actual excursion of P_M - E_M outside [0, I]).
Both are rigorous; delta <= eps, so L_delta >= L_eps.  Checked on the small-GMN points (t ~ 0.0125-0.025, pi/2 - those) that the first
certified rerun failed to certify, plus K4 control points.  Prints L_eps, L_delta, U and the primal estimate after fixed numbers of ADMM iterations.
L_ss_box / L_ss_aff: shift-and-scale certificate on the boxed (affinely repaired) and on the affine iterate.
Pass: L_eps <= L_delta <= max(L_ss_box, L_ss_aff) <= U (+1e-9) everywhere (the larger of the two is what the solver uses)."""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, gmn_admm as A, store
xp = np; nq = 3; d = 8; cuts = A.bipartition_cuts(nq); K = len(cuts)

def run(rho, checkpoints, pen=0.003, alpha=1.0):
    B = rho.shape[0]; z0 = lambda: np.zeros((K, B, d, d), complex); Wz = np.zeros((B, d, d), complex); Pz, Pu, Qz, Qu = z0(), z0(), z0(), z0(); res = {}
    for it in range(1, max(checkpoints) + 1):
        vW = Wz - rho / pen; vP = Pz - Pu; vQ = Qz - Qu
        r = vP + A._pt_stack(xp, vQ, cuts, nq) - vW[None]; y = 0.5 * (r - r.sum(0)[None] / (2.0 + K))
        xW = vW + y.sum(0); xP = vP - y; xQ = vQ - A._pt_stack(xp, y, cuts, nq)
        hW = alpha * xW + (1 - alpha) * Wz; hP = alpha * xP + (1 - alpha) * Pz; hQ = alpha * xQ + (1 - alpha) * Qz
        proj = A._box(xp, np.concatenate([hP + Pu, hQ + Qu]).reshape(2 * K * B, d, d)).reshape(2 * K, B, d, d); nP, nQ = proj[:K], proj[K:]
        Pu = Pu + hP - nP; Qu = Qu + hQ - nQ; Wz, Pz, Qz = hW, nP, nQ
        if it in checkpoints:
            WM = nP + A._pt_stack(xp, nQ, cuts, nq); Wt = WM.mean(0); Ed = WM - Wt[None]; herm = A._herm
            eps = np.abs(np.linalg.eigvalsh(herm(xp, Ed))).max(axis=(0, 2)); lam_e = 2 * eps / (1 + 2 * eps)
            ev = np.linalg.eigvalsh(herm(xp, nP - Ed)); dl = np.maximum(np.clip(-ev.min(axis=(0, 2)), 0, None), np.clip(ev.max(axis=(0, 2)) - 1, 0, None)) + 1e-12; lam_d = 2 * dl / (1 + 2 * dl)
            tr = np.real(np.einsum("bij,bji->b", Wt, rho)); trr = np.real(np.einsum("bii->b", rho))
            Le = np.maximum(0, -(1 - lam_e) * tr - lam_e * trr); Ld = np.maximum(0, -(1 - lam_d) * tr - lam_d * trr)
            Lam = pen * y; U = A.dual_upper_bound(xp, rho, Lam, cuts, nq); Lg = np.maximum(0, -np.real(np.einsum("bij,bji->b", xW, rho)))
            Lc_b = A.certified_lower(xp, Wt, nP - Ed, nQ, rho); Lc_a = A.certified_lower(xp, xW, xP, xQ, rho)
            res[it] = (Le, Ld, U, Lg, eps, dl, Lc_b, Lc_a)
    return res

CASES = [("star", "abd", 0.0165), ("path", "abc", 0.0165), ("cycle", "abc", 0.0125), ("cycle", "abc", 0.0249), ("diamond", "abc", 0.0125), ("paw", "abc", 0.0165), ("K4", "abc", 0.4), ("K4", "abc", 0.78)]
ok = True; CK = (500, 2000, 10000, 30000)
print(f"{'case':22s} {'iters':>6s} {'L_est':>10s} {'L_eps':>10s} {'L_delta':>10s} {'L_ss_box':>10s} {'L_ss_aff':>10s} {'U':>10s}")
for name, tri, t in CASES:
    R = store.load(os.path.join(Q4, "results", "xx_vacuum", name)); psi = X.evolution(np, [tuple(map(int, e)) for e in R.edges], 4, np.array([t])); rho = C.ptrace_pure(np, psi, 4, [ord(c) - 97 for c in tri])
    res = run(rho, CK)
    for it in CK:
        Le, Ld, U, Lg, eps, dl, Lb, La = (v[0] for v in res[it]); Lc = max(Lb, La); good = Ld >= Le - 1e-10 and Lc >= Ld - 1e-10 and Lc <= U + 1e-9; ok &= bool(good)
        print(f"{name+' '+tri+' t='+str(t):22s} {it:6d} {Lg:10.3e} {Le:10.3e} {Ld:10.3e} {Lb:10.3e} {La:10.3e} {U:10.3e}  {'ok' if good else 'FAIL'}")
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
