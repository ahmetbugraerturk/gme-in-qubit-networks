"""Batched ADMM solver for the GMN SDP (Jungnitsch-Moroder-Guhne) with
per-problem convergence stopping and a dual certificate. Array-module generic
(`xp` = numpy for CPU, cupy for GPU); any number of qubits nq (d = 2**nq).

  primal:  min Re Tr(W rho)  s.t.  W = P_m + Q_m^{T_m},  0 <= P_m, Q_m <= I   (m over bipartitions)
  GMN(rho) = max(0, -min).
  dual  :  p >= sum_m [ neg(L_m) + neg(L_m^{T_m}) ]  for ANY L_m with sum_m L_m = rho
           (neg(X) = sum of negative eigenvalues).  Hence for any such L,
           GMN <= U(L) = sum_m [ |neg(L_m)| + |neg(L_m^{T_m})| ]  -- a rigorous upper bound.

ADMM splitting: x=(W,P_m,Q_m) on the affine set A x = 0 (closed-form projection, since
A A^H = 2I + 1 1^T (x) I  =>  inverse = (I - J/(2+K))/2), z = box projections (eigh clipping).
Scaled duals give the dual multipliers: L_m ~ -rho_admm * u_P^m; they are repaired so that
sum L_m = rho exactly, then U is evaluated.  Primal value from x (exactly satisfies the
equalities, boxes only to within the residual) gives the estimate L = max(0,-<W,rho>);
a problem is "converged" when  U - L <= tol_gap  and the primal residual <= tol_res.
Converged problems are removed from the active batch (compaction), so easy states stop early.
"""
import itertools
import numpy as np


def bipartition_cuts(nq):
    """Distinct bipartitions M|rest (smaller side listed; for even nq the half/half split
    keeps the side containing qubit 0).  nq=3 -> 3 cuts, 4 -> 7, 5 -> 15."""
    cuts = []
    for size in range(1, nq // 2 + 1):
        for c in itertools.combinations(range(nq), size):
            if 2 * size == nq and 0 not in c:
                continue
            cuts.append(c)
    return cuts


def _pt(xp, X, cut, nq):
    B, d = X.shape[0], X.shape[1]
    Y = X.reshape((B,) + (2,) * (2 * nq))
    for q in cut:
        Y = xp.swapaxes(Y, 1 + q, 1 + nq + q)
    return Y.reshape(B, d, d)


def _pt_stack(xp, X, cuts, nq):
    return xp.stack([_pt(xp, X[m], cuts[m], nq) for m in range(len(cuts))])


def _herm(xp, X):
    return 0.5 * (X + xp.conj(xp.swapaxes(X, -1, -2)))


def _box(xp, X):
    w, V = xp.linalg.eigh(_herm(xp, X))
    w = xp.clip(w, 0.0, 1.0)
    return (V * w[..., None, :]) @ xp.conj(xp.swapaxes(V, -1, -2))


def _neg_mass(xp, X):
    """sum of |negative eigenvalues|, over last two dims."""
    w = xp.linalg.eigvalsh(_herm(xp, X))
    return -xp.sum(xp.minimum(w, 0.0), axis=-1)


def dual_upper_bound(xp, rho, Pu, Qu, cuts, nq, rho_admm):
    """Rigorous GMN upper bound from ADMM scaled duals (see module docstring). rho (B,d,d)."""
    K = len(cuts)
    Lam = -rho_admm * 0.5 * (Pu + _pt_stack(xp, Qu, cuts, nq))      # (K,B,d,d)
    Lam = _herm(xp, Lam)
    Lam = Lam + (rho - Lam.sum(0))[None] / K                          # enforce sum L_m = rho exactly
    U = _neg_mass(xp, Lam).sum(0) + _neg_mass(xp, _pt_stack(xp, Lam, cuts, nq)).sum(0)
    return U                                                         # (B,)


def solve_batch(xp, rho, nq=None, rho_admm=0.03, alpha=1.5, max_iter=20000, check=100,
                tol_gap=1e-7, tol_res=1e-8):
    """rho: (B,d,d) complex on xp's device. Returns dict of (B,) arrays (on xp's device):
    gmn (primal estimate), gmn_upper (rigorous dual bound), iters, converged."""
    B0, d = rho.shape[0], rho.shape[1]
    nq = nq or int(round(np.log2(d)))
    cuts = bipartition_cuts(nq); K = len(cuts)
    rho = _herm(xp, rho)
    out_g = xp.zeros(B0); out_u = xp.full((B0,), np.inf); out_it = xp.zeros(B0, dtype=int)
    out_ok = xp.zeros(B0, dtype=bool)
    gid = xp.arange(B0)                       # global ids of the active problems
    B = B0
    Wz = xp.zeros((B, d, d), complex); Wu = xp.zeros_like(Wz)
    Pz = xp.zeros((K, B, d, d), complex); Pu = xp.zeros_like(Pz); Qz = xp.zeros_like(Pz); Qu = xp.zeros_like(Pz)
    rho_a = rho
    for it in range(1, max_iter + 1):
        vW = Wz - Wu - rho_a / rho_admm
        vP = Pz - Pu; vQ = Qz - Qu
        r = vP + _pt_stack(xp, vQ, cuts, nq) - vW[None]
        y = 0.5 * (r - r.sum(0)[None] / (2.0 + K))
        xW = vW + y.sum(0)
        xP = vP - y
        xQ = vQ - _pt_stack(xp, y, cuts, nq)
        hW = alpha * xW + (1 - alpha) * Wz
        hP = alpha * xP + (1 - alpha) * Pz
        hQ = alpha * xQ + (1 - alpha) * Qz
        newW = hW + Wu
        proj = _box(xp, xp.concatenate([hP + Pu, hQ + Qu]).reshape(2 * K * B, d, d)).reshape(2 * K, B, d, d)
        newP, newQ = proj[:K], proj[K:]
        check_now = (it % check == 0) or it == max_iter
        if check_now:
            pres = xp.maximum(xp.abs(xP - newP).max(axis=(0, 2, 3)), xp.abs(xQ - newQ).max(axis=(0, 2, 3)))
        Wu = Wu + hW - newW; Pu = Pu + hP - newP; Qu = Qu + hQ - newQ
        Wz, Pz, Qz = newW, newP, newQ
        if check_now:
            val = xp.real(xp.einsum("bij,bji->b", xW, rho_a))
            Lg = xp.maximum(0.0, -val)
            U = dual_upper_bound(xp, rho_a, Pu, Qu, cuts, nq, rho_admm)
            done = ((U - Lg) <= tol_gap) & (pres <= tol_res)
            if it == max_iter:
                done = xp.ones_like(done)
            ids = gid[done]
            out_g[ids] = Lg[done]; out_u[ids] = U[done]; out_it[ids] = it
            out_ok[ids] = ((U - Lg) <= tol_gap)[done] & (pres <= tol_res)[done]
            keep = ~done
            if not bool(keep.any()):
                break
            if int(keep.sum()) < B:
                gid = gid[keep]; rho_a = rho_a[keep]; B = int(keep.sum())
                Wz, Wu = Wz[keep], Wu[keep]
                Pz, Pu, Qz, Qu = Pz[:, keep], Pu[:, keep], Qz[:, keep], Qu[:, keep]
    return dict(gmn=out_g, gmn_upper=out_u, iters=out_it, converged=out_ok, cuts=K)
