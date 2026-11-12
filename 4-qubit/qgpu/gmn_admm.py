"""Batched ADMM solver for the GMN SDP (Jungnitsch-Moroder-Guhne) with per-problem convergence stopping,
a dual certificate and optional per-problem adaptive penalty.  `xp` = numpy (CPU) or cupy (GPU); any nq.

  primal:  min Re Tr(W rho)  s.t.  W = P_m + Q_m^{T_m},  0 <= P_m, Q_m <= I   (m over bipartitions)
  GMN(rho) = max(0, -min).
  dual  :  for ANY Hermitian L_m with sum_m L_m = rho:  GMN <= U(L) = sum_m [|nu(L_m)| + |nu(L_m^{T_m})|]
           (nu(X) = sum of the negative eigenvalues of X).  U is a rigorous upper bound.

Splitting: x = (W, P_m, Q_m) on the affine set A x = 0 (closed-form projection: A A^H = 2I + 1 1^T (x) I, so
(A A^H)^-1 = (I - J/(2+K))/2);  z = (W, boxed P_m, boxed Q_m).  W is unconstrained, so z_W = x^_W + u_W and the
scaled dual u_W is identically 0: it is NOT stored (and z_W is the only W state).
Dual estimate (dual="y"): the equality multiplier of the x-step, L_m = pen * y_m, satisfies
sum_m L_m = rho + pen * (x_W - z_W) exactly, i.e. it is already dual-feasible up to the (vanishing) W residual,
so the repair step  L_m += (rho - sum L)/K  (= Frobenius projection onto {sum L = rho}) is tiny.
"""
import itertools
import numpy as np


def bipartition_cuts(nq):
    """Distinct bipartitions M|rest (smaller side; for even nq the half/half split keeps the side containing
    qubit 0).  nq=3 -> 3 cuts, 4 -> 7, 5 -> 15."""
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
    return -xp.sum(xp.minimum(xp.linalg.eigvalsh(_herm(xp, X)), 0.0), axis=-1)


def dual_upper_bound(xp, rho, Lam, cuts, nq):
    """Rigorous GMN upper bound from raw multiplier estimates Lam (K,B,d,d): repair to sum = rho, evaluate U."""
    K = len(cuts)
    Lam = _herm(xp, Lam)
    Lam = Lam + (rho - Lam.sum(0))[None] / K
    return _neg_mass(xp, Lam).sum(0) + _neg_mass(xp, _pt_stack(xp, Lam, cuts, nq)).sum(0)


def _mx(xp, A, axes):
    return xp.abs(A).max(axis=axes)


def certified_lower(xp, Wc, P, Q, rs, margin=1e-12):
    """Certified lower bound on GMN(rho) from ANY pair (P_M, Q_M) with P_M + Q_M^{T_M} = Wc for every cut M (shift and scale into the box [0, I]).
    d_lo / d_hi: largest excursion of P_M (Q_M) below 0 / above 1;  s = 1 + max(d_lo_P + d_hi_P, d_lo_Q + d_hi_Q);
    P' = (P + d_lo_P I)/s, Q' = (Q + d_lo_Q I)/s are boxed and P'_M + Q'_M^{T_M} = (Wc + (d_lo_P + d_lo_Q) I)/s for every M (I is invariant under partial transpose),
    so GMN >= -Tr(W' rho) = (-Re Tr(Wc rho) - (d_lo_P + d_lo_Q) Tr rho)/s.  `margin` covers floating-point round-off of the eigenvalues."""
    def exc(X):
        ev = xp.linalg.eigvalsh(_herm(xp, X))                                  # (K, B, d)
        return xp.clip(-ev.min(axis=(0, 2)), 0.0, None) + margin, xp.clip(ev.max(axis=(0, 2)) - 1.0, 0.0, None) + margin
    lp, hp = exc(P); lq, hq = exc(Q); sc = 1.0 + xp.maximum(lp + hp, lq + hq)
    num = -xp.real(xp.einsum("bij,bji->b", _herm(xp, Wc), rs)) - (lp + lq) * xp.real(xp.einsum("bii->b", rs))
    return xp.maximum(0.0, num / sc)



def solve_batch(xp, rho, nq=None, rho_admm=0.003, alpha=1.0, max_iter=40000, check=100,
                tol_gap=1e-7, tol_res=1e-8, dual="y", adaptive_rho=False, mu=10.0, tau=2.0,
                rho_bounds=(1e-4, 1e2), compact=True, strict_gap=False, rigorous_lower=False, rel_gap=0.0):
    """rho: (B,d,d) complex on xp's device.  Returns dict of (B,) arrays on xp's device:
    gmn (primal estimate L), gmn_upper (rigorous U), iters, converged, pres (primal residual, max-abs),
    aff (affine violation of the clipped variables, max-abs), gap (U-L), pen (final penalty per problem).
    Defaults (rho_admm=0.003, alpha=1.0, max_iter=40000) were chosen on hard K4/K5 states: converged 99.5% (3-qubit
    subset) and 87.3% (4-qubit whole state) vs 95.5% / 67.3% for the earlier rho=0.03, alpha=1.5.
    dual in {"y","u_avg","u_P"}: how the multipliers are read off.  compact=False keeps every problem in the
    batch (masking only) -- same results, used to measure the cost/benefit of compaction.
    strict_gap=True : stop on |U - L| <= tol_gap.  The default test U - L <= tol_gap also accepts L > U (a negative gap), which stopped many
                      runs prematurely with L above the certified bound (see report/xx_study_report.pdf, 'stopping-rule defect').
    rigorous_lower=True : also return out["gmn_lower"] (= out["gmn"], the uncertified primal estimate moves to out["gmn_primal"]), a CERTIFIED lower bound obtained by shrinking the boxed iterate towards the strictly
                      feasible point W = I (P'_M = (1-l)(P_M - E_M) + l/2, Q'_M = (1-l) Q_M + l/2, l = 2 d/(1 + 2 d), d = largest excursion of the repaired P_M - E_M outside [0, I]; d <= eps = max_M ||W_M - mean W||),
                      and stop on U - gmn_lower <= tol_gap: the returned bracket [gmn_lower, gmn_upper] is then a proof, not an estimate."""
    B0, d = rho.shape[0], rho.shape[1]
    nq = nq or int(round(np.log2(d)))
    cuts = bipartition_cuts(nq); K = len(cuts)
    rho = _herm(xp, rho)
    f = lambda v, dt=float: xp.full((B0,), v, dtype=dt)
    out = {k: f(np.nan) for k in ("gmn", "gmn_upper", "pres", "aff", "gap", "pen")}
    out["iters"] = f(0, int); out["converged"] = f(False, bool); out["gmn_lower"] = f(np.nan); out["gmn_primal"] = f(np.nan)
    gid = xp.arange(B0); B = B0
    pen = f(rho_admm)                                    # per-problem penalty
    Wz = xp.zeros((B, d, d), complex)
    Pz = xp.zeros((K, B, d, d), complex); Pu = xp.zeros_like(Pz); Qz = xp.zeros_like(Pz); Qu = xp.zeros_like(Pz)
    rs = rho
    fin = xp.zeros(B, dtype=bool)                        # only used when compact=False
    for it in range(1, max_iter + 1):
        p3 = pen[:, None, None]; p4 = pen[None, :, None, None]
        vW = Wz - rs / p3
        vP = Pz - Pu; vQ = Qz - Qu
        r = vP + _pt_stack(xp, vQ, cuts, nq) - vW[None]
        y = 0.5 * (r - r.sum(0)[None] / (2.0 + K))
        xW = vW + y.sum(0)
        xP = vP - y
        xQ = vQ - _pt_stack(xp, y, cuts, nq)
        hW = alpha * xW + (1 - alpha) * Wz
        hP = alpha * xP + (1 - alpha) * Pz
        hQ = alpha * xQ + (1 - alpha) * Qz
        newW = hW                                        # z_W = x^_W (+ u_W == 0)
        proj = _box(xp, xp.concatenate([hP + Pu, hQ + Qu]).reshape(2 * K * B, d, d)).reshape(2 * K, B, d, d)
        newP, newQ = proj[:K], proj[K:]
        chk = (it % check == 0) or it == max_iter
        if chk:
            pres = xp.maximum(xp.maximum(_mx(xp, xP - newP, (0, 2, 3)), _mx(xp, xQ - newQ, (0, 2, 3))), _mx(xp, xW - newW, (1, 2)))
            dres = pen * xp.maximum(xp.maximum(_mx(xp, newP - Pz, (0, 2, 3)), _mx(xp, newQ - Qz, (0, 2, 3))), _mx(xp, newW - Wz, (1, 2)))
            aff = _mx(xp, newP + _pt_stack(xp, newQ, cuts, nq) - newW[None], (0, 2, 3))
            if dual == "y":
                Lam = p4 * y
            elif dual == "u_P":
                Lam = -p4 * (Pu + hP - newP)
            else:  # "u_avg"
                Lam = -p4 * 0.5 * ((Pu + hP - newP) + _pt_stack(xp, Qu + hQ - newQ, cuts, nq))
            U = dual_upper_bound(xp, rs, Lam, cuts, nq)
            val = xp.real(xp.einsum("bij,bji->b", xW, rs)); Lg = xp.maximum(0.0, -val)
            gapv = xp.abs(U - Lg) if strict_gap else (U - Lg)
            if rigorous_lower:
                WM = newP + _pt_stack(xp, newQ, cuts, nq); Wt = WM.mean(0); Ed = WM - Wt[None]
                # two certified candidates, the larger one is kept (both are proofs): (a) boxed iterate repaired affinely, P_M - E_M with Q_M = newQ (P_rep + Q^{T_M} = Wt);
                # (b) the affine iterate (xP, xQ, xW), which satisfies the coupling exactly and violates only the box.
                Lrig = xp.maximum(certified_lower(xp, Wt, newP - Ed, newQ, rs), certified_lower(xp, xW, xP, xQ, rs))
                ok = (U - Lrig) <= xp.maximum(tol_gap, rel_gap * U)      # rel_gap > 0: relative stop for strongly entangled states (absolute floor tol_gap)
            else:
                Lrig = None; ok = (gapv <= tol_gap) & (pres <= tol_res)
            last = it == max_iter
            newly = (ok | last) if compact else ((ok | last) & ~fin)
            ids = gid[newly]
            for k, v in (("gmn", Lg), ("gmn_upper", U), ("pres", pres), ("aff", aff), ("gap", U - Lg), ("pen", pen)):
                out[k][ids] = v[newly]
            out["iters"][ids] = it; out["converged"][ids] = ok[newly]
            if Lrig is not None:
                out["gmn_lower"][ids] = Lrig[newly]; out["gmn_primal"][ids] = Lg[newly]
                out["gmn"][ids] = Lrig[newly]; out["gap"][ids] = (U - Lrig)[newly]       # certified bracket: gmn <= GMN <= gmn_upper (L > U cannot occur)
            if compact:
                keep = ~newly
                nk = int(keep.sum())
                if nk == 0:
                    break
                if nk < B:
                    gid = gid[keep]; rs = rs[keep]; pen = pen[keep]; B = nk
                    Wz = newW[keep]; Pu = (Pu + hP - newP)[:, keep]; Qu = (Qu + hQ - newQ)[:, keep]
                    Pz = newP[:, keep]; Qz = newQ[:, keep]
                    continue
            else:
                fin = fin | newly
                if bool(fin.all()):
                    break
            if adaptive_rho:
                # residual balancing, per problem; relative residuals as in OSQP
                pn = pres / xp.maximum(1e-12, xp.maximum(_mx(xp, xP, (0, 2, 3)), _mx(xp, newP, (0, 2, 3))))
                dn = dres / xp.maximum(1e-12, pen * xp.maximum(_mx(xp, Pu + hP - newP, (0, 2, 3)), 1e-30))
                up = pn > mu * dn; dn_ = dn > mu * pn
                fac = xp.where(up, tau, xp.where(dn_, 1.0 / tau, 1.0))
                newpen = xp.clip(pen * fac, *rho_bounds)
                sc = pen / newpen                         # keep unscaled multiplier y = pen*u fixed
                Pu = (Pu + hP - newP) * sc[None, :, None, None]; Qu = (Qu + hQ - newQ) * sc[None, :, None, None]
                pen = newpen; Wz, Pz, Qz = newW, newP, newQ
                continue
        Pu = Pu + hP - newP; Qu = Qu + hQ - newQ
        Wz, Pz, Qz = newW, newP, newQ
    return out


ZERO, POSITIVE, NEAR_THRESHOLD = 0, 1, 2


def classify(xp, L, U, tol=1e-6, certified=False):
    """Decision at threshold `tol` from the bracket [L, U] returned by solve_batch.
    ZERO (0)           : U <= tol                       -- rigorous (U is a certified upper bound on GMN).
    POSITIVE (1)       : L - (U - L) > tol              -- margin rule: L must clear the threshold by the bracket
                                                           width.  NOT rigorous (L is a primal estimate, not a
                                                           certified lower bound); validated empirically only.
    certified=True     : L is the certified lower bound (rigorous_lower=True): POSITIVE iff L > tol, a proof.
    NEAR_THRESHOLD (2) : otherwise -- report the bracket [L, U] instead of forcing a yes/no.
    """
    zero = U <= tol
    pos = (~zero) & ((L > tol) if certified else ((2.0 * L - U) > tol))
    return xp.where(zero, ZERO, xp.where(pos, POSITIVE, NEAR_THRESHOLD))
