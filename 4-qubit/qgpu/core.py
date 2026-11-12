"""Spectral time evolution, reduced states, concurrence, PPT/negativity and Meyer-Wallach, written
against an array module `xp` (numpy = CPU, cupy = GPU).  Numerically identical to atlas/qnet
(checked in check_vs_atlas.py); see ../../gpu_bench/REPORT.md for the speed comparison."""
import numpy as np


def get_xp(device="gpu"):
    if device == "gpu":
        import cupy as cp
        return cp
    return np


def to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def evolve(xp, H, psi0, times):
    """psi(t) = V exp(-iEt) V^H psi0 for all t. H (d,d) Hermitian, psi0 (d,), times (T,). -> (d,T)."""
    H = xp.asarray(H); psi0 = xp.asarray(psi0); times = xp.asarray(times)
    e, V = xp.linalg.eigh(H)
    c = V.conj().T @ psi0
    return V @ (xp.exp(-1j * xp.outer(e, times)) * c[:, None])


def ptrace_pure(xp, psi_t, n, keep):
    """Reduced states of a pure-state trajectory: (d,T) -> (T,2^|keep|,2^|keep|)."""
    T = psi_t.shape[1]; keep = list(keep); rest = [q for q in range(n) if q not in keep]
    a = psi_t.T.reshape((T,) + (2,) * n)
    a = xp.transpose(a, [0] + [1 + q for q in keep] + [1 + q for q in rest]).reshape(T, 2 ** len(keep), -1)
    return a @ xp.conj(xp.swapaxes(a, 1, 2))


def concurrence(xp, rho):
    """Wootters concurrence of a batch (T,4,4) via the Hermitian form sqrt(sqrt(rho) rho~ sqrt(rho))."""
    sy = xp.asarray([[0, -1j], [1j, 0]]); YY = xp.kron(sy, sy)
    rt = YY @ xp.conj(rho) @ YY
    w, U = xp.linalg.eigh(rho); w = xp.clip(w, 0, None)
    sq = (U * xp.sqrt(w)[:, None, :]) @ xp.conj(xp.swapaxes(U, 1, 2))
    s = xp.sqrt(xp.clip(xp.linalg.eigvalsh(sq @ rt @ sq), 0, None))     # ascending
    return xp.clip(s[:, 3] - s[:, 2] - s[:, 1] - s[:, 0], 0, None)


def partial_transpose(xp, rho, nq, qubits):
    """Partial transpose of (T,d,d) states of nq qubits on the given qubits."""
    T, d = rho.shape[0], rho.shape[1]
    Y = rho.reshape((T,) + (2,) * (2 * nq))
    for q in qubits:
        Y = xp.swapaxes(Y, 1 + q, 1 + nq + q)
    return Y.reshape(T, d, d)


def min_eig_pt(xp, rho, nq, qubits):
    pt = partial_transpose(xp, rho, nq, qubits)
    return xp.linalg.eigvalsh(0.5 * (pt + xp.conj(xp.swapaxes(pt, 1, 2))))[:, 0]


def negativity(xp, rho, nq, qubits):
    pt = partial_transpose(xp, rho, nq, qubits)
    w = xp.linalg.eigvalsh(0.5 * (pt + xp.conj(xp.swapaxes(pt, 1, 2))))
    return -xp.sum(xp.minimum(w, 0.0), axis=1)


def meyer_wallach(xp, psi_t, n):
    acc = 0
    for i in range(n):
        r = ptrace_pure(xp, psi_t, n, [i]); acc = acc + xp.real(xp.einsum("tab,tba->t", r, r))
    return 2.0 * (1.0 - acc / n)


def concurrence_checked(xp, rho):
    """Wootters concurrence of (T,4,4) states (Hermitian form).  Returns (C, worst_negative) where worst_negative is the
    largest negative eigenvalue magnitude that had to be clipped (round-off should be < 1e-12; flag anything larger)."""
    sy = xp.asarray([[0, -1j], [1j, 0]]); YY = xp.kron(sy, sy)
    rt = YY @ xp.conj(rho) @ YY
    w, U = xp.linalg.eigh(rho)
    sq = (U * xp.sqrt(xp.clip(w, 0, None))[:, None, :]) @ xp.conj(xp.swapaxes(U, 1, 2))
    m = sq @ rt @ sq
    mu = xp.linalg.eigvalsh(0.5 * (m + xp.conj(xp.swapaxes(m, 1, 2))))
    worst = max(0.0, -float(mu.min()), -float(w.min()))
    s = xp.sqrt(xp.clip(mu, 0, None))                                    # ascending
    return xp.clip(s[:, 3] - s[:, 2] - s[:, 1] - s[:, 0], 0, None), worst


def pure_cut(xp, psi_t, n, A):
    """Purity of rho_A and negativity across A|B for a pure-state trajectory psi_t (d,T), from the Schmidt coefficients s_i
    (singular values of psi reshaped to a d_A x d_B matrix; lambda_i = s_i^2).  Negativity N = sum_{i<j} s_i s_j.
    Using the SVD (not sqrt of eigenvalues of rho_A) keeps round-off at 1e-16: a round-off eigenvalue of 1e-17 would otherwise
    give sqrt = 3e-9 of spurious negativity at exactly-product cuts."""
    T = psi_t.shape[1]; A = list(A); rest = [q for q in range(n) if q not in A]
    a = psi_t.T.reshape((T,) + (2,) * n)
    a = xp.transpose(a, [0] + [1 + q for q in A] + [1 + q for q in rest]).reshape(T, 2 ** len(A), -1)
    sv = xp.linalg.svd(a, compute_uv=False)
    s1 = xp.sum(sv, axis=1); s2 = xp.sum(sv ** 2, axis=1)
    return xp.sum(sv ** 4, axis=1), xp.clip((s1 ** 2 - s2) / 2.0, 0.0, None)
