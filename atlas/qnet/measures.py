"""Entanglement measures: Wootters concurrence (own vectorized implementation,
cross-checked against qt.concurrence), partial transpose / negativity (used
for the PPT pre-check and the GMN sampling proxy), C_GME and Meyer-Wallach Q
for the whole pure state.
"""
from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np
import qutip as qt

SIGMA_Y = np.array([[0, -1j], [1j, 0]])
YY = np.kron(SIGMA_Y, SIGMA_Y)


def clean_rho(rho: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Enforce Hermiticity and clip tiny negative eigenvalues, renormalize trace."""
    rho = 0.5 * (rho + rho.conj().T)
    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    tr = w.sum()
    if tr > tol:
        w = w / tr
    out = (V * w) @ V.conj().T
    return 0.5 * (out + out.conj().T)


def concurrence_single(rho: np.ndarray) -> float:
    """Wootters concurrence of a single 2-qubit density matrix (4x4)."""
    rho = clean_rho(rho)
    rho_tilde = YY @ rho.conj() @ YY
    M = rho @ rho_tilde
    evals = np.linalg.eigvals(M)
    evals = np.clip(evals.real, 0.0, None)
    lambdas = np.sqrt(np.sort(evals)[::-1])
    C = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]
    return float(max(0.0, C))


def concurrence_batch(rho_t: np.ndarray) -> np.ndarray:
    """rho_t: shape (T, 4, 4). Returns C(t), shape (T,)."""
    T = rho_t.shape[0]
    out = np.empty(T)
    for t in range(T):
        out[t] = concurrence_single(rho_t[t])
    return out


def concurrence_qutip_cross_check(rho: np.ndarray) -> float:
    q = qt.Qobj(rho, dims=[[2, 2], [2, 2]])
    return float(qt.concurrence(q))


def partial_transpose(rho: np.ndarray, dims: Sequence[int], sys: Sequence[int]) -> np.ndarray:
    """Explicit partial transpose over the parties listed in `sys`."""
    k = len(dims)
    shape = tuple(dims) + tuple(dims)
    T = rho.reshape(shape)
    perm = list(range(2 * k))
    for s in sys:
        perm[s], perm[k + s] = perm[k + s], perm[s]
    T2 = np.transpose(T, perm)
    return np.ascontiguousarray(T2).reshape(rho.shape)


def min_eig_partial_transpose(rho: np.ndarray, dims: Sequence[int], sys: Sequence[int]) -> float:
    pt = partial_transpose(rho, dims, sys)
    pt = 0.5 * (pt + pt.conj().T)
    return float(np.min(np.linalg.eigvalsh(pt)))


def negativity(rho: np.ndarray, dims: Sequence[int], sys: Sequence[int]) -> float:
    """N = sum of |negative eigenvalues| of the partial transpose over `sys`."""
    pt = partial_transpose(rho, dims, sys)
    pt = 0.5 * (pt + pt.conj().T)
    w = np.linalg.eigvalsh(pt)
    return float(-np.sum(w[w < 0]))


def partial_transpose_batch(rho_t: np.ndarray, dims: Sequence[int], sys: Sequence[int]) -> np.ndarray:
    """Batched partial transpose. rho_t: (T, d, d) with d = prod(dims)."""
    T = rho_t.shape[0]
    k = len(dims)
    shape = (T,) + tuple(dims) + tuple(dims)
    Tt = rho_t.reshape(shape)
    perm = [0] + list(range(1, 2 * k + 1))
    for s in sys:
        perm[1 + s], perm[1 + k + s] = perm[1 + k + s], perm[1 + s]
    Tt2 = np.transpose(Tt, perm)
    return np.ascontiguousarray(Tt2).reshape(rho_t.shape)


def min_eig_partial_transpose_batch(rho_t: np.ndarray, dims: Sequence[int], sys: Sequence[int]) -> np.ndarray:
    """numpy.linalg.eigvalsh is natively batched over leading dims."""
    pt = partial_transpose_batch(rho_t, dims, sys)
    pt = 0.5 * (pt + np.swapaxes(pt.conj(), 1, 2))
    return np.linalg.eigvalsh(pt)[:, 0]  # eigvalsh returns ascending order


def negativity_batch(rho_t: np.ndarray, dims: Sequence[int], sys: Sequence[int]) -> np.ndarray:
    pt = partial_transpose_batch(rho_t, dims, sys)
    pt = 0.5 * (pt + np.swapaxes(pt.conj(), 1, 2))
    w = np.linalg.eigvalsh(pt)  # (T, d) ascending
    return -np.sum(np.clip(w, None, 0.0), axis=1)


def purity_of_reduced(rho_full: np.ndarray, n: int, keep: Sequence[int]) -> float:
    """Tr(rho_S^2) for S = `keep`, from the full k-qubit density matrix,
    tracing out the rest (for mixed states; use measures on reduced states
    already computed elsewhere for pure global states)."""
    raise NotImplementedError("use evolve.partial_trace_pure_batch for pure global states")


def c_gme_pure_single(psi: np.ndarray, n: int, max_size: int | None = None) -> float:
    """C_GME = min over bipartitions S (nonempty, |S| <= n/2) of
    sqrt(2*(1 - Tr(rho_S^2))), for ONE pure state vector (dim 2**n,)."""
    from .evolve import partial_trace_pure_batch

    if max_size is None:
        max_size = n // 2
    best = np.inf
    psi_col = psi.reshape(-1, 1)
    nodes = list(range(n))
    for size in range(1, max_size + 1):
        for S in itertools.combinations(nodes, size):
            rho_S = partial_trace_pure_batch(psi_col, n, S)[0]
            purity = float(np.real(np.trace(rho_S @ rho_S)))
            val = np.sqrt(max(0.0, 2.0 * (1.0 - purity)))
            best = min(best, val)
    return float(best)


def c_gme_pure_batch(psi_t: np.ndarray, n: int, max_size: int | None = None) -> np.ndarray:
    """Vectorized over time: psi_t shape (2**n, T)."""
    from .evolve import partial_trace_pure_batch

    if max_size is None:
        max_size = n // 2
    T = psi_t.shape[1]
    best = np.full(T, np.inf)
    nodes = list(range(n))
    for size in range(1, max_size + 1):
        for S in itertools.combinations(nodes, size):
            rho_S_t = partial_trace_pure_batch(psi_t, n, S)  # (T, d, d)
            purity_t = np.real(np.einsum("tab,tba->t", rho_S_t, rho_S_t))
            val_t = np.sqrt(np.clip(2.0 * (1.0 - purity_t), 0.0, None))
            best = np.minimum(best, val_t)
    return best


def meyer_wallach_Q_batch(psi_t: np.ndarray, n: int) -> np.ndarray:
    """Q(t) = 2*(1 - (1/n) sum_i Tr(rho_i(t)^2)), single-qubit reduced states."""
    from .evolve import partial_trace_pure_batch

    T = psi_t.shape[1]
    acc = np.zeros(T)
    for i in range(n):
        rho_i_t = partial_trace_pure_batch(psi_t, n, [i])
        purity_t = np.real(np.einsum("tab,tba->t", rho_i_t, rho_i_t))
        acc += purity_t
    return 2.0 * (1.0 - acc / n)
