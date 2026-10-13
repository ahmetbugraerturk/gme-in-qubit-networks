"""Exact time evolution via one-time spectral decomposition of H, vectorized
over a time grid. Includes cross-checks against qt.sesolve, vectorized
partial trace from a pure state (cross-checked against Qobj.ptrace), and
application of local Hadamards to a reduced density matrix (used for the
XX <-> ZZ symmetry shortcut of section 1, applied cheaply on the kept
subsystem rather than the full n-qubit state).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import qutip as qt

HAD2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def diagonalize(H: qt.Qobj) -> Tuple[np.ndarray, np.ndarray]:
    """Dense Hermitian diagonalization. Returns (evals, evecs) with
    H = evecs @ diag(evals) @ evecs.conj().T, evecs column-orthonormal.
    """
    Hm = H.full()
    Hm = 0.5 * (Hm + Hm.conj().T)  # enforce exact Hermiticity
    evals, evecs = np.linalg.eigh(Hm)
    return evals, evecs


def make_time_grid(evals: np.ndarray, t_max: float, dt_cap: float = 0.02, dt_factor: float = 0.1) -> np.ndarray:
    """dt = dt_factor / DeltaE_max, capped at dt_cap. DeltaE_max = Emax-Emin."""
    dE = float(evals.max() - evals.min())
    dt = dt_cap if dE < 1e-12 else min(dt_factor / dE, dt_cap)
    n_steps = max(int(np.ceil(t_max / dt)) + 1, 2)
    times = np.linspace(0.0, t_max, n_steps)
    return times


def evolve_batch(evals: np.ndarray, evecs: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """psi(t) = evecs @ diag(exp(-i*evals*t)) @ evecs^H @ psi0, vectorized
    over the whole time grid at once. Returns array of shape (dim, ntimes).
    """
    coeffs = evecs.conj().T @ psi0  # (dim,)
    phases = np.exp(-1j * np.outer(evals, times))  # (dim, ntimes)
    psi_t = evecs @ (phases * coeffs[:, None])
    return psi_t


def evolve_single(evals: np.ndarray, evecs: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    coeffs = evecs.conj().T @ psi0
    phases = np.exp(-1j * evals * t)
    return evecs @ (phases * coeffs)


def partial_trace_pure_batch(psi_t: np.ndarray, n: int, keep: Sequence[int]) -> np.ndarray:
    """Vectorized partial trace of a batch of pure states over all qubits
    except `keep`. psi_t has shape (2**n, T). Returns rho_keep of shape
    (T, 2**k, 2**k), k = len(keep).
    """
    keep = tuple(keep)
    T = psi_t.shape[1]
    trace_out = tuple(i for i in range(n) if i not in keep)
    order = keep + trace_out
    psi_resh = psi_t.reshape((2,) * n + (T,))
    perm = list(order) + [n]
    psi_perm = np.transpose(psi_resh, perm)
    k = len(keep)
    dim_keep = 2 ** k
    dim_trace = 2 ** (n - k)
    M = np.ascontiguousarray(psi_perm).reshape(dim_keep, dim_trace, T)
    rho = np.einsum("ait,bit->tab", M, M.conj())
    return rho


def local_unitary_tensor(U: np.ndarray, k: int) -> np.ndarray:
    """U^{(x)k}, Kronecker power of a single-qubit 2x2 unitary."""
    out = np.array([[1.0]], dtype=complex)
    for _ in range(k):
        out = np.kron(out, U)
    return out


def apply_local_unitary_batch(rho_t: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply the same local unitary U (already the full tensor-power matrix
    matching rho's dimension) to a batch of density matrices: U rho U^dagger
    for each time slice. rho_t shape (T, d, d).
    """
    Uc = U.conj().T
    return np.einsum("ab,tbc,cd->tad", U, rho_t, Uc)


def hadamard_transform_batch(rho_t: np.ndarray, k: int) -> np.ndarray:
    """H^{(x)k} rho H^{(x)k} for a batch of k-qubit density matrices."""
    Hk = local_unitary_tensor(HAD2, k)
    return apply_local_unitary_batch(rho_t, Hk)


def enforce_physical(rho: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Enforce Hermiticity and clip tiny negative eigenvalues (renormalize trace)."""
    rho = 0.5 * (rho + rho.conj().T)
    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    tr = w.sum()
    if tr > tol:
        w = w / tr
    rho_fixed = (V * w) @ V.conj().T
    return 0.5 * (rho_fixed + rho_fixed.conj().T)


# ---------------------------------------------------------------------------
# Cross-checks (used by tests, section 6 item 3 / engineering requirement 3).
# ---------------------------------------------------------------------------

def cross_check_sesolve(H: qt.Qobj, psi0: qt.Qobj, times: np.ndarray, n: int) -> float:
    """Compare spectral-decomposition evolution against qt.sesolve. Returns
    the max |psi_spectral - psi_sesolve| (up to global phase) over the grid.
    """
    evals, evecs = diagonalize(H)
    psi_spec = evolve_batch(evals, evecs, psi0.full().ravel(), times)

    res = qt.sesolve(H, psi0, times, options={"atol": 1e-14, "rtol": 1e-12})
    max_err = 0.0
    for idx, t in enumerate(times):
        a = psi_spec[:, idx]
        b = res.states[idx].full().ravel()
        # remove global phase before comparing
        overlap = np.vdot(b, a)
        phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
        err = np.linalg.norm(a - phase * b)
        max_err = max(max_err, err)
    return max_err


def cross_check_ptrace(psi: np.ndarray, n: int, keep: Sequence[int]) -> float:
    """Compare partial_trace_pure_batch (single time slice) against Qobj.ptrace."""
    rho_vec = partial_trace_pure_batch(psi.reshape(-1, 1), n, keep)[0]
    psi_q = qt.Qobj(psi.reshape(-1, 1), dims=[[2] * n, [1] * n])
    rho_qt = psi_q.ptrace(list(keep)).full()
    return float(np.max(np.abs(rho_vec - rho_qt)))
