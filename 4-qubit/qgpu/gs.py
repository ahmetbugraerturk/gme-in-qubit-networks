"""Guhne-Seevinck genuine-multipartite-entanglement criteria for three qubits
(arXiv:0905.1349): GHZ-type (Observation 2-ish, antipodal-pair inequality)
and W-type (Observation 3, Hamming-neighbourhood inequality), evaluated in
all 27 local bases U_a (x) U_b (x) U_c with U in {I, H, H S^dagger}
(computational/Z, X, Y eigenbases).

Index convention: product-basis state b in {0,1}^3 <-> integer 0..7 via
b = 4*b0 + 2*b1 + b2 (qubit 0 most significant), matching qt.tensor order.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

I2 = np.eye(2, dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SDAG = np.array([[1, 0], [0, -1j]], dtype=complex)
HY = HAD @ SDAG  # rotates to the Y eigenbasis

LOCAL_UNITARIES: Dict[str, np.ndarray] = {"Z": I2, "X": HAD, "Y": HY}
BASIS_NAMES = list(LOCAL_UNITARIES.keys())

ANTIPODAL_PAIRS = [(a, 7 - a) for a in range(4)]  # (0,7),(1,6),(2,5),(3,4)


def _neighbours(v: int):
    n1 = [w for w in range(8) if bin(v ^ w).count("1") == 1]
    n2 = [w for w in range(8) if bin(v ^ w).count("1") == 2]
    return n1, n2


N1_N2 = {v: _neighbours(v) for v in range(8)}


def ghz_margins(rho: np.ndarray) -> np.ndarray:
    """4 margins (one per antipodal pair); margin>1e-9 anywhere => GME."""
    diag = np.real(np.diag(rho))
    margins = np.zeros(4)
    for idx, (a, abar) in enumerate(ANTIPODAL_PAIRS):
        lhs = abs(rho[a, abar])
        rhs = 0.0
        for jdx, (c, cbar) in enumerate(ANTIPODAL_PAIRS):
            if jdx == idx:
                continue
            rhs += np.sqrt(max(diag[c], 0.0) * max(diag[cbar], 0.0))
        margins[idx] = lhs - rhs
    return margins


def w_margins(rho: np.ndarray) -> np.ndarray:
    """8 margins (one per reference state v)."""
    diag = np.real(np.diag(rho))
    margins = np.zeros(8)
    for v in range(8):
        n1, n2 = N1_N2[v]
        lhs = 0.0
        for a in range(3):
            for b in range(a + 1, 3):
                k, l = n1[a], n1[b]
                lhs += abs(rho[k, l])
        rhs = sum(np.sqrt(max(diag[v], 0.0) * max(diag[t], 0.0)) for t in n2)
        rhs += 0.5 * sum(diag[k] for k in n1)
        margins[v] = lhs - rhs
    return margins


def scan_all_bases(rho: np.ndarray) -> dict:
    """Scan all 27 local bases; return the certifying max margin and where it
    was achieved, plus the computational-basis-only (Z,Z,Z) result."""
    best_margin = -np.inf
    best_basis = None
    best_criterion = None
    best_detail = None
    comp_margin = None
    comp_criterion = None
    for ua in BASIS_NAMES:
        for ub in BASIS_NAMES:
            for uc in BASIS_NAMES:
                U = np.kron(np.kron(LOCAL_UNITARIES[ua], LOCAL_UNITARIES[ub]), LOCAL_UNITARIES[uc])
                rho_t = U @ rho @ U.conj().T
                gm = ghz_margins(rho_t)
                wm = w_margins(rho_t)
                g_i, g_v = int(np.argmax(gm)), float(np.max(gm))
                w_i, w_v = int(np.argmax(wm)), float(np.max(wm))
                if w_v > g_v:
                    local_margin, local_crit, local_detail = w_v, "W", w_i
                else:
                    local_margin, local_crit, local_detail = g_v, "GHZ", g_i
                if local_margin > best_margin:
                    best_margin = local_margin
                    best_basis = (ua, ub, uc)
                    best_criterion = local_crit
                    best_detail = local_detail
                if (ua, ub, uc) == ("Z", "Z", "Z"):
                    comp_margin, comp_criterion = local_margin, local_crit
    return {
        "max_margin": best_margin,
        "basis": best_basis,
        "criterion": best_criterion,
        "detail": best_detail,
        "certified": best_margin > 1e-9,
        "computational_margin": comp_margin,
        "computational_criterion": comp_criterion,
        "computational_certified": comp_margin > 1e-9,
    }


# ---------------------------------------------------------------------------
# Vectorized (batched over time) versions for production scanning.
# ---------------------------------------------------------------------------

def _transform_batch(rho_t: np.ndarray, U: np.ndarray) -> np.ndarray:
    return np.einsum("ab,tbc,cd->tad", U, rho_t, U.conj().T)


def ghz_margins_batch(rho_t: np.ndarray) -> np.ndarray:
    T = rho_t.shape[0]
    diag = np.real(np.diagonal(rho_t, axis1=1, axis2=2))  # (T,8)
    margins = np.zeros((T, 4))
    for idx, (a, abar) in enumerate(ANTIPODAL_PAIRS):
        lhs = np.abs(rho_t[:, a, abar])
        rhs = np.zeros(T)
        for jdx, (c, cbar) in enumerate(ANTIPODAL_PAIRS):
            if jdx == idx:
                continue
            rhs += np.sqrt(np.clip(diag[:, c], 0, None) * np.clip(diag[:, cbar], 0, None))
        margins[:, idx] = lhs - rhs
    return margins


def w_margins_batch(rho_t: np.ndarray) -> np.ndarray:
    T = rho_t.shape[0]
    diag = np.real(np.diagonal(rho_t, axis1=1, axis2=2))
    margins = np.zeros((T, 8))
    for v in range(8):
        n1, n2 = N1_N2[v]
        lhs = np.zeros(T)
        for a in range(3):
            for b in range(a + 1, 3):
                k, l = n1[a], n1[b]
                lhs += np.abs(rho_t[:, k, l])
        rhs = np.zeros(T)
        for t_ in n2:
            rhs += np.sqrt(np.clip(diag[:, v], 0, None) * np.clip(diag[:, t_], 0, None))
        rhs += 0.5 * sum(diag[:, k] for k in n1)
        margins[:, v] = lhs - rhs
    return margins


def scan_all_bases_batch(rho_t: np.ndarray) -> dict:
    """Batched version of scan_all_bases over a time axis. rho_t: (T,8,8).
    Returns dict of arrays: max_margin(T,), basis(T,) [string], criterion(T,)
    [string], detail(T,) [int], computational_margin(T,), computational_criterion(T,).
    """
    T = rho_t.shape[0]
    best_margin = np.full(T, -np.inf)
    best_basis = np.array([""] * T, dtype=object)
    best_criterion = np.array([""] * T, dtype=object)
    best_detail = np.zeros(T, dtype=int)
    comp_margin = np.zeros(T)
    comp_criterion = np.array([""] * T, dtype=object)

    for ua in BASIS_NAMES:
        for ub in BASIS_NAMES:
            for uc in BASIS_NAMES:
                U = np.kron(np.kron(LOCAL_UNITARIES[ua], LOCAL_UNITARIES[ub]), LOCAL_UNITARIES[uc])
                rho_tb = _transform_batch(rho_t, U)
                gm = ghz_margins_batch(rho_tb)
                wm = w_margins_batch(rho_tb)
                g_idx = np.argmax(gm, axis=1)
                g_val = gm[np.arange(T), g_idx]
                w_idx = np.argmax(wm, axis=1)
                w_val = wm[np.arange(T), w_idx]
                take_w = w_val > g_val
                local_margin = np.where(take_w, w_val, g_val)
                local_detail = np.where(take_w, w_idx, g_idx)
                local_crit = np.where(take_w, "W", "GHZ")

                improve = local_margin > best_margin
                best_margin = np.where(improve, local_margin, best_margin)
                best_detail = np.where(improve, local_detail, best_detail)
                basis_str = f"{ua}{ub}{uc}"
                best_basis = np.where(improve, basis_str, best_basis)
                best_criterion = np.where(improve, local_crit, best_criterion)

                if (ua, ub, uc) == ("Z", "Z", "Z"):
                    comp_margin = local_margin
                    comp_criterion = local_crit

    return {
        "max_margin": best_margin,
        "basis": best_basis,
        "criterion": best_criterion,
        "detail": best_detail,
        "certified": best_margin > 1e-9,
        "computational_margin": comp_margin,
        "computational_criterion": comp_criterion,
        "computational_certified": comp_margin > 1e-9,
    }
