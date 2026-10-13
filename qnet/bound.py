"""Negativity-sum upper bound on GMN (Jungnitsch, Moroder & Guhne, PRL 106,
190502 (2010), arXiv:1010.6049):

    GMN(rho) <= min_M N_M(rho),   N_M(rho) := sum of |negative eigenvalues| of rho^{T_M}

for the SAME SDP (variables P_M, Q_M with W = P_M + Q_M^{T_M} for every cut
M, 0<=P_M<=I, 0<=Q_M<=I; GMN = max(0, -min Tr(W rho))) implemented in gmn.py.
This is an EXACT, rigorous upper bound -- not a heuristic -- so whenever
min_M N_M(rho) falls below the GMN decision threshold, GMN(rho) is PROVABLY
below that threshold too, and the SDP does not need to be solved at all.

Generic in the number of qubits in the subsystem (n_sub) and which cuts are
considered (all bipartitions of size 1..n_sub//2), so the same code serves
3-qubit triples now and larger k-partite subsets later (the --whole C_GME
machinery already walks the analogous bipartition list for the full system;
this module is the per-cut negativity primitive that GMN's own SDP sampling
proxy was already informally computing).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_EIG_TOL = 1e-12


def partial_transpose_batch(rho_batch: np.ndarray, dims: Sequence[int], cut: Sequence[int]) -> np.ndarray:
    """Partial transpose over the subsystems listed in `cut`, batched over a
    leading time axis. rho_batch: (T, d, d) with d = prod(dims)."""
    T = rho_batch.shape[0]
    k = len(dims)
    shape = (T,) + tuple(dims) + tuple(dims)
    Tt = rho_batch.reshape(shape)
    perm = [0] + list(range(1, 2 * k + 1))
    for s in cut:
        perm[1 + s], perm[1 + k + s] = perm[1 + k + s], perm[1 + s]
    Tt2 = np.transpose(Tt, perm)
    return np.ascontiguousarray(Tt2).reshape(rho_batch.shape)


def negativity_sum_batch(rho_batch: np.ndarray, dims: Sequence[int], cut: Sequence[int],
                          tol: float = DEFAULT_EIG_TOL) -> np.ndarray:
    """N_M(t) = sum of |negative eigenvalues| of rho(t)^{T_M}, batched via
    numpy's natively-batched eigvalsh. Eigenvalues > -tol are treated as
    exactly zero (numerical noise floor), matching the GMN_TOL-scale
    bookkeeping used elsewhere in the pipeline.
    """
    pt = partial_transpose_batch(rho_batch, dims, cut)
    pt = 0.5 * (pt + np.swapaxes(pt.conj(), 1, 2))  # Hermitize: (X+X^dagger)/2
    eigs = np.linalg.eigvalsh(pt)  # (T, d), ascending, batched over T
    neg = np.where(eigs < -tol, -eigs, 0.0)
    return neg.sum(axis=1)


def all_cuts(n_sub: int) -> List[Tuple[int, ...]]:
    """All bipartition cuts M (as the smaller-or-equal side's qubit indices)
    with 1 <= |M| <= n_sub//2. For n_sub=3 this is the 3 single-qubit cuts
    used by the GMN SDP; for larger n_sub it also includes 2-qubit-vs-rest
    cuts etc., for future genuine-multipartite-entanglement generalizations
    beyond triples.
    """
    nodes = list(range(n_sub))
    cuts: List[Tuple[int, ...]] = []
    for size in range(1, n_sub // 2 + 1):
        cuts.extend(itertools.combinations(nodes, size))
    return cuts


def gmn_upper_bound_batch(rho_batch: np.ndarray, n_sub: int,
                           tol: float = DEFAULT_EIG_TOL) -> Tuple[np.ndarray, Dict[Tuple[int, ...], np.ndarray]]:
    """U(t) = min over all cuts M of N_M(rho(t)), an exact upper bound on
    GMN(rho(t)). Returns (U, per_cut) where per_cut maps each cut -> its
    N_M(t) array (kept for diagnostics/plots, per spec).
    """
    dims = [2] * n_sub
    per_cut: Dict[Tuple[int, ...], np.ndarray] = {}
    T = rho_batch.shape[0]
    best = np.full(T, np.inf)
    for cut in all_cuts(n_sub):
        n_m = negativity_sum_batch(rho_batch, dims, cut, tol=tol)
        per_cut[cut] = n_m
        best = np.minimum(best, n_m)
    return best, per_cut
