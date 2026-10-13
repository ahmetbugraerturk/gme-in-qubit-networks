"""Genuine multipartite negativity (GMN) for 3-qubit states, via the SDP of
Jungnitsch, Moroder & Guhne, PRL 106, 190502 (2010), arXiv:1010.6049.

  Variables (Hermitian, 8x8): P_M, Q_M for each single-party cut M in {0,1,2}.
  Shared W with  W = P_M + (Q_M)^{T_M}  for every M.
  Constraints:  0 <= P_M <= I,  0 <= Q_M <= I.
  Minimize Re Tr(W rho).  GMN(rho) = max(0, -minimum).

Also implements the full per-triple pipeline: PPT pre-check, Guhne-Seevinck
scan, and SDP only on a configurable sample of "undecided" time points
(uniform subsample + refinement around local maxima of a cheap negativity
proxy), per the cost-control policy in the spec.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import cvxpy as cp
import numpy as np

from . import bound as BOUND
from . import gs, measures

logger = logging.getLogger("qnet.gmn")

DIMS3 = (2, 2, 2)
GMN_TOL = 1e-6  # threshold above which we call GMN "nonzero"
PPT_TOL = 1e-10
GS_TOL = 1e-9

# Per-timestep triple classification codes (stored in results/*.npz and
# summarized in summary_triples.csv):
LABEL_PPT = 0           # some bipartition is PPT -> GMN=0, exact, free
LABEL_GS = 1            # Guhne-Seevinck witness certified GME in some local basis
LABEL_GMN_POS = 2       # SDP solved: GMN > GMN_TOL
LABEL_GMN_ZERO = 3      # SDP solved: GMN <= GMN_TOL
LABEL_UNDECIDED = 4     # PPT/GS/bound inconclusive, and skipped by the SDP sampling budget
LABEL_BOUND_ZERO = 5    # negativity-sum upper bound (Jungnitsch-Moroder-Guhne) < GMN_TOL -> GMN=0, exact, free


class _GmnProblem:
    """Builds the GMN SDP ONCE as a DPP-parameterized cvxpy Problem (rho is a
    cp.Parameter), so repeated solves over many time points only re-run the
    numerical solve, not cvxpy's Python-side canonicalization/compilation.
    Rebuilding the Problem from scratch per call was measured at ~0.4-0.5s
    (dominated by CLARABEL's cold solve on a freshly compiled problem); the
    cached+parameterized version with SCS solves in ~0.01s, a ~40x speedup,
    which matters because a single n=5 triple can need O(100) SDP solves.
    """

    def __init__(self):
        d = 8
        self.rho_param = cp.Parameter((d, d), hermitian=True)
        self.W = cp.Variable((d, d), hermitian=True)
        constraints = []
        for m in range(3):
            P = cp.Variable((d, d), hermitian=True)
            Q = cp.Variable((d, d), hermitian=True)
            constraints += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]
            QT = cp.partial_transpose(Q, dims=DIMS3, axis=m)
            constraints.append(self.W == P + QT)
        objective = cp.Minimize(cp.real(cp.trace(self.W @ self.rho_param)))
        self.prob = cp.Problem(objective, constraints)
        assert self.prob.is_dcp(dpp=True), "GMN SDP is not DPP; parameter caching would not help"

    def solve(self, rho: np.ndarray, solver_order: Tuple[str, ...]) -> dict:
        rho_h = 0.5 * (rho + rho.conj().T)
        self.rho_param.value = rho_h
        solver_kwargs = {
            "SCS": dict(solver=cp.SCS, eps=1e-7, max_iters=50000),
            "CLARABEL": dict(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9),
        }
        last_err = None
        for name in solver_order:
            kwargs = solver_kwargs.get(name, dict(solver=name))
            try:
                val = self.prob.solve(**kwargs)
                if self.prob.status in ("optimal", "optimal_inaccurate") and val is not None:
                    gmn = max(0.0, -float(val))
                    return {"gmn": gmn, "status": self.prob.status, "solver": name, "raw_value": float(val)}
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        return {"gmn": float("nan"), "status": "failed", "solver": None, "raw_value": None, "error": str(last_err)}


_problem_cache: dict = {}


def _get_problem() -> _GmnProblem:
    # Per-process cache (safe under multiprocessing: each worker process gets
    # its own fresh module globals, so no cross-process sharing is needed).
    if "default" not in _problem_cache:
        _problem_cache["default"] = _GmnProblem()
    return _problem_cache["default"]


def solve_gmn(rho: np.ndarray, solver_order: Tuple[str, ...] = ("SCS", "CLARABEL")) -> dict:
    """Solve the GMN SDP for one 3-qubit density matrix. SCS first (fast,
    eps=1e-7, well below our GMN_TOL=1e-6 decision threshold); falls back to
    CLARABEL (slower, tighter) only if SCS fails to reach optimal/inaccurate.
    """
    return _get_problem().solve(rho, solver_order)


def select_sdp_times(undecided_idx: np.ndarray, proxy: np.ndarray, n_uniform: int = 100,
                      n_refine: int = 20) -> np.ndarray:
    """Sampling policy: ~n_uniform evenly spaced undecided points, plus up to
    n_refine local maxima of `proxy` (restricted to undecided points).
    """
    if len(undecided_idx) == 0:
        return np.array([], dtype=int)
    if len(undecided_idx) <= n_uniform:
        uniform_sel = undecided_idx
    else:
        pos = np.unique(np.round(np.linspace(0, len(undecided_idx) - 1, n_uniform)).astype(int))
        uniform_sel = undecided_idx[pos]

    sub_proxy = proxy[undecided_idx]
    local_max_mask = np.zeros(len(undecided_idx), dtype=bool)
    if len(undecided_idx) >= 3:
        local_max_mask[1:-1] = (sub_proxy[1:-1] > sub_proxy[:-2]) & (sub_proxy[1:-1] > sub_proxy[2:])
    local_max_idx = undecided_idx[local_max_mask]
    if len(local_max_idx) > n_refine:
        vals = proxy[local_max_idx]
        top = np.argsort(vals)[::-1][:n_refine]
        local_max_idx = local_max_idx[top]

    return np.unique(np.concatenate([uniform_sel, local_max_idx]))


@dataclass
class SamplingConfig:
    n_uniform: int = 100
    n_refine: int = 20
    use_gs: bool = True   # Guhne-Seevinck witness pass (sufficient-only, "proves positive")
    use_bound: bool = True  # negativity-sum upper bound pre-filter (exact, "proves zero")
    sample: bool = True   # if False, solve the SDP at EVERY remaining point (no subsampling)


@dataclass
class TripleAnalysis:
    labels: np.ndarray  # (T,) int, codes 0..5 (see LABEL_* constants above)
    gs_max_margin: np.ndarray  # (T,); NaN-filled if use_gs=False
    gs_basis: np.ndarray  # (T,) str
    gs_criterion: np.ndarray  # (T,) str
    gs_computational_margin: np.ndarray  # (T,)
    gmn: np.ndarray  # (T,) float, nan where not solved
    solved_mask: np.ndarray  # (T,) bool
    proxy: np.ndarray  # (T,) == the negativity-sum upper bound min_M N_M(rho)
    min_eigs_pt: np.ndarray  # (T,3) min eig of partial transpose per cut
    bound_per_cut: np.ndarray  # (T,3) per-cut negativity sum N_M(rho), M in {0},{1},{2}
    solver_log: List[tuple] = field(default_factory=list)


def analyze_triple(rho_t: np.ndarray, times: np.ndarray, config: SamplingConfig = SamplingConfig()) -> TripleAnalysis:
    """Full per-triple pipeline over a time grid. rho_t: (T, 8, 8).

    Cascade: PPT (exact, free) -> negativity-sum upper bound (exact, free;
    skippable via config.use_bound) -> Guhne-Seevinck witness (sufficient
    only, "proves positive"; skippable via config.use_gs) -> SDP, either on
    a cost-controlled sample of the remaining points (config.sample=True,
    default) or on every single remaining point (config.sample=False).
    """
    T = rho_t.shape[0]
    min_eigs = np.stack(
        [measures.min_eig_partial_transpose_batch(rho_t, DIMS3, [p]) for p in range(3)], axis=1
    )  # (T,3)
    ppt_mask = (min_eigs >= -PPT_TOL).any(axis=1)

    labels = np.full(T, LABEL_UNDECIDED, dtype=int)
    labels[ppt_mask] = LABEL_PPT
    remaining = ~ppt_mask

    # Exact upper bound GMN(rho) <= min_M N_M(rho) (Jungnitsch-Moroder-Guhne):
    # below GMN_TOL means GMN is PROVABLY below GMN_TOL too -- no SDP needed.
    # This is also exactly the negativity quantity previously used only as an
    # SDP-sampling refinement proxy; it now does double duty as that proxy.
    bound_u, per_cut = BOUND.gmn_upper_bound_batch(rho_t, n_sub=3)
    bound_per_cut = np.stack([per_cut[c] for c in BOUND.all_cuts(3)], axis=1)
    proxy = bound_u
    if config.use_bound:
        bound_cert = remaining & (bound_u < GMN_TOL)
        labels[bound_cert] = LABEL_BOUND_ZERO
        remaining = remaining & (~bound_cert)

    if config.use_gs:
        gs_result = gs.scan_all_bases_batch(rho_t)
        gs_certified = gs_result["certified"] & remaining
        labels[gs_certified] = LABEL_GS
        remaining = remaining & (~gs_certified)
    else:
        gs_result = {
            "max_margin": np.full(T, np.nan), "basis": np.array([""] * T, dtype=object),
            "criterion": np.array([""] * T, dtype=object), "computational_margin": np.full(T, np.nan),
        }

    undecided_idx = np.where(remaining)[0]

    if config.sample:
        solve_idx = select_sdp_times(undecided_idx, proxy, config.n_uniform, config.n_refine)
    else:
        solve_idx = undecided_idx  # solve every remaining point, no subsampling

    gmn_vals = np.full(T, np.nan)
    solved_mask = np.zeros(T, dtype=bool)
    solver_log: List[tuple] = []
    for idx in solve_idx:
        res = solve_gmn(rho_t[int(idx)])
        gmn_vals[idx] = res["gmn"]
        solved_mask[idx] = True
        solver_log.append((int(idx), float(times[idx]), res["solver"], res["status"], res["gmn"]))
        if np.isnan(res["gmn"]):
            continue
        labels[idx] = LABEL_GMN_POS if res["gmn"] > GMN_TOL else LABEL_GMN_ZERO

    return TripleAnalysis(
        labels=labels,
        gs_max_margin=gs_result["max_margin"],
        gs_basis=gs_result["basis"],
        gs_criterion=gs_result["criterion"],
        gs_computational_margin=gs_result["computational_margin"],
        gmn=gmn_vals,
        solved_mask=solved_mask,
        proxy=proxy,
        min_eigs_pt=min_eigs,
        bound_per_cut=bound_per_cut,
        solver_log=solver_log,
    )


# ---------------------------------------------------------------------------
# Per-triple checkpointing (section 8: "checkpoint per task and make runs
# resumable" -- for the "always" GMN mode a single triple's full SDP sweep
# can itself take tens of minutes to hours, so checkpointing needs to be
# finer than one file per (graph, interaction, init) task).
# ---------------------------------------------------------------------------

def save_triple_analysis(analysis: TripleAnalysis, path: str) -> None:
    # np.savez_compressed silently appends ".npz" if the name doesn't already
    # end with it, so the tmp name must already carry that suffix or the
    # subsequent os.replace(tmp_path, path) can't find what was actually written.
    assert path.endswith(".npz")
    tmp_path = path[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        labels=analysis.labels, gs_max_margin=analysis.gs_max_margin,
        gs_basis=analysis.gs_basis, gs_criterion=analysis.gs_criterion,
        gs_computational_margin=analysis.gs_computational_margin,
        gmn=analysis.gmn, solved_mask=analysis.solved_mask, proxy=analysis.proxy,
        min_eigs_pt=analysis.min_eigs_pt, bound_per_cut=analysis.bound_per_cut,
    )
    os.replace(tmp_path, path)  # atomic: a reader never sees a half-written file


def load_triple_analysis(path: str) -> TripleAnalysis:
    d = np.load(path, allow_pickle=True)
    return TripleAnalysis(
        labels=d["labels"], gs_max_margin=d["gs_max_margin"], gs_basis=d["gs_basis"],
        gs_criterion=d["gs_criterion"], gs_computational_margin=d["gs_computational_margin"],
        gmn=d["gmn"], solved_mask=d["solved_mask"], proxy=d["proxy"],
        min_eigs_pt=d["min_eigs_pt"], bound_per_cut=d["bound_per_cut"], solver_log=[],
    )
