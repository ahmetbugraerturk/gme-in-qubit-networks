"""Per-task production pipeline: one task = (n, graph_index, interaction, init).
Builds H, evolves exactly, computes pair (edge-orbit) and triple entanglement
data, writes a checkpointed .npz, per section 4/5/7 of the spec.

xx and ex are simulated directly. zz is DERIVED from the corresponding xx run
via the Hadamard symmetry of section 1 (applied locally and cheaply to the
reduced pair/triple density matrices, not by re-diagonalizing an n-qubit H):
    zz[init=plus](t) = Hn xx[init=0](t)   Hn        (same time grid as xx[init=0])
    zz[init=0]   (t) = Hn xx[init=plus](t) Hn
"""
from __future__ import annotations

import logging
import os
import shutil
import time as _time
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar

from . import graphs as G
from . import hamiltonians as H
from . import evolve as E
from . import measures as M
from . import gmn as GMN

logger = logging.getLogger("qnet.pipeline")

XX_T_MAX = np.pi / 2
EX_T_MAX = 100.0
EX_T_MAX_EXTENDED = 200.0
EX_FLAG_TOL = 1e-4

ZZ_SOURCE = {"plus": "0", "0": "plus"}  # zz[init] <- xx[ZZ_SOURCE[init]]


@dataclass
class TaskConfig:
    data_dir: str = "data/graphs"
    results_dir: str = "results"
    whole: bool = True  # section 5 optional --whole flag, default ON for n<=5
    gmn_n_uniform: int = 100
    gmn_n_refine: int = 20
    force: bool = False
    # "Always" GMN mode (new experiment): skip the Guhne-Seevinck witness
    # pass and the SDP subsampling entirely -- solve the exact SDP at every
    # single remaining (non-PPT, non-bound-zero) grid point. The PPT
    # pre-check and the negativity-sum upper-bound pre-filter (qnet/bound.py)
    # stay on regardless (both are exact/free, never approximate).
    always_gmn: bool = False

    def sampling_config(self) -> "GMN.SamplingConfig":
        return GMN.SamplingConfig(
            n_uniform=self.gmn_n_uniform, n_refine=self.gmn_n_refine,
            use_gs=not self.always_gmn, use_bound=True, sample=not self.always_gmn,
        )


def _time_grid_and_evolution(Gph: nx.Graph, interaction: str, init: str, t_max: float):
    n = Gph.number_of_nodes()
    Hmat = H.build_hamiltonian(Gph, interaction)
    psi0 = H.initial_state(n, init)
    evals, evecs = E.diagonalize(Hmat)
    times = E.make_time_grid(evals, t_max)
    psi_t = E.evolve_batch(evals, evecs, psi0.full().ravel(), times)
    return times, psi_t, evals, evecs, psi0.full().ravel()


def _refine_pair_max(evals, evecs, psi0, n, i, j, times, C):
    """Refine the grid maximum with scipy.optimize.minimize_scalar (bounded)
    on the bracket around the grid maximum."""
    idx = int(np.argmax(C))
    lo = times[max(idx - 1, 0)]
    hi = times[min(idx + 1, len(times) - 1)]
    if hi <= lo:
        return float(times[idx]), float(C[idx])

    def neg_c(t):
        psi = E.evolve_single(evals, evecs, psi0, t)
        rho = E.partial_trace_pure_batch(psi.reshape(-1, 1), n, (i, j))[0]
        return -M.concurrence_single(rho)

    res = minimize_scalar(neg_c, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-10})
    t_opt = float(res.x)
    c_opt = float(-res.fun)
    if c_opt < C[idx]:  # refinement should never do worse than the grid point
        return float(times[idx]), float(C[idx])
    return t_opt, c_opt


def _first_time_at_or_above(times: np.ndarray, C: np.ndarray, c_max: float) -> float:
    thresh = c_max - 1e-3 * c_max
    idx = np.argmax(C >= thresh)
    return float(times[idx])


def compute_pair_results(Gph: nx.Graph, psi_t: np.ndarray, times: np.ndarray, n: int,
                          evals, evecs, psi0) -> dict:
    orbits = G.edge_orbits(Gph)
    K = len(orbits)
    T = len(times)
    orbit_rep = np.zeros((K, 2), dtype=int)
    orbit_mult = np.zeros(K, dtype=int)
    orbit_C = np.zeros((K, T))
    orbit_Cmax = np.zeros(K)
    orbit_tfirst = np.zeros(K)
    orbit_targmax = np.zeros(K)
    for k, orbit in enumerate(orbits):
        i, j = orbit[0]
        rho_t = E.partial_trace_pure_batch(psi_t, n, (i, j))
        C = M.concurrence_batch(rho_t)
        c_max = float(np.max(C))
        t_first = _first_time_at_or_above(times, C, c_max)
        t_argmax, c_refined = _refine_pair_max(evals, evecs, psi0, n, i, j, times, C)
        orbit_rep[k] = (i, j)
        orbit_mult[k] = len(orbit)
        orbit_C[k] = C
        orbit_Cmax[k] = max(c_max, c_refined)
        orbit_tfirst[k] = t_first
        orbit_targmax[k] = t_argmax
    return dict(
        orbit_rep=orbit_rep, orbit_mult=orbit_mult, orbit_C=orbit_C,
        orbit_Cmax=orbit_Cmax, orbit_tfirst=orbit_tfirst, orbit_targmax=orbit_targmax,
        orbits=orbits,
    )


def derive_zz_pair_results(Gph: nx.Graph, psi_t_xx_source: np.ndarray, times: np.ndarray, n: int) -> dict:
    """Derive zz pair results from an xx run's full state, by Hadamard-
    transforming each orbit's 2-qubit reduced density matrix."""
    orbits = G.edge_orbits(Gph)
    K = len(orbits)
    T = len(times)
    orbit_rep = np.zeros((K, 2), dtype=int)
    orbit_mult = np.zeros(K, dtype=int)
    orbit_C = np.zeros((K, T))
    orbit_Cmax = np.zeros(K)
    orbit_tfirst = np.zeros(K)
    orbit_targmax = np.zeros(K)
    for k, orbit in enumerate(orbits):
        i, j = orbit[0]
        rho_xx_t = E.partial_trace_pure_batch(psi_t_xx_source, n, (i, j))
        rho_zz_t = E.hadamard_transform_batch(rho_xx_t, 2)
        C = M.concurrence_batch(rho_zz_t)
        c_max = float(np.max(C))
        t_first = _first_time_at_or_above(times, C, c_max)
        t_argmax = float(times[int(np.argmax(C))])  # grid-level; zz derived, no extra refine pass
        orbit_rep[k] = (i, j)
        orbit_mult[k] = len(orbit)
        orbit_C[k] = C
        orbit_Cmax[k] = c_max
        orbit_tfirst[k] = t_first
        orbit_targmax[k] = t_argmax
    return dict(
        orbit_rep=orbit_rep, orbit_mult=orbit_mult, orbit_C=orbit_C,
        orbit_Cmax=orbit_Cmax, orbit_tfirst=orbit_tfirst, orbit_targmax=orbit_targmax,
        orbits=orbits,
    )


def _analyze_triples(triples: list, rho_t_fn, times: np.ndarray, config: TaskConfig,
                      checkpoint_dir: Optional[str] = None) -> dict:
    """Shared core for compute_triple_results/derive_zz_triple_results: given
    the list of (i,j,k) triples and a function rho_t_fn(i,j,k) -> rho_t
    (T,8,8), run GMN.analyze_triple per triple and stack the results.

    If checkpoint_dir is given, each triple's full analysis is cached to its
    own file there as soon as it finishes (section 8: resumable runs) -- for
    config.always_gmn, a single triple's SDP sweep can itself take tens of
    minutes to hours, so per-task checkpointing alone is too coarse.
    """
    M_ = len(triples)
    T = len(times)
    sampling = config.sampling_config()

    labels = np.zeros((M_, T), dtype=np.int8)
    gs_max_margin = np.zeros((M_, T))
    gs_basis = np.empty((M_, T), dtype="<U3")
    gs_criterion = np.empty((M_, T), dtype="<U3")
    gs_comp_margin = np.zeros((M_, T))
    gmn_vals = np.full((M_, T), np.nan)
    solved_mask = np.zeros((M_, T), dtype=bool)
    proxy = np.zeros((M_, T))
    min_eigs_pt = np.zeros((M_, T, 3))
    bound_per_cut = np.zeros((M_, T, 3))
    solver_log_all = []

    for m, (i, j, k) in enumerate(triples):
        ckpt_path = os.path.join(checkpoint_dir, f"triple_{m}.npz") if checkpoint_dir else None
        if ckpt_path and os.path.exists(ckpt_path):
            analysis = GMN.load_triple_analysis(ckpt_path)
            logger.info("[ckpt] resumed triple %d (%d,%d,%d) from %s", m, i, j, k, ckpt_path)
        else:
            rho_t = rho_t_fn(i, j, k)
            analysis = GMN.analyze_triple(rho_t, times, sampling)
            if ckpt_path:
                os.makedirs(checkpoint_dir, exist_ok=True)
                GMN.save_triple_analysis(analysis, ckpt_path)
                logger.info("[ckpt] saved triple %d (%d,%d,%d) -> %s", m, i, j, k, ckpt_path)
        labels[m] = analysis.labels
        gs_max_margin[m] = analysis.gs_max_margin
        gs_basis[m] = analysis.gs_basis
        gs_criterion[m] = analysis.gs_criterion
        gs_comp_margin[m] = analysis.gs_computational_margin
        gmn_vals[m] = analysis.gmn
        solved_mask[m] = analysis.solved_mask
        proxy[m] = analysis.proxy
        min_eigs_pt[m] = analysis.min_eigs_pt
        bound_per_cut[m] = analysis.bound_per_cut
        solver_log_all.append((m, (i, j, k), analysis.solver_log))

    return dict(
        triples=np.array(triples, dtype=int) if triples else np.zeros((0, 3), dtype=int),
        labels=labels, gs_max_margin=gs_max_margin, gs_basis=gs_basis,
        gs_criterion=gs_criterion, gs_comp_margin=gs_comp_margin, gmn=gmn_vals,
        solved_mask=solved_mask, proxy=proxy, min_eigs_pt=min_eigs_pt,
        bound_per_cut=bound_per_cut, solver_log=solver_log_all,
    )


def compute_triple_results(Gph: nx.Graph, psi_t: np.ndarray, times: np.ndarray, n: int,
                            config: TaskConfig, checkpoint_dir: Optional[str] = None) -> dict:
    triples = G.induced_connected_triples(Gph)
    return _analyze_triples(
        triples, lambda i, j, k: E.partial_trace_pure_batch(psi_t, n, (i, j, k)), times, config,
        checkpoint_dir=checkpoint_dir)


def derive_zz_triple_results(Gph: nx.Graph, psi_t_xx_source: np.ndarray, times: np.ndarray, n: int,
                              config: TaskConfig, checkpoint_dir: Optional[str] = None) -> dict:
    """Same as compute_triple_results, but the 3-qubit reduced states are
    Hadamard-transformed (xx -> zz) before running PPT/bound/GS/GMN."""
    triples = G.induced_connected_triples(Gph)

    def rho_t_fn(i, j, k):
        rho_xx_t = E.partial_trace_pure_batch(psi_t_xx_source, n, (i, j, k))
        return E.hadamard_transform_batch(rho_xx_t, 3)

    return _analyze_triples(triples, rho_t_fn, times, config, checkpoint_dir=checkpoint_dir)


def compute_whole_results(psi_t: np.ndarray, n: int) -> dict:
    c_gme = M.c_gme_pure_batch(psi_t, n)
    q_mw = M.meyer_wallach_Q_batch(psi_t, n)
    return dict(c_gme=c_gme, meyer_wallach_Q=q_mw)


def result_path(config: TaskConfig, n: int, idx: int, interaction: str, init: str) -> str:
    d = os.path.join(config.results_dir, str(n))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"g{idx}_{interaction}_{init}.npz")


def _save_npz(path: str, meta: dict, pair: dict, triple: dict, whole: Optional[dict], times: np.ndarray):
    kwargs = dict(
        times=times,
        meta_n=meta["n"], meta_idx=meta["idx"], meta_graph6=meta["graph6"],
        meta_interaction=meta["interaction"], meta_init=meta["init"],
        meta_dt=meta["dt"], meta_t_max=meta["t_max"], meta_delta_e_max=meta["delta_e_max"],
        meta_w0=1.0, meta_runtime_s=meta["runtime_s"],
        orbit_rep=pair["orbit_rep"], orbit_mult=pair["orbit_mult"], orbit_C=pair["orbit_C"],
        orbit_Cmax=pair["orbit_Cmax"], orbit_tfirst=pair["orbit_tfirst"], orbit_targmax=pair["orbit_targmax"],
        triples=triple["triples"], triple_labels=triple["labels"],
        triple_gs_max_margin=triple["gs_max_margin"], triple_gs_basis=triple["gs_basis"],
        triple_gs_criterion=triple["gs_criterion"], triple_gs_comp_margin=triple["gs_comp_margin"],
        triple_gmn=triple["gmn"], triple_solved_mask=triple["solved_mask"],
        triple_proxy=triple["proxy"], triple_min_eigs_pt=triple["min_eigs_pt"],
        triple_bound_per_cut=triple["bound_per_cut"],
    )
    if "ex_flag" in meta:
        kwargs["ex_flagged"] = meta["ex_flag"]["flagged"]
        kwargs["ex_window2_max"] = meta["ex_flag"]["window2_max"]
    if "n3_gmn_cgme_mismatches" in meta:
        kwargs["n3_gmn_cgme_mismatches"] = meta["n3_gmn_cgme_mismatches"]
    if whole is not None:
        kwargs["whole_c_gme"] = whole["c_gme"]
        kwargs["whole_meyer_wallach_Q"] = whole["meyer_wallach_Q"]
    np.savez_compressed(path, **kwargs)


def check_ex_convergence(Gph: nx.Graph, n: int, init: str) -> dict:
    """Section 3: evaluate ex also on [0, 2T]; flag edges where max increases
    by more than 1e-4 relative to [0, T]."""
    times1, psi1, evals, evecs, psi0 = _time_grid_and_evolution(Gph, "ex", init, EX_T_MAX)
    times2, psi2, _, _, _ = _time_grid_and_evolution(Gph, "ex", init, EX_T_MAX_EXTENDED)
    orbits = G.edge_orbits(Gph)
    flagged = np.zeros(len(orbits), dtype=bool)
    window2_max = np.zeros(len(orbits))
    for k, orbit in enumerate(orbits):
        i, j = orbit[0]
        c1 = np.max(M.concurrence_batch(E.partial_trace_pure_batch(psi1, n, (i, j))))
        c2 = np.max(M.concurrence_batch(E.partial_trace_pure_batch(psi2, n, (i, j))))
        window2_max[k] = c2
        flagged[k] = (c2 - c1) > EX_FLAG_TOL
    return {"flagged": flagged, "window2_max": window2_max}


def run_task(n: int, idx: int, interaction: str, init: str, config: TaskConfig) -> str:
    """Run (or skip, if checkpointed) one production task. Returns the output path."""
    path = result_path(config, n, idx, interaction, init)
    if os.path.exists(path) and not config.force:
        logger.info("[skip] %s already exists", path)
        return path

    t_start = _time.time()
    graphs_list = G.load_graphs(n, data_dir=config.data_dir)
    Gph = graphs_list[idx]
    graph6 = nx.to_graph6_bytes(Gph, header=False).decode("ascii").strip()

    # Per-triple checkpointing (always_gmn only): a single triple's full SDP
    # sweep can itself take tens of minutes to hours there, so resumability
    # needs to be finer than one file per task. Cleaned up once the task's
    # final .npz is written successfully.
    checkpoint_dir = (os.path.join(config.results_dir, str(n), f".ckpt_g{idx}_{interaction}_{init}")
                       if config.always_gmn else None)

    if interaction in ("xx", "ex"):
        t_max = XX_T_MAX if interaction == "xx" else EX_T_MAX
        times, psi_t, evals, evecs, psi0 = _time_grid_and_evolution(Gph, interaction, init, t_max)
        pair = compute_pair_results(Gph, psi_t, times, n, evals, evecs, psi0)
        triple = compute_triple_results(Gph, psi_t, times, n, config, checkpoint_dir=checkpoint_dir)
        # n=2's only bipartition is the pair itself (redundant with C_ij), so
        # whole-system C_GME/Meyer-Wallach is only meaningful for n>=3.
        whole = compute_whole_results(psi_t, n) if (config.whole and n >= 3) else None
        dE = float(evals.max() - evals.min())
        meta = dict(n=n, idx=idx, graph6=graph6, interaction=interaction, init=init,
                    dt=float(times[1] - times[0]) if len(times) > 1 else 0.0,
                    t_max=t_max, delta_e_max=dE, runtime_s=_time.time() - t_start)
        if interaction == "ex":
            meta["ex_flag"] = check_ex_convergence(Gph, n, init)
    elif interaction == "zz":
        source_init = ZZ_SOURCE[init]
        times, psi_t, evals, evecs, psi0 = _time_grid_and_evolution(Gph, "xx", source_init, XX_T_MAX)
        pair = derive_zz_pair_results(Gph, psi_t, times, n)
        triple = derive_zz_triple_results(Gph, psi_t, times, n, config, checkpoint_dir=checkpoint_dir)
        if config.whole and n >= 3:
            Hn = H.hadamard_all(n).full()
            psi_zz_t = Hn @ psi_t
            whole = compute_whole_results(psi_zz_t, n)
        else:
            whole = None
        dE = float(evals.max() - evals.min())
        meta = dict(n=n, idx=idx, graph6=graph6, interaction=interaction, init=init,
                    dt=float(times[1] - times[0]) if len(times) > 1 else 0.0,
                    t_max=XX_T_MAX, delta_e_max=dE, runtime_s=_time.time() - t_start)
    else:
        raise ValueError(f"unknown interaction {interaction!r}")

    # Section 5c: for n=3 the single triple IS the whole pure state, so GMN>0
    # must agree with C_GME>0 wherever GMN was actually solved. Check it here
    # (not just in the unit tests) and record any disagreement for REPORT.md.
    n3_mismatches = None
    if n == 3 and whole is not None and triple["triples"].shape[0] == 1:
        solved = triple["solved_mask"][0]
        if np.any(solved):
            gmn_pos = triple["gmn"][0][solved] > GMN.GMN_TOL
            cgme_pos = whole["c_gme"][solved] > 1e-6
            mismatch = gmn_pos != cgme_pos
            n3_mismatches = int(mismatch.sum())
            if n3_mismatches:
                logger.warning("n=3 GMN<=>C_GME mismatch at %d/%d solved times for %s",
                                n3_mismatches, int(solved.sum()), path)

    meta["runtime_s"] = _time.time() - t_start
    if n3_mismatches is not None:
        meta["n3_gmn_cgme_mismatches"] = n3_mismatches
    _save_npz(path, meta, pair, triple, whole, times)
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    logger.info("[done] %s (%.2fs)", path, meta["runtime_s"])
    return path


def all_tasks(n_values):
    """Enumerate all (n, idx, interaction, init) tasks for the given n values."""
    tasks = []
    for n in n_values:
        n_graphs = len(G.load_graphs(n))
        for idx in range(n_graphs):
            for interaction in ("xx", "zz", "ex"):
                for init in ("0", "plus"):
                    tasks.append((n, idx, interaction, init))
    return tasks
