#!/usr/bin/env python3
"""Benchmark: QuTiP-native (sesolve + ptrace + concurrence, looped) vs the
vectorized numpy pipeline (qnet.evolve + qnet.measures), for n=2 and n=3.
Compares wall time AND numerical agreement (not just speed -- a faster
implementation that's wrong isn't a win).

Deliberately excludes the GMN SDP stage: that part (cvxpy/SCS) is identical
regardless of how rho(t) was obtained, so including it would just dilute the
one thing this benchmark is meant to isolate -- the cost of evolution +
partial trace + concurrence done the QuTiP-native way vs the vectorized way.
"""
import time

import numpy as np

from qnet import graphs as G
from qnet import hamiltonians as H
from qnet import evolve as E
from qnet import measures as M
from qnet_qutip import simulate as QS


def bench_one(n, graph_idx, interaction, init, t_max):
    Gph = G.load_graphs(n)[graph_idx]
    edges = [tuple(sorted(e)) for e in Gph.edges()]
    triples = G.induced_connected_triples(Gph) if n >= 3 else []

    Hmat = H.build_hamiltonian(Gph, interaction)
    evals, evecs = E.diagonalize(Hmat)
    times = E.make_time_grid(evals, t_max)

    # --- vectorized numpy pipeline ---
    t0 = time.time()
    psi0 = H.initial_state(n, init)
    psi_t = E.evolve_batch(evals, evecs, psi0.full().ravel(), times)
    edge_C_numpy = {}
    for e in edges:
        rho_t = E.partial_trace_pure_batch(psi_t, n, e)
        edge_C_numpy[e] = M.concurrence_batch(rho_t)
    triple_rho_numpy = {trip: E.partial_trace_pure_batch(psi_t, n, trip) for trip in triples}
    t_numpy = time.time() - t0

    # --- QuTiP-native pipeline ---
    t0 = time.time()
    states = QS.evolve_qutip(Hmat, psi0, times)
    edge_C_qutip = QS.pair_concurrences_qutip(states, edges)
    triple_rho_qutip = QS.triple_reduced_states_qutip(states, triples) if triples else {}
    t_qutip = time.time() - t0

    # --- agreement check ---
    max_c_err = 0.0
    for e in edges:
        max_c_err = max(max_c_err, float(np.max(np.abs(edge_C_numpy[e] - edge_C_qutip[e]))))

    max_rho_err = 0.0
    for trip in triples:
        rho_np = triple_rho_numpy[trip]  # (T, 8, 8)
        for ti, rho_qt in enumerate(triple_rho_qutip[trip]):
            max_rho_err = max(max_rho_err, float(np.max(np.abs(rho_np[ti] - rho_qt.full()))))

    graph6 = Gph.__dict__.get("name", "")
    print(f"n={n} g{graph_idx} {interaction}|{init}>  T={len(times)}  "
          f"edges={len(edges)} triples={len(triples)}")
    print(f"  numpy pipeline : {t_numpy:8.4f} s")
    print(f"  qutip pipeline : {t_qutip:8.4f} s   ({t_qutip / max(t_numpy, 1e-9):6.1f}x numpy time)")
    print(f"  max |C_numpy - C_qutip|       = {max_c_err:.2e}")
    if triples:
        print(f"  max |rho_numpy - rho_qutip|   = {max_rho_err:.2e}")
    print()
    return dict(n=n, graph_idx=graph_idx, interaction=interaction, init=init,
                T=len(times), t_numpy=t_numpy, t_qutip=t_qutip,
                max_c_err=max_c_err, max_rho_err=max_rho_err)


def main():
    cases = [
        # n=2: only pairs
        (2, 0, "xx", "0", np.pi / 2),
        (2, 0, "zz", "plus", np.pi / 2),
        (2, 0, "ex", "plus", 100.0),
        # n=3: pairs + triples
        (3, 0, "xx", "0", np.pi / 2),
        (3, 1, "xx", "0", np.pi / 2),
        (3, 0, "ex", "plus", 100.0),
        (3, 1, "ex", "plus", 100.0),
    ]
    results = []
    for n, idx, interaction, init, t_max in cases:
        results.append(bench_one(n, idx, interaction, init, t_max))

    total_numpy = sum(r["t_numpy"] for r in results)
    total_qutip = sum(r["t_qutip"] for r in results)
    worst_c_err = max(r["max_c_err"] for r in results)
    worst_rho_err = max((r["max_rho_err"] for r in results), default=0.0)
    print("=" * 60)
    print(f"TOTAL: numpy={total_numpy:.4f}s  qutip={total_qutip:.4f}s  "
          f"ratio={total_qutip / max(total_numpy, 1e-9):.1f}x")
    print(f"worst concurrence disagreement: {worst_c_err:.2e}")
    print(f"worst triple-rho disagreement:  {worst_rho_err:.2e}")


if __name__ == "__main__":
    main()
