"""QuTiP-native implementation of the n<=3 simulation pipeline: time
evolution via qt.sesolve, reduced states via Qobj.ptrace, pair concurrence
via qt.concurrence -- all called directly in per-timestep loops, the way
you'd write this with QuTiP as the only tool.

This exists purely to performance- and accuracy-compare against the
vectorized numpy pipeline (qnet.evolve + qnet.measures). The Hamiltonian/
initial-state construction (qnet.hamiltonians) is reused unchanged in both,
since that part is identical either way (a one-time Qobj build, not a
per-timestep hot loop).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import qutip as qt

from qnet import hamiltonians as H

# Same tight tolerances the cross-check in qnet/evolve.py needed to match
# the spectral method to 1e-8: a fair, accuracy-matched comparison rather
# than pitting a loose-but-fast sesolve against an exact spectral method.
SESOLVE_OPTIONS = {"atol": 1e-14, "rtol": 1e-12}


def evolve_qutip(Hmat: qt.Qobj, psi0: qt.Qobj, times: np.ndarray,
                  options: dict = SESOLVE_OPTIONS) -> List[qt.Qobj]:
    """List of Qobj kets, one per time in `times`, via qt.sesolve."""
    res = qt.sesolve(Hmat, psi0, times, options=options)
    return res.states


def pair_concurrences_qutip(states: List[qt.Qobj], edges: Sequence[Tuple[int, int]]
                             ) -> Dict[Tuple[int, int], np.ndarray]:
    """edge -> C(t) array, via Qobj.ptrace(edge) + qt.concurrence, looped
    over every timestep (the natural QuTiP-only way to do this)."""
    T = len(states)
    out = {e: np.zeros(T) for e in edges}
    for e in edges:
        keep = list(e)
        for ti, psi in enumerate(states):
            rho_e = psi.ptrace(keep)
            out[e][ti] = qt.concurrence(rho_e)
    return out


def triple_reduced_states_qutip(states: List[qt.Qobj], triples: Sequence[Tuple[int, int, int]]
                                 ) -> Dict[Tuple[int, int, int], List[qt.Qobj]]:
    """triple -> list of Qobj rho_ijk(t), via Qobj.ptrace, looped."""
    out = {trip: [] for trip in triples}
    for trip in triples:
        keep = list(trip)
        for psi in states:
            out[trip].append(psi.ptrace(keep))
    return out


def run_pipeline_qutip(Gph, n: int, interaction: str, init: str, times: np.ndarray):
    """End-to-end QuTiP-native run: evolve, then every pair concurrence and
    every triple's reduced density matrix. Mirrors what the numpy pipeline's
    evolve_batch + partial_trace_pure_batch + concurrence_batch compute."""
    from qnet import graphs as G

    Hmat = H.build_hamiltonian(Gph, interaction)
    psi0 = H.initial_state(n, init)
    states = evolve_qutip(Hmat, psi0, times)

    edges = [tuple(sorted(e)) for e in Gph.edges()]
    triples = G.induced_connected_triples(Gph) if n >= 3 else []

    edge_C = pair_concurrences_qutip(states, edges)
    triple_rho = triple_reduced_states_qutip(states, triples) if triples else {}
    return edge_C, triple_rho
