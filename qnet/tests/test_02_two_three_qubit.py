"""Section 6, items 2, 3, 7."""
import networkx as nx
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from qnet import hamiltonians as H
from qnet import evolve as E
from qnet import measures as M


def _concurrence_curve(G, interaction, init, t_max, n_pts=2000):
    n = G.number_of_nodes()
    Hmat = H.build_hamiltonian(G, interaction)
    psi0 = H.initial_state(n, init)
    evals, evecs = E.diagonalize(Hmat)
    times = np.linspace(0, t_max, n_pts)
    psi_t = E.evolve_batch(evals, evecs, psi0.full().ravel(), times)
    return times, psi_t


def test_n2_xx_sin2t():
    G = nx.path_graph(2)
    times, psi_t = _concurrence_curve(G, "xx", "0", np.pi)
    rho_t = E.partial_trace_pure_batch(psi_t, 2, [0, 1])
    C = M.concurrence_batch(rho_t)
    expected = np.abs(np.sin(2 * times))
    assert np.max(np.abs(C - expected)) < 1e-6
    assert abs(np.max(C) - 1.0) < 1e-6
    t_at_max = times[np.argmax(C)]
    assert abs(t_at_max - np.pi / 4) < (times[1] - times[0]) + 1e-9


def test_triangle_xx_000():
    G = nx.complete_graph(3)
    times, psi_t = _concurrence_curve(G, "xx", "0", np.pi / 2, n_pts=4000)
    for (i, j) in G.edges():
        rho_t = E.partial_trace_pure_batch(psi_t, 3, [i, j])
        C = M.concurrence_batch(rho_t)
        assert abs(np.max(C) - 1.0 / 3.0) < 1e-3


def test_path3_xx_000():
    G = nx.path_graph(3)  # edges (0,1),(1,2); (0,2) not connected
    times, psi_t = _concurrence_curve(G, "xx", "0", np.pi / 2, n_pts=4000)
    for (i, j) in [(0, 1), (1, 2)]:
        rho_t = E.partial_trace_pure_batch(psi_t, 3, [i, j])
        C = M.concurrence_batch(rho_t)
        assert abs(np.max(C) - 0.5) < 1e-3
    rho_t_02 = E.partial_trace_pure_batch(psi_t, 3, [0, 2])
    C_02 = M.concurrence_batch(rho_t_02)
    assert np.max(C_02) < 1e-6


def test_exchange_single_excitation():
    """Unit test only: exchange Hamiltonian restricted to the single-excitation
    sector is a tight-binding model; C_ij(t) = 2|a_i(t) a_j(t)| where a(t) is
    the single-excitation amplitude vector evolved under the hopping matrix
    (h_ex couples |e_i> <-> |e_j> with amplitude 1/2 for each edge, so the
    effective single-particle Hamiltonian is (1/2) * Adjacency).
    """
    G = nx.path_graph(4)
    n = 4
    Hmat = H.build_hamiltonian(G, "ex")
    # initial state |1000>  (excitation on qubit 0)
    psi0 = H.initial_state(n, "0")
    # flip qubit 0 to |1>
    from qutip import tensor, basis
    kets = [basis(2, 0) for _ in range(n)]
    kets[0] = basis(2, 1)
    psi0 = tensor(kets)

    evals, evecs = E.diagonalize(Hmat)
    times = np.linspace(0, 10, 500)
    psi_t = E.evolve_batch(evals, evecs, psi0.full().ravel(), times)

    # Single-particle picture: amplitude a_i(t) = <e_i|psi(t)> where |e_i> is
    # the computational basis state with a single 1 at site i.
    def basis_index(excited):
        idx = 0
        for bit in range(n):
            idx = idx * 2 + (1 if bit == excited else 0)
        return idx

    a = np.array([psi_t[basis_index(i), :] for i in range(n)])  # (n, T)

    for (i, j) in G.edges():
        rho_t = E.partial_trace_pure_batch(psi_t, n, [i, j])
        C = M.concurrence_batch(rho_t)
        expected = 2 * np.abs(a[i] * a[j])
        assert np.max(np.abs(C - expected)) < 1e-6
