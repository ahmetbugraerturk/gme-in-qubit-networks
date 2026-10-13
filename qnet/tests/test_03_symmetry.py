"""Section 6, items 4, 5, 6, 8."""
import itertools

import networkx as nx
import numpy as np
import pytest

from qnet import hamiltonians as H
from qnet import evolve as E
from qnet import measures as M
from qnet import graphs as G


def _simulate(Gph, interaction, init, t_max, n_pts=1500):
    n = Gph.number_of_nodes()
    Hmat = H.build_hamiltonian(Gph, interaction)
    psi0 = H.initial_state(n, init)
    evals, evecs = E.diagonalize(Hmat)
    times = np.linspace(0, t_max, n_pts)
    psi_t = E.evolve_batch(evals, evecs, psi0.full().ravel(), times)
    return times, psi_t


ALL_TEST_GRAPHS = []
for n in range(2, 6):
    gl = G.load_graphs(n)
    for g in gl:
        ALL_TEST_GRAPHS.append((n, g))


@pytest.mark.parametrize("n,Gph", ALL_TEST_GRAPHS)
def test_symmetry_shortcut(n, Gph):
    """XX|0..0> & Hn -> ZZ|+..+>, and XX|+..+> & Hn -> ZZ|0..0>.
    Also: XX|+..+>, ZZ|0..0>, EX|0..0> are stationary (C=0).
    """
    t_max = 2.0
    Hn = H.hadamard_all(n).full()

    # --- stationary cases ---
    for interaction, init in [("xx", "plus"), ("zz", "0"), ("ex", "0")]:
        times, psi_t = _simulate(Gph, interaction, init, t_max, n_pts=50)
        # state should be constant (up to global phase) -> all pairwise overlaps = 1
        psi0 = psi_t[:, 0]
        overlaps = np.abs(np.conj(psi0) @ psi_t)
        assert np.min(overlaps) > 1 - 1e-7, f"{interaction}/{init} not stationary on {Gph.edges()}"
        for (i, j) in Gph.edges():
            rho_t = E.partial_trace_pure_batch(psi_t, n, [i, j])
            C = M.concurrence_batch(rho_t)
            assert np.max(C) < 1e-6

    # --- symmetry mapping: XX|0> -> ZZ|+> ---
    times, psi_xx0 = _simulate(Gph, "xx", "0", t_max, n_pts=200)
    psi_zz_plus_transformed = Hn @ psi_xx0  # (dim, T)
    _, psi_zz_plus_direct = _simulate(Gph, "zz", "plus", t_max, n_pts=200)
    overlap = np.abs(np.sum(np.conj(psi_zz_plus_direct) * psi_zz_plus_transformed, axis=0))
    assert np.min(overlap) > 1 - 1e-6

    # --- symmetry mapping: XX|+> -> ZZ|0> ---
    times, psi_xx_plus = _simulate(Gph, "xx", "plus", t_max, n_pts=200)
    psi_zz0_transformed = Hn @ psi_xx_plus
    _, psi_zz0_direct = _simulate(Gph, "zz", "0", t_max, n_pts=200)
    overlap2 = np.abs(np.sum(np.conj(psi_zz0_direct) * psi_zz0_transformed, axis=0))
    assert np.min(overlap2) > 1 - 1e-6


@pytest.mark.parametrize("n,Gph", ALL_TEST_GRAPHS)
def test_zz_edge_marginal_closed_form(n, Gph):
    """Closed form (section 6 item 5) specialized to the 2-qubit reduced
    density matrix rho_ij, which is what actually enters the concurrence:
    rho_ij[s_i,s_j; s_i',s_j'] = 1/4 * exp(-i t A_ij (s_i s_j - s_i' s_j'))
      * prod_{k != i,j} E_k[cos(t*[A_ik(s_i-s_i') + A_jk(s_j-s_j')])]
    where E_k[.] = 1 since qubit k is traced out and its +/- outcomes split
    evenly for the |+..+> state; this reduces (after tracing) to exactly the
    formula with the product over k != i,j of cos(...), because each traced
    qubit k contributes a factor cos(t*(...)) from summing its two branches.
    """
    A = H.adjacency_matrix(Gph)
    t_max = 1.3
    times, psi_t = _simulate(Gph, "zz", "plus", t_max, n_pts=25)

    for (i, j) in Gph.edges():
        rho_t = E.partial_trace_pure_batch(psi_t, n, [i, j])
        others = [k for k in range(n) if k not in (i, j)]
        max_err = 0.0
        for ti, t in enumerate(times):
            for si in (1, -1):
                for sj in (1, -1):
                    for spi in (1, -1):
                        for spj in (1, -1):
                            row = 0 if si == 1 else 1
                            row = row * 2 + (0 if sj == 1 else 1)
                            col = 0 if spi == 1 else 1
                            col = col * 2 + (0 if spj == 1 else 1)
                            val = 0.25 * np.exp(-1j * t * A[i, j] * (si * sj - spi * spj))
                            for k in others:
                                arg = t * (A[i, k] * (si - spi) + A[j, k] * (sj - spj))
                                val *= np.cos(arg)
                            err = abs(val - rho_t[ti, row, col])
                            max_err = max(max_err, err)
        assert max_err < 1e-10, f"edge {(i,j)} n={n} g={Gph.edges()} err={max_err}"


def test_kn_ex_zz_equivalence():
    """K_n: EX|+..+> max pair concurrence == ZZ|+..+> max (one-axis-twisting
    equivalence); should differ for non-complete graphs (P4, C4)."""
    for n in range(3, 6):
        Gph = nx.complete_graph(n)
        t_max = 20.0
        _, psi_ex = _simulate(Gph, "ex", "plus", t_max, n_pts=4000)
        _, psi_zz = _simulate(Gph, "zz", "plus", t_max, n_pts=4000)
        rho_ex = E.partial_trace_pure_batch(psi_ex, n, [0, 1])
        rho_zz = E.partial_trace_pure_batch(psi_zz, n, [0, 1])
        C_ex = np.max(M.concurrence_batch(rho_ex))
        C_zz = np.max(M.concurrence_batch(rho_zz))
        assert abs(C_ex - C_zz) < 5e-3, f"n={n}: C_ex={C_ex} C_zz={C_zz}"


def test_p4_c4_ex_zz_differ():
    # P4: expect ~0.5 (zz) vs ~0.586 (ex) on an edge
    G4 = nx.path_graph(4)
    t_max = 30.0
    _, psi_ex = _simulate(G4, "ex", "plus", t_max, n_pts=6000)
    _, psi_zz = _simulate(G4, "zz", "plus", t_max, n_pts=6000)
    rho_ex = E.partial_trace_pure_batch(psi_ex, 4, [0, 1])
    rho_zz = E.partial_trace_pure_batch(psi_zz, 4, [0, 1])
    C_ex = np.max(M.concurrence_batch(rho_ex))
    C_zz = np.max(M.concurrence_batch(rho_zz))
    assert abs(C_zz - 0.5) < 1e-2
    assert abs(C_ex - 0.586) < 2e-2

    # C4: expect ~0.309 (zz) vs ~0.355 (ex)
    G4c = nx.cycle_graph(4)
    _, psi_ex_c = _simulate(G4c, "ex", "plus", t_max, n_pts=6000)
    _, psi_zz_c = _simulate(G4c, "zz", "plus", t_max, n_pts=6000)
    rho_ex_c = E.partial_trace_pure_batch(psi_ex_c, 4, [0, 1])
    rho_zz_c = E.partial_trace_pure_batch(psi_zz_c, 4, [0, 1])
    C_ex_c = np.max(M.concurrence_batch(rho_ex_c))
    C_zz_c = np.max(M.concurrence_batch(rho_zz_c))
    assert abs(C_zz_c - 0.309) < 1e-2
    assert abs(C_ex_c - 0.355) < 2e-2


@pytest.mark.parametrize("n,Gph", ALL_TEST_GRAPHS)
def test_edge_orbit_consistency(n, Gph):
    """Concurrences must be equal within an edge orbit (Aut(G) symmetry of
    the |0..0>/|+..+> initial states)."""
    orbits = G.edge_orbits(Gph)
    t_max = np.pi / 2
    _, psi_t = _simulate(Gph, "xx", "0", t_max, n_pts=300)
    for orbit in orbits:
        curves = []
        for (i, j) in orbit:
            rho_t = E.partial_trace_pure_batch(psi_t, n, [i, j])
            curves.append(M.concurrence_batch(rho_t))
        base = curves[0]
        for c in curves[1:]:
            assert np.max(np.abs(c - base)) < 1e-6


@pytest.mark.parametrize("n,Gph", ALL_TEST_GRAPHS)
def test_xx_period_pi(n, Gph):
    """max over [0, pi/2] == max over [0, pi] for XX|0..0>."""
    _, psi_half = _simulate(Gph, "xx", "0", np.pi / 2, n_pts=800)
    _, psi_full = _simulate(Gph, "xx", "0", np.pi, n_pts=1600)
    for (i, j) in Gph.edges():
        rho_half = E.partial_trace_pure_batch(psi_half, n, [i, j])
        rho_full = E.partial_trace_pure_batch(psi_full, n, [i, j])
        C_half_max = np.max(M.concurrence_batch(rho_half))
        C_full_max = np.max(M.concurrence_batch(rho_full))
        assert abs(C_half_max - C_full_max) < 1e-3
