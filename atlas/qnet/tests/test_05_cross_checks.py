"""Section 3: cross-check spectral evolution against qt.sesolve (5 random
cases, tol 1e-8), and vectorized partial trace against Qobj.ptrace. Also
cross-check our concurrence against qt.concurrence.
"""
import random

import networkx as nx
import numpy as np
import pytest

from qnet import hamiltonians as H
from qnet import evolve as E
from qnet import measures as M
from qnet import graphs as G


def _random_cases(n_cases=5, seed=0):
    rng = random.Random(seed)
    cases = []
    for _ in range(n_cases):
        n = rng.randint(2, 5)
        graphs = G.load_graphs(n)
        Gph = rng.choice(graphs)
        interaction = rng.choice(["xx", "zz", "ex"])
        init = rng.choice(["0", "plus"])
        cases.append((n, Gph, interaction, init))
    return cases


@pytest.mark.parametrize("case_idx,case", list(enumerate(_random_cases())))
def test_sesolve_cross_check(case_idx, case):
    n, Gph, interaction, init = case
    Hmat = H.build_hamiltonian(Gph, interaction)
    psi0 = H.initial_state(n, init)
    t_max = np.pi / 2 if interaction == "xx" else 2.0
    times = np.linspace(0, t_max, 15)
    err = E.cross_check_sesolve(Hmat, psi0, times, n)
    assert err < 1e-8, f"case {case}: sesolve mismatch {err}"


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_ptrace_cross_check(n):
    graphs = G.load_graphs(n)
    Gph = graphs[min(1, len(graphs) - 1)]
    Hmat = H.build_hamiltonian(Gph, "ex")
    psi0 = H.initial_state(n, "plus")
    evals, evecs = E.diagonalize(Hmat)
    psi = E.evolve_single(evals, evecs, psi0.full().ravel(), 1.234)
    for keep_size in range(1, (n // 2) + 1):
        keep = tuple(sorted(random.sample(range(n), keep_size)))
        err = E.cross_check_ptrace(psi, n, keep)
        assert err < 1e-10, f"n={n} keep={keep} err={err}"


def test_concurrence_qutip_cross_check():
    G2 = nx.path_graph(2)
    Hmat = H.build_hamiltonian(G2, "xx")
    psi0 = H.initial_state(2, "0")
    evals, evecs = E.diagonalize(Hmat)
    for t in [0.1, 0.3, 0.5, 0.7853981633974483]:
        psi = E.evolve_single(evals, evecs, psi0.full().ravel(), t)
        rho = np.outer(psi, psi.conj())
        c_own = M.concurrence_single(rho)
        c_qt = M.concurrence_qutip_cross_check(rho)
        assert abs(c_own - c_qt) < 1e-8
