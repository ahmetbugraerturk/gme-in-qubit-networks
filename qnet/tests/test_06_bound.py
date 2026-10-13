"""qnet/bound.py: the negativity-sum upper bound GMN(rho) <= min_M N_M(rho)
must actually bound the SDP-computed GMN value on every test state we have a
ground truth for, and must be exactly 0 on PPT/product states."""
import numpy as np

from qnet import bound as B
from qnet import gmn as GMN

SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)


def ghz3():
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / SQRT2
    psi[7] = 1 / SQRT2
    return psi


def w3():
    psi = np.zeros(8, dtype=complex)
    for b in (1, 2, 4):
        psi[b] = 1 / SQRT3
    return psi


def product_zero_phiplus():
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / SQRT2
    psi[3] = 1 / SQRT2
    return psi


def noisy(psi, p):
    rho_pure = np.outer(psi, psi.conj())
    return (1 - p) * rho_pure + p * np.eye(8) / 8


def test_bound_zero_on_product_state():
    rho = np.outer(product_zero_phiplus(), product_zero_phiplus().conj())
    u, per_cut = B.gmn_upper_bound_batch(rho[np.newaxis, :, :], n_sub=3)
    assert u[0] < 1e-9
    # the A|BC cut (qubit 0) should itself be exactly PPT/zero, since the
    # state is literally a product across that cut
    assert per_cut[(0,)][0] < 1e-9


def test_bound_is_a_true_upper_bound_on_known_states():
    cases = [ghz3(), w3(), noisy(ghz3(), 0.4), noisy(ghz3(), 0.6),
             noisy(w3(), 0.3), product_zero_phiplus()]
    for psi in cases:
        rho = np.outer(psi, psi.conj()) if psi.ndim == 1 else psi
        res = GMN.solve_gmn(rho)
        u, _ = B.gmn_upper_bound_batch(rho[np.newaxis, :, :], n_sub=3)
        assert res["gmn"] <= u[0] + 1e-6, (res["gmn"], u[0])


def test_bound_batched_matches_single():
    rho_list = [np.outer(ghz3(), ghz3().conj()), np.outer(w3(), w3().conj()),
                noisy(ghz3(), 0.5)]
    rho_batch = np.stack(rho_list, axis=0)
    u_batch, per_cut_batch = B.gmn_upper_bound_batch(rho_batch, n_sub=3)
    for i, rho in enumerate(rho_list):
        u_single, _ = B.gmn_upper_bound_batch(rho[np.newaxis, :, :], n_sub=3)
        assert abs(u_batch[i] - u_single[0]) < 1e-12


def test_all_cuts_triple():
    assert B.all_cuts(3) == [(0,), (1,), (2,)]
