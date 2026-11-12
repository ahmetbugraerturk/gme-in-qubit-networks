"""Section 6, item 9: Guhne-Seevinck and GMN tests."""
import numpy as np
import pytest

from qnet import gs
from qnet import gmn
from qnet import measures as M

SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)


def ghz3() -> np.ndarray:
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / SQRT2  # |000>
    psi[7] = 1 / SQRT2  # |111>
    return psi


def w3() -> np.ndarray:
    psi = np.zeros(8, dtype=complex)
    for b in (1, 2, 4):  # |001>,|010>,|100>
        psi[b] = 1 / SQRT3
    return psi


def product_zero_phiplus() -> np.ndarray:
    """|0>_A (x) |Phi+>_BC = (|000> + |011>)/sqrt2 -- biseparable (A|BC)."""
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / SQRT2  # |000>
    psi[3] = 1 / SQRT2  # |011>
    return psi


def noisy(psi: np.ndarray, p: float) -> np.ndarray:
    rho_pure = np.outer(psi, psi.conj())
    return (1 - p) * rho_pure + p * np.eye(8) / 8


# ---------------------------------------------------------------------------
# GS thresholds
# ---------------------------------------------------------------------------

def test_gs_ghz3_threshold():
    p_thresh = 4.0 / 7.0
    rho_below = noisy(ghz3(), p_thresh - 0.02)
    rho_above = noisy(ghz3(), p_thresh + 0.02)
    assert gs.scan_all_bases(rho_below)["certified"]
    assert not gs.scan_all_bases(rho_above)["certified"]


def test_gs_w3_threshold():
    p_thresh = 8.0 / 17.0
    rho_below = noisy(w3(), p_thresh - 0.02)
    rho_above = noisy(w3(), p_thresh + 0.02)
    assert gs.scan_all_bases(rho_below)["certified"]
    assert not gs.scan_all_bases(rho_above)["certified"]


# ---------------------------------------------------------------------------
# GMN SDP
# ---------------------------------------------------------------------------

def test_gmn_pure_ghz_w():
    res_ghz = gmn.solve_gmn(np.outer(ghz3(), ghz3().conj()))
    res_w = gmn.solve_gmn(np.outer(w3(), w3().conj()))
    assert res_ghz["gmn"] > 1e-3, res_ghz
    assert res_w["gmn"] > 1e-3, res_w


def test_gmn_biseparable_zero():
    rho = np.outer(product_zero_phiplus(), product_zero_phiplus().conj())
    res = gmn.solve_gmn(rho)
    assert res["gmn"] < 1e-6, res


def test_gmn_noisy_ghz():
    res_low = gmn.solve_gmn(noisy(ghz3(), 0.4))
    res_high = gmn.solve_gmn(noisy(ghz3(), 0.6))
    assert res_low["gmn"] > 1e-3, res_low
    assert res_high["gmn"] < 1e-6, res_high


# ---------------------------------------------------------------------------
# C_GME <=> GMN>0 consistency, for the canonical 3-qubit states above.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("psi,expect_entangled", [
    (ghz3(), True),
    (w3(), True),
    (product_zero_phiplus(), False),
])
def test_cgme_gmn_consistency(psi, expect_entangled):
    c_gme = M.c_gme_pure_single(psi, 3)
    res = gmn.solve_gmn(np.outer(psi, psi.conj()))
    gmn_pos = res["gmn"] > 1e-6
    cgme_pos = c_gme > 1e-6
    assert gmn_pos == cgme_pos == expect_entangled, (c_gme, res)
