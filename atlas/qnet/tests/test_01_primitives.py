"""Section 6, item 1: s+/s- convention and h matrices."""
import numpy as np

from qnet import hamiltonians as H


def test_sp_sm_convention():
    assert np.allclose(H.SP.full(), np.array([[0, 0], [1, 0]]))  # |1><0|
    assert np.allclose(H.SM.full(), np.array([[0, 1], [0, 0]]))  # |0><1|
    # X, Z standard Pauli matrices in the {|0>,|1>} basis
    assert np.allclose(H.X.full(), np.array([[0, 1], [1, 0]]))
    assert np.allclose(H.Z.full(), np.array([[1, 0], [0, -1]]))


def test_h_ex_block():
    h_ex = H.two_qubit_term("ex").full()
    # basis order |00>,|01>,|10>,|11>
    expected = np.zeros((4, 4))
    expected[1, 2] = 0.5  # <01|h|10>
    expected[2, 1] = 0.5  # <10|h|01>
    assert np.allclose(h_ex, expected)


def test_h_xx_zz():
    h_xx = H.two_qubit_term("xx").full()
    assert np.allclose(h_xx, np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]))
    h_zz = H.two_qubit_term("zz").full()
    assert np.allclose(h_zz, np.kron([[1, 0], [0, -1]], [[1, 0], [0, -1]]))
