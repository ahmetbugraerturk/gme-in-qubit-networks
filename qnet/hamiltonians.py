"""Pauli-like operators built explicitly from kets (per spec: do NOT use
qt.sigmap()/qt.sigmam(), whose convention is the opposite of ours), and
Hamiltonian construction for networks of qubits on a graph.

Convention:
  s+ = |1><0|   (our "raising" operator, from the 0 computational state to 1)
  s- = |0><1|
  X  = |0><1| + |1><0|   (flips the computational basis)
  Z  = |0><0| - |1><1|   (diagonal, +1 on |0>, -1 on |1>)
"""
from __future__ import annotations

import itertools
from typing import List, Literal, Tuple

import networkx as nx
import numpy as np
import qutip as qt

ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)

SP = ket1 * ket0.dag()  # |1><0|
SM = ket0 * ket1.dag()  # |0><1|
X = ket0 * ket1.dag() + ket1 * ket0.dag()
Z = ket0 * ket0.dag() - ket1 * ket1.dag()
PLUS = (ket0 + ket1).unit()
HADAMARD = (ket0 * ket0.dag() + ket0 * ket1.dag() + ket1 * ket0.dag() - ket1 * ket1.dag()) / np.sqrt(2)

InteractionType = Literal["xx", "zz", "ex"]
InitType = Literal["0", "plus"]


def two_qubit_term(interaction: InteractionType) -> qt.Qobj:
    """h_ij for the given interaction type, as a 4x4 Qobj (qubit i tensor qubit j)."""
    if interaction == "xx":
        return qt.tensor(X, X)
    if interaction == "zz":
        return qt.tensor(Z, Z)
    if interaction == "ex":
        return 0.5 * (qt.tensor(SP, SM) + qt.tensor(SM, SP))
    raise ValueError(f"unknown interaction {interaction!r}")


def build_hamiltonian(G: nx.Graph, interaction: InteractionType, w0: float = 1.0) -> qt.Qobj:
    """H = w0 * sum_{(i,j) in E(G)} h_ij, qubit order = node order 0..n-1."""
    n = G.number_of_nodes()
    dims = [2] * n
    h = two_qubit_term(interaction)
    H = qt.qzero([2] * n)
    for (i, j) in sorted(tuple(sorted(e)) for e in G.edges()):
        H = H + qt.expand_operator(h, dims, [i, j])
    return w0 * H


def initial_state(n: int, init: InitType) -> qt.Qobj:
    if init == "0":
        return qt.tensor([ket0] * n)
    if init == "plus":
        return qt.tensor([PLUS] * n)
    raise ValueError(f"unknown init {init!r}")


def hadamard_all(n: int) -> qt.Qobj:
    return qt.tensor([HADAMARD] * n)


def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for (i, j) in G.edges():
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A
