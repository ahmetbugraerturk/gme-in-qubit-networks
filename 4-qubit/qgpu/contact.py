"""Contact (ancilla) model of the research plan, Setting I: XX network, ZZ contact at one site v, fresh ancilla in |1> or |+> (plan Sec. 3.1-3.2, 3.4).

For the ZZ contact the ancilla operator Z_a commutes with the interaction, so (plan Eq. 3) one collision is the random-unitary map
    E(rho) = p_+ U_+ rho U_+^dag + p_- U_- rho U_-^dag,   U_(+-) = exp(-i (H_G +- g Z_v) tau),   p_(+-) = (1 +- <Z_a>)/2.
  ancilla |1>  : <Z_a> = -1, p_- = 1  -> a deterministic unitary U_-, the network stays pure; n collisions = continuous evolution under H_G - g Z_v at t = n tau
  ancilla |0>  : the same with +g
  ancilla |+>  : <Z_a> = 0, p_+ = p_- = 1/2 -> random field sign at every collision (dephasing); the network becomes mixed
The continuous-time counterpart (one retained ancilla, plan [1] second protocol) is rho(t) = sum_s p_s |psi_s(t)><psi_s(t)| with psi_s(t) = exp(-i (H_G + s g Z_v) t)|psi_0>
(the field sign is fixed for the whole time): for |1> identical to the collision model at t = n tau, for |+> a mixture of two pure trajectories.
Qubit 0 = a is the leftmost tensor factor, as everywhere in qgpu.  NumPy / CuPy generic where it matters (the evolution itself is tiny and done on the host)."""
import itertools, math
import numpy as np
from . import core as C, xxvac as X, gmn_admm as A, extras as E, gs

LET = "abcd"
_PX = np.array([[0, 1], [1, 0]], complex); _PZ = np.diag([1.0, -1.0]).astype(complex); _I2 = np.eye(2, dtype=complex)


def op(o, i, n):
    out = np.array([[1.0 + 0j]])
    for k in range(n): out = np.kron(out, o if k == i else _I2)
    return out


def xx_hamiltonian(edges, n=4):
    return sum(op(_PX, i, n) @ op(_PX, j, n) for i, j in edges)


def contact_sites(g):
    """Symmetry-inequivalent contact sites of graph g: list of (representative, orbit) under Aut(G).  11 in total over the six four-vertex graphs (plan Sec. 7.1)."""
    autos = X.automorphisms(g); seen = set(); out = []
    for v in sorted(g.nodes()):
        if v in seen: continue
        orb = sorted({p[v] for p in autos}); seen |= set(orb); out.append((v, orb))
    return out


def stabilizer(autos, v):
    """Automorphisms fixing the contact site: the symmetry group of the problem with a contact (plan Sec. 5.1)."""
    return [p for p in autos if p[v] == v]


def eig(Hm):
    e, V = np.linalg.eigh(0.5 * (Hm + Hm.conj().T)); return e, V


def unitary(Hm, t, cache=None):
    e, V = eig(Hm) if cache is None else cache
    return (V * np.exp(-1j * e * t)) @ V.conj().T


def state_at(edges, t0, n=4):
    """Pre-contact state exp(-i H_G t0)|0000> (closed form, plan Eq. 4)."""
    return X.evolution(np, edges, n, np.array([float(t0)]))[:, 0]


def fields(edges, v, g, n=4):
    H0 = xx_hamiltonian(edges, n); Zv = op(_PZ, v, n)
    return H0 + g * Zv, H0 - g * Zv


def collide(psi0, edges, v, g, tau, nmax, ancilla):
    """States after n = 0..nmax collisions.  ancilla '1' / '0': returns ('pure', psi (nmax+1, d)); '+' (or any p_+ in (0,1) via float): ('mixed', rho (nmax+1, d, d))."""
    Hp, Hm = fields(edges, v, g)
    if ancilla in ("1", "0"):
        U = unitary(Hm if ancilla == "1" else Hp, tau); psi = np.empty((nmax + 1, psi0.size), complex); psi[0] = psi0
        for k in range(nmax): psi[k + 1] = U @ psi[k]
        return "pure", psi
    pp = 0.5 if ancilla == "+" else float(ancilla)
    Up, Um = unitary(Hp, tau), unitary(Hm, tau); rho = np.empty((nmax + 1,) + (psi0.size,) * 2, complex); rho[0] = np.outer(psi0, psi0.conj())
    for k in range(nmax):
        r = rho[k]; rho[k + 1] = pp * (Up @ r @ Up.conj().T) + (1 - pp) * (Um @ r @ Um.conj().T)
    return "mixed", rho


def continuous(psi0, edges, v, g, times, ancilla):
    """Retained ancilla.  Returns (weights, trajectories) with trajectories (S, d, T); rho(t) = sum_s w_s |psi_s(t)><psi_s(t)|."""
    Hp, Hm = fields(edges, v, g)
    if ancilla == "1": return [1.0], [C.evolve(np, Hm, psi0, times)]
    if ancilla == "0": return [1.0], [C.evolve(np, Hp, psi0, times)]
    pp = 0.5 if ancilla == "+" else float(ancilla)      # p_+ = probability of the field sign +g (p_- = 1 - p_+)
    return [pp, 1 - pp], [C.evolve(np, Hp, psi0, times), C.evolve(np, Hm, psi0, times)]


def rho_from(weights, trajs, idx=None):
    """(T, d, d) density matrices of the mixture, for the time indices idx (all if None)."""
    out = 0
    for w, P in zip(weights, trajs):
        Pi = P if idx is None else P[:, idx]; out = out + w * (Pi.T[:, :, None] * np.conj(Pi.T)[:, None, :])
    return out


def rho_of_pure(psi):
    """(T, d) pure states -> (T, d, d)."""
    return psi[:, :, None] * np.conj(psi)[:, None, :]


# ---------------------------------------------------------------------------------------------------- explicit 5-qubit versions (validation)
def five_qubit_unitary(edges, v, g, t):
    """exp(-i t (H_G + g Z_a Z_v)) on the ancilla (qubit 0) and the network (vertex k = qubit k + 1)."""
    H5 = sum(op(_PX, i + 1, 5) @ op(_PX, j + 1, 5) for i, j in edges) + g * op(_PZ, 0, 5) @ op(_PZ, v + 1, 5)
    return unitary(H5, t)


def ptrace_anc(rho5):
    """Trace out qubit 0 of a (32,32) density matrix."""
    return np.einsum("aiaj->ij", rho5.reshape(2, 16, 2, 16))


def kraus_collision(rho, edges, v, g, tau, eta):
    """E(rho) = Tr_a[ U (eta (x) rho) U^dag ] with the explicit five-qubit unitary and the 2x2 ancilla state eta (plan Eq. 2)."""
    U = five_qubit_unitary(edges, v, g, tau); r5 = np.kron(eta, rho); return ptrace_anc(U @ r5 @ U.conj().T)


def retained_ancilla(psi0, edges, v, g, t, eta_vec):
    """Network state at time t when one ancilla in the pure state eta_vec (2,) stays and interacts continuously."""
    U = five_qubit_unitary(edges, v, g, t); phi = U @ np.kron(eta_vec, psi0); return ptrace_anc(np.outer(phi, phi.conj()))


# ---------------------------------------------------------------------------------------------------- reduced states of density matrices
def ptrace_rho(xp, rho, n, keep):
    """Reduced states of (T, 2^n, 2^n) density matrices on the qubits `keep` (sorted)."""
    T = rho.shape[0]; keep = list(keep); rest = [q for q in range(n) if q not in keep]
    a = rho.reshape((T,) + (2,) * (2 * n))
    perm = [0] + [1 + q for q in keep] + [1 + q for q in rest] + [1 + n + q for q in keep] + [1 + n + q for q in rest]
    a = xp.transpose(a, perm).reshape(T, 2 ** len(keep), 2 ** len(rest), 2 ** len(keep), 2 ** len(rest))
    return xp.einsum("tikjk->tij", a)


def fidelities_rho(xp, k, rho):
    """{name: F(t)} of (T, 2^k, 2^k) states against the Bell (k = 2) or GHZ / W (k >= 3) targets of xxvac plus the phase-optimised Bell / GHZ variants (same definitions as xxvac.fidelities)."""
    tg = X.targets(k); names = list(tg); Phi = xp.asarray(np.stack([tg[nm][1] for nm in names], 1))
    F = xp.real(xp.einsum("dn,tde,en->nt", xp.conj(Phi), rho, Phi))
    if k == 2: pairs_ab = {"Phi~": (np.eye(4)[0], np.eye(4)[3]), "Psi~": (np.eye(4)[1], np.eye(4)[2])}
    else: pairs_ab = {f"GHZ~{b}": (X._kron([X.BASES[b][0]] * k), X._kron([X.BASES[b][1]] * k)) for b in "zxy"}
    Ab = xp.asarray(np.stack([np.stack(v, 1) for v in pairs_ab.values()], 0).astype(complex))
    raa = xp.real(xp.einsum("ed,tdf,ef->et", xp.conj(Ab[:, :, 0]), rho, Ab[:, :, 0])); rbb = xp.real(xp.einsum("ed,tdf,ef->et", xp.conj(Ab[:, :, 1]), rho, Ab[:, :, 1]))
    rab = xp.einsum("ed,tdf,ef->et", xp.conj(Ab[:, :, 0]), rho, Ab[:, :, 1]); Fe = 0.5 * (raa + rbb) + xp.abs(rab)
    out = {nm: F[i] for i, nm in enumerate(names)}
    for i, key in enumerate(pairs_ab): out[key] = Fe[i]
    return out


# ---------------------------------------------------------------------------------------------------- general models: network family, contact kind, several contact sites (plan Sec. 3.1-3.3)
_PY = np.array([[0, -1j], [1j, 0]], complex)
PP = {"z": {"1": 0.0, "0": 1.0, "+": 0.5, "-": 0.5}, "x": {"+": 1.0, "-": 0.0, "1": 0.5, "0": 0.5}}      # probability p_+ of the field sign +g, by contact kind and ancilla state


def net_hamiltonian(edges, family, n=4):
    """'xx': sum X_i X_j (norm 1 per bond).  'ex': sum (sigma+_i sigma-_j + h.c.)/2 = (X_i X_j + Y_i Y_j)/4 (norm 1/2 per bond, plan Sec. 3.1: compare coupling families only with this normalisation stated)."""
    if family == "xx": return sum(op(_PX, i, n) @ op(_PX, j, n) for i, j in edges) if edges else np.zeros((2 ** n,) * 2, complex)
    if family == "ex": return sum(0.25 * (op(_PX, i, n) @ op(_PX, j, n) + op(_PY, i, n) @ op(_PY, j, n)) for i, j in edges) if edges else np.zeros((2 ** n,) * 2, complex)
    raise ValueError(family)


def field_operator(sites, kind, n=4):
    """sum over the contact sites of Z_j ('z') or X_j ('x'): the ancilla operator P_a (Z_a or X_a) commutes with the whole Hamiltonian, so a collision is a random unitary (plan Eq. 3)."""
    P = _PZ if kind == "z" else _PX
    return sum(op(P, j, n) for j in sites)


def model_fields(model, g):
    """(H0 + g F, H0 - g F) for model = dict(edges, net, kind, sites, n)."""
    n = model.get("n", 4); H0 = net_hamiltonian(model["edges"], model["net"], n); F = field_operator(model["sites"], model["kind"], n); return H0 + g * F, H0 - g * F


def collide_model(psi0, model, g, tau, nmax, pp):
    """States after n = 0..nmax collisions with probability pp of the field sign +g (pp in {0, 1}: coherent, returns ('pure', psi (nmax+1, d)); otherwise ('mixed', rho))."""
    Hp, Hm = model_fields(model, g)
    if pp in (0.0, 1.0):
        U = unitary(Hp if pp == 1.0 else Hm, tau); psi = np.empty((nmax + 1, psi0.size), complex); psi[0] = psi0
        for k in range(nmax): psi[k + 1] = U @ psi[k]
        return "pure", psi
    Up, Um = unitary(Hp, tau), unitary(Hm, tau); rho = np.empty((nmax + 1,) + (psi0.size,) * 2, complex); rho[0] = np.outer(psi0, psi0.conj())
    for k in range(nmax): r = rho[k]; rho[k + 1] = pp * (Up @ r @ Up.conj().T) + (1 - pp) * (Um @ r @ Um.conj().T)
    return "mixed", rho


def continuous_model(psi0, model, g, times, pp):
    """Retained ancilla: (weights, trajectories (S, d, T)); rho(t) = sum_s w_s |psi_s(t)><psi_s(t)| with the field sign fixed for the whole time."""
    Hp, Hm = model_fields(model, g)
    if pp == 1.0: return [1.0], [C.evolve(np, Hp, psi0, times)]
    if pp == 0.0: return [1.0], [C.evolve(np, Hm, psi0, times)]
    return [pp, 1 - pp], [C.evolve(np, Hp, psi0, times), C.evolve(np, Hm, psi0, times)]


def channel_spectrum(model, g, tau, pp):
    """Spectrum of the collision channel as a 4N x 4N superoperator (plan Sec. 5.5): E = sum_s p_s U_s (x) conj(U_s).  Returns number of eigenvalues equal to 1 (fixed-point space), number on the unit circle,
    the largest |lambda| < 1 (slowest decay) and the corresponding 1/e time in collisions (inf if none)."""
    Hp, Hm = model_fields(model, g); Up, Um = unitary(Hp, tau), unitary(Hm, tau); S = pp * np.kron(Up, Up.conj()) + (1 - pp) * np.kron(Um, Um.conj()); w = np.linalg.eigvals(S); a = np.abs(w)
    nfix = int((np.abs(w - 1) < 1e-8).sum()); nuni = int((np.abs(a - 1) < 1e-8).sum()); rest = a[a < 1 - 1e-8]; sec = float(rest.max()) if len(rest) else 0.0
    return dict(n_fixed=nfix, n_unit=nuni, second=sec, one_over_e=(float(-1 / np.log(sec)) if 0 < sec < 1 else float("inf")))


def five_qubit_generic(model, g, t, P=_PZ):
    """exp(-i t (H_net + g P_a sum_j sigma_j)) with the ancilla as qubit 0 and network vertex k as qubit k + 1 (validation).  P is the ancilla operator (Z for kind 'z', X for 'x')."""
    n = model.get("n", 4); H0 = net_hamiltonian(model["edges"], model["net"], n); I2n = np.eye(2 ** n); H5 = np.kron(_I2, H0)
    sig = _PZ if model["kind"] == "z" else _PX
    for j in model["sites"]: H5 = H5 + g * np.kron(P, op(sig, j, n))
    return unitary(H5, t)


def kraus_generic(rho, model, g, tau, eta):
    P = _PZ if model["kind"] == "z" else _PX; U = five_qubit_generic(model, g, tau, P); r5 = np.kron(eta, rho); out = U @ r5 @ U.conj().T; return np.einsum("aiaj->ij", out.reshape(2, 16, 2, 16))
