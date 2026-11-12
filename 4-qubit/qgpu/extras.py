"""Tools of the research plan that go beyond the basic toolbox (NumPy, CPU): graph-state fidelity targets (plan Sec. 4.6, 4.7), the quantum Fisher
information with the k-producibility bounds (Sec. 4.9), cut ranks of graph states (Sec. 5.3).  Qubit 0 is the leftmost tensor factor, as everywhere in qgpu."""
import itertools
import numpy as np

I2 = np.eye(2, dtype=complex)
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.array([[1, 0], [0, -1]], complex)]
HAD = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
SGATE = np.diag([1, 1j]).astype(complex)


def _kron(ms):
    out = np.array([[1.0 + 0j]])
    for m in ms: out = np.kron(out, m)
    return out


# ------------------------------------------------------------------------------------------ graph-state targets
def connected_graphs(k):
    """All connected graphs on the labelled vertices 0..k-1 (as edge tuples)."""
    allE = list(itertools.combinations(range(k), 2)); out = []
    for m in range(k - 1, len(allE) + 1):
        for E in itertools.combinations(allE, m):
            seen = {0}; stack = [0]
            while stack:
                v = stack.pop()
                for (a, b) in E:
                    for u, w in ((a, b), (b, a)):
                        if u == v and w not in seen: seen.add(w); stack.append(w)
            if len(seen) == k: out.append(E)
    return out


def graph_state(k, E, frame="x"):
    """|G> = prod_{(i,j) in E} CZ_ij |+>^k.  frame 'x': H^{(x)k}|G>, the x-basis graph state (plan Sec. 3.3);
    frame 'dyn': H^{(x)k} prod_v S_v^{deg(v)} |G>, which is exactly what exp(-i pi/4 sum XX) |0..0> gives for the graph E (exp(-i pi/4 ZZ) = e^{-i pi/4} (S x S) CZ)."""
    idx = np.arange(2 ** k); bits = ((idx[:, None] >> (k - 1 - np.arange(k))) & 1)
    ph = np.zeros(2 ** k, int)
    for (i, j) in E: ph += bits[:, i] * bits[:, j]
    v = ((-1.0) ** ph) / np.sqrt(2 ** k) + 0j
    if frame == "dyn":
        deg = np.zeros(k, int)
        for (i, j) in E: deg[i] += 1; deg[j] += 1
        v = v * (1j ** (bits @ deg))
    return _kron([HAD] * k) @ v


def alpha(vec, k):
    """Largest squared Schmidt coefficient over all cuts: <psi|rho|psi> > alpha proves GME (plan Eq. 11)."""
    a = 0.0; t = vec.reshape((2,) * k)
    for m in range(1, k // 2 + 1):
        for A in itertools.combinations(range(k), m):
            B = [q for q in range(k) if q not in A]
            s = np.linalg.svd(np.transpose(t, list(A) + B).reshape(2 ** m, -1), compute_uv=False)
            a = max(a, float(s[0] ** 2))
    return a


def graph_targets(k):
    """{name: (vector, alpha)} for every connected graph on k labelled vertices in the two frames."""
    out = {}
    for E in connected_graphs(k):
        for fr in ("x", "dyn"):
            v = graph_state(k, E, fr); out[f"{fr}:" + ",".join(f"{i}{j}" for i, j in E)] = (v, alpha(v, k))
    return out


# ------------------------------------------------------------------------------------------ quantum Fisher information
def generators(k):
    """The 3k operators sigma_alpha(qubit i)/2 as a (3k, d, d) array, ordered (qubit 0: x,y,z), (qubit 1: x,y,z), ..."""
    return np.stack([_kron([PAULI[a] / 2 if q == i else I2 for q in range(k)]) for i in range(k) for a in range(3)])


def qfi_matrix(rho, k, chunk=400, eps=1e-12):
    """F_ab = 2 sum_{pq: l_p+l_q>eps} (l_p-l_q)^2/(l_p+l_q) Re[<p|A_a|q><q|A_b|p>] for states rho (T,d,d): F_Q[rho, J_n] = n^T F n for J_n = sum_a n_a A_a (plan Eq. 12)."""
    A = generators(k); T = rho.shape[0]; out = np.empty((T, 3 * k, 3 * k))
    for s in range(0, T, chunk):
        r = rho[s:s + chunk]; lam, V = np.linalg.eigh(0.5 * (r + np.conj(np.swapaxes(r, 1, 2)))); lam = np.clip(lam, 0, None)
        M = np.einsum("tdp,ade,teq->tapq", np.conj(V), A, V, optimize=True)
        ssum = lam[:, :, None] + lam[:, None, :]; dif = lam[:, :, None] - lam[:, None, :]
        w = np.where(ssum > eps, 2 * dif ** 2 / np.where(ssum > eps, ssum, 1.0), 0.0)
        out[s:s + chunk] = np.real(np.einsum("tpq,tapq,tbpq->tab", w, M, np.conj(M), optimize=True))
    return out


def qfi_max(F, k, starts=24, iters=80, seed=1):
    """max over unit vectors n_i (one per qubit) of n^T F n.  Block ascent (n_i <- normalised sum_j F_ij n_j) is monotone; random starts plus the all-x/y/z starts.  Returns (T,) lower bounds on the maximum (a valid value of F_Q, so the depth certificate it gives is rigorous)."""
    T = F.shape[0]; rng = np.random.default_rng(seed)
    n = rng.normal(size=(T, starts, 3 * k)); n[:, :3, :] = 0
    for a in range(3):
        for i in range(k): n[:, a, 3 * i + a] = 1.0
    n = n.reshape(T, starts, k, 3); n /= np.linalg.norm(n, axis=-1, keepdims=True); n = n.reshape(T, starts, 3 * k)
    for _ in range(iters):
        for i in range(k):
            g = np.einsum("tab,tsb->tsa", F[:, 3 * i:3 * i + 3, :], n); nn = np.linalg.norm(g, axis=-1, keepdims=True)
            n[:, :, 3 * i:3 * i + 3] = np.where(nn > 1e-14, g / np.where(nn > 1e-14, nn, 1.0), n[:, :, 3 * i:3 * i + 3])
    val = np.einsum("tsa,tab,tsb->ts", n, F, n)
    return val.max(axis=1)


def kprod_bound(N, k):
    """Every k-producible state of N qubits obeys F_Q <= s k^2 + r^2 with N = s k + r, 0 <= r < k (plan Sec. 4.9)."""
    s, r = divmod(N, k); return s * k * k + r * r


def depth_from_qfi(Fmax, N, margin=1e-9):
    """Lower bound on the entanglement depth: D >= k+1 for the largest k with F_Q > bound(k).  Fmax array -> int array (>= 1)."""
    D = np.ones(Fmax.shape, int)
    for k in range(1, N):
        D = np.where(Fmax > kprod_bound(N, k) + margin, k + 1, D)
    return D


# ------------------------------------------------------------------------------------------ cut ranks of graph states
def cut_ranks_from_purity(purity):
    """GF(2) cut rank r of a graph state from the purity of a cut: Tr rho_A^2 = 2^-r."""
    return int(round(-np.log2(purity)))
