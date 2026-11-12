"""XX network from |0000>: exact evolution and every entanglement quantity of the plan (sec. 4), computed once per
symmetry orbit and shared by all equivalent subsets/cuts/partitions.

H = sum_{(i,j) in E} X_i X_j  (J = 1).  The terms commute, and Hd X Hd = Z for Hd = H^{(x)n}, so
    psi(t) = Hd [ 2^{-n/2} exp(-i t E(x)) ]_x ,   E(x) = sum_{(i,j)} (-1)^{x_i + x_j}
exactly (no eigendecomposition).  All entanglement measures are invariant under complex conjugation (psi(-t) = psi(t)^*)
and local unitaries, and psi(t + pi/2) is psi(t) with X applied on the odd-degree vertices, so every curve is symmetric
about t = pi/4: only [0, pi/4] is computed and mirrored.
Qubit order: qubit 0 = leftmost tensor factor (a), then b, c, d.
"""
import os
import itertools, math, time
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from . import core as C, gmn_admm as A

LET = "abcdefgh"


# ------------------------------------------------------------------ graphs and symmetry
def load_g6(path):
    return [nx.from_graph6_bytes(l.strip()) for l in open(path, "rb").read().split() if l.strip()]


def graph_name(g):
    m, deg = g.number_of_edges(), sorted(d for _, d in g.degree())
    return {(3, (1, 1, 1, 3)): "star", (3, (1, 1, 2, 2)): "path", (4, (1, 2, 2, 3)): "paw", (4, (2, 2, 2, 2)): "cycle",
            (5, (2, 2, 3, 3)): "diamond", (6, (3, 3, 3, 3)): "K4"}[(m, tuple(deg))]


def automorphisms(g):
    n = g.number_of_nodes()
    return [tuple(m[i] for i in range(n)) for m in GraphMatcher(g, g).isomorphisms_iter()]


def act_set(s, p): return frozenset(p[i] for i in s)
def act_tcut(it, p): return (act_set(it[0], p), p[it[1]])
def act_part(P, p): return frozenset(act_set(b, p) for b in P)


def orbit_reps(items, autos, act):
    """-> (list of representatives, dict item -> representative)"""
    rep, reps = {}, []
    for it in items:
        if it in rep: continue
        reps.append(it)
        for p in autos: rep[act(it, p)] = it
    return reps, rep


def set_partitions(s):
    s = list(s)
    if len(s) == 1:
        yield [s]; return
    first, rest = s[0], s[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def partitions(sub):
    """All set partitions of the qubit tuple `sub`, as frozenset of frozensets."""
    return [frozenset(frozenset(b) for b in P) for P in set_partitions(list(sub))]


def block_cuts(P):
    """Distinct bipartitions that are unions of blocks of P (each as frozenset({A, B}))."""
    blocks = list(P); out = set()
    for r in range(1, len(blocks)):
        for sel in itertools.combinations(blocks, r):
            A = frozenset().union(*sel); B = frozenset().union(*[b for b in blocks if b not in sel])
            out.add(frozenset({A, B}))
    return sorted(out, key=lambda c: sorted(sorted(x) for x in c))


def label(s): return "".join(LET[i] for i in sorted(s))
def cut_label(c): return "|".join(label(x) for x in sorted(c, key=lambda x: (len(x), sorted(x))))
def part_label(P): return "|".join(label(b) for b in sorted(P, key=lambda b: (-len(b), sorted(b))))


# ------------------------------------------------------------------ evolution
def evolution(xp, edges, n, th):
    d = 2 ** n; x = np.arange(d)
    bits = (x[:, None] >> (n - 1 - np.arange(n))) & 1
    E = np.zeros(d, int)
    for i, j in edges: E += 1 - 2 * (bits[:, i] ^ bits[:, j])
    levels = np.unique(E)
    S = 1 - 2 * ((bits @ bits.T) % 2)                                   # (-1)^{popcount(y & x)}
    M = (S / 2 ** (n / 2)) @ (E[:, None] == levels[None, :]).astype(float) / 2 ** (n / 2)
    return xp.asarray(M) @ xp.exp(-1j * xp.asarray(levels, float)[:, None] * xp.asarray(th)[None, :])


# ------------------------------------------------------------------ GMN with certified bracket
# ADMM penalty / over-relaxation tuned on the hard paw and diamond states for a relative stop of 1e-5 (tests/test_tuning_hard.py, jobs 1991535, 1991560):
# 3 qubits: 0.1 / 1.5 stopped 85% (old default 0.003 / 1.0: 3%); 4 qubits: 0.03 / 1.8 stopped 75-78% (old default: 1-3%)
REL_GAP = float(os.environ.get("XX_REL_GAP", "1e-5"))          # relative stop of the certified bracket: U - L <= max(1e-7, REL_GAP * U); override with the environment variable XX_REL_GAP
TUNED = {3: dict(rho_admm=0.1, alpha=1.5), 4: dict(rho_admm=0.03, alpha=1.8)}


def gmn_curve(xp, rho, nq, cutN, delta=1e-9, tol=1e-6, tol_gap=1e-7, rel_gap=None, log=None):
    """rho (T,d,d); cutN (T,K) negativities of ALL bipartitions.  PPT across a cut => GMN = 0 exactly; min_M N_M < delta =>
    GMN < delta (bound); otherwise batched ADMM with certified bracket [L, U] (U also capped by min_M N_M)."""
    rel_gap = REL_GAP if rel_gap is None else rel_gap
    T = rho.shape[0]
    minN = cutN.min(axis=1)
    L = xp.zeros(T); U = minN.copy()
    conv = xp.ones(T, dtype=bool); iters = xp.zeros(T, dtype=int); status = xp.ones(T, dtype=int)   # 1 = bound
    ppt = minN < 1e-10
    U[ppt] = 0.0; status[ppt] = 0                                                    # 0 = PPT across a cut: exact zero
    idx = xp.nonzero(minN >= delta)[0]
    if len(idx):
        t0 = time.perf_counter()
        r = A.solve_batch(xp, rho[idx], nq=nq, tol_gap=tol_gap, strict_gap=True, rigorous_lower=True, rel_gap=rel_gap, **TUNED[nq])     # L = certified lower bound, so L <= U
        L[idx] = r["gmn"]; U[idx] = xp.minimum(r["gmn_upper"], minN[idx])
        conv[idx] = r["converged"]; iters[idx] = r["iters"]; status[idx] = 2        # 2 = SDP
        if log: log(f"      SDP nq={nq}: {len(idx)} states, {time.perf_counter()-t0:.1f}s, converged {float(r['converged'].mean()):.1%}")
    bad = (L - U) > 1e-9                                                  # cannot happen with a certified L (only float round-off); kept as a guard
    L = xp.where(L > U, U, L); conv = conv & ~bad
    cls = A.classify(xp, L, U, tol, certified=True)                       # zero: U <= tol; positive: L_cert > tol (both proofs); else near threshold
    return dict(L=L, U=U, cls=cls, conv=conv, iters=iters, status=status)


# ------------------------------------------------------------------ fidelity targets (plan sec. 4.6)
_s = 1 / np.sqrt(2)
BASES = {"z": (np.array([1, 0], complex), np.array([0, 1], complex)),
         "x": (np.array([1, 1], complex) * _s, np.array([1, -1], complex) * _s),
         "y": (np.array([1, 1j], complex) * _s, np.array([1, -1j], complex) * _s)}


def _kron(vs):
    out = np.array([1.0 + 0j])
    for v in vs: out = np.kron(out, v)
    return out


def targets(k):
    """Ordered {name: (family, state vector)} for a k-qubit subset.  k=2: the four Bell states.  k>=3: GHZ_k^(+/-) and W_k
    (and the flipped W_k') in the z, x and y local bases.  All members of a family are local-unitary equivalent, so they share
    one witness threshold (plan sec. 4.6)."""
    out = {}
    if k == 2:
        z = np.zeros(4, complex)
        for nm, (i, j, sg) in {"Phi+": (0, 3, 1), "Phi-": (0, 3, -1), "Psi+": (1, 2, 1), "Psi-": (1, 2, -1)}.items():
            v = z.copy(); v[i] = _s; v[j] = sg * _s; out[nm] = ("Bell", v)
        return out
    for b, (a0, a1) in BASES.items():
        for sg, tag in ((1, "+"), (-1, "-")):
            out[f"GHZ{tag}{b}"] = ("GHZ", (_kron([a0] * k) + sg * _kron([a1] * k)) / np.sqrt(2))
        for fl, tag in ((False, ""), (True, "'")):
            lo, hi = (a1, a0) if fl else (a0, a1)
            out[f"W{b}{tag}"] = ("W", sum(_kron([hi if q == i else lo for q in range(k)]) for i in range(k)) / np.sqrt(k))
    return out


def extra_names(k):
    """Phase-optimised variants (name, family).  XX evolution conserves parity, so the relative phase between the two branches
    matters (the pair state is cos t|00> - i sin t|11>, whose fidelity with every real Bell state is exactly 1/2).  For
    GHZ-type targets (|a> + e^{i phi}|b>)/sqrt2 the optimum over phi is closed form: F = (rho_aa + rho_bb)/2 + |rho_ab|.
    They are local-unitary equivalent to Bell / GHZ, so the witness thresholds are unchanged."""
    return [("Phi~", "Bell"), ("Psi~", "Bell")] if k == 2 else [(f"GHZ~{b}", "GHZ") for b in "zxy"]


def witness_threshold(family, k):
    """F above this proves entanglement of the subset (Bell, k=2) or genuine multipartite entanglement (GHZ, W)."""
    return 0.5 if family in ("Bell", "GHZ") else (k - 1) / k


def fidelities(xp, k, psi_full, sub, n):
    """{name: F(t)} for the reduced state of qubits `sub` (or the pure state if len(sub) == n) against every target,
    including the phase-optimised Bell / GHZ variants (see extra_names)."""
    tg = targets(k); names = list(tg)
    Phi = xp.asarray(np.stack([tg[nm][1] for nm in names], 1))                 # (d, Nt)
    # pairs of basis states (a, b) whose superposition (|a> + e^{i phi} |b>)/sqrt2 is optimised over phi
    if k == 2:
        pairs_ab = {"Phi~": (np.eye(4)[0], np.eye(4)[3]), "Psi~": (np.eye(4)[1], np.eye(4)[2])}
    else:
        pairs_ab = {f"GHZ~{b}": (_kron([BASES[b][0]] * k), _kron([BASES[b][1]] * k)) for b in "zxy"}
    Ab = xp.asarray(np.stack([np.stack(v, 1) for v in pairs_ab.values()], 0).astype(complex))    # (Ne, d, 2)
    if len(sub) == n:
        F = xp.abs(Phi.conj().T @ psi_full) ** 2                                 # (Nt, T)
        amp = xp.abs(xp.einsum("edj,dt->ejt", xp.conj(Ab), psi_full))             # (Ne, 2, T): |<a|psi>|, |<b|psi>|
        Fe = 0.5 * (amp[:, 0] + amp[:, 1]) ** 2                                   # (Ne, T) = (|al|^2+|be|^2)/2 + |al||be|
        out = {nm: F[i] for i, nm in enumerate(names)}
    else:
        rho = C.ptrace_pure(xp, psi_full, n, list(sub))                          # (T, d, d)
        F = xp.real(xp.einsum("dn,tde,en->nt", xp.conj(Phi), rho, Phi))
        raa = xp.real(xp.einsum("ed,tdf,ef->et", xp.conj(Ab[:, :, 0]), rho, Ab[:, :, 0]))
        rbb = xp.real(xp.einsum("ed,tdf,ef->et", xp.conj(Ab[:, :, 1]), rho, Ab[:, :, 1]))
        rab = xp.einsum("ed,tdf,ef->et", xp.conj(Ab[:, :, 0]), rho, Ab[:, :, 1])
        Fe = 0.5 * (raa + rbb) + xp.abs(rab)
        out = {nm: F[i] for i, nm in enumerate(names)}
    for i, key in enumerate(pairs_ab): out[key] = Fe[i]
    return out


# ------------------------------------------------------------------ everything for one graph
def analyze(g, xp, n_intervals=2000, tol=1e-6, log=print, gmn_triples=True, gmn_whole=True):
    n = 4; edges = sorted(g.edges()); autos = automorphisms(g)
    th = np.linspace(0.0, math.pi / 4, n_intervals // 2 + 1)
    psi = evolution(xp, edges, n, th)
    h = C.to_host
    res = dict(name=graph_name(g), edges=edges, n_autos=len(autos), th=th, n_intervals=n_intervals)

    # pairs: every pair, one computation per orbit
    pairs = [frozenset(p) for p in itertools.combinations(range(n), 2)]
    reps, prep = orbit_reps(pairs, autos, act_set)
    pr = {}
    worst = 0.0
    for r in reps:
        rho = C.ptrace_pure(xp, psi, n, sorted(r))
        c, w = C.concurrence_checked(xp, rho); worst = max(worst, w)
        pr[r] = dict(C=h(c), N=h(C.negativity(xp, rho, 2, [0])))
    res.update(pair_reps=reps, pair_rep=prep, pair=pr, conc_roundoff=worst)

    # triples: cut negativities per orbit of (triple, v); GMN per triple orbit
    triples = [frozenset(t) for t in itertools.combinations(range(n), 3)]
    tcuts = [(t, v) for t in triples for v in sorted(t)]
    treps_c, tcrep = orbit_reps(tcuts, autos, act_tcut)
    tc = {}
    for (t, v) in treps_c:
        sub = sorted(t); rho = C.ptrace_pure(xp, psi, n, sub)
        tc[(t, v)] = h(C.negativity(xp, rho, 3, [sub.index(v)]))
    res.update(tcut_reps=treps_c, tcut_rep=tcrep, tcut=tc)
    treps, trep = orbit_reps(triples, autos, act_set)
    tg = {}
    for t in treps:
        sub = sorted(t); rho = C.ptrace_pure(xp, psi, n, sub)
        cutN = xp.asarray(np.stack([tc[tcrep[(t, v)]] for v in sub], 1))
        if log: log(f"    triple {label(t)} ({sum(1 for x in triples if trep[x]==t)} equivalent): GMN")
        out = gmn_curve(xp, rho, 3, cutN, tol=tol, log=log) if gmn_triples else None
        tg[t] = {k: h(v) for k, v in out.items()} if out else None
    res.update(triple_reps=treps, triple_rep=trep, triple_gmn=tg)

    # whole state: 7 bipartitions (orbit reps), purities -> Q and C_GME, negativities, GMN
    cutsA = A.bipartition_cuts(n)
    qcuts = [frozenset({frozenset(a), frozenset(set(range(n)) - set(a))}) for a in cutsA]
    qreps, qrep = orbit_reps(qcuts, autos, act_part)
    qd = {}
    for qc in qreps:
        a = sorted(min(qc, key=lambda s: (len(s), sorted(s))))
        pu, ne = C.pure_cut(xp, psi, n, a); qd[qc] = dict(purity=h(pu), N=h(ne))
    Ncut = np.stack([qd[qrep[c]]["N"] for c in qcuts]); Pcut = np.stack([qd[qrep[c]]["purity"] for c in qcuts])
    single_pur = np.stack([qd[qrep[c]]["purity"] for c in qcuts if min(len(s) for s in c) == 1])
    res.update(qcuts=qcuts, qcut_rep=qrep, qcut=qd, Q=2 * (1 - single_pur.mean(0)),
               cgme2=np.clip(2 * (1 - Pcut.max(0)), 0, None))                 # C_GME^2 = min over cuts of 2(1 - Tr rho_A^2)
    rho4 = psi.T[:, :, None] * xp.conj(psi.T)[:, None, :]
    if log: log("    whole 4-qubit state: GMN")
    out = gmn_curve(xp, rho4, 4, xp.asarray(Ncut.T), tol=tol, log=log) if gmn_whole else None
    res["gmn4"] = {k: h(v) for k, v in out.items()} if out else None

    # fidelities with Bell / GHZ / W targets on the FULL grid (a fixed target is not mirror symmetric); one run per subset orbit
    th_full = np.linspace(0.0, math.pi / 2, n_intervals + 1); psi_full = evolution(xp, edges, n, th_full)
    fid = dict(th=th_full, pair={}, triple={}, whole=None, names={}, family={}, alpha={})
    for k in (2, 3, 4):
        tg = targets(k); fid["names"][k] = list(tg) + [nm for nm, _ in extra_names(k)]
        for nm, (fam, _) in tg.items(): fid["family"][(k, nm)] = fam
        for nm, fam in extra_names(k): fid["family"][(k, nm)] = fam
        for fam in {f for f, _ in tg.values()}: fid["alpha"][(k, fam)] = witness_threshold(fam, k)
    for r in reps:
        fid["pair"][r] = {nm: h(v) for nm, v in fidelities(xp, 2, psi_full, sorted(r), n).items()}
    for t in treps:
        fid["triple"][t] = {nm: h(v) for nm, v in fidelities(xp, 3, psi_full, sorted(t), n).items()}
    fid["whole"] = {nm: h(v) for nm, v in fidelities(xp, 4, psi_full, list(range(n)), n).items()}
    res["fid"] = fid

    # all 15 partitions: largest block-respecting cut negativity (> 1e-9 => partition excluded, plan sec. 4.8)
    pl = {}
    for P in partitions(range(n)):
        cs = block_cuts(P)
        pl[P] = np.max(np.stack([Ncut[qcuts.index(c)] for c in cs]), 0) if cs else np.zeros(len(th))
    res["part_maxN"] = pl
    return res


def mirror(a):
    """[0, pi/4] -> [0, pi/2] using f(pi/2 - t) = f(t)."""
    a = np.asarray(a)
    return np.concatenate([a, a[..., :-1][..., ::-1]], axis=-1)
