#!/usr/bin/env python3
"""Analytic predictions versus simulation for the XX-from-|0000> study (CPU, NumPy; runs in seconds).
Every test states the closed-form prediction it checks (see report/xx_tests_report.tex, section 2 for the derivations)."""
import itertools, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)   # HERE = 4-qubit/tests
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, gmn_admm as A, xxvac as X

JSONL = os.path.join(HERE, "results", "test_analytic.jsonl"); os.makedirs(os.path.dirname(JSONL), exist_ok=True); open(JSONL, "w").close()
fails = []; SEC = [""]
def section(t): SEC[0] = t; print("\n== " + t, flush=True)
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    with open(JSONL, "a") as fh: fh.write(json.dumps(dict(section=SEC[0], name=name, ok=bool(cond), detail=detail)) + "\n")
    if not cond: fails.append(name)
gs = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); NAMES = [X.graph_name(g) for g in gs]

def gf2_rank(M):
    rows = [int("".join(str(int(x)) for x in r), 2) for r in M] if len(M) else []; rank = 0
    for bit in reversed(range(max(M.shape[1], 1))):
        piv = next((i for i, r in enumerate(rows) if (r >> bit) & 1), None)
        if piv is None: continue
        pr = rows.pop(piv); rank += 1; rows = [r ^ pr if (r >> bit) & 1 else r for r in rows]
    return rank

section("(A1) one XX bond: psi(t) = cos t|00> - i sin t|11>")
th = np.linspace(0, math.pi / 2, 401); psi = X.evolution(np, [(0, 1)], 2, th)
an = np.zeros((4, len(th)), complex); an[0] = np.cos(th); an[3] = -1j * np.sin(th)
report("closed-form evolution equals cos t|00> - i sin t|11>", np.abs(psi - an).max() < 1e-14, f"max diff {np.abs(psi-an).max():.1e}")
rho = C.ptrace_pure(np, psi, 2, [0, 1]); Cn, _ = C.concurrence_checked(np, rho)
report("concurrence C(t) = |sin 2t|", np.abs(Cn - np.abs(np.sin(2 * th))).max() < 1e-7, f"max diff {np.abs(Cn-np.abs(np.sin(2*th))).max():.1e} (sqrt round-off ~1e-8)")
Nn = C.pure_cut(np, psi, 2, [0])[1]
report("negativity N(t) = |sin 2t| / 2  (= C/2 for a pure pair)", np.abs(Nn - np.abs(np.sin(2 * th)) / 2).max() < 1e-14, f"max diff {np.abs(Nn-np.abs(np.sin(2*th))/2).max():.1e}")
Fb = {nm: np.abs(v.conj() @ psi) ** 2 for nm, (f, v) in X.targets(2).items()}
report("F with Phi+ and Phi- is exactly 1/2 at all t; with Psi+- it is 0", all(np.abs(Fb[k] - .5).max() < 1e-14 for k in ("Phi+", "Phi-")) and all(np.abs(Fb[k]).max() < 1e-14 for k in ("Psi+", "Psi-")),
       "real Bell states never certify entanglement of this pair (witness threshold is also 1/2)")
rho2 = C.ptrace_pure(np, psi, 2, [0, 1]); raa = rho2[:, 0, 0].real; rbb = rho2[:, 3, 3].real; rab = np.abs(rho2[:, 0, 3])
report("phase-optimised Bell fidelity = (1 + C)/2 for a pure pair", np.abs(0.5 * (raa + rbb) + rab - (1 + np.abs(np.sin(2 * th))) / 2).max() < 1e-14, "F_max = (rho_00,00 + rho_11,11)/2 + |rho_00,11|")

section("(A2) exact symmetries of the state")
for g, nm in zip(gs, NAMES):
    e = sorted(g.edges()); deg = np.array([d for _, d in sorted(g.degree())]); t = np.linspace(0.0, 1.3, 7)
    p0 = X.evolution(np, e, 4, t); p1 = X.evolution(np, e, 4, t + math.pi / 2); pm = X.evolution(np, e, 4, -t)
    Xodd = np.array([[1.0]])
    for q in range(4): Xodd = np.kron(Xodd, np.array([[0, 1], [1, 0]]) if deg[q] % 2 else np.eye(2))
    pred = (-1j) ** len(e) * (Xodd @ p0)
    report(f"psi(t + pi/2) = (-i)^|E| (prod over odd-degree vertices of X) psi(t), {nm}", np.abs(p1 - pred).max() < 1e-13, f"|E|={len(e)}, odd-degree vertices {[LET for LET, d in zip('abcd', deg) if d % 2]}, max diff {np.abs(p1-pred).max():.1e}")
    report(f"psi(-t) = conj(psi(t)), {nm}", np.abs(pm - p0.conj()).max() < 1e-14, f"max diff {np.abs(pm-p0.conj()).max():.1e}")
    pmir = X.evolution(np, e, 4, math.pi / 2 - t); report(f"psi(pi/2 - t) = (-i)^|E| X_odd conj(psi(t)), {nm}", np.abs(pmir - (-1j) ** len(e) * (Xodd @ p0.conj())).max() < 1e-13, "=> every LU- and conjugation-invariant measure is symmetric about pi/4")

section("(A3) graph state at Jt = pi/4: GF(2)-rank formulas  (purity 2^-r, negativity (2^r - 1)/2)")
for g, nm in zip(gs, NAMES):
    e = sorted(g.edges()); adj = np.zeros((4, 4), int)
    for i, j in e: adj[i, j] = adj[j, i] = 1
    psi = X.evolution(np, e, 4, np.array([math.pi / 4])); worst_p = worst_n = 0; ranks = {}
    pur_all = []
    for a in A.bipartition_cuts(4):
        B = [q for q in range(4) if q not in a]; r = gf2_rank(adj[np.ix_(a, B)]); ranks["".join("abcd"[q] for q in a) + "|" + "".join("abcd"[q] for q in B)] = r
        pu, ne = C.pure_cut(np, psi, 4, list(a)); worst_p = max(worst_p, abs(float(pu[0]) - 2.0 ** -r)); worst_n = max(worst_n, abs(float(ne[0]) - (2 ** r - 1) / 2)); pur_all.append(2.0 ** -r)
    report(f"cut purities 2^-r and negativities (2^r-1)/2, {nm}", max(worst_p, worst_n) < 1e-13, f"ranks {ranks}; max diff {max(worst_p, worst_n):.1e}")
    singles = [2.0 ** -gf2_rank(adj[np.ix_([q], [x for x in range(4) if x != q])]) for q in range(4)]
    Q = 2 * (1 - np.mean(singles)); cg2 = 2 * (1 - max(pur_all))
    pu_all = np.stack([C.pure_cut(np, psi, 4, list(a))[0] for a in A.bipartition_cuts(4)]); Qn = 2 * (1 - C.pure_cut(np, psi, 4, [0])[0][0])
    report(f"Q = 1 and C_GME^2 = 1 (connected graph: every vertex cut has rank 1), {nm}", abs(Q - 1) < 1e-12 and abs(cg2 - 1) < 1e-12 and abs(2 * (1 - pu_all.max()) - 1) < 1e-12, f"predicted Q={Q:.3f}, C_GME^2={cg2:.3f}")
    cmax = max(float(C.concurrence_checked(np, C.ptrace_pure(np, psi, 4, list(p)))[0][0]) for p in itertools.combinations(range(4), 2))
    report(f"all six pair concurrences vanish at pi/4 (each pair reduction is maximally mixed or classically correlated), {nm}", cmax < 1e-7, f"max C = {cmax:.1e}")

section("(A4) pure-state negativity: Schmidt coefficients, and why the SVD is used")
g = gs[1]; e = sorted(g.edges()); t = np.array([0.0, 0.3, 0.9, math.pi / 4, math.pi / 2]); psi = X.evolution(np, e, 4, t)
rho = np.einsum("it,jt->tij", psi, psi.conj()); w = 0.0
for a in ([0], [1], [0, 1], [0, 2], [0, 3]):
    w = max(w, np.abs(C.pure_cut(np, psi, 4, a)[1] - C.negativity(np, rho, 4, a)).max())
report("N = sum_{i<j} s_i s_j (SVD) equals the partial-transpose negativity of the 16x16 state", w < 1e-14, f"max diff {w:.1e}")
worst_old = 0.0; worst_svd = 0.0
for g in gs:
    e = sorted(g.edges()); tt = np.linspace(0, math.pi / 2, 2001); ps = X.evolution(np, e, 4, tt); rh = np.einsum("it,jt->tij", ps, ps.conj())
    for a in ([0], [1], [2], [3], [0, 1], [0, 2], [0, 3]):
        lam = np.clip(np.linalg.eigvalsh(C.ptrace_pure(np, ps, 4, a)), 0, None); old = ((np.sqrt(lam).sum(1)) ** 2 - 1) / 2
        ref = C.negativity(np, rh, 4, a); worst_old = max(worst_old, np.abs(old - ref).max()); worst_svd = max(worst_svd, np.abs(C.pure_cut(np, ps, 4, a)[1] - ref).max())
report("SVD route agrees with the partial-transpose negativity to 1e-14 on the full grid (all graphs, 7 cuts)", worst_svd < 1e-14, f"max diff {worst_svd:.1e}")
report("the sqrt-of-eigenvalue route does not (round-off eigenvalues ~1e-17 become ~1e-9 after the square root)", worst_old > 1e-10, f"max diff {worst_old:.1e}  (this is what failed the mirror test at 1e-8)")

section("(A5) algebra of the GMN solver")
rng = np.random.default_rng(0); nq = 3; d = 8; cuts = A.bipartition_cuts(nq); K = len(cuts); B = 2
def herm(): X_ = rng.normal(size=(B, d, d)) + 1j * rng.normal(size=(B, d, d)); return 0.5 * (X_ + X_.conj().transpose(0, 2, 1))
def Aop(W, P, Q): return [P[m] + A._pt(np, Q[m], cuts[m], nq) - W for m in range(K)]
def Aadj(y): return -sum(y), [y[m] for m in range(K)], [A._pt(np, y[m], cuts[m], nq) for m in range(K)]
ip = lambda a, b: float(np.real(np.sum(a.conj() * b)))
W, P, Q = herm(), [herm() for _ in range(K)], [herm() for _ in range(K)]; y = [herm() for _ in range(K)]
lhs = sum(ip(a, b) for a, b in zip(Aop(W, P, Q), y)); aw, ap, aq = Aadj(y); rhs = ip(W, aw) + sum(ip(P[m], ap[m]) + ip(Q[m], aq[m]) for m in range(K))
report("<A x, y> = <x, A^H y>  (A^H y = (-sum y_m ; y_m ; T_m y_m))", abs(lhs - rhs) < 1e-10, f"|diff| {abs(lhs-rhs):.1e}")
aay = Aop(*Aadj(y)); aay_pred = [2 * y[m] + sum(y) for m in range(K)]
report("A A^H = 2 I + (1 1^T) (x) I  (since T_m T_m = id)", max(np.abs(a - b).max() for a, b in zip(aay, aay_pred)) < 1e-12, f"max diff {max(np.abs(a-b).max() for a, b in zip(aay, aay_pred)):.1e}")
inv = lambda r: [0.5 * (r[m] - sum(r) / (2.0 + K)) for m in range(K)]
rr = Aop(W, P, Q); yy = inv(rr); rec = Aop(*Aadj(yy)); report("(A A^H)^-1 = (I - J/(2+K))/2 reproduces the residual", max(np.abs(a - b).max() for a, b in zip(rec, rr)) < 1e-10, f"max diff {max(np.abs(a-b).max() for a, b in zip(rec, rr)):.1e}")
aw, ap, aq = Aadj(yy); Wp, Pp, Qp = W - aw, [P[m] - ap[m] for m in range(K)], [Q[m] - aq[m] for m in range(K)]
report("projection x - A^H (A A^H)^-1 A x lies in ker A", max(np.abs(a).max() for a in Aop(Wp, Pp, Qp)) < 1e-10, f"max |A Pi x| {max(np.abs(a).max() for a in Aop(Wp, Pp, Qp)):.1e}")
ghz = np.zeros(8, complex); ghz[0] = ghz[7] = 1 / np.sqrt(2); rho_g = np.outer(ghz, ghz.conj())[None]
ok = True; mn = 1e9
for _ in range(400):
    Lam = []; 
    for m in range(K):
        Z = rng.normal(size=(1, d, d)) + 1j * rng.normal(size=(1, d, d)); Lam.append(0.5 * (Z + Z.conj().transpose(0, 2, 1)) * rng.uniform(0.05, 1.0))
    U = float(A.dual_upper_bound(np, rho_g, np.stack(Lam), cuts, nq)[0]); mn = min(mn, U)
report("weak duality: every dual-feasible Lambda gives U(Lambda) >= GMN(GHZ3) = 1/2", mn >= 0.5 - 1e-12, f"smallest of 400 random U = {mn:.4f}")
Ut = float(A.dual_upper_bound(np, rho_g, np.stack([rho_g / 3.0] * K), cuts, nq)[0])
report("tight certificate: Lambda_m = rho/3 gives U = 1/2 exactly, so GMN(GHZ3) <= 1/2 analytically (each |nu(rho^{T_m})| = 1/2)", abs(Ut - 0.5) < 1e-12, f"U = {Ut:.12f}")

section("(A6) counting facts of the plan")
report("Bell number: 15 partitions of four qubits", len(X.partitions(range(4))) == 15, f"{len(X.partitions(range(4)))}")
report("2^(n-1) - 1 cuts: 1, 3, 7 for n = 2, 3, 4", [len(A.bipartition_cuts(n)) for n in (2, 3, 4)] == [1, 3, 7], str([len(A.bipartition_cuts(n)) for n in (2, 3, 4)]))
exp = {"star": 6, "path": 2, "cycle": 8, "paw": 2, "diamond": 4, "K4": 24}
report("|Aut(G)| equals the plan's table (sec. 7.1)", all(len(X.automorphisms(g)) == exp[nm] for g, nm in zip(gs, NAMES)), str({nm: len(X.automorphisms(g)) for g, nm in zip(gs, NAMES)}))
print("\nOVERALL:", "PASS" if not fails else "FAIL: " + ", ".join(fails)); sys.exit(0 if not fails else 1)
