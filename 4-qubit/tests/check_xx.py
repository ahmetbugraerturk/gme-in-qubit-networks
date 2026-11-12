#!/usr/bin/env python3
"""Validation gate for the XX-from-|0000> study (plan sec. 4.5 table + our symmetry shortcuts + second-solver check)."""
import itertools, math, os, sys, time
import numpy as np, cupy as cp, cvxpy as cvx
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)   # HERE = 4-qubit/tests
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, gmn_admm as A, xxvac as X
xp = cp; fails = []
import json
_SUITE = os.path.splitext(os.path.basename(__file__))[0]
_JSONL = os.path.join(HERE, "results", _SUITE + ".jsonl"); os.makedirs(os.path.dirname(_JSONL), exist_ok=True); open(_JSONL, "w").close()
_SEC = [""]
def section(title): _SEC[0] = title
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    with open(_JSONL, "a") as fh: fh.write(json.dumps(dict(section=_SEC[0], name=name, ok=bool(cond), detail=detail)) + "\n")
    if not cond: fails.append(name)
def gmn_state(rho):
    nq = int(round(math.log2(rho.shape[0]))); rb = xp.asarray(np.asarray(rho)[None])
    cutN = xp.stack([C.negativity(xp, rb, nq, list(c)) for c in A.bipartition_cuts(nq)], 1)
    o = X.gmn_curve(xp, rb, nq, cutN); return float(o["L"][0]), float(o["U"][0])
def ket(v): v = np.asarray(v, complex); return np.outer(v / np.linalg.norm(v), (v / np.linalg.norm(v)).conj())
ghz3 = np.zeros(8); ghz3[0] = ghz3[7] = 1
w3 = np.zeros(8); w3[[1, 2, 4]] = 1
bell = np.zeros(4); bell[0] = bell[3] = 1; phi0 = np.kron(bell, [1, 0])
psim = np.zeros(4); psim[1] = 1; psim[2] = -1

print("== (a) plan sec. 4.5 validation table", flush=True); section("(a) plan sec. 4.5 validation table")
L, U = gmn_state(ket(ghz3)); report("GMN(GHZ3) = 0.500", abs(L - 0.5) < 1e-4 and U >= 0.5 - 1e-6, f"L={L:.5f} U={U:.5f}")
L, U = gmn_state(ket(w3)); report("GMN(W3) = 0.443", abs(L - 0.443) < 2e-3, f"L={L:.5f} U={U:.5f}")
L, U = gmn_state(ket(phi0)); report("GMN(Phi+ x |0>) = 0", U <= 1e-9, f"L={L:.2e} U={U:.2e}")
r = C.ptrace_pure(xp, xp.asarray(ket(phi0)[:, :1] * 0 + (np.kron(bell, [1, 0]) / np.sqrt(2))[:, None]), 3, [0, 1]); c, _ = C.concurrence_checked(xp, r)
report("C12(Phi+ x |0>) = 1", abs(float(c[0]) - 1) < 1e-9, f"C={float(c[0]):.6f}")
for p, want, tag in [(0.30, 0.0, "p=0.30 (<3/7)"), (3 / 7, 0.0, "p=3/7 boundary"), (0.44, 0.010, "p=0.44")]:
    L, U = gmn_state(p * ket(ghz3) + (1 - p) * np.eye(8) / 8)
    report(f"GMN(GHZ3 + noise) {tag} = {want}", (U <= 1e-5) if want == 0 else abs(L - want) < 1.5e-3, f"L={L:.5f} U={U:.5f}")
L, U = gmn_state(0.6 * ket(w3) + 0.4 * np.eye(8) / 8); report("GMN(W3 + noise) p=0.6 = 0.088", abs(L - 0.088) < 1.5e-3, f"L={L:.5f} U={U:.5f}")
for p in (0.2, 0.5, 0.8, 1.0):
    rho = p * ket(psim) + (1 - p) * np.eye(4) / 4; c, _ = C.concurrence_checked(xp, xp.asarray(rho[None]))
    report(f"Werner concurrence p={p}", abs(float(c[0]) - max(0, (3 * p - 1) / 2)) < 1e-9, f"C={float(c[0]):.6f}")

gs = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6"))
print("\n== (b) mirror symmetry f(pi/2 - t) = f(t) on the full grid (cheap quantities, every graph)", flush=True); section("(b) mirror symmetry f(pi/2 - t) = f(t) on the full grid (cheap quantities, every graph)")
for g in gs:
    edges = sorted(g.edges()); N = 2000; full = np.linspace(0, math.pi / 2, N + 1); half = full[: N // 2 + 1]
    psi_f = X.evolution(xp, edges, 4, full); psi_h = X.evolution(xp, edges, 4, half)
    dconc = 0.0; dneg = 0.0
    for pr in itertools.combinations(range(4), 2):
        cf = C.to_host(C.concurrence_checked(xp, C.ptrace_pure(xp, psi_f, 4, pr))[0]); ch = C.to_host(C.concurrence_checked(xp, C.ptrace_pure(xp, psi_h, 4, pr))[0])
        dconc = max(dconc, float(np.abs(cf - X.mirror(ch)).max()))
    for a in ([0], [1], [0, 1], [0, 2]):
        nf = C.to_host(C.pure_cut(xp, psi_f, 4, a)[1]); nh = C.to_host(C.pure_cut(xp, psi_h, 4, a)[1])
        dneg = max(dneg, float(np.abs(nf - X.mirror(nh)).max()))
    # concurrence = sqrt of eigenvalues that are round-off (~1e-16) near zero, so it carries ~1e-8 noise; negativities do not
    report(f"mirror symmetry, {X.graph_name(g)}", dconc < 1e-7 and dneg < 1e-10, f"concurrence {dconc:.1e} (<1e-7), negativity {dneg:.1e} (<1e-10)")

print("\n== (c) GMN mirror symmetry and (d) orbit equality of triple GMN (path, paw, star)", flush=True); section("(c) GMN mirror symmetry and (d) orbit equality of triple GMN (path, paw, star)")
rng = np.random.default_rng(1)
for g in gs:
    if X.graph_name(g) not in ("path", "paw", "star"): continue
    edges = sorted(g.edges()); autos = X.automorphisms(g)
    ta = np.sort(rng.uniform(0.05, math.pi / 4 - 0.05, 12)); tb = math.pi / 2 - ta
    triples = [frozenset(t) for t in itertools.combinations(range(4), 3)]
    def triple_gmn(th, t):
        sub = sorted(t); psi = X.evolution(xp, edges, 4, th); rho = C.ptrace_pure(xp, psi, 4, sub)
        cutN = xp.stack([C.negativity(xp, rho, 3, [i]) for i in range(3)], 1)
        o = X.gmn_curve(xp, rho, 3, cutN); return C.to_host(o["L"]), C.to_host(o["U"])
    t0 = sorted(triples[0]); La, Ua = triple_gmn(ta, triples[0]); Lb, Ub = triple_gmn(tb, triples[0])
    report(f"GMN mirror, {X.graph_name(g)} triple {X.label(triples[0])}", float(np.abs(La - Lb).max()) < 2e-5, f"max|dL|={float(np.abs(La-Lb).max()):.1e}")
    reps, rep = X.orbit_reps(triples, autos, X.act_set); worst = 0.0
    for t in triples:
        Lt, _ = triple_gmn(ta, t); Lr, _ = triple_gmn(ta, rep[t]); worst = max(worst, float(np.abs(Lt - Lr).max()))
    report(f"triple-GMN orbit equality, {X.graph_name(g)}", worst < 2e-5, f"max diff {worst:.1e}, {len(reps)} orbits")

print("\n== (e) second solver (SCS) on network triples; bracket must contain it", flush=True); section("(e) second solver (SCS) on network triples; bracket must contain it")
def scs_gmn(rho):
    d = 8; rp = cvx.Parameter((d, d), hermitian=True); W = cvx.Variable((d, d), hermitian=True); cons = []
    for c in A.bipartition_cuts(3):
        P = cvx.Variable((d, d), hermitian=True); Q = cvx.Variable((d, d), hermitian=True)
        cons += [P >> 0, np.eye(d) - P >> 0, Q >> 0, np.eye(d) - Q >> 0]; QT = Q
        for q in c: QT = cvx.partial_transpose(QT, dims=(2, 2, 2), axis=q)
        cons.append(W == P + QT)
    rp.value = 0.5 * (rho + rho.conj().T); pr = cvx.Problem(cvx.Minimize(cvx.real(cvx.trace(W @ rp))), cons)
    return max(0.0, -float(pr.solve(solver=cvx.SCS, eps=1e-8, max_iters=200000)))
bad = 0; n_ok = 0
for g in gs:
    if X.graph_name(g) not in ("star", "path", "K4"): continue
    edges = sorted(g.edges()); th = np.array([0.2, 0.45, 0.6, 0.75]); psi = X.evolution(xp, edges, 4, th)
    for sub in ([0, 1, 2], [1, 2, 3]):
        rho = C.ptrace_pure(xp, psi, 4, sub); cutN = xp.stack([C.negativity(xp, rho, 3, [i]) for i in range(3)], 1)
        o = X.gmn_curve(xp, rho, 3, cutN); Lh, Uh = C.to_host(o["L"]), C.to_host(o["U"]); rh = C.to_host(rho)
        for k in range(len(th)):
            s = scs_gmn(rh[k]); n_ok += 1
            if not (Lh[k] - 5e-6 <= s <= Uh[k] + 5e-6): bad += 1; print("   mismatch", X.graph_name(g), sub, th[k], Lh[k], s, Uh[k])
report("SCS value inside the ADMM bracket [L, U]", bad == 0, f"{n_ok - bad}/{n_ok}")

print("\n== (f) pure four-qubit state: C_GME > 0 exactly when the state is not a product across a cut; GMN4 > 0 iff GME", flush=True); section("(f) pure four-qubit state: C_GME > 0 exactly when the state is not a product across a cut; GMN4 > 0 iff GME")
g = gs[5]; th = np.linspace(0.02, math.pi / 4, 40); psi = X.evolution(xp, sorted(g.edges()), 4, th)
rho4 = psi.T[:, :, None] * cp.conj(psi.T)[:, None, :]
cutN = cp.stack([C.negativity(xp, rho4, 4, list(c)) for c in A.bipartition_cuts(4)], 1); o = X.gmn_curve(xp, rho4, 4, cutN)
pur = cp.stack([C.pure_cut(xp, psi, 4, list(c))[0] for c in A.bipartition_cuts(4)], 1); cgme2 = C.to_host(2 * (1 - pur.max(1)))
report("GMN4 > 0 wherever C_GME^2 > 1e-6 (K4, 40 times)", bool(np.all((C.to_host(o["L"]) > 0) | (cgme2 < 1e-6))), f"min C_GME^2={cgme2.min():.2e}")
for g2 in gs:
    if X.graph_name(g2) not in ("star", "K4"): continue
    ps = X.evolution(xp, sorted(g2.edges()), 4, np.array([math.pi / 4]))
    cm = max(float(C.concurrence_checked(xp, C.ptrace_pure(xp, ps, 4, pr))[0][0]) for pr in itertools.combinations(range(4), 2))
    report(f"GHZ-class graph state at Jt=pi/4 has all pair concurrences 0 ({X.graph_name(g2)})", cm < 1e-9, f"max C={cm:.1e}")
print("\n== (g) fidelity targets, checked independently against QuTiP", flush=True); section("(g) fidelity targets, checked independently against QuTiP")
import qutip as qt
for k in (2, 3, 4):
    tg = X.targets(k); names = list(tg); V = np.stack([tg[nm][1] for nm in names], 1)
    norms = np.abs(np.linalg.norm(V, axis=0) - 1).max()
    report(f"targets normalised, k={k} ({len(names)} states)", norms < 1e-12, f"max |norm-1| = {norms:.1e}")
bellG = np.abs(np.stack([v for _, v in X.targets(2).values()], 1).conj().T @ np.stack([v for _, v in X.targets(2).values()], 1) - np.eye(4)).max()
report("the four Bell targets are orthonormal", bellG < 1e-12, f"{bellG:.1e}")
g = gs[2]; edges = sorted(g.edges()); th = np.array([0.1, 0.5, 1.0, 1.4]); psi = X.evolution(xp, edges, 4, th)
dq = 0.0; dqf = 0.0
for sub in ([0, 1], [1, 2, 3], [0, 1, 2, 3]):
    k = len(sub); mine = {nm: C.to_host(v) for nm, v in X.fidelities(xp, k, psi, sub, 4).items()}; ph = C.to_host(psi)
    for ti in range(len(th)):
        rho = qt.Qobj(np.outer(ph[:, ti], ph[:, ti].conj()), dims=[[2] * 4, [2] * 4])
        rs = rho if k == 4 else rho.ptrace(sub)
        for nm, (fam, vec) in X.targets(k).items():
            ket = qt.Qobj(vec, dims=[[2] * k, [1] * k])
            dq = max(dq, abs(float(np.real(qt.expect(qt.ket2dm(ket), rs))) - mine[nm][ti]))          # exact: tr(|t><t| rho)
            dqf = max(dqf, abs(qt.fidelity(rs, ket) ** 2 - mine[nm][ti]))                              # qt.fidelity takes sqrtm: ~1e-8 accurate
report("fidelities equal QuTiP expect(|target><target|, rho) (paw, 3 subsets, 4 times)", dq < 1e-12, f"max diff {dq:.1e}")
report("fidelities equal qt.fidelity(rho, |target>)^2 to QuTiP's own accuracy", dqf < 1e-6, f"max diff {dqf:.1e} (qt.fidelity uses matrix square roots, ~1e-8)")
p0 = X.evolution(xp, edges, 4, np.array([0.0]))
f0 = {nm: float(v[0]) for nm, v in X.fidelities(xp, 4, p0, [0, 1, 2, 3], 4).items()}
report("t=0: F(|0000>, GHZ+z) = 1/2 and F with W = 0", abs(f0["GHZ+z"] - 0.5) < 1e-12 and abs(f0["Wz"]) < 1e-12, f"GHZ+z {f0['GHZ+z']:.3f}, Wz {f0['Wz']:.1e}")
print("\nOVERALL:", "PASS" if not fails else "FAIL: " + ", ".join(fails), flush=True); sys.exit(0 if not fails else 1)
