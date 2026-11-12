#!/usr/bin/env python3
"""Validation of the Setting I completion: (T1) intermediate polarization p_- = 1/4: collision recurrence and continuous mixture against the explicit five-qubit Kraus map;
(T2) the nine coupling pairs table (zero rows to round-off, (EX, XX) 1.454); (T3) second-order effective Hamiltonian H_eff = H_{G-v} + s Z_v (g + J^2 B^2/(2g)) converges to the exact contact state as g grows (error ~ 1/g^2 per unit time, checked by scaling);
(T4) the slow-check bracket: fidelities in report/slow_check.json are inside [0, 1] and the unpolarized purity stays in [1/2, 1] for g = 20."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4); sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import contact as K, xxvac as X, core as C
fails = []
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    if not cond: fails.append(name)
G4 = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6"))
print("== (T1) p_- = 1/4: collisions against the explicit five-qubit Kraus map")
worst = 0.0; n = 0
for G in G4:
    edges = sorted(G.edges())
    for v, orb in K.contact_sites(G):
        g, tau = 5.0, 0.4; psi0 = K.state_at(edges, 0.0); kind, rho = K.collide(psi0, edges, v, g, tau, 12, "0.25")
        model = dict(edges=edges, net="xx", kind="z", sites=[v], n=4); eta = np.diag([0.25, 0.75]).astype(complex); r = np.outer(psi0, psi0.conj())
        for k in range(12): r = K.kraus_generic(r, model, g, tau, eta)
        worst = max(worst, float(np.abs(r - rho[12]).max())); n += 1
report("collision recurrence p_-=1/4 equals the Kraus map (all 11 sites, 12 collisions)", worst < 1e-12, f"{worst:.1e}")
edges = sorted(G4[0].edges()); v = 0; psi0 = K.state_at(edges, 0.0); t = np.array([0.0, 1.0, 5.0]); w, tr = K.continuous(psi0, edges, v, 5.0, t, "0.25")
report("continuous mixture weights are (1/4, 3/4)", abs(w[0] - 0.25) < 1e-15 and abs(w[1] - 0.75) < 1e-15, str(w))
Hp, Hm = K.fields(edges, v, 5.0); r_ex = 0.25 * np.outer(C.evolve(np, Hp, psi0, t)[:, 2], C.evolve(np, Hp, psi0, t)[:, 2].conj()) + 0.75 * np.outer(C.evolve(np, Hm, psi0, t)[:, 2], C.evolve(np, Hm, psi0, t)[:, 2].conj())
r_m = K.rho_from(w, tr)[2]; report("continuous state equals the p_+ = 1/4 mixture of the two field signs", float(np.abs(r_m - r_ex).max()) < 1e-12)
print("\n== (T2) nine coupling pairs table")
p = os.path.join(Q4, "report", "data", "pairs_table.json")
if os.path.exists(p):
    rows = json.load(open(p)); mx = lambda a, b: max(r["max_cut_neg"] for r in rows if r["net"] == a and r["contact"] == b)
    for a, b in (("ZZ", "ZZ"), ("ZZ", "XX"), ("ZZ", "EX"), ("EX", "ZZ")): report(f"({a}, {b}): zero negativity on all cuts for every ancilla", mx(a, b) < 1e-12, f"{mx(a, b):.1e}")
    report("(EX, XX) largest cut negativity 1.45 (plan pilot)", abs(mx("EX", "XX") - 1.454) < 0.01, f"{mx('EX', 'XX'):.4f}")
    report("(XX, ZZ), (XX, EX), (EX, EX) are active", min(mx("XX", "ZZ"), mx("XX", "EX"), mx("EX", "EX")) > 0.1)
else: report("pairs table present (run run_pairs_table.py)", False)
print("\n== (T3) second-order effective Hamiltonian converges with g")
def evolve(H, p0, t):
    e, V = np.linalg.eigh(H); c = V.conj().T @ p0; return (V @ (c[:, None] * np.exp(-1j * np.outer(e, t)))).T
edges = sorted(G4[0].edges()); st = X.graph_name(G4[0]); errs = []
for name in ("star", "K4"):
    G = {X.graph_name(g): g for g in G4}[name]; edges = sorted(G.edges()); v = 3 if name == "star" else 0; nb = sorted({j for i, j in edges if i == v} | {i for i, j in edges if j == v}); Zv = K.op(K._PZ, v, 4); B = sum(K.op(K._PX, j, 4) for j in nb)
    Hp_ = K.xx_hamiltonian([e for e in edges if v not in e], 4); psi0 = K.state_at(edges, 0.0); T = np.array([0.5 * 20 / 1.0])
    e1 = []
    for g in (10.0, 20.0, 40.0):
        T = np.array([5.0]); ex = evolve(K.xx_hamiltonian(edges, 4) - g * Zv, psi0, T)[0]; fs = evolve(Hp_ - Zv @ (g * np.eye(16) + B @ B / (2 * g)), psi0, T)[0]; e1.append(1 - abs(np.vdot(ex, fs)) ** 2)
    report(f"{name}: infidelity of H_eff decreases by >= 3x per doubling of g (O(1/g^2))", e1[0] / e1[1] > 3 and e1[1] / e1[2] > 3, " ".join(f"{x:.1e}" for x in e1))
print("\n== (T4) slow_check data sanity")
sc = json.load(open(os.path.join(Q4, "report", "data", "slow_check.json"))); ok = all(0 <= r["coh_slow_min"] <= 1 + 1e-9 and 0 <= r["unp_slow_min"] <= 1 + 1e-9 and 0.5 - 1e-9 <= r["unp_purity_min"] <= 1 + 1e-9 for r in sc)
report("fidelities in [0,1], unpolarized purity in [1/2, 1]", ok, f"{len(sc)} rows")
report("g = 20: H_eff fidelity >= 0.99 over Jt <= 80 for all 11 sites, both t0, coherent and unpolarized", all(r["coh_slow_min"] >= 0.99 and r["unp_slow_min"] >= 0.99 for r in sc if r["g"] == 20))
print(f"\n{len(fails)} failures"); sys.exit(1 if fails else 0)
