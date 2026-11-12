#!/usr/bin/env python3
"""Checks of qgpu/extras.py against known values: QFI (shot noise N, Heisenberg N^2, mixed qubit r^2, two Bell pairs = 2-producible bound), k-producibility
bounds of the plan (4, 8, 10 for N = 4), graph-state targets (alpha = 1/2 for connected graphs; the 'dyn' target equals the XX state at pi/4), cut ranks of the six graphs."""
import itertools, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); sys.path.insert(0, Q4)
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, extras as E, store
_JSONL = os.path.join(HERE, "results", "test_extras.jsonl"); os.makedirs(os.path.dirname(_JSONL), exist_ok=True); open(_JSONL, "w").close()
fails = []; SEC = [""]
def section(s): SEC[0] = s; print("\n== " + s, flush=True)
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    with open(_JSONL, "a") as fh: fh.write(json.dumps(dict(section=SEC[0], name=name, ok=bool(cond), detail=detail)) + "\n")
    if not cond: fails.append(name)
def rho_of(v): v = np.asarray(v, complex); return (np.outer(v, v.conj()) / np.vdot(v, v))[None]
def fq(rho, k, **kw): return float(E.qfi_max(E.qfi_matrix(rho, k), k, **kw)[0])

section("(E1) quantum Fisher information: known values")
zero = np.zeros(16, complex); zero[0] = 1; report("|0000>: F_Q = 4 (shot noise N)", abs(fq(rho_of(zero), 4) - 4) < 1e-9, f"{fq(rho_of(zero), 4):.12f}")
ghz = np.zeros(16, complex); ghz[0] = ghz[15] = 1; report("GHZ4: F_Q = 16 (Heisenberg N^2)", abs(fq(rho_of(ghz), 4) - 16) < 1e-8, f"{fq(rho_of(ghz), 4):.10f}")
ghz3 = np.zeros(8, complex); ghz3[0] = ghz3[7] = 1; report("GHZ3: F_Q = 9", abs(fq(rho_of(ghz3), 3) - 9) < 1e-8, f"{fq(rho_of(ghz3), 3):.10f}")
bb = np.zeros(16, complex); bb[[0, 3, 12, 15]] = 0.5; report("two Bell pairs: F_Q = 8, the 2-producible bound", abs(fq(rho_of(bb), 4) - 8) < 1e-8, f"{fq(rho_of(bb), 4):.10f}")
r = 0.6; rq = 0.5 * (np.eye(2) + r * E.PAULI[0])[None]; report("mixed qubit, Bloch vector (0.6,0,0): max F_Q over directions = 0.36", abs(fq(rq, 1) - r * r) < 1e-9, f"{fq(rq, 1):.10f}")
rq2 = np.diag([0.7, 0.3]).astype(complex)[None]; report("commuting (diagonal) qubit state: F_Q along z = 0, max = r^2 = 0.16", abs(fq(rq2, 1) - 0.16) < 1e-9, f"{fq(rq2, 1):.10f}")
rng = np.random.default_rng(3); v = rng.normal(size=16) + 1j * rng.normal(size=16); v /= np.linalg.norm(v); A = E.generators(4); n = rng.normal(size=12).reshape(4, 3); n /= np.linalg.norm(n, axis=1, keepdims=True)
J = sum(n.reshape(12)[a] * A[a] for a in range(12)); var = np.vdot(v, J @ J @ v).real - np.vdot(v, J @ v).real ** 2
Fm = E.qfi_matrix(rho_of(v), 4)[0]; report("pure random state: n^T F n = 4 Var(J_n)", abs(n.reshape(12) @ Fm @ n.reshape(12) - 4 * var) < 1e-9, f"{abs(n.reshape(12) @ Fm @ n.reshape(12) - 4 * var):.1e}")
section("(E2) k-producibility bounds of the plan")
report("N=4 bounds for k=1,2,3: 4, 8, 10", [E.kprod_bound(4, k) for k in (1, 2, 3)] == [4, 8, 10], str([E.kprod_bound(4, k) for k in (1, 2, 3)]))
report("N=3 bounds for k=1,2: 3, 5", [E.kprod_bound(3, k) for k in (1, 2)] == [3, 5], str([E.kprod_bound(3, k) for k in (1, 2)]))
report("depth from QFI (F_Q > bound(k) gives D >= k+1): 4.0->1, 8.5->3, 10.5->4, 16->4 (N=4)", list(E.depth_from_qfi(np.array([4.0, 8.5, 10.5, 16.0]), 4)) == [1, 3, 4, 4], str(list(E.depth_from_qfi(np.array([4.0, 8.5, 10.5, 16.0]), 4))))
section("(E3) graph-state targets")
T4 = E.graph_targets(4); T3 = E.graph_targets(3)
report("38 connected labelled graphs on 4 vertices, 4 on 3 (x two frames)", len(T4) == 76 and len(T3) == 8, f"{len(T4)}, {len(T3)}")
report("alpha = 1/2 for every connected graph-state target (k = 3, 4)", all(abs(a - 0.5) < 1e-12 for _, a in list(T4.values()) + list(T3.values())), "all 84 targets")
LET = "abcd"; worst = 0.0
for name in ["star", "path", "paw", "cycle", "diamond", "K4"]:
    R = store.load(os.path.join(Q4, "results", "xx_vacuum", name)); edges = [tuple(map(int, e)) for e in R.edges]
    psi = X.evolution(np, edges, 4, np.array([math.pi / 4]))[:, 0]; tgt = E.graph_state(4, [tuple(sorted(e)) for e in edges], "dyn"); f = abs(np.vdot(tgt, psi)) ** 2; worst = max(worst, 1 - f)
    report(f"{name}: the XX state at Jt = pi/4 equals the 'dyn' graph state of G (fidelity 1)", 1 - f < 1e-12, f"1-F = {1 - f:.1e}")
section("(E4) graph-state classes (plan Sec. 5.3): ranks of the three 2|2 cuts at Jt = pi/4")
ranks = {}
for name in ["star", "path", "paw", "cycle", "diamond", "K4"]:
    R = store.load(os.path.join(Q4, "results", "xx_vacuum", name)); edges = [tuple(map(int, e)) for e in R.edges]; psi = X.evolution(np, edges, 4, np.array([math.pi / 4]))
    rk = sorted(E.cut_ranks_from_purity(float(C.pure_cut(np, psi, 4, list(A))[0][0])) for A in ([0, 1], [0, 2], [0, 3])); ranks[name] = rk
    singles = [E.cut_ranks_from_purity(float(C.pure_cut(np, psi, 4, [q])[0][0])) for q in range(4)]
    exp = [1, 1, 1] if name in ("star", "K4") else [1, 2, 2]
    report(f"{name}: 2|2 cut ranks {rk}, single-qubit cuts {singles} -> {'GHZ' if rk == [1, 1, 1] else 'linear-cluster'} class", rk == exp and singles == [1, 1, 1, 1], f"expected {exp}")
section("(E5) consistency of the stored extras (results/xx_vacuum/*.extras.npz, plan_extras.py)")
for name in ["star", "path", "paw", "cycle", "diamond", "K4"]:
    pre = os.path.join(Q4, "results", "xx_vacuum", name)
    if not os.path.exists(pre + ".extras.npz"): report(f"{name}: extras file present", False, "run plan_extras.py"); continue
    ex = np.load(pre + ".extras.npz"); ej = json.load(open(pre + ".extras.json")); R = store.load(pre); ck = ej["checks"]
    report(f"{name}: state checks (norm, trace, Hermiticity <= 1e-12, min eigenvalue >= -1e-12)", max(ck["norm"], ck["trace"], ck["herm"]) < 1e-12 and ck["mineig"] > -1e-12, f"norm {ck['norm']:.1e}, trace {ck['trace']:.1e}, herm {ck['herm']:.1e}, mineig {ck['mineig']:.1e}")
    g = R.gmn4(); pure = R.cgme2 > 1e-12
    report(f"{name}: GMN4 positive implies the exact pure-state GME test (C_GME^2 > 0)", not np.any((g["cls"] == 1) & ~pure), f"{ej['whole_gmn_not_pure']} violations")
    lab = ex["label"]; report(f"{name}: label 4-qubit GME at every sample except Jt = 0 and pi/2", int((lab == 4).sum()) == len(lab) - 2 and lab[0] == 0 and lab[-1] == 0, f"{int((lab == 4).sum())} of {len(lab)}")
    ok = True
    for rep in sorted(set(R.meta["triple_map"].values())):
        gt = R.triple_gmn(rep); ok &= not np.any((gt["cls"] == 1) & (ex[f"status/tri/{rep}"] != 3))
    report(f"{name}: every triple with certified GMN > 0 has status 'GME certified'", ok)
    ps = np.load(pre + ".psi.npz"); report(f"{name}: stored psi(t) is normalised and has the full grid", ps["psi"].shape == (len(R.t), 16) and np.abs(np.linalg.norm(ps["psi"], axis=1) - 1).max() < 1e-12, f"{ps['psi'].shape}")
    d = ex["depth_qfi"]; report(f"{name}: QFI depth bound never exceeds 4 and is 1 at Jt = 0", d.max() <= 4 and d[0] == 1, f"max {int(d.max())}")
print("\nOVERALL:", "PASS" if not fails else "FAIL: " + "; ".join(fails)); sys.exit(1 if fails else 0)
