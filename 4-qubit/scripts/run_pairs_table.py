#!/usr/bin/env python3
"""Plan Task 3.1 / Sec. 3.3: all nine (network, contact) pairs on P4 from |0000>, 60 collisions, J tau = 0.4, g/J = 5, contact at an end and at a middle vertex, ancilla eta in {|0>, |1>, |+>, 1/2}.
Collision map by the explicit five-qubit unitary and partial trace over the ancilla (no conservation-law shortcut).  Reports the largest cut negativity of the four-qubit state over all 7 cuts and all collisions, and the largest pair concurrence.
Output: report/pairs_table.json, report/pairs_table.tex."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import contact as K, core as C
I2, X, Y, Z = K._I2, K._PX, K._PY, K._PZ
def kron(*a):
    out = np.eye(1)
    for m in a: out = np.kron(out, m)
    return out
EDGES = [(0, 1), (1, 2), (2, 3)]; N = 4; G, TAU, NC = 5.0, 0.4, 60
def bond(fam, A, B):
    """two-site operator for family fam on operators A (ancilla/left) and B"""
    if fam == "xx": return np.kron(X, X) if A is None else None
def net_H(fam):
    H = np.zeros((16, 16), complex)
    for i, j in EDGES:
        if fam == "xx": H += K.op(X, i, N) @ K.op(X, j, N)
        elif fam == "zz": H += K.op(Z, i, N) @ K.op(Z, j, N)
        else: H += 0.25 * (K.op(X, i, N) @ K.op(X, j, N) + K.op(Y, i, N) @ K.op(Y, j, N))
    return H
def contact_H(fam, v):
    """ancilla = qubit 0 of the five-qubit space, network qubit k -> k + 1"""
    def at(P, k): return K.op(P, k + 1, N + 1)
    a = lambda P: K.op(P, 0, N + 1)
    if fam == "xx": return a(X) @ at(X, v)
    if fam == "zz": return a(Z) @ at(Z, v)
    return 0.25 * (a(X) @ at(X, v) + a(Y) @ at(Y, v))
ETA = {"|0>": np.diag([1, 0]).astype(complex), "|1>": np.diag([0, 1]).astype(complex), "|+>": 0.5 * np.ones((2, 2), complex), "1/2": 0.5 * np.eye(2, dtype=complex)}
def cut_neg(rho):
    return max(float(C.negativity(np, rho[None], 4, list(c)).max()) for c in ([0], [1], [2], [3], [0, 1], [0, 2], [0, 3]))
def pair_conc(rho):
    best = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rp = K.ptrace_rho(np, rho[None], 4, [i, j]); c, _ = C.concurrence_checked(np, rp); best = max(best, float(np.max(c)))
    return best
rows = []
for net in ("xx", "zz", "ex"):
    for con in ("zz", "xx", "ex"):
        for vname, v in (("end", 0), ("middle", 1)):
            H5 = np.kron(I2, net_H(net)) + G * contact_H(con, v); U = K.unitary(H5, TAU)
            for en, eta in ETA.items():
                rho = np.zeros((16, 16), complex); rho[0, 0] = 1; mx = 0.0; mc = 0.0; arg = 0
                for n in range(NC + 1):
                    if n > 0:
                        r5 = np.kron(eta, rho); o = U @ r5 @ U.conj().T; rho = np.einsum("aiaj->ij", o.reshape(2, 16, 2, 16))
                        assert abs(np.trace(rho) - 1) < 1e-10
                    cn = cut_neg(rho); cc = pair_conc(rho)
                    if cn > mx: mx, arg = cn, n
                    mc = max(mc, cc)
                rows.append(dict(net=net.upper(), contact=con.upper(), site=vname, eta=en, max_cut_neg=mx, at_n=arg, max_conc=mc))
json.dump(rows, open(os.path.join(HERE, "report", "data", "pairs_table.json"), "w"), indent=1)
print("network contact | max over site, eta of the largest cut negativity (best eta) | all-eta zero?")
L = [r"\begin{small}\begin{tabular}{@{}llrrrrl@{}}\toprule", r"network & contact & \multicolumn{4}{c}{largest cut negativity (end / middle contact)} & plan Sec.~3.3\\", r" & & $|0\rangle$ & $|1\rangle$ & $|+\rangle$ & $\mathbb 1/2$ & \\\midrule"]
PLAN = {("XX", "ZZ"): "Setting I", ("EX", "EX"): "Setting II", ("EX", "XX"): "Setting II", ("XX", "EX"): "optional", ("XX", "XX"): "No (free XX only)", ("ZZ", "XX"): "No", ("ZZ", "EX"): "No", ("ZZ", "ZZ"): "No", ("EX", "ZZ"): "No"}
for net in ("XX", "ZZ", "EX"):
    for con in ("ZZ", "XX", "EX"):
        sub = [r for r in rows if r["net"] == net and r["contact"] == con]; cells = []
        for en in ETA: cells.append(" / ".join(f"{[r for r in sub if r['eta'] == en and r['site'] == s][0]['max_cut_neg']:.3g}" for s in ("end", "middle")))
        L.append(f"{net} & {con} & " + " & ".join(cells) + f" & {PLAN[(net, con)]}\\\\")
        zero = max(r["max_cut_neg"] for r in sub) < 1e-12; print(net, con, f"max {max(r['max_cut_neg'] for r in sub):.3e}", "all zero" if zero else "active", "| plan:", PLAN[(net, con)])
L.append(r"\bottomrule\end{tabular}\end{small}"); open(os.path.join(HERE, "report", "tables", "pairs_table.tex"), "w").write("\n".join(L) + "\n")
