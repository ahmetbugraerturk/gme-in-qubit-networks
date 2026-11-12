#!/usr/bin/env python3
"""Items of the research plan that are not part of the basic run (run after run_xx.py and witness_certs.py; CPU, about a minute per graph):
  * stores psi(t) on the full grid (<graph>.psi.npz): the plan asks for the raw states; every density matrix follows by partial trace
  * state checks (plan Task 1): norm, trace, Hermiticity and positivity of every reduced state (<graph>.extras.json)
  * graph-state fidelity targets for the whole state and the triples (plan Sec. 4.6/4.7): graph states of G and of its connected subgraphs, two local frames, alpha = 1/2
  * quantum Fisher information with the k-producibility bounds (Sec. 4.9), whole state (pure) and triples
  * per-subset status (Sec. 4.7), reporting label (Sec. 4.10) and entanglement-depth lower bounds, versus time
Writes results/<dir>/<graph>.extras.npz (+ .extras.json, .psi.npz).  Usage: python plan_extras.py [--results results/xx_vacuum]"""
import argparse, itertools, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, extras as E, store
LET = "abcd"; TAU = 1e-6; EPS_ENT = 1e-12; EPS_PROD = 1e-10
ap = argparse.ArgumentParser(); ap.add_argument("--results", default=os.path.join(HERE, "results", "xx_vacuum")); ap.add_argument("--only", default=None); a = ap.parse_args()
T4 = E.graph_targets(4); T3 = E.graph_targets(3)
M4 = np.stack([v for v, _ in T4.values()], 1); N4 = list(T4); A4 = np.array([al for _, al in T4.values()])
M3 = np.stack([v for v, _ in T3.values()], 1); N3 = list(T3); A3 = np.array([al for _, al in T3.values()])
def idx(s): return [LET.index(c) for c in s]

def product_across(rho, sub_n, cut):
    """max over t of || rho - rho_A (x) rho_B ||_max for the cut A|B of the sub_n-qubit states rho (T,d,d)."""
    A = cut; Bq = [q for q in range(sub_n) if q not in A]; T = rho.shape[0]
    def red(keep):
        t = rho.reshape((T,) + (2,) * (2 * sub_n)); drop = [q for q in range(sub_n) if q not in keep]
        for c, q in enumerate(sorted(drop, reverse=True)): t = np.trace(t, axis1=1 + q, axis2=1 + q + (sub_n - c))
        return t.reshape(T, 2 ** len(keep), 2 ** len(keep))
    rA, rB = red(A), red(Bq)
    # rho in qubit order 0..n-1; build rA (x) rB in the order A then B and permute back
    full = np.einsum("tij,tkl->tikjl", rA, rB).reshape(T, 2 ** sub_n, 2 ** sub_n)
    perm = list(A) + Bq; inv = np.argsort(perm)
    r = full.reshape((T,) + (2,) * (2 * sub_n)); r = np.transpose(r, [0] + [1 + p for p in inv] + [1 + sub_n + p for p in inv]).reshape(T, 2 ** sub_n, 2 ** sub_n)
    return np.abs(rho - r).max(axis=(1, 2))

for name, R in store.load_all(a.results).items():
    if a.only and name != a.only: continue
    edges = [tuple(map(int, e)) for e in R.edges]; t = R.t; Tn = len(t)
    psi = X.evolution(np, edges, 4, t)                                           # (16, T)
    np.savez_compressed(os.path.join(a.results, name + ".psi.npz"), t=t, psi=psi.T, edges=np.array(edges), note="psi(t) = exp(-i t sum_{(i,j) in E} X_i X_j)|0000>, qubit 0 = a = leftmost tensor factor; J = 1")
    out = {}; info = dict(graph=name, checks={})
    # ---- state checks on every reduced state
    worst = dict(norm=float(np.abs(np.linalg.norm(psi, axis=0) - 1).max()), trace=0.0, herm=0.0, mineig=0.0)
    for k in (1, 2, 3):
        for sub in itertools.combinations(range(4), k):
            rho = C.ptrace_pure(np, psi, 4, list(sub)); worst["trace"] = max(worst["trace"], float(np.abs(np.trace(rho, axis1=1, axis2=2) - 1).max()))
            worst["herm"] = max(worst["herm"], float(np.abs(rho - np.conj(np.swapaxes(rho, 1, 2))).max())); worst["mineig"] = min(worst["mineig"], float(np.linalg.eigvalsh(rho)[:, 0].min()))
    info["checks"] = worst
    # ---- entanglement of pairs, triple cuts, whole cuts
    pairs = sorted(set(R.meta["pair_map"].values())); triples = sorted(set(R.meta["triple_map"].values()))
    pair_ent = {p: R.pair(p, "N") > EPS_ENT for p in pairs}
    anypair = np.zeros(Tn, bool)
    for p in pairs: anypair |= pair_ent[p]
    qcuts = sorted(set(R.meta["qcut_map"].values())); anycut4 = np.zeros(Tn, bool)
    for c in qcuts: anycut4 |= R.z[f"qcut/{c}/N"] > EPS_ENT
    # ---- per triple orbit: status, QFI, graph-state fidelity
    tri_gme = np.zeros(Tn, bool); anytri_ent = np.zeros(Tn, bool); depth_q3 = np.ones(Tn, int)
    for rep in triples:
        g = R.triple_gmn(rep); w = R.cert_triple(rep); gme = (g["cls"] == 1) | ((w > 0) if w is not None else False)
        rho = C.ptrace_pure(np, psi, 4, idx(rep)); cutN = [R.tcut_N(rep, v) for v in rep]
        inner = np.zeros(Tn, bool)
        for pr in itertools.combinations(rep, 2): inner |= R.pair("".join(pr), "N") > EPS_ENT
        other = inner | np.any(np.stack([c > EPS_ENT for c in cutN]), 0)
        prod = np.zeros(Tn, bool)
        for A in ([0], [1], [2]): prod |= product_across(rho, 3, A) < EPS_PROD
        st = np.where(gme, 3, np.where(other, 2, np.where(prod, 1, 0))).astype(np.int8)
        out[f"status/tri/{rep}"] = st; tri_gme |= gme; anytri_ent |= other | gme
        F3 = E.qfi_matrix(rho, 3); q3 = E.qfi_max(F3, 3); out[f"qfi/tri/{rep}"] = q3; depth_q3 = np.maximum(depth_q3, E.depth_from_qfi(q3, 3))
        gf = np.real(np.einsum("dn,tde,en->nt", np.conj(M3), rho, M3)); out[f"gfid/tri/{rep}"] = gf.max(0); info[f"gfid_best_tri_{rep}"] = N3[int(gf.max(1).argmax())]
        out[f"gfid_cert/tri/{rep}"] = (gf.max(0) > 0.5 + 1e-9).astype(np.int8)
    for p in pairs: out[f"status/pair/{p}"] = pair_ent[p].astype(np.int8)
    # ---- whole state (pure): exact GME test, status, QFI, graph-state fidelity
    cg = R.cgme2; gme4_pure = cg > EPS_ENT
    gm4 = R.gmn4(); wh = R.cert_whole(); gme4_gmn = (gm4["cls"] == 1) if gm4 is not None else np.zeros(Tn, bool); gme4_fid = (wh > 0) if wh is not None else np.zeros(Tn, bool)
    info["whole_gme_pure_not_gmn"] = int((gme4_pure & ~gme4_gmn).sum()); info["whole_gmn_not_pure"] = int((gme4_gmn & ~gme4_pure).sum()); info["whole_gme_pure_not_fid_when_fid"] = int((gme4_fid & ~gme4_pure).sum())
    rho4 = psi.T[:, :, None] * np.conj(psi.T)[:, None, :]
    prod4 = np.zeros(Tn, bool)
    for A in ([0], [1], [2], [3], [0, 1], [0, 2], [0, 3]): prod4 |= np.abs(C.pure_cut(np, psi, 4, A)[0] - 1) < EPS_PROD
    st4 = np.where(gme4_pure, 3, np.where(prod4, 1, 0)).astype(np.int8); out["status/whole"] = st4
    F4 = E.qfi_matrix(rho4, 4); q4 = E.qfi_max(F4, 4); out["qfi/whole"] = q4; dq4 = E.depth_from_qfi(q4, 4)
    gf4 = np.abs(np.conj(M4).T @ psi) ** 2; out["gfid/whole"] = gf4.max(0); out["gfid_cert/whole"] = (gf4.max(0) > 0.5 + 1e-9).astype(np.int8)
    ipi = int(np.argmin(np.abs(t - math.pi / 4))); info["gfid_best_whole_at_pi4"] = N4[int(gf4[:, ipi].argmax())]; info["gfid_whole_at_pi4"] = float(gf4[:, ipi].max()); info["gfid_whole_max"] = float(gf4.max())
    info["gfid_best_whole_overall"] = N4[int(gf4.max(1).argmax())]
    # ---- label (plan Sec. 4.10) and depth
    lab = np.where(gme4_pure | gme4_gmn | gme4_fid, 4, np.where(tri_gme, 3, np.where(anypair | anytri_ent | anycut4, 2, 0))).astype(np.int8); out["label"] = lab
    dsub = np.where(lab == 4, 4, np.where(lab == 3, 3, np.where(lab == 2, 2, 1))); out["depth_subsets"] = dsub.astype(np.int8); out["depth_qfi"] = np.maximum(dq4, depth_q3).astype(np.int8)
    out["depth"] = np.maximum(dsub, np.maximum(dq4, depth_q3)).astype(np.int8)
    info["qfi_whole_max"] = float(q4.max()); info["qfi_whole_tmax_over_pi"] = float(t[int(q4.argmax())] / math.pi)
    np.savez_compressed(os.path.join(a.results, name + ".extras.npz"), t=t, **out); json.dump(info, open(os.path.join(a.results, name + ".extras.json"), "w"), indent=1)
    print(f"{name:8s} norm {worst['norm']:.1e} trace {worst['trace']:.1e} herm {worst['herm']:.1e} min eig {worst['mineig']:.1e} | labels 4q/D3/D2/none = " + "/".join(str(int((lab == v).sum())) for v in (4, 3, 2, 0)) + f" | QFI max {q4.max():.3f} | graph-state F at pi/4 {info['gfid_whole_at_pi4']:.6f} ({info['gfid_best_whole_at_pi4']})", flush=True)
