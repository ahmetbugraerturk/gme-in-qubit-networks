"""All quantities of the plan's toolbox (Sec. 4) for a series of four-qubit network states rho(t) (pure or mixed), with symmetry sharing under a group of
automorphisms (the stabiliser of the contact site, plan Sec. 5.1).  `cheap` is fast and run on every sample; `deep` adds the expensive items (GMN of triples and, for mixed
states, of the whole state; Guhne-Seevinck, fidelity and graph-state witnesses; QFI; status labels and depth) on the samples it is given.
rho arrays live on the device `xp` (NumPy or CuPy); outputs are host arrays in a flat dict (keys 'pair/ab/C', 'tri/abd/L', ...) plus a meta dict with the orbit maps."""
import itertools, math
import numpy as np
from . import core as C, xxvac as X, gmn_admm as A, extras as E, gs, contact as K

LET = "abcd"; EPS_ENT = 1e-12; EPS_PROD = 1e-10; TAU = 1e-6
_T3 = None; _T4 = None
def _graph_targets():
    global _T3, _T4
    if _T3 is None:
        t3 = E.graph_targets(3); t4 = E.graph_targets(4)
        _T3 = (np.stack([v for v, _ in t3.values()], 1), list(t3)); _T4 = (np.stack([v for v, _ in t4.values()], 1), list(t4))
    return _T3, _T4
def lab(s): return "".join(LET[i] for i in sorted(s))
h = C.to_host


def _orbits(autos, n=4):
    pairs = [frozenset(p) for p in itertools.combinations(range(n), 2)]; triples = [frozenset(t) for t in itertools.combinations(range(n), 3)]
    tcuts = [(t, v) for t in triples for v in sorted(t)]
    cutsA = A.bipartition_cuts(n); qcuts = [frozenset({frozenset(a), frozenset(set(range(n)) - set(a))}) for a in cutsA]
    o = {}
    o["pair_reps"], o["pair_rep"] = X.orbit_reps(pairs, autos, X.act_set); o["triple_reps"], o["triple_rep"] = X.orbit_reps(triples, autos, X.act_set)
    o["tcut_reps"], o["tcut_rep"] = X.orbit_reps(tcuts, autos, X.act_tcut); o["qcut_reps"], o["qcut_rep"] = X.orbit_reps(qcuts, autos, X.act_part)
    o["qcuts"] = qcuts; return o


def _maps(o):
    return dict(pair_map={lab(p): lab(o["pair_rep"][p]) for p in o["pair_rep"]}, triple_map={lab(t): lab(o["triple_rep"][t]) for t in o["triple_rep"]},
                tcut_map={lab(t) + "|" + LET[v]: lab(o["tcut_rep"][(t, v)][0]) + "|" + LET[o["tcut_rep"][(t, v)][1]] for (t, v) in o["tcut_rep"]},
                qcut_map={X.cut_label(c): X.cut_label(o["qcut_rep"][c]) for c in o["qcut_rep"]})


def _qcut_N(xp, rho, qc):
    a = sorted(min(qc, key=lambda s: (len(s), sorted(s)))); return C.negativity(xp, rho, 4, a)


def cheap(xp, rho, autos, psi=None):
    """pairs (C, N, best Bell fidelity), triple cut negativities, the 7 cut negativities, the 15 partitions, purity of the whole state, Q and C_GME^2 if the state is pure (psi (d,T) given)."""
    o = _orbits(autos); out = {}; T = rho.shape[0]
    for r in o["pair_reps"]:
        rp = K.ptrace_rho(xp, rho, 4, sorted(r)); c, _ = C.concurrence_checked(xp, rp); l = lab(r)
        out[f"pair/{l}/C"] = h(c); out[f"pair/{l}/N"] = h(C.negativity(xp, rp, 2, [0]))
        fb = K.fidelities_rho(xp, 2, rp); out[f"pair/{l}/Fbell"] = h(xp.max(xp.stack([v for v in fb.values()]), axis=0))
    for (t, v) in o["tcut_reps"]:
        sub = sorted(t); r3 = K.ptrace_rho(xp, rho, 4, sub); out[f"tcut/{lab(t)}|{LET[v]}/N"] = h(C.negativity(xp, r3, 3, [sub.index(v)]))
    Nq = {}
    for qc in o["qcut_reps"]: Nq[qc] = _qcut_N(xp, rho, qc); out[f"qcut/{X.cut_label(qc)}/N"] = h(Nq[qc])
    Ncut = xp.stack([Nq[o["qcut_rep"][c]] for c in o["qcuts"]], 1)
    for P in X.partitions(range(4)):
        cs = X.block_cuts(P); out["part/" + X.part_label(P)] = h(xp.max(xp.stack([Ncut[:, o["qcuts"].index(c)] for c in cs]), 0)) if cs else np.zeros(T)
    out["purity"] = h(xp.real(xp.einsum("tab,tba->t", rho, rho)))
    if psi is not None:
        out["Q"] = h(C.meyer_wallach(xp, psi, 4)); pur = [C.pure_cut(xp, psi, 4, list(a))[0] for a in A.bipartition_cuts(4)]; out["cgme2"] = np.clip(2 * (1 - h(xp.max(xp.stack(pur), 0))), 0, None)
    meta = _maps(o)
    return out, meta, o, Ncut


def _fid_cert(xp, k, rho):
    """GHZ / W fidelity witnesses: (certified bool (T,), best GHZ, best W) with alpha = 1/2 and (k-1)/k."""
    fid = K.fidelities_rho(xp, k, rho); fam = {nm: f for nm, f in [(n, X.targets(k)[n][0]) for n in X.targets(k)]}
    fam.update({nm: f for nm, f in X.extra_names(k)}); best = {}
    for nm, v in fid.items():
        f = fam[nm]; best[f] = v if f not in best else xp.maximum(best[f], v)
    cert = None
    for f, v in best.items():
        c = v > X.witness_threshold(f, k) + 1e-9 if f != "Bell" else v > 0.5 + 1e-9; cert = c if cert is None else (cert | c)
    return h(cert), h(best.get("GHZ", xp.zeros(rho.shape[0]))), h(best.get("W", xp.zeros(rho.shape[0])))


def _product_across(rho, n, A_):
    """max abs of rho - rho_A (x) rho_B for the cut A|B (host arrays)."""
    Bq = [q for q in range(n) if q not in A_]; T = rho.shape[0]
    rA = K.ptrace_rho(np, rho, n, list(A_)); rB = K.ptrace_rho(np, rho, n, Bq)
    full = np.einsum("tij,tkl->tikjl", rA, rB).reshape(T, 2 ** n, 2 ** n); perm = list(A_) + Bq; inv = np.argsort(perm)
    r = full.reshape((T,) + (2,) * (2 * n)); r = np.transpose(r, [0] + [1 + p for p in inv] + [1 + n + p for p in inv]).reshape(T, 2 ** n, 2 ** n)
    return np.abs(rho - r).max(axis=(1, 2))


def deep(xp, rho, autos, pure, log=None, whole_gmn=None):
    """Expensive items on the given samples.  whole_gmn: compute the GMN of the whole state (default: only for mixed states; for a pure state the exact test C_GME^2 > 0 decides)."""
    o = _orbits(autos); out = {}; Tn = rho.shape[0]; (M3, N3), (M4, N4) = _graph_targets(); whole_gmn = (not pure) if whole_gmn is None else whole_gmn
    rhoh = h(rho); anypair = np.zeros(Tn, bool); anytri_gme = np.zeros(Tn, bool); anyother = np.zeros(Tn, bool); depth_q = np.ones(Tn, int)
    for r in o["pair_reps"]: anypair |= h(C.negativity(xp, K.ptrace_rho(xp, rho, 4, sorted(r)), 2, [0])) > EPS_ENT
    for rep in o["triple_reps"]:
        sub = sorted(rep); l = lab(rep); r3 = K.ptrace_rho(xp, rho, 4, sub); cut = xp.stack([C.negativity(xp, r3, 3, [q]) for q in range(3)], 1)
        if log: log(f"    triple {l}: GMN")
        g = X.gmn_curve(xp, r3, 3, cut); gg = {k: h(v) for k, v in g.items()}
        for k in ("L", "U", "cls", "conv"): out[f"tri/{l}/{k}"] = gg[k]
        out[f"tri/{l}/cutN"] = h(cut)
        r3h = h(r3); gsr = gs.scan_all_bases_batch(r3h); out[f"tri/{l}/gs"] = gsr["certified"].astype(np.int8)
        fc, gh, wv = _fid_cert(xp, 3, r3); out[f"tri/{l}/fid"] = fc.astype(np.int8)
        gf = np.real(np.einsum("dn,tde,en->nt", np.conj(M3), r3h, M3)); out[f"tri/{l}/gfid"] = gf.max(0)
        q3 = E.qfi_max(E.qfi_matrix(r3h, 3), 3); out[f"tri/{l}/qfi"] = q3; depth_q = np.maximum(depth_q, E.depth_from_qfi(q3, 3))
        gme = (gg["cls"] == 1) | (out[f"tri/{l}/gs"] > 0) | fc | (gf.max(0) > 0.5 + 1e-9); out[f"tri/{l}/gme"] = gme.astype(np.int8)
        inner = np.zeros(Tn, bool)
        for pr in itertools.combinations(sub, 2): inner |= h(C.negativity(xp, K.ptrace_rho(xp, rho, 4, list(pr)), 2, [0])) > EPS_ENT
        other = inner | (h(cut) > EPS_ENT).any(1); prod = np.zeros(Tn, bool)
        for q in range(3): prod |= _product_across(r3h, 3, [q]) < EPS_PROD
        out[f"tri/{l}/status"] = np.where(gme, 3, np.where(other, 2, np.where(prod, 1, 0))).astype(np.int8)
        anytri_gme |= gme; anyother |= other
    # whole state
    Ncut = np.stack([h(_qcut_N(xp, rho, qc)) for qc in o["qcuts"]], 1); anycut = (Ncut > EPS_ENT).any(1)
    if pure:
        pur = np.max(np.stack([np.real(np.einsum("tab,tba->t", ra, ra)) for ra in (K.ptrace_rho(np, rhoh, 4, list(a_)) for a_ in A.bipartition_cuts(4))]), 0); gme4 = (1 - pur) > EPS_ENT      # exact: all cut purities < 1
        out["whole/gme_pure"] = gme4.astype(np.int8)
    else: gme4 = np.zeros(Tn, bool)
    if whole_gmn:
        if log: log("    whole 4-qubit state: GMN")
        g4 = X.gmn_curve(xp, rho, 4, xp.asarray(Ncut)); gg4 = {k: h(v) for k, v in g4.items()}
        for k in ("L", "U", "cls", "conv"): out[f"gmn4/{k}"] = gg4[k]
        gme4 = gme4 | (gg4["cls"] == 1)
    fc4, gh4, wv4 = _fid_cert(xp, 4, rho); out["whole/fid"] = fc4.astype(np.int8); out["whole/ghz"] = gh4; out["whole/w"] = wv4
    gf4 = np.real(np.einsum("dn,tde,en->nt", np.conj(M4), rhoh, M4)); out["whole/gfid"] = gf4.max(0)
    gme4 = gme4 | fc4 | (gf4.max(0) > 0.5 + 1e-9)
    q4 = E.qfi_max(E.qfi_matrix(rhoh, 4), 4); out["whole/qfi"] = q4; dq4 = E.depth_from_qfi(q4, 4)
    prod4 = np.zeros(Tn, bool)
    for a_ in A.bipartition_cuts(4): prod4 |= _product_across(rhoh, 4, list(a_)) < EPS_PROD
    out["whole/status"] = np.where(gme4, 3, np.where(anycut, 2, np.where(prod4, 1, 0))).astype(np.int8)
    label = np.where(gme4, 4, np.where(anytri_gme, 3, np.where(anypair | anyother | anycut, 2, 0))).astype(np.int8); out["label"] = label
    dsub = np.where(label == 4, 4, np.where(label == 3, 3, np.where(label == 2, 2, 1))); out["depth"] = np.maximum(dsub, np.maximum(dq4, depth_q)).astype(np.int8)
    out["depth_subsets"] = dsub.astype(np.int8); out["depth_qfi"] = np.maximum(dq4, depth_q).astype(np.int8)
    return out
