#!/usr/bin/env python3
"""Direct comparison of the new XX-from-|0000> results with the earlier atlas (phase 1) results for n = 4, xx|0>:
atlas/results/4/g{idx}_xx_0.npz  versus  results/xx_vacuum/<graph>.npz.
Atlas stores: concurrence of the edge orbits, GMN of the connected triples (SDP value only where it solved, plus a class label per
time), whole-state C_GME and Meyer-Wallach Q, on a 96-point grid.  Our curves (2001 points) are interpolated onto the atlas times.
Both use graph4c.g6 in file order with the same vertex labels (the atlas run stored the graph6 string of every run, checked below), so no relabelling is needed.
(atlas.qnet.graphs.load_graphs can fall back to NetworkX's own atlas with other labels; that list is deliberately not used here.)"""
import json, os, sys
import numpy as np, networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import store, xxvac as X

SUF = os.environ.get("XX_RESULTS", "xx_vacuum"); TAG = "" if SUF == "xx_vacuum" else "_" + SUF.replace("xx_vacuum_", "")   # XX_RESULTS=xx_vacuum_cert compares the certified rerun
JSONL = os.path.join(HERE, "results", f"compare_atlas{TAG}.jsonl"); os.makedirs(os.path.dirname(JSONL), exist_ok=True); open(JSONL, "w").close()
fails = []; SEC = [""]
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    with open(JSONL, "a") as fh: fh.write(json.dumps(dict(section=SEC[0], name=name, ok=bool(cond), detail=detail)) + "\n")
    if not cond: fails.append(name)
G6 = os.path.join(ROOT, "atlas/data/graphs/graph4c.g6"); G6LINES = [l.decode() for l in open(G6, "rb").read().split()]; FILE = X.load_g6(G6)
RES = os.path.join(Q4, "results", SUF); have = {f[: -len(".meta.json")] for f in os.listdir(RES) if f.endswith(".meta.json")}
LET = "abcd"; TOL = 2e-5; TOL_UP = 2e-6; TAU = 1e-5      # TOL: interpolation onto the atlas grid; TOL_UP: SCS accuracy for the rigorous side
for idx, og in enumerate(FILE):
    name = X.graph_name(og); SEC[0] = f"atlas g{idx} = {name}"
    print(f"\n== atlas g{idx} = {name}", flush=True)
    if name not in have: print("   not yet in the new store: skipped"); continue
    d = np.load(os.path.join(ROOT, f"atlas/results/4/g{idx}_xx_0.npz"), allow_pickle=True); R = store.load(os.path.join(RES, name)); ta = d["times"]; ip = lambda a: np.interp(ta, R.t, a)
    report(f"atlas run used the same graph ({str(d['meta_graph6'])} = file line {idx + 1})", str(d["meta_graph6"]) == G6LINES[idx], f"{str(d['meta_graph6'])} vs {G6LINES[idx]}")
    lab = lambda vs: "".join(sorted(LET[int(v)] for v in vs))
    wc = 0.0
    for k, rep in enumerate(d["orbit_rep"]): wc = max(wc, float(np.abs(d["orbit_C"][k] - ip(R.pair(lab(rep)))).max()))
    report(f"edge-orbit concurrence ({len(d['orbit_rep'])} orbits, {len(ta)} times)", wc < TOL, f"max |difference| {wc:.1e}")
    dq = float(np.abs(d["whole_meyer_wallach_Q"] - ip(R.Q)).max()); report("whole-state Meyer-Wallach Q", dq < TOL, f"max |difference| {dq:.1e}")
    dc = float(np.abs(d["whole_c_gme"] - np.sqrt(np.clip(ip(R.cgme2), 0, None))).max()); report("whole-state C_GME (atlas) vs sqrt(C_GME^2) (new)", dc < 1e-4, f"max |difference| {dc:.1e} (sqrt amplifies the 1e-8 noise near zero)")
    up = 0.0; lo = 0.0; nsolved = 0; nbad = 0; zero_bad = 0; nzero = 0; gs_bad = 0; ngs = 0
    for k, tri in enumerate(d["triples"]):
        gm = R.triple_gmn(lab(tri))
        if gm is None: continue
        L, U = ip(gm["L"]), ip(gm["U"]); bad = ip(gm["bad"].astype(float)) > 0; ga = d["triple_gmn"][k]; sm = d["triple_solved_mask"][k].astype(bool); lb = d["triple_labels"][k]
        if sm.any():
            up = max(up, float(np.max(np.maximum(0, ga[sm] - U[sm]))))                                    # rigorous side: atlas <= our certified U
            ok_ = sm & ~bad; lo = max(lo, float(np.max(np.maximum(0, L[ok_] - TOL - ga[ok_]))) if ok_.any() else 0.0)  # L is a primal estimate: compared only where the bracket is consistent
            nbad += int((sm & bad).sum())
        nsolved += int(sm.sum())
        z = (lb == 0) | (lb == 5); zero_bad += int((U[z] > TAU).sum()); nzero += int(z.sum())            # atlas: exact zero (PPT or bound)
        gsm = lb == 1; gs_bad += int((L[gsm] <= 0).sum()); ngs += int(gsm.sum())                          # atlas: Guhne-Seevinck certified positive
    report(f"GMN, rigorous side: every atlas SDP value is <= our certified U up to SCS accuracy ({nsolved} solved points)", up < TOL_UP, f"largest excess of the atlas value over U {up:.1e} (tolerance {TOL_UP:.0e})" + ("" if nsolved else " (no point needed an SDP in the atlas run)"))
    report(f"GMN, estimate side: atlas SDP value >= our L where our bracket is consistent", lo < 5e-5, f"largest excess of L {lo:.1e}; {nbad} of {nsolved} solved points have an unconverged run with L > U (L clipped to U, never classed positive)")
    report(f"GMN: atlas exact zeros (PPT/bound, {nzero} points) have our U <= 1e-5", zero_bad == 0, f"{zero_bad} violations")
    report(f"GMN: atlas Guhne-Seevinck positives ({ngs} points) have our L > 0", gs_bad == 0, f"{gs_bad} violations")
print("\nOVERALL:", "PASS" if not fails else "FAIL: " + "; ".join(fails)); sys.exit(0 if not fails else 1)
