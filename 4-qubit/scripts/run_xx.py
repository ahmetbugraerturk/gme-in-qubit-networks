#!/usr/bin/env python3
"""Run the XX-from-|0000> study for all graphs of graph4c.g6.  Writes, per graph, results/xx_vacuum/<name>.npz + <name>.meta.json (see qgpu/store.py: full grid, plain arrays, loadable with
qgpu.store.load) and <name>.csv (table view of the mirrored curves).  Plots and the artifact are built from the store only."""
import argparse, json, math, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Q4 = HERE; ROOT = os.path.dirname(Q4)   # HERE = 4-qubit
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
from qgpu import core as C, xxvac as X, store

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"]); ap.add_argument("--n-intervals", type=int, default=2000)
ap.add_argument("--no-gmn", action="store_true"); ap.add_argument("--out", default=os.path.join(HERE, "results", "xx_vacuum"))
ap.add_argument("--graphs", default=os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); ap.add_argument("--only", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
xp = C.get_xp(a.device)

def rnd(v, nd=5):
    v = np.asarray(v, float); v = np.where(np.abs(v) < 1e-12, 0.0, v); return [float(f"{x:.{nd}g}") for x in v]
def gm(d, keys=("L", "U")):
    return None if d is None else dict(L=rnd(d["L"][::5]), U=rnd(d["U"][::5]), cls=[int(x) for x in d["cls"][::5]], conv=[bool(x) for x in d["conv"][::5]])

for g in X.load_g6(a.graphs):
    name = X.graph_name(g)
    if a.only and name != a.only: continue
    print(f"\n=== {name}  edges {[X.label(e) for e in sorted(g.edges())]}  |Aut|={len(X.automorphisms(g))}", flush=True)
    t0 = time.time(); r = X.analyze(g, xp, n_intervals=a.n_intervals, log=lambda s: print(s, flush=True), gmn_triples=not a.no_gmn, gmn_whole=not a.no_gmn)
    print(f"  analysis time {time.time()-t0:.1f}s; concurrence round-off {r['conc_roundoff']:.1e} (plan: flag > 1e-12)", flush=True)
    store.save(r, a.out, params=dict(n_intervals=a.n_intervals, gmn_tol=1e-6, gmn_rel_gap=X.REL_GAP, gmn_abs_gap=X.gmn_curve.__defaults__[2], gmn_admm_tuned={str(k): v for k, v in X.TUNED.items()}, device=a.device, solver="batched ADMM (qgpu.gmn_admm defaults)"))
    # table view: mirrored curves, every 5th point of the full grid
    cols = {"Jt": X.mirror(r["th"])}
    for k, v in r["pair"].items(): cols[f"C[{X.label(k)}]"] = X.mirror(v["C"])
    for k, v in r["tcut"].items(): cols[f"N[{X.label(k[0])}|{X.LET[k[1]]}]"] = X.mirror(v)
    for k, v in r["qcut"].items(): cols[f"N[{X.cut_label(k)}]"] = X.mirror(v["N"])
    cols["Q"] = X.mirror(r["Q"]); cols["C_GME^2"] = X.mirror(r["cgme2"])
    for k, v in r["triple_gmn"].items():
        if v: cols[f"GMN_L[{X.label(k)}]"] = X.mirror(v["L"]); cols[f"GMN_U[{X.label(k)}]"] = X.mirror(v["U"])
    if r["gmn4"]: cols["GMN_L[abcd]"] = X.mirror(r["gmn4"]["L"]); cols["GMN_U[abcd]"] = X.mirror(r["gmn4"]["U"])
    names = list(cols); M = np.stack([cols[k] for k in names], 1)[::5]
    np.savetxt(os.path.join(a.out, f"{name}.csv"), M, delimiter=",", header=",".join(names), comments="", fmt="%.6g")
    if r["gmn4"] is not None:
        c = r["gmn4"]["cls"]; print(f"  whole-state GMN classes over [0,pi/4]: zero {(c==0).sum()}  positive {(c==1).sum()}  near-threshold {(c==2).sum()}; "
                                    f"max GMN4 {r['gmn4']['L'].max():.4f}", flush=True)
print("\nDONE", flush=True)
