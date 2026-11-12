#!/usr/bin/env python3
"""Text summary of the contact study (results/contact): per (graph, site, ancilla, g, t0) collision run at J tau = 0.4 against the free (no contact) control, and the check of the plan's
fast-cluster prediction (Sec. 5.4 / 6.1): fidelity of every component S of G - v with the graph state of G[S] at n = 2.  Usage: python contact_summary.py [--root results/contact] [--tau 0.4]"""
import argparse, glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import networkx as nx
from qgpu import core as C, xxvac as X, contact as K, extras as E
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(HERE, "results", "contact")); ap.add_argument("--tau", default="0.4"); a = ap.parse_args()
def load(p): z = np.load(p); return z, json.loads(str(z["meta"]))
def peak_pairs(z): return {k.split("/")[1]: float(z[k][1:].max()) for k in z.files if k.startswith("pair/") and k.endswith("/C")}
def gmn_peak(z):
    tri = {k.split("/")[2]: float(np.nanmax(z[k][1:])) for k in z.files if k.startswith("deep/tri/") and k.endswith("/L")}
    return tri, (float(np.nanmax(z["deep/gmn4/L"][1:])) if "deep/gmn4/L" in z.files else None)
def comp_fid(z, m, n_idx=2):
    """fidelity of each component S of G - v (|S| >= 2) with the best graph state of G[S] (two frames) at collision number n_idx."""
    G = nx.Graph(); G.add_nodes_from(range(4)); G.add_edges_from(m["edges"]); v = "abcd".index(m["site"]); H = G.copy(); H.remove_node(v); out = {}
    r = z["rho"][n_idx]; r = np.outer(r, r.conj()) if r.ndim == 1 else r
    for comp in nx.connected_components(H):
        S = sorted(comp)
        if len(S) < 2: continue
        rs = K.ptrace_rho(np, r[None], 4, S)[0]; k = len(S); best = 0.0
        for fr in ("x", "dyn"):
            idx = {q: i for i, q in enumerate(S)}; Es = [(idx[i], idx[j]) for i, j in G.subgraph(S).edges()]
            tv = E.graph_state(k, Es, fr); best = max(best, float(np.real(tv.conj() @ rs @ tv)))
        out["".join("abcd"[q] for q in S)] = best
    return out
rows = []
for gdir in sorted(glob.glob(os.path.join(a.root, "*"))):
    gname = os.path.basename(gdir)
    for sdir in sorted(d for d in glob.glob(os.path.join(gdir, "*")) if os.path.isdir(d)):
        for p in sorted(glob.glob(os.path.join(sdir, f"*_coll_tau{a.tau}.npz"))):
            z, m = load(p); lab = z["deep/label"][1:]; fr = [float((lab == k).mean()) for k in (4, 3, 2, 0)]
            tri, w4 = gmn_peak(z)
            cf = comp_fid(z, m)
            print(f"{gname:8s} {m['site']} anc {m['ancilla']} g {m['g']:>4g} t0 {'0' if m['t0'] == 0 else 'pi/4'}: labels 4q/D3/D2/none {fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}/{fr[3]:.2f}  peak triple GMN L " + ", ".join(f"{k}:{v:.3f}" for k, v in tri.items()) + (f"  whole GMN L {w4:.3f}" if w4 is not None else f"  whole C_GME^2 max {float(z['cgme2'][1:].max()):.3f}") + "  comp. graph-state F(n=2): " + (", ".join(f"{k}:{v:.3f}" for k, v in cf.items()) or "-"))
