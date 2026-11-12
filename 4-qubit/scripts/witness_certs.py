#!/usr/bin/env python3
"""Independent positivity certificates that do NOT come from the GMN solver: for every triple orbit (3 qubits) and the whole state,
  bit 1 : Guhne-Seevinck witness (arXiv:0905.1349, GHZ and W type, 27 local bases) certifies genuine multipartite entanglement   [triples only]
  bit 2 : fidelity witness F > alpha with a GHZ / W target (alpha = 1/2 GHZ, (k-1)/k W) certifies genuine multipartite entanglement
A state certified this way has GMN > 0 (these witnesses prove GME), but no value of GMN is claimed.
Reads results/<dir>/<graph>.npz (+ .meta.json), writes results/<dir>/<graph>.cert.npz: int8 arrays on the full grid, keys tri/<rep> and gmn4 (+ tri/<rep>/gs_margin).
Usage: python witness_certs.py [--results results/xx_vacuum_cert2]   (cheap, CPU, a few seconds per graph)"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, gs, store
MARGIN = 1e-9                      # fidelities are exact closed forms (~1e-15); the margin only keeps a witness from firing on round-off
ap = argparse.ArgumentParser(); ap.add_argument("--results", default=os.path.join(HERE, "results", "xx_vacuum")); a = ap.parse_args()
for name, R in store.load_all(a.results).items():
    edges = [tuple(map(int, e)) for e in R.edges]; psi = X.evolution(np, edges, 4, R.t); out = {}; LET = "abcd"
    def fid_cert(sub):
        k = len(sub); c = np.zeros(len(R.t), bool)
        for fam in ("GHZ", "W"):
            if f"{k}:{fam}" in R.meta["witness_threshold"]: c |= R.fid_best(sub, fam) > R.threshold(k, fam) + MARGIN
        return c
    for rep in sorted(set(R.meta["triple_map"].values())):
        rho = C.ptrace_pure(np, psi, 4, [LET.index(ch) for ch in rep]); g = gs.scan_all_bases_batch(rho)
        out[f"tri/{rep}"] = (g["certified"].astype(np.int8) | (fid_cert(rep).astype(np.int8) << 1)).astype(np.int8); out[f"tri/{rep}/gs_margin"] = np.asarray(g["max_margin"], float)
    out["gmn4"] = (fid_cert("abcd").astype(np.int8) << 1).astype(np.int8)
    np.savez_compressed(os.path.join(a.results, name + ".cert.npz"), **out)
    print(f"{name:8s} " + "  ".join(f"{k[4:] if k.startswith('tri/') and '/' not in k[4:] else k}: GS {int(((v&1)>0).sum())} fid {int(((v&2)>0).sum())} either {int((v>0).sum())}/{len(v)}" for k, v in out.items() if not k.endswith("margin")))
