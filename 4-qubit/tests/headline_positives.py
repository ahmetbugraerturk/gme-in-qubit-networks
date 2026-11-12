#!/usr/bin/env python3
"""Headline check (GMN solver alone, and with the witness certificates if results/<dir>/<graph>.cert.npz exists): of the atlas-certified positive triple points (label 1 = Guhne-Seevinck, label 2 = SDP positive), how many do we classify POSITIVE?
Usage: XX_RESULTS=xx_vacuum_cert python tests/headline_positives.py   (default results/xx_vacuum).  Also lists every point we miss."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4)
sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import store, xxvac as X
SUF = os.environ.get("XX_RESULTS", "xx_vacuum"); RES = os.path.join(Q4, "results", SUF); LET = "abcd"
FILE = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); tot = pos = posw = 0; miss = []
for idx, og in enumerate(FILE):
    name = X.graph_name(og); d = np.load(os.path.join(ROOT, f"atlas/results/4/g{idx}_xx_0.npz"), allow_pickle=True); R = store.load(os.path.join(RES, name)); ta = d["times"]
    for k, tri in enumerate(d["triples"]):
        gm = R.triple_gmn("".join(sorted(LET[int(v)] for v in tri)))
        if gm is None: continue
        j = np.array([int(np.argmin(np.abs(R.t - t))) for t in ta]); cls = gm["cls"][j]; lb = d["triple_labels"][k]
        lbl = "".join(sorted(LET[int(v)] for v in tri)); ct = R.cert_triple(lbl); wit = (ct[j] > 0) if ct is not None else np.zeros(len(j), bool)
        for kind, lab in (("GS", 1), ("SDP", 2)):
            m = lb == lab; n = int(m.sum()); p = int((cls[m] == 1).sum()); tot += n; pos += p; posw += int(((cls == 1) | wit)[m].sum())
            print(f"{name:8s} triple {''.join(sorted(LET[int(v)] for v in tri))} atlas {kind:3s}: {p}/{n} classed positive")
            for q in np.nonzero(m & (cls != 1) & ~wit)[0]: miss.append((name, kind, ta[q], float(gm["L"][j[q]]), float(gm["U"][j[q]]), bool(gm["conv"][j[q]])))
print(f"\n{SUF}: GMN solver alone {pos}/{tot} = {100*pos/tot:.1f}%;  with the witness certificates (GS / fidelity, results/*.cert.npz) {posw}/{tot} = {100*posw/tot:.1f}%  of the atlas-certified positives are certified positive")
for m in miss[:40]: print("  missed", m[0], m[1], f"t={m[2]:.4f} L={m[3]:.3e} U={m[4]:.3e} conv={m[5]}")
if len(miss) > 40: print(f"  ... {len(miss)-40} more")
