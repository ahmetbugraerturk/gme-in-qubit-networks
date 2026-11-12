#!/usr/bin/env python3
"""Figure for one contact run: collision model (J tau = 0.4, dots) against the continuous retained-ancilla evolution (line) and the free control (grey), for pairs, triples and the whole state.
Usage: python plot_contact.py --graph path --site b --g 20 --t0 0 [--tmax 20] [--out figures/contact]"""
import argparse, glob, json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--graph", required=True); ap.add_argument("--site", required=True); ap.add_argument("--g", type=float, required=True); ap.add_argument("--t0", default="0")
ap.add_argument("--tau", default="0.4"); ap.add_argument("--tmax", type=float, default=20.0); ap.add_argument("--root", default=os.path.join(HERE, "results", "contact")); ap.add_argument("--out", default=os.path.join(HERE, "figures", "contact")); a = ap.parse_args()
tn = "0" if a.t0 == "0" else "pi_4"; stem = f"p_g{a.g:g}_t0{tn}"; sd = os.path.join(a.root, a.graph, a.site); fd = os.path.join(a.root, a.graph)
L = lambda p: np.load(p)
col = L(os.path.join(sd, f"{stem}_coll_tau{a.tau}.npz")); con = L(os.path.join(sd, f"{stem}_cont.npz")); fcol = L(os.path.join(fd, f"free_t0{tn}_coll_tau{a.tau}.npz")); fcon = L(os.path.join(fd, f"free_t0{tn}_cont.npz"))
m = json.loads(str(col["meta"])); tau = float(a.tau); n = np.arange(len(col["purity"])); tc = n * tau; t = con["t"]; di = con["deep_idx"]; td = t[di]
mc = json.loads(str(con["meta"]))
fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.2, 1.0, 0.35]))
pairs = sorted({k.split("/")[1] for k in col.files if k.startswith("pair/") and k.endswith("/C")}); cm = plt.cm.tab10
pm = m["pair_map"]; eq = {}
for k, r in pm.items(): eq.setdefault(r, []).append(k)
for i, p in enumerate(pairs):
    ax[0].plot(t, con[f"pair/{p}/C"], color=cm(i), lw=1.0, label=" = ".join(eq[p])); ax[0].plot(tc, col[f"pair/{p}/C"], "o", color=cm(i), ms=3)
ax[0].set_ylabel("pair concurrence C"); ax[0].legend(fontsize=7, ncol=3)
tris = sorted({k.split("/")[2] for k in col.files if k.startswith("deep/tri/") and k.endswith("/L")}); tm = m["triple_map"]; eqt = {}
for k, r in tm.items(): eqt.setdefault(r, []).append(k)
for i, p in enumerate(tris):
    ax[1].plot(td, con[f"deep/tri/{p}/L"], color=cm(i), lw=1.0, label=" = ".join(eqt[p])); ax[1].plot(tc, col[f"deep/tri/{p}/L"], "o", color=cm(i), ms=3)
    if f"deep/tri/{p}/L" in fcon.files: ax[1].plot(td, fcon[f"deep/tri/{p}/L"], color="grey", lw=0.6, alpha=0.6)
ax[1].set_ylabel("triple GMN (certified L)"); ax[1].legend(fontsize=7, ncol=3)
ax[2].plot(td, con["deep/gmn4/L"], color="C3", lw=1.0, label="whole state, continuous"); ax[2].plot(tc, col["deep/gmn4/L"], "o", color="C3", ms=3, label="collisions"); ax[2].plot(td, 0 * td, alpha=0)
ax[2].set_ylabel("whole-state GMN (L)"); ax[2].legend(fontsize=7)
lab = col["deep/label"]; cols = {4: "#2a78d6", 3: "#1baf7a", 2: "#eda100", 0: "#bbbbbb"}
for k, c in zip(tc, lab): ax[3].axvspan(k - tau / 2, k + tau / 2, color=cols[int(c)], lw=0)
ax[3].set_yticks([]); ax[3].set_ylabel("label"); ax[3].set_xlabel("Jt after the first contact (collision n at Jt = n tau)")
for x in ax[:3]: x.set_xlim(0, a.tmax); x.grid(alpha=0.3)
ax[0].set_title(f"{a.graph}, contact at {a.site}, ancilla |+>, g/J = {a.g:g}, t0 = {a.t0}, J tau = {a.tau}  (blue strip: 4-qubit GME, green: depth 3, amber: depth 2)", fontsize=9)
os.makedirs(a.out, exist_ok=True); p = os.path.join(a.out, f"contact_{a.graph}_{a.site}_g{a.g:g}_t0{tn}.png"); fig.tight_layout(); fig.savefig(p, dpi=110); print("wrote", p)
