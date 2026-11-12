#!/usr/bin/env python3
"""Pair concurrence of the collision model at ALL times, not only at the collision ends.  During collision k (interaction time tau) the network has evolved under H_G + s g Z_v for the time t' < tau with the
random sign s of that collision, so the reduced network state at k tau + t' is  rho(k tau + t') = 1/2 sum_s U_s(t') rho_k U_s(t')^dag  (rho_k: state before collision k; the ancilla is traced out).  The fine series ends in
rho_(k+1) at t' = tau, which is the stored post-collision state (checked).  Writes <run>_collfine.npz next to each |+> collision run (J tau = 0.4):  t, and  pair/<rep>/C  on the grid dt = 0.01 (8001 points)."""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, contact as K
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(HERE, "results", "contact")); ap.add_argument("--dt", type=float, default=0.01); ap.add_argument("--tau", default="0.4"); a = ap.parse_args()
n = 0; worst = 0.0
for f in sorted(glob.glob(os.path.join(a.root, "*", "*", f"p_*_coll_tau{a.tau}.npz"))):
    out = f.replace(f"_coll_tau{a.tau}.npz", "_collfine.npz")
    if os.path.exists(out): continue
    z = np.load(f); m = json.loads(str(z["meta"])); edges = [tuple(e) for e in m["edges"]]; v = "abcd".index(m["site"]); g = m["g"]; tau = float(m["tau"]); rho = z["rho"]
    Hp, Hm = K.fields(edges, v, g); ep, Vp = K.eig(Hp); em, Vm = K.eig(Hm); sub = int(round(tau / a.dt)); tp = a.dt * np.arange(1, sub + 1)
    Up = np.stack([(Vp * np.exp(-1j * ep * t)) @ Vp.conj().T for t in tp]); Um = np.stack([(Vm * np.exp(-1j * em * t)) @ Vm.conj().T for t in tp])      # (sub, 16, 16)
    nmax = len(rho) - 1; fine = np.empty((nmax * sub + 1, 16, 16), complex); fine[0] = rho[0]
    for k in range(nmax):
        fine[k * sub + 1:(k + 1) * sub + 1] = 0.5 * (Up @ rho[k] @ Up.conj().transpose(0, 2, 1) + Um @ rho[k] @ Um.conj().transpose(0, 2, 1))
    worst = max(worst, float(np.abs(fine[sub::sub] - rho[1:]).max()))                      # fine series ends in the stored post-collision states
    res = {"t": a.dt * np.arange(len(fine))}
    for rep in sorted({k.split("/")[1] for k in z.files if k.startswith("pair/") and k.endswith("/C")}):
        i, j = ["abcd".index(ch) for ch in rep]; c, _ = C.concurrence_checked(np, K.ptrace_rho(np, fine, 4, sorted([i, j]))); res[f"pair/{rep}/C"] = c
        if np.abs(c[sub::sub] - z[f"pair/{rep}/C"][1:]).max() > 1e-6: print("MISMATCH", f, rep)
    np.savez_compressed(out, meta=json.dumps(m), **res); n += 1
print(f"{n} files written; max |fine state at collision ends - stored states| = {worst:.1e}")
