#!/usr/bin/env python3
"""GMN of the collision model BETWEEN collisions (ancilla |+>, g/J = 20, J tau = 0.4).  During collision k the network state is rho(k tau + t') = 1/2 sum_s U_s(t') rho_k U_s(t')^dag
(average over the field sign of the running collision).  Evaluated every --step (0.1) in Jt, i.e. 4 states per collision: 801 states from Jt = 0 to 80 (every fourth is a post-collision state).
For each state: GMN of the triples and of the whole (mixed) state, witnesses, QFI, labels (qgpu.contact_analysis.deep).  Output: <run>_collgmn.npz next to the collision runs; existing files are skipped.
One task = all contact sites in turn (use --task K for one site); restartable."""
import argparse, glob, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, contact as K, contact_analysis as CA
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(HERE, "results", "contact")); ap.add_argument("--g", type=float, default=20.0); ap.add_argument("--tau", default="0.4"); ap.add_argument("--step", type=float, default=0.1)
ap.add_argument("--task", type=int, default=None); ap.add_argument("--n", type=int, default=None, help="only the first n collisions (test)"); ap.add_argument("--device", default="gpu"); ap.add_argument("--graphs", default=os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); a = ap.parse_args()
xp = C.get_xp(a.device); log = lambda s: print(s, flush=True)
TASKS = [(X.graph_name(g), g, v) for g in X.load_g6(a.graphs) for v, orb in K.contact_sites(g)]; tau = float(a.tau); sub = int(round(tau / a.step)); T0 = {"0": 0.0, "pi_4": math.pi / 4}
for ti, (nm, G, v) in enumerate(TASKS):
    if a.task is not None and ti != a.task: continue
    site = "abcd"[v]; edges = sorted(G.edges()); stab = K.stabilizer(X.automorphisms(G), v); d = os.path.join(a.root, nm, site)
    items = []
    for t0n in T0:
        src = os.path.join(d, f"p_g{a.g:g}_t0{t0n}_coll_tau{a.tau}.npz"); out = os.path.join(d, f"p_g{a.g:g}_t0{t0n}_collgmn.npz")
        if not os.path.exists(src) or os.path.exists(out): continue
        z = np.load(src); m = json.loads(str(z["meta"])); rho = z["rho"]; nmax = len(rho) - 1 if a.n is None else a.n
        Hp, Hm = K.fields(edges, v, a.g); ep, Vp = K.eig(Hp); em, Vm = K.eig(Hm); tp = a.step * np.arange(sub)
        Up = np.stack([(Vp * np.exp(-1j * ep * t)) @ Vp.conj().T for t in tp]); Um = np.stack([(Vm * np.exp(-1j * em * t)) @ Vm.conj().T for t in tp])
        st = []
        for k in range(nmax): st.append(0.5 * (Up @ rho[k] @ Up.conj().transpose(0, 2, 1) + Um @ rho[k] @ Um.conj().transpose(0, 2, 1)))          # t' = 0, step, .. < tau (t' = 0 is rho_k itself)
        st = np.concatenate(st + [rho[nmax][None]]); items.append((out, st, m))
    if not items: continue
    t = time.perf_counter(); log(f"=== {nm}, contact {site}: {sum(len(s) for _, s, _ in items)} states"); allst = np.concatenate([s for _, s, _ in items]); deep = CA.deep(xp, xp.asarray(allst), stab, False, log=log); off = 0
    for out, st, m in items:
        n = len(st); res = {"deep/" + k: v[off:off + n] for k, v in deep.items()}; res["t"] = a.step * np.arange(n); off += n
        np.savez_compressed(out, meta=json.dumps(dict(m, kind="collision_between", step=a.step)), **res); log(f"  wrote {os.path.relpath(out, a.root)}")
    log(f"  {time.perf_counter() - t:.1f}s")
log("DONE")
