#!/usr/bin/env python3
"""Contact (ancilla) study, plan Setting I: XX network from |0000>, ZZ contact at one site, fresh ancilla |1> (coherent field -g Z_v) or |+> (random field sign), g/J, pre-contact time t0,
collision model (J tau = 0.4 and 0.1, n = 0..200) and continuous (retained ancilla) evolution, plus the free (no contact) control.  All quantities of the plan's toolbox for every subset,
symmetry-shared under the stabiliser of the contact site.  One task = one (graph, contact site); `--task K` picks the K-th of the 11 sites, `--list` prints them.
Results: results/contact/<graph>/<site>/<anc>_g<g>_t0<t0>_<mode>.npz  (mode coll_tau<tau> or cont) and results/contact/<graph>/free_t0<t0>.npz.  Existing files are skipped (restartable)."""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, contact as K, contact_analysis as CA
ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=None); ap.add_argument("--list", action="store_true"); ap.add_argument("--out", default=os.path.join(HERE, "results", "contact"))
ap.add_argument("--g", default="1,2,5,20"); ap.add_argument("--t0", default="0,pi/4"); ap.add_argument("--tau", default="0.4,0.1"); ap.add_argument("--anc", default="+")
ap.add_argument("--nmax", type=int, default=200); ap.add_argument("--cont-T", type=float, default=80.0); ap.add_argument("--cont-dt", type=float, default=0.01)
ap.add_argument("--gmn-window", type=float, default=20.0); ap.add_argument("--gmn-dt", type=float, default=0.05); ap.add_argument("--device", default="gpu"); ap.add_argument("--no-free", action="store_true")
ap.add_argument("--graphs", default=os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); ap.add_argument("--no-cont", action="store_true"); ap.add_argument("--cont-suffix", default=""); ap.add_argument("--no-coll", action="store_true")
a = ap.parse_args()
GRAPHS = X.load_g6(a.graphs); TASKS = []
for g in GRAPHS:
    for v, orb in K.contact_sites(g): TASKS.append((X.graph_name(g), g, v, orb))
if a.list:
    for i, (nm, g, v, orb) in enumerate(TASKS): print(i, nm, "contact", "abcd"[v], "orbit", "".join("abcd"[u] for u in orb), "edges", ["abcd"[i] + "abcd"[j] for i, j in sorted(g.edges())])
    sys.exit(0)
xp = C.get_xp(a.device); logf = lambda s: print(s, flush=True)
tof = lambda s: eval(s.replace("pi", "math.pi")); GS = [float(x) for x in a.g.split(",")]; T0 = [(s, tof(s)) for s in a.t0.split(",")]; TAUS = [float(x) for x in a.tau.split(",")]; ANC = a.anc.split(",")
tname = lambda s: s.replace("/", "_")
def save(path, out, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True); np.savez_compressed(path, meta=json.dumps(meta), **out)
def tgrid():
    t = np.arange(0, int(round(a.cont_T / a.cont_dt)) + 1) * a.cont_dt; deep = np.nonzero((t <= a.gmn_window + 1e-9) & (np.arange(len(t)) % int(round(a.gmn_dt / a.cont_dt)) == 0))[0]; return t, deep
CH = 8000
def analyze_many(items, autos, pure):
    """items: list of dict(rho (T,16,16), psi (T,16) or None, deep (indices into T) or None = all).  All series of one call share the group (pure / mixed) and the symmetry, so they are concatenated,
    analysed in one go (the GMN solver runs on the whole batch: its cost is dominated by the iteration count, not the batch size) and sliced again.  -> list of (out, meta)."""
    rho = np.concatenate([it["rho"] for it in items]); psi = np.concatenate([it["psi"] for it in items]) if pure else None; offs = np.cumsum([0] + [len(it["rho"]) for it in items])
    parts = []; meta = None
    for s0 in range(0, len(rho), CH):
        o, meta, _, _ = CA.cheap(xp, xp.asarray(rho[s0:s0 + CH]), autos, xp.asarray(psi[s0:s0 + CH].T) if pure else None); parts.append(o)
    cheap = {k: np.concatenate([o[k] for o in parts]) for k in parts[0]}
    di = [np.arange(len(it["rho"])) if it.get("deep") is None else np.asarray(it["deep"]) for it in items]; doffs = np.cumsum([0] + [len(d) for d in di])
    rows = np.concatenate([offs[k] + di[k] for k in range(len(items))]); logf(f"    deep analysis on {len(rows)} of {len(rho)} states"); deep = CA.deep(xp, xp.asarray(rho[rows]), autos, pure, log=logf)
    res = []
    for k in range(len(items)):
        out = {kk: v[offs[k]:offs[k + 1]] for kk, v in cheap.items()}
        for kk, v in deep.items(): out["deep/" + kk] = v[doffs[k]:doffs[k + 1]]
        out["deep_idx"] = di[k]; res.append((out, meta))
    return res
def task_name(path): return os.path.relpath(path, a.out)

tasks = [TASKS[a.task]] if a.task is not None else TASKS; tg, deep_c = tgrid()
for (nm, g, v, orb) in tasks:
    edges = sorted(g.edges()); autos = X.automorphisms(g); stab = K.stabilizer(autos, v); site = "abcd"[v]; base = os.path.join(a.out, nm)
    logf(f"=== {nm}, contact {site} (orbit {''.join('abcd'[u] for u in orb)}), |stabiliser| = {len(stab)}, edges {[ 'abcd'[i] + 'abcd'[j] for i, j in edges]}")
    # ---------------- free control (shared by all sites of the graph; full symmetry)
    if not a.no_free:
        items = []; paths = []
        for tn, t0 in T0:
            if not a.no_coll:
                for tau in TAUS:
                    path = os.path.join(base, f"free_t0{tname(tn)}_coll_tau{tau}.npz")
                    if os.path.exists(path): continue
                    psi = X.evolution(np, edges, 4, t0 + tau * np.arange(a.nmax + 1)); items.append(dict(rho=K.rho_of_pure(psi.T), psi=psi.T, deep=None)); paths.append((path, dict(t0=t0, tau=tau, n=list(range(a.nmax + 1)), mode="collision"), None))
            if not a.no_cont:
                path = os.path.join(base, f"free_t0{tname(tn)}_cont{a.cont_suffix}.npz")
                if not os.path.exists(path):
                    psi = X.evolution(np, edges, 4, t0 + tg); items.append(dict(rho=K.rho_of_pure(psi.T), psi=psi.T, deep=deep_c)); paths.append((path, dict(t0=t0, mode="continuous"), tg))
        if items:
            t = time.perf_counter(); res = analyze_many(items, autos, True)
            for (path, md, tt), (out, meta) in zip(paths, res):
                meta = dict(meta); meta.update(dict(graph=nm, kind="free", edges=edges, **md))
                if tt is not None: out["t"] = tt
                save(path, out, meta)
            logf(f"  free controls: {len(items)} series, {time.perf_counter() - t:.1f}s")
    # ---------------- contact runs: one batched analysis per (ancilla, mode)
    adir = os.path.join(base, site)
    for anc in ANC:
        for mode in (["coll"] if not a.no_coll else []) + (["cont"] if not a.no_cont else []):
            items = []; paths = []; pure = None
            for gg in GS:
                for tn, t0 in T0:
                    psi0 = K.state_at(edges, t0); stem = f"{'p' if anc == '+' else anc}_g{gg:g}_t0{tname(tn)}"
                    common = dict(graph=nm, site=site, orbit="".join("abcd"[u] for u in orb), ancilla=anc, g=gg, t0=t0, edges=edges, stabiliser=len(stab))
                    if mode == "coll":
                        for tau in TAUS:
                            path = os.path.join(adir, f"{stem}_coll_tau{tau}.npz")
                            if os.path.exists(path): continue
                            kind, st = K.collide(psi0, edges, v, gg, tau, a.nmax, anc); pure = kind == "pure"; rho = K.rho_of_pure(st) if pure else st
                            items.append(dict(rho=rho, psi=st if pure else None, deep=None)); paths.append((path, dict(common, tau=tau, kind="collision", pure=pure, n=list(range(a.nmax + 1))), dict(rho=(st if pure else st))))
                    else:
                        path = os.path.join(adir, f"{stem}_cont{a.cont_suffix}.npz")
                        if os.path.exists(path): continue
                        w, tr = K.continuous(psi0, edges, v, gg, tg, anc); pure = len(w) == 1; rho = K.rho_from(w, tr)
                        items.append(dict(rho=rho, psi=tr[0].T if pure else None, deep=deep_c))
                        paths.append((path, dict(common, kind="continuous", pure=pure, cont_T=a.cont_T, cont_dt=a.cont_dt, gmn_window=a.gmn_window, gmn_dt=a.gmn_dt), dict(t=tg, traj=np.stack(tr).astype(np.complex64), weights=np.array(w))))
            if not items: continue
            t = time.perf_counter(); logf(f"  ancilla {anc}, {mode}: {len(items)} series")
            res = analyze_many(items, stab, pure)
            for (path, md, extra), (out, meta) in zip(paths, res):
                out.update(extra); save(path, out, dict(meta, **md))
            logf(f"  ancilla {anc}, {mode}: {time.perf_counter() - t:.1f}s")
logf("DONE")
