#!/usr/bin/env python3
"""Additional contact setups (research plan Setting II and a multi-contact extension).  Collision model (J tau = 0.4) and continuous (retained ancilla) evolution, ancillas |+> and |1>, all toolbox quantities.
 setup A : exchange (EX) network, XX contact at one site v (11 symmetry-inequivalent sites of the six four-vertex graphs).  EX bond = (XX + YY)/4 (norm 1/2), contact g X_a X_v.  From the vacuum |0000> the network is inert,
           so there is no pre-contact time and no free control (the free state is |0000> at all times).  X_a commutes with H: ancilla |+> gives a coherent drive +g X_v, ancilla |1> a random field sign (p_+ = 1/2).
           g/J = 1,2,5,20; n = 0..400 collisions (Jt <= 160); continuous Jt <= 160 (cheap quantities, step 0.02) and deep analysis for Jt <= 80 (step 0.2).
 setup B : the 58 symmetry-inequivalent (connected five-vertex graph, ancilla vertex a) pairs.  The four other vertices form the XX network (induced edges, possibly disconnected); every edge at a becomes a ZZ contact g Z_a Z_j,
           so the ancilla touches all neighbours of a at once; Z_a commutes with H: ancilla |1> is a coherent field -g sum_j Z_j, |+> a random sign.  g/J = 5,20; t0 = 0; n = 0..200; continuous Jt <= 80 (cheap quantities).
           Free control: the induced network without contact (closed form), cached by its edge set.
One task = one (graph, site) / (graph, ancilla vertex) pair; `--task K`, `--list`.  Results: results/setII/<graph>/<site>/ and results/five/g<k>_a<v>/ ; existing files are skipped."""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from qgpu import core as C, xxvac as X, contact as K, contact_analysis as CA
ap = argparse.ArgumentParser(); ap.add_argument("--setup", required=True, choices=["A", "B"]); ap.add_argument("--task", type=int, default=None); ap.add_argument("--list", action="store_true"); ap.add_argument("--out", default=None)
ap.add_argument("--g", default=None); ap.add_argument("--anc", default="+,1"); ap.add_argument("--nmax", type=int, default=None); ap.add_argument("--tau", type=float, default=0.4); ap.add_argument("--device", default="gpu"); ap.add_argument("--no-cont", action="store_true"); ap.add_argument("--no-coll", action="store_true")
a = ap.parse_args(); xp = C.get_xp(a.device); logf = lambda s: print(s, flush=True)
SETA = a.setup == "A"; GS = [float(x) for x in (a.g or ("1,2,5,20" if SETA else "5,20")).split(",")]; NMAX = a.nmax or (400 if SETA else 200); ANC = a.anc.split(","); KIND = "x" if SETA else "z"; NET = "ex" if SETA else "xx"
OUT = a.out or os.path.join(HERE, "results", "setII" if SETA else "five"); TAU = a.tau
def tasks():
    out = []
    if SETA:
        for g in X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")):
            for v, orb in K.contact_sites(g): out.append(dict(name=X.graph_name(g), dir=f"{X.graph_name(g)}/{'abcd'[v]}", edges=sorted(g.edges()), sites=[v], stab=K.stabilizer(X.automorphisms(g), v), info=f"contact {'abcd'[v]} (orbit {''.join('abcd'[u] for u in orb)})", base=None))
    else:
        for gi, g in enumerate(X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph5c.g6"))):
            autos = X.automorphisms(g); seen = set()
            for v in sorted(g.nodes()):
                if v in seen: continue
                orb = sorted({p[v] for p in autos}); seen |= set(orb); rest = [q for q in sorted(g.nodes()) if q != v]; idx = {q: i for i, q in enumerate(rest)}
                edges = sorted((idx[i], idx[j]) for i, j in g.edges() if v not in (i, j)); sites = sorted(idx[j] for j in g.neighbors(v))
                stab = [tuple(idx[p[rest[i]]] for i in range(4)) for p in autos if p[v] == v]
                out.append(dict(name=f"g{gi}", dir=f"g{gi}_a{v}", edges=edges, sites=sites, stab=stab, info=f"graph {gi} edges {sorted(g.edges())}, ancilla vertex {v} (orbit {orb}), network edges {edges}, contact sites {sites}", base=g))
    return out
TASKS = tasks()
if a.list:
    for i, t in enumerate(TASKS): print(i, t["dir"], t["info"])
    sys.exit(0)
def vac(): v = np.zeros(16, complex); v[0] = 1; return v
CH = 8000
def analyze_many(items, autos, pure):
    rho = np.concatenate([it["rho"] for it in items]); psi = np.concatenate([it["psi"] for it in items]) if pure else None; offs = np.cumsum([0] + [len(it["rho"]) for it in items]); parts = []; meta = None
    for s0 in range(0, len(rho), CH):
        o, meta, _, _ = CA.cheap(xp, xp.asarray(rho[s0:s0 + CH]), autos, xp.asarray(psi[s0:s0 + CH].T) if pure else None); parts.append(o)
    cheap = {k: np.concatenate([o[k] for o in parts]) for k in parts[0]}
    di = [np.arange(len(it["rho"])) if it.get("deep") is None else np.asarray(it["deep"]) for it in items]; doffs = np.cumsum([0] + [len(d) for d in di]); rows = np.concatenate([offs[k] + di[k] for k in range(len(items))])
    deep = CA.deep(xp, xp.asarray(rho[rows]), autos, pure, log=logf) if len(rows) else {}; logf(f"    deep analysis on {len(rows)} of {len(rho)} states")
    res = []
    for k in range(len(items)):
        out = {kk: v[offs[k]:offs[k + 1]] for kk, v in cheap.items()}
        for kk, v in deep.items(): out["deep/" + kk] = v[doffs[k]:doffs[k + 1]]
        out["deep_idx"] = di[k]; res.append((out, meta))
    return res
def save(path, out, meta): os.makedirs(os.path.dirname(path), exist_ok=True); np.savez_compressed(path, meta=json.dumps(meta), **out)
tg = np.arange(0, int(round((160.0 if SETA else 80.0) / 0.02)) + 1) * 0.02
deep_c = np.nonzero((tg <= 80 + 1e-9) & (np.arange(len(tg)) % 10 == 0))[0] if SETA else np.array([], int)
for tk in ([TASKS[a.task]] if a.task is not None else TASKS):
    edges = tk["edges"]; sites = tk["sites"]; stab = tk["stab"]; base = os.path.join(OUT, tk["dir"]); model = dict(edges=edges, net=NET, kind=KIND, sites=sites, n=4)
    logf(f"=== setup {a.setup}: {tk['dir']}: {tk['info']} |stabiliser| = {len(stab)}")
    # free control (setup B only; in setup A the network is inert from the vacuum)
    if not SETA and not a.no_coll:
        fpath = os.path.join(OUT, "_free", "_".join(f"{i}{j}" for i, j in edges) + "_coll.npz" if edges else "none_coll.npz")
        if not os.path.exists(fpath):
            Gn = nx.Graph(); Gn.add_nodes_from(range(4)); Gn.add_edges_from(edges); fa = [tuple(m[i] for i in range(4)) for m in GraphMatcher(Gn, Gn).isomorphisms_iter()]
            psi = X.evolution(np, edges, 4, TAU * np.arange(NMAX + 1)) if edges else np.tile(vac()[:, None], (1, NMAX + 1)); res = analyze_many([dict(rho=K.rho_of_pure(psi.T), psi=psi.T, deep=None)], fa, True)
            out, meta = res[0]; save(fpath, out, dict(meta, kind="free", edges=edges, tau=TAU, n=list(range(NMAX + 1)))); logf("  free control written")
    for anc in ANC:
        pp = K.PP[KIND][anc]; pure = pp in (0.0, 1.0); stem = "p" if anc == "+" else anc
        for mode in (["coll"] if not a.no_coll else []) + (["cont"] if not a.no_cont else []):
            items = []; paths = []
            for g in GS:
                path = os.path.join(base, f"{stem}_g{g:g}_{mode}" + (f"_tau{TAU}" if mode == "coll" else "") + ".npz"); common = dict(setup=a.setup, graph=tk["name"], dir=tk["dir"], edges=edges, sites=sites, net=NET, kind=KIND, ancilla=anc, p_plus=pp, g=g, pure=pure, stabiliser=len(stab), t0=0.0)
                if os.path.exists(path): continue
                psi0 = vac()
                if mode == "coll":
                    kind, st = K.collide_model(psi0, model, g, TAU, NMAX, pp); rho = K.rho_of_pure(st) if kind == "pure" else st; sp = K.channel_spectrum(model, g, TAU, pp)
                    items.append(dict(rho=rho, psi=st if pure else None, deep=None)); paths.append((path, dict(common, tau=TAU, kind="collision", n=list(range(NMAX + 1)), spectrum=sp), dict(rho=st)))
                else:
                    w, tr = K.continuous_model(psi0, model, g, tg, pp); rho = K.rho_from(w, tr); items.append(dict(rho=rho, psi=tr[0].T if pure else None, deep=deep_c))
                    paths.append((path, dict(common, kind="continuous", cont_T=float(tg[-1]), cont_dt=0.02), dict(t=tg, traj=np.stack(tr).astype(np.complex64), weights=np.array(w))))
            if not items: continue
            t = time.perf_counter(); logf(f"  ancilla {anc}, {mode}: {len(items)} series"); res = analyze_many(items, stab, pure)
            for (path, md, extra), (out, meta) in zip(paths, res): out.update(extra); save(path, out, dict(meta, **md))
            logf(f"  ancilla {anc}, {mode}: {time.perf_counter() - t:.1f}s")
logf("DONE")
