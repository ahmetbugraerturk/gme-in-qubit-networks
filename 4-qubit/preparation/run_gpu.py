#!/usr/bin/env python3
"""GPU (or CPU) entanglement run for one n-qubit graph: pair concurrence, Meyer-Wallach, PPT per cut for
every connected k-qubit subset, and (with --gmn) the GMN SDP for the subsets that are not PPT, solved by the
batched ADMM with a rigorous dual upper bound, classified as zero / positive / near-threshold.  Saves one .npz.
  python run_gpu.py --n 4 --graph-index 5 --interaction ex --init plus --gmn --device gpu
"""
import argparse, itertools, os, sys, time
import numpy as np, networkx as nx
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE)   # HERE = 4-qubit/preparation
sys.path.insert(0, os.path.join(os.path.dirname(Q4), "atlas")); sys.path.insert(0, Q4)
from qnet import graphs as G, hamiltonians as Hm, evolve as E          # graph / Hamiltonian construction only
from qgpu import core as C, gmn_admm as A

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4); ap.add_argument("--graph-index", type=int, required=True)
ap.add_argument("--interaction", default="ex"); ap.add_argument("--init", default="plus")
ap.add_argument("--t-max", type=float, default=100.0); ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--gmn", action="store_true"); ap.add_argument("--subset-sizes", default="3,4")
ap.add_argument("--gmn-tol", type=float, default=1e-6); ap.add_argument("--out", default=None)
a = ap.parse_args()

xp = C.get_xp(a.device); sync = (lambda: xp.cuda.Stream.null.synchronize()) if a.device == "gpu" else (lambda: None)
Gph = G.load_graphs(a.n)[a.graph_index]; edges = [tuple(sorted(e)) for e in Gph.edges()]
Hmat = Hm.build_hamiltonian(Gph, a.interaction); psi0 = Hm.initial_state(a.n, a.init).full().ravel()
ev, _ = E.diagonalize(Hmat); times = E.make_time_grid(ev, a.t_max)     # time grid rule shared with atlas
t0 = time.perf_counter()
psi_t = C.evolve(xp, Hmat.full(), psi0, times)
res = {"times": times, "edges": np.array(edges)}
res["concurrence"] = np.stack([C.to_host(C.concurrence(xp, C.ptrace_pure(xp, psi_t, a.n, e))) for e in edges])
res["meyer_wallach"] = C.to_host(C.meyer_wallach(xp, psi_t, a.n)); sync()
print(f"evolution + pairs + MW: {time.perf_counter()-t0:.3f}s  (T={len(times)}, {len(edges)} edges)", flush=True)

for k in [int(s) for s in a.subset_sizes.split(",") if int(s) <= a.n]:
    cuts = A.bipartition_cuts(k)
    for sub in itertools.combinations(range(a.n), k):
        if not nx.is_connected(Gph.subgraph(sub)): continue
        t1 = time.perf_counter()
        rho = C.ptrace_pure(xp, psi_t, a.n, sub) if k < a.n else (psi_t.T[:, :, None] * xp.conj(psi_t.T)[:, None, :])
        me = xp.stack([C.min_eig_pt(xp, rho, k, c) for c in cuts], 1)           # (T,K)
        ppt = (me >= -1e-10).any(1)                                             # some cut PPT => GMN = 0
        tag = "".join(map(str, sub)); res[f"pt_mineig_{tag}"] = C.to_host(me)
        out = f"  subset {sub}: {int((~ppt).sum())}/{len(times)} non-PPT"
        if a.gmn and bool((~ppt).any()):
            idx = xp.nonzero(~ppt)[0]
            r = A.solve_batch(xp, rho[idx], nq=k, tol_gap=1e-7)
            h = C.to_host(idx)
            gmn = np.zeros(len(times)); upper = np.zeros(len(times)); ok = np.ones(len(times), bool)
            cls = np.zeros(len(times), int)                                      # PPT time points: certified zero
            gmn[h] = C.to_host(r["gmn"]); upper[h] = C.to_host(r["gmn_upper"]); ok[h] = C.to_host(r["converged"])
            cls[h] = C.to_host(A.classify(xp, r["gmn"], r["gmn_upper"], a.gmn_tol))
            res[f"gmn_{tag}"] = gmn; res[f"gmn_upper_{tag}"] = upper; res[f"gmn_converged_{tag}"] = ok; res[f"gmn_class_{tag}"] = cls
            out += (f", GMN max={gmn.max():.4g}, converged {ok[h].mean():.0%}; classes: zero {int((cls==A.ZERO).sum())}, "
                    f"positive {int((cls==A.POSITIVE).sum())}, near-threshold {int((cls==A.NEAR_THRESHOLD).sum())}")
        sync(); print(out + f"  [{time.perf_counter()-t1:.2f}s]", flush=True)
path = a.out or os.path.join(HERE, f"run_n{a.n}_g{a.graph_index}_{a.interaction}_{a.init}.npz")
np.savez_compressed(path, **res); print("saved", path)
