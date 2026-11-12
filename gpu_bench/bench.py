#!/usr/bin/env python3
"""Run ONE backend on a list of cases; print JSON lines. Reference for accuracy
is computed in-process with the numpy spectral pipeline."""
import argparse, json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B
from qnet import graphs as G, hamiltonians as H, evolve as E

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--cases", default="3:1:ex:plus:100,4:5:ex:plus:100,5:20:ex:plus:100")
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

for spec in a.cases.split(","):
    n, gi, inter, init, tmax = spec.split(":"); n, gi, tmax = int(n), int(gi), float(tmax)
    Gph = G.load_graphs(n)[gi]
    edges = [tuple(sorted(e)) for e in Gph.edges()]
    Hm = H.build_hamiltonian(Gph, inter); psi0 = H.initial_state(n, init)
    ev, _ = E.diagonalize(Hm); times = E.make_time_grid(ev, tmax)
    ref = B.numpy_spectral(Hm, psi0, times, edges, n)["C"]
    row = dict(backend=a.backend, n=n, graph=gi, inter=inter, init=init, T=len(times), edges=len(edges))
    try:
        runs = []
        for r in range(a.repeats + 1):   # run 0 = warm-up / JIT / first call
            runs.append(B.BACKENDS[a.backend](Hm, psi0, times, edges, n))
        row["first_total"] = runs[0]["t_evolve"] + runs[0]["t_post"]
        best = min(runs[1:], key=lambda d: d["t_evolve"] + d["t_post"])
        row.update(t_evolve=best["t_evolve"], t_post=best["t_post"],
                   t_total=best["t_evolve"] + best["t_post"],
                   max_C_err=max(float(np.max(np.abs(best["C"][e] - ref[e]))) for e in edges))
    except Exception as ex:
        row["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    print(json.dumps(row), flush=True)
