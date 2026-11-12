#!/usr/bin/env python3
"""Stage-by-stage comparison of what we implemented vs the corresponding QuTiP alternative.
ex|+> on the complete graph K_n (n=3,4,5), T=5001.  Stages (per n):
  S1 time evolution          : ours numpy | ours CuPy | qt.sesolve (CPU) | qutip-jax (GPU)
  S2 partial trace, all pairs: ours numpy | ours CuPy | Qobj.ptrace loop (CPU)
  S3 pair concurrence        : ours numpy | ours CuPy | qt.concurrence loop (CPU)
  S4 PPT min-eig over all single-qubit cuts of all connected triples
                             : ours numpy | ours CuPy | qt.partial_transpose + eigvalsh loop (CPU)
  S5 Meyer-Wallach Q(t)      : ours numpy | ours CuPy | Qobj.ptrace-per-qubit loop (CPU)
GPU inputs are device-resident (no PCIe in the timing); CuPy timings include a device sync.
Slow QuTiP loops are timed once, fast batched code is best-of-5 after a warm-up."""
import itertools, json, os, sys, time
import numpy as np, networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qutip as qt, cupy as cp
import backends as B
from qnet import graphs as G, hamiltonians as H, evolve as E, measures as M

rows = []
def emit(**r):
    rows.append(r); print(json.dumps(r), flush=True)
def best(fn, rep=5, sync=None):
    fn(); sync and sync()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); out = fn(); sync and sync(); ts.append(time.perf_counter() - t0)
    return min(ts), out
def once(fn):
    t0 = time.perf_counter(); out = fn(); return time.perf_counter() - t0, out
sync = cp.cuda.Stream.null.synchronize
gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

def pt_gpu(rho, dims_n, q):                       # partial transpose on qubit q of (T,d,d)
    T, d = rho.shape[0], rho.shape[1]
    Y = cp.swapaxes(rho.reshape((T,) + (2,) * (2 * dims_n)), 1 + q, 1 + dims_n + q)
    return Y.reshape(T, d, d)

def mw_gpu(psi_t, n):
    acc = 0
    for i in range(n):
        r = B._ptrace_gpu(psi_t, n, [i]); acc = acc + cp.real(cp.einsum("tab,tba->t", r, r))
    return 2.0 * (1.0 - acc / n)

jax_cases = []
for n, gi in [(3, 1), (4, 5), (5, 20)]:
    Gph = G.load_graphs(n)[gi]; edges = [tuple(sorted(e)) for e in Gph.edges()]
    triples = [s for s in itertools.combinations(range(n), 3) if nx.is_connected(Gph.subgraph(s))]
    Hm = H.build_hamiltonian(Gph, "ex"); psi0 = H.initial_state(n, "plus")
    ev0, _ = E.diagonalize(Hm); times = E.make_time_grid(ev0, 100.0); T = len(times)
    meta = dict(n=n, T=T, edges=len(edges), triples=len(triples), gpu=gpu)
    jax_cases.append((n, Hm, psi0, times, meta))
    p0 = psi0.full().ravel()

    # ---------- S1 evolution
    def ev_np():
        e, V = E.diagonalize(Hm); return E.evolve_batch(e, V, p0, times)
    t, psi_t = best(ev_np); emit(stage="S1 evolution", impl="ours numpy", time_s=t, err=0.0, **meta)
    def ev_cp():
        Hg = cp.asarray(Hm.full()); e, V = cp.linalg.eigh(Hg); c = V.conj().T @ cp.asarray(p0)
        return V @ (cp.exp(-1j * cp.outer(e, cp.asarray(times))) * c[:, None])
    t, pg = best(ev_cp, sync=sync); emit(stage="S1 evolution", impl="ours CuPy", time_s=t, err=float(np.abs(cp.asnumpy(pg) - psi_t).max()), **meta)
    t, st = once(lambda: qt.sesolve(Hm, psi0, times, options=B.TOL).states)
    emit(stage="S1 evolution", impl="QuTiP sesolve (CPU)", time_s=t,
         err=float(np.abs(np.stack([s_.full().ravel() for s_ in st], 1) - psi_t).max()), **meta)

    # ---------- S2 ptrace (pairs)
    t, rho2 = best(lambda: [E.partial_trace_pure_batch(psi_t, n, e) for e in edges]); emit(stage="S2 partial trace (pairs)", impl="ours numpy", time_s=t, err=0.0, **meta)
    t, r2g = best(lambda: [B._ptrace_gpu(pg, n, e) for e in edges], sync=sync)
    emit(stage="S2 partial trace (pairs)", impl="ours CuPy", time_s=t, err=float(max(np.abs(cp.asnumpy(a) - b).max() for a, b in zip(r2g, rho2))), **meta)
    kets = [qt.Qobj(psi_t[:, k], dims=[[2] * n, [1] * n]) for k in range(T)]
    t, rq = once(lambda: {e: [k_.ptrace(list(e)) for k_ in kets] for e in edges})
    emit(stage="S2 partial trace (pairs)", impl="QuTiP ptrace loop (CPU)", time_s=t,
         err=float(max(np.abs(np.stack([r.full() for r in rq[e]]) - rho2[i]).max() for i, e in enumerate(edges))), **meta)

    # ---------- S3 concurrence
    t, c_np = best(lambda: [M.concurrence_batch(r) for r in rho2]); emit(stage="S3 concurrence (pairs)", impl="ours numpy", time_s=t, err=0.0, **meta)
    t, c_g = best(lambda: [B._concurrence_gpu(r) for r in r2g], sync=sync)
    emit(stage="S3 concurrence (pairs)", impl="ours CuPy", time_s=t, err=float(max(np.abs(cp.asnumpy(a) - b).max() for a, b in zip(c_g, c_np))), **meta)
    t, c_q = once(lambda: {e: np.array([qt.concurrence(r) for r in rq[e]]) for e in edges})
    emit(stage="S3 concurrence (pairs)", impl="QuTiP concurrence loop (CPU)", time_s=t, err=float(max(np.abs(c_q[e] - c_np[i]).max() for i, e in enumerate(edges))), **meta)

    # ---------- S4 PPT min-eig, single-qubit cuts of all connected triples
    rho3 = [E.partial_trace_pure_batch(psi_t, n, s) for s in triples]; rho3g = [cp.asarray(r) for r in rho3]
    def ppt_np(): return [M.min_eig_partial_transpose_batch(r, (2, 2, 2), [q]) for r in rho3 for q in range(3)]
    t, m_np = best(ppt_np); emit(stage="S4 PPT min-eig (triples)", impl="ours numpy", time_s=t, err=0.0, **meta)
    def ppt_cp(): return [cp.linalg.eigvalsh(0.5 * (lambda p: p + cp.conj(cp.swapaxes(p, 1, 2)))(pt_gpu(r, 3, q)))[:, 0] for r in rho3g for q in range(3)]
    t, m_g = best(ppt_cp, sync=sync); emit(stage="S4 PPT min-eig (triples)", impl="ours CuPy", time_s=t, err=float(max(np.abs(cp.asnumpy(a) - b).max() for a, b in zip(m_g, m_np))), **meta)
    rq3 = [[qt.Qobj(r[k], dims=[[2] * 3, [2] * 3]) for k in range(T)] for r in rho3]
    def ppt_q():
        return [np.array([np.linalg.eigvalsh(qt.partial_transpose(rk, [1 if j == q else 0 for j in range(3)]).full())[0] for rk in rl]) for rl in rq3 for q in range(3)]
    t, m_q = once(ppt_q); emit(stage="S4 PPT min-eig (triples)", impl="QuTiP partial_transpose loop (CPU)", time_s=t, err=float(max(np.abs(a - b).max() for a, b in zip(m_q, m_np))), **meta)

    # ---------- S5 Meyer-Wallach
    t, q_np = best(lambda: M.meyer_wallach_Q_batch(psi_t, n)); emit(stage="S5 Meyer-Wallach Q(t)", impl="ours numpy", time_s=t, err=0.0, **meta)
    t, q_g = best(lambda: mw_gpu(pg, n), sync=sync); emit(stage="S5 Meyer-Wallach Q(t)", impl="ours CuPy", time_s=t, err=float(np.abs(cp.asnumpy(q_g) - q_np).max()), **meta)
    def mw_q():
        out = np.zeros(T)
        for k, ket in enumerate(kets):
            out[k] = 2.0 * (1.0 - np.mean([(ket.ptrace(i) ** 2).tr().real for i in range(n)]))
        return out
    t, q_q = once(mw_q); emit(stage="S5 Meyer-Wallach Q(t)", impl="QuTiP ptrace loop (CPU)", time_s=t, err=float(np.abs(q_q - q_np).max()), **meta)

# ---------- S1 with qutip-jax last (it switches QuTiP's global default dtype)
import jax, diffrax, qutip_jax as qj
jax.config.update("jax_enable_x64", True); qj.set_as_default()
for n, Hm, psi0, times, meta in jax_cases:
    def run():
        res = qt.sesolve(Hm.to("jax"), psi0.to("jax"), times, options={"method": "diffrax", "normalize_output": False, "max_steps": 10_000_000,
                          "stepsize_controller": diffrax.PIDController(atol=1e-12, rtol=1e-10)})
        jax.block_until_ready(res.states[-1].data._jxa); return res
    t, res = once(run)
    emit(stage="S1 evolution", impl="qutip-jax diffrax (GPU)", time_s=t, err=float("nan"), **meta)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stages_results_" + gpu.split("-")[0].split()[-1] + ".json"), "w"), indent=1)
