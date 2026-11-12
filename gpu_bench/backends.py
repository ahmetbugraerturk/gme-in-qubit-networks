"""One function per backend: run(Hmat, psi0, times, edges, n) -> dict with
per-stage timings and edge->C(t) arrays. Stages: evolve, ptrace+concurrence.
Each backend is run in its own process (the cuQuantum/JAX backends change
QuTiP's global default dtype), selected via bench.py --backend NAME."""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atlas"))
import qutip as qt
from qnet import evolve as E, measures as M

TOL = {"atol": 1e-14, "rtol": 1e-12}


def _sync_cupy():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


# ---------- ours, CPU ----------
def numpy_spectral(Hm, psi0, times, edges, n):
    t0 = time.perf_counter()
    ev, V = E.diagonalize(Hm)
    psi_t = E.evolve_batch(ev, V, psi0.full().ravel(), times)
    t1 = time.perf_counter()
    C = {e: M.concurrence_batch(E.partial_trace_pure_batch(psi_t, n, e)) for e in edges}
    t2 = time.perf_counter()
    return dict(t_evolve=t1 - t0, t_post=t2 - t1, C=C)


# ---------- QuTiP core, CPU ----------
def qutip_cpu(Hm, psi0, times, edges, n):
    t0 = time.perf_counter()
    states = qt.sesolve(Hm, psi0, times, options=TOL).states
    t1 = time.perf_counter()
    C = {e: np.array([qt.concurrence(s.ptrace(list(e))) for s in states]) for e in edges}
    t2 = time.perf_counter()
    return dict(t_evolve=t1 - t0, t_post=t2 - t1, C=C)


# ---------- ours, CuPy GPU (same spectral algorithm, hand-written) ----------
def _ptrace_gpu(psi_t, n, keep):
    import cupy as cp
    T = psi_t.shape[1]
    keep = list(keep); rest = [q for q in range(n) if q not in keep]
    a = psi_t.T.reshape((T,) + (2,) * n)
    a = cp.transpose(a, [0] + [1 + q for q in keep] + [1 + q for q in rest])
    dk = 2 ** len(keep)
    a = a.reshape(T, dk, -1)
    return a @ cp.conj(cp.swapaxes(a, 1, 2))


def _concurrence_gpu(rho):
    import cupy as cp
    sy = cp.array([[0, -1j], [1j, 0]]); YY = cp.kron(sy, sy)
    rt = YY @ cp.conj(rho) @ YY
    w, U = cp.linalg.eigh(rho)
    w = cp.clip(w, 0, None)
    sq = (U * cp.sqrt(w)[:, None, :]) @ cp.conj(cp.swapaxes(U, 1, 2))
    lam = cp.linalg.eigvalsh(sq @ rt @ sq)
    s = cp.sqrt(cp.clip(lam, 0, None))  # ascending
    return cp.clip(s[:, 3] - s[:, 2] - s[:, 1] - s[:, 0], 0, None)


def cupy_spectral(Hm, psi0, times, edges, n):
    import cupy as cp
    t0 = time.perf_counter()
    Hg = cp.asarray(Hm.full())
    ev, V = cp.linalg.eigh(Hg)
    p0 = cp.asarray(psi0.full().ravel()); tg = cp.asarray(times)
    coeffs = V.conj().T @ p0
    psi_t = V @ (cp.exp(-1j * cp.outer(ev, tg)) * coeffs[:, None])
    _sync_cupy(); t1 = time.perf_counter()
    C = {e: _concurrence_gpu(_ptrace_gpu(psi_t, n, e)) for e in edges}
    C = {e: cp.asnumpy(v) for e, v in C.items()}
    t2 = time.perf_counter()
    return dict(t_evolve=t1 - t0, t_post=t2 - t1, C=C)


# ---------- QuTiP + cuQuantum (qutip-cuquantum) ----------
_cuq_ready = False
def _cuq_setup():
    global _cuq_ready
    if not _cuq_ready:
        import qutip_cuquantum as qc
        from cuquantum.densitymat import WorkStream
        qc.set_as_default(WorkStream())
        _cuq_ready = True


def qutip_cuquantum(Hm, psi0, times, edges, n):
    _cuq_setup()
    t0 = time.perf_counter()
    Hc = qt.Qobj(Hm.full(), dims=Hm.dims).to("cuDensity") if False else Hm
    res = qt.sesolve(Hc, psi0, times, options=TOL)
    _sync_cupy()
    t1 = time.perf_counter()
    # post-processing: QuTiP-native ptrace/concurrence need CPU data -> transfer
    states = [qt.Qobj(np.asarray(s.full()), dims=psi0.dims) for s in res.states]
    C = {e: np.array([qt.concurrence(s.ptrace(list(e))) for s in states]) for e in edges}
    t2 = time.perf_counter()
    return dict(t_evolve=t1 - t0, t_post=t2 - t1, C=C)


# ---------- QuTiP + JAX (qutip-jax, diffrax) ----------
_jax_ready = False
def qutip_jax(Hm, psi0, times, edges, n):
    global _jax_ready
    import jax, diffrax, qutip_jax as qj
    if not _jax_ready:
        jax.config.update("jax_enable_x64", True)
        qj.set_as_default()
        _jax_ready = True
    t0 = time.perf_counter()
    Hj = Hm.to("jax"); pj = psi0.to("jax")
    res = qt.sesolve(Hj, pj, times, options={
        "method": "diffrax", "normalize_output": False, "max_steps": 10_000_000,
        "stepsize_controller": diffrax.PIDController(atol=1e-12, rtol=1e-10)})
    jax.block_until_ready(res.states[-1].data._jxa)
    t1 = time.perf_counter()
    states = [qt.Qobj(np.asarray(s.full()), dims=psi0.dims) for s in res.states]
    C = {e: np.array([qt.concurrence(s.ptrace(list(e))) for s in states]) for e in edges}
    t2 = time.perf_counter()
    return dict(t_evolve=t1 - t0, t_post=t2 - t1, C=C)


BACKENDS = dict(numpy_spectral=numpy_spectral, qutip_cpu=qutip_cpu, cupy_spectral=cupy_spectral,
                qutip_cuquantum=qutip_cuquantum, qutip_jax=qutip_jax)
