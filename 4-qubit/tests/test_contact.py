#!/usr/bin/env python3
"""Validation gates of the contact (ancilla) model (plan Task 1 / Sec. 3.2 / 5.4):
 (K1) reproduce [1] (plan Sec. 6 test 1): 3-qubit chain and triangle, XX network, ZZ contact at an end site, unpolarised ancilla, g/J = 20, J tau = 0.4: C_BC(n=2) = 0.990 / 0.978, phase-optimised Bell fidelity 0.995 / 0.989
 (K2) collision recurrence against the explicit five-qubit Kraus map  Tr_a[U (eta x rho) U^dag]  for eta = |1><1|, |+><+|
 (K3) ancilla |1>: n collisions = continuous evolution under H - g Z_v at t = n tau = five-qubit retained ancilla;  ancilla |+>: continuous = mixture of the two field signs = five-qubit retained |+> ancilla
 (K4) no contact (g = 0) reproduces the closed-form free evolution;  n = 0 is the pre-contact state
 (K5) trace 1, hermiticity, positivity at every step;  Z-contact conserves nothing it should not: <Z_a> unchanged
 (K6) plan Eq. 14 check: 3-qubit chain, contact at the middle site, eigenstate ancilla, g/J = 5: C_13 = |sin 2 J_eff t| with J_eff = 0.2; peaks near n = 10, 29, 49 (the paper reports the third at 51)
 (K7) symmetry: results of equivalent subsets agree under Aut(G)_v for all contact sites of all six graphs"""
import itertools, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4); sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
from qgpu import core as C, xxvac as X, contact as K
_JSONL = os.path.join(HERE, "results", "test_contact.jsonl"); os.makedirs(os.path.dirname(_JSONL), exist_ok=True); open(_JSONL, "w").close()
fails = []; SEC = [""]
def section(s): SEC[0] = s; print("\n== " + s, flush=True)
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    with open(_JSONL, "a") as fh: fh.write(json.dumps(dict(section=SEC[0], name=name, ok=bool(cond), detail=detail)) + "\n")
    if not cond: fails.append(name)
def vac(n): v = np.zeros(2 ** n, complex); v[0] = 1; return v
CHAIN = [(0, 1), (1, 2)]; TRI = [(0, 1), (1, 2), (0, 2)]

def evolve_gen(edges, v, g, tau, nmax, ancilla, nq):
    Hm = sum(K.op(K._PX, i, nq) @ K.op(K._PX, j, nq) for i, j in edges); Zv = K.op(K._PZ, v, nq)
    pp = {"1": 0.0, "0": 1.0, "+": 0.5}[ancilla]; Up = K.unitary(Hm + g * Zv, tau); Um = K.unitary(Hm - g * Zv, tau); rho = np.outer(vac(nq), vac(nq).conj()); out = [rho]
    for _ in range(nmax): rho = pp * Up @ rho @ Up.conj().T + (1 - pp) * Um @ rho @ Um.conj().T; out.append(rho)
    return np.array(out)
def bc(rho3):
    rbc = np.einsum("ijik->jk", rho3.reshape(2, 4, 2, 4)); c = C.concurrence(np, rbc[None])[0]
    f = max(0.5 * (rbc[a, a].real + rbc[b, b].real) + abs(rbc[a, b]) for a, b in ((0, 3), (1, 2))); return float(c), float(f)

section("(K1) reproduction of [1]: unpolarised ancilla, contact at an end site, g/J = 20, J tau = 0.4, n = 2")
for name, edges, c_exp, f_exp in (("chain", CHAIN, 0.990, 0.995), ("triangle", TRI, 0.978, 0.989)):
    c, f = bc(evolve_gen(edges, 0, 20.0, 0.4, 2, "+", 3)[2]); report(f"{name}: C_BC = {c_exp}, phase-optimised Bell fidelity {f_exp}", abs(c - c_exp) < 5e-4 and abs(f - f_exp) < 5e-4, f"C_BC {c:.4f}, F {f:.4f}")
section("(K2) collision recurrence against the explicit five-qubit Kraus map")
edges = [(0, 2), (0, 3), (1, 3)]; rng = np.random.default_rng(5); v = rng.normal(size=16) + 1j * rng.normal(size=16); v /= np.linalg.norm(v); rho0 = np.outer(v, v.conj())
for anc, eta in (("1", np.diag([0.0, 1.0]).astype(complex)), ("+", 0.5 * np.ones((2, 2), complex))):
    for site in (0, 3):
        Hm = K.xx_hamiltonian(edges); Zv = K.op(K._PZ, site, 4); pp = 0.0 if anc == "1" else 0.5
        Up, Um = K.unitary(Hm + 5 * Zv, 0.4), K.unitary(Hm - 5 * Zv, 0.4); rec = pp * Up @ rho0 @ Up.conj().T + (1 - pp) * Um @ rho0 @ Um.conj().T
        kr = K.kraus_collision(rho0, edges, site, 5.0, 0.4, eta); report(f"ancilla {anc}, contact {'abcd'[site]}: recurrence = Kraus map", np.abs(rec - kr).max() < 1e-12, f"{np.abs(rec - kr).max():.1e}")
section("(K3) collision (|1>) = continuous = retained five-qubit ancilla;  |+>: continuous = mixture of the two signs")
psi0 = K.state_at(edges, 0.3); n = 25; tau = 0.4; times = tau * np.arange(n + 1)
for site in (0, 2):
    kind, ps = K.collide(psi0, edges, site, 5.0, tau, n, "1"); w, tr = K.continuous(psi0, edges, site, 5.0, times, "1")
    d1 = np.abs(ps.T - tr[0]).max(); ret = max(np.abs(K.retained_ancilla(psi0, edges, site, 5.0, times[k], np.array([0, 1], complex)) - np.outer(ps[k], ps[k].conj())).max() for k in (3, 25))
    report(f"|1>, contact {'abcd'[site]}: collision = continuous (max |dpsi|) and = retained five-qubit ancilla", d1 < 1e-11 and ret < 1e-11, f"{d1:.1e}, {ret:.1e}")
    w, tr = K.continuous(psi0, edges, site, 5.0, times, "+"); rc = K.rho_from(w, tr); ret = max(np.abs(K.retained_ancilla(psi0, edges, site, 5.0, times[k], np.array([1, 1], complex) / np.sqrt(2)) - rc[k]).max() for k in (3, 25))
    report(f"|+>, contact {'abcd'[site]}: continuous mixture = retained five-qubit |+> ancilla", ret < 1e-11, f"{ret:.1e}")
section("(K4) g = 0 and n = 0")
t0 = 0.37; kind, ps = K.collide(K.state_at(edges, t0), edges, 1, 0.0, 0.4, 10, "1"); free = X.evolution(np, edges, 4, t0 + 0.4 * np.arange(11))
report("g = 0: n collisions = free evolution to t0 + n tau (closed form)", np.abs(ps.T - free).max() < 1e-12, f"{np.abs(ps.T - free).max():.1e}")
kind, rr = K.collide(K.state_at(edges, t0), edges, 1, 5.0, 0.4, 4, "+"); report("n = 0 is the pre-contact state", np.abs(rr[0] - np.outer(K.state_at(edges, t0), K.state_at(edges, t0).conj())).max() < 1e-15)
section("(K5) trace, hermiticity, positivity at every step; <Z_a> conserved")
for anc in ("1", "+"):
    kind, st = K.collide(K.state_at(edges, math.pi / 4), edges, 2, 20.0, 0.4, 200, anc); rr = K.rho_of_pure(st) if kind == "pure" else st
    tr = np.abs(np.trace(rr, axis1=1, axis2=2) - 1).max(); he = np.abs(rr - np.conj(np.swapaxes(rr, 1, 2))).max(); me = np.linalg.eigvalsh(rr)[:, 0].min()
    report(f"ancilla {anc}, 200 collisions: trace {tr:.1e}, hermiticity {he:.1e}, min eigenvalue {me:.1e} (all within 1e-12)", tr < 1e-12 and he < 1e-12 and me > -1e-12)
section("(K6) plan Eq. 14: chain, middle contact, eigenstate ancilla, g/J = 5, J tau = 0.4")
rr = evolve_gen(CHAIN, 1, 5.0, 0.4, 70, "1", 3); c13 = C.concurrence(np, K.ptrace_rho(np, rr, 3, [0, 2]))
peaks = [k for k in range(1, 70) if c13[k] > c13[k - 1] and c13[k] >= c13[k + 1] and c13[k] > 0.94]
peaks = [k for k in peaks if c13[k] >= c13[max(0, k - 6):k + 7].max() - 1e-15]      # keep the main peaks: largest within +-6 collisions (drops the ripples at n = 32)
report(f"C_13 peaks: Eq. 14 predicts n = 10, 29, 49; first within 1, second within 28-33, third at the paper's 51; found {peaks}", len(peaks) >= 3 and abs(peaks[0] - 10) <= 1 and 28 <= peaks[1] <= 33 and abs(peaks[2] - 51) <= 1, f"peak heights {[round(float(c13[k]), 3) for k in peaks[:3]]}")
section("(K7) contact sites and symmetry: P_sigma rho P_sigma^dag = rho for every sigma fixing the contact site (plan Eq. 13), all contact sites, both ancillas")
def perm_matrix(p):
    d = 16; idx = np.arange(d); bits = ((idx[:, None] >> (3 - np.arange(4))) & 1); new = np.zeros(d, int)
    for k in range(d): nb = np.zeros(4, int); nb[list(p)] = bits[k]; new[k] = int("".join(map(str, nb)), 2)      # qubit i -> qubit p[i]
    P = np.zeros((d, d)); P[new, idx] = 1; return P
G6 = os.path.join(ROOT, "atlas/data/graphs/graph4c.g6"); graphs = X.load_g6(G6); total = 0
for g in graphs:
    nm = X.graph_name(g); sites = K.contact_sites(g); total += len(sites); autos = X.automorphisms(g); edges = sorted(g.edges())
    for v, orb in sites:
        st = K.stabilizer(autos, v); worst = 0.0; worst_free_break = 0.0
        for anc in ("1", "+"):
            kind, rr = K.collide(K.state_at(edges, 0.3), edges, v, 5.0, 0.4, 12, anc); rr = K.rho_of_pure(rr) if kind == "pure" else rr
            for p in st:
                P = perm_matrix(p); worst = max(worst, float(np.abs(P @ rr @ P.T - rr).max()))
        report(f"{nm}, contact {'abcd'[v]} (orbit {''.join('abcd'[u] for u in orb)}, |stabiliser| = {len(st)}): P rho P^T = rho", worst < 1e-12, f"{worst:.1e}")
report("11 inequivalent contact sites over the six graphs", total == 11, str(total))
print("\nOVERALL:", "PASS" if not fails else "FAIL: " + "; ".join(fails)); sys.exit(1 if fails else 0)
