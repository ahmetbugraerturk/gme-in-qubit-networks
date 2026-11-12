#!/usr/bin/env python3
"""Validation of the additional contact setups: (A) exchange (EX) network with XX contact, ancilla |+> (coherent drive +gX_v) or |1> (random sign); (B) five-vertex connected graphs, one vertex made the ancilla
(its edges become ZZ contacts to several network qubits), the other four the XX network.
 (S1) EX normalisation and the inert vacuum;  (S2) plan pilot: P4, (EX, XX), |+>, g/J = 5, 60 collisions: largest cut negativity 1.45
 (S3) collision recurrence against the explicit five-qubit Kraus map for contact kinds x and z, several contact sites, ancillas |+>, |1>, |->
 (S4) five-vertex mapping: the explicit five-qubit graph evolution equals the network + ancilla model for all 58 (graph, ancilla-vertex orbit) pairs
 (S5) symmetry P rho P^dag = rho under the automorphisms fixing the ancilla / contact site;  (S6) counts;  (S7) channel spectrum sanity"""
import itertools, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q4 = os.path.dirname(HERE); ROOT = os.path.dirname(Q4); sys.path.insert(0, Q4); sys.path.insert(0, os.path.join(ROOT, "atlas"))
import warnings; warnings.filterwarnings("ignore")
import networkx as nx
from qgpu import core as C, xxvac as X, contact as K
_JSONL = os.path.join(HERE, "results", "test_setups.jsonl"); os.makedirs(os.path.dirname(_JSONL), exist_ok=True); open(_JSONL, "w").close()
fails = []; SEC = [""]
def section(s): SEC[0] = s; print("\n== " + s, flush=True)
def report(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    with open(_JSONL, "a") as fh: fh.write(json.dumps(dict(section=SEC[0], name=name, ok=bool(cond), detail=detail)) + "\n")
    if not cond: fails.append(name)
ETA = {"+": 0.5 * np.ones((2, 2), complex), "1": np.diag([0.0, 1.0]).astype(complex), "-": 0.5 * np.array([[1, -1], [-1, 1]], complex)}
G4 = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph4c.g6")); G5 = X.load_g6(os.path.join(ROOT, "atlas/data/graphs/graph5c.g6"))
def vac(n): v = np.zeros(2 ** n, complex); v[0] = 1; return v

section("(S1) exchange coupling: normalisation and inert vacuum")
ev = np.linalg.eigvalsh(K.net_hamiltonian([(0, 1)], "ex", 2)); report("one EX bond has spectrum {-1/2, 0, 0, 1/2} (norm 1/2)", np.allclose(sorted(ev), [-0.5, 0, 0, 0.5]), str(np.round(ev, 3)))
ok = all(np.abs(K.net_hamiltonian(sorted(g.edges()), "ex") @ vac(4)).max() < 1e-15 for g in G4); report("the EX network leaves |0000> invariant on all six graphs", ok)
ev = np.linalg.eigvalsh(K.net_hamiltonian([(0, 1)], "xx", 2)); report("one XX bond has spectrum {-1, -1... norm 1}", abs(max(abs(ev)) - 1) < 1e-12)

section("(S2) plan pilot: P4 (1-2-3-4), (EX, XX) contact at site 2, ancilla |+>, g/J = 5, J tau = 0.4, 60 collisions: largest cut negativity 1.45")
P4 = [(0, 1), (1, 2), (2, 3)]; mod = dict(edges=P4, net="ex", kind="x", sites=[1], n=4); kind, psi = K.collide_model(vac(4), mod, 5.0, 0.4, 60, K.PP["x"]["+"]); rho = K.rho_of_pure(psi)
best = max(float(C.negativity(np, rho, 4, list(c)).max()) for c in ([0], [1], [2], [3], [0, 1], [0, 2], [0, 3])); report("largest cut negativity over the 60 collisions", abs(best - 1.45) < 0.01, f"{best:.4f}")

section("(S3) collision recurrence against the explicit five-qubit Kraus map")
rng = np.random.default_rng(7); v = rng.normal(size=16) + 1j * rng.normal(size=16); v /= np.linalg.norm(v); rho0 = np.outer(v, v.conj()); worst = 0.0
for kind_, net, edges, sites in (("x", "ex", P4, [1]), ("x", "ex", P4, [0]), ("z", "xx", sorted(G4[1].edges()), [0, 2]), ("z", "xx", sorted(G4[5].edges()), [0, 1, 2, 3]), ("x", "xx", P4, [2])):
    mod = dict(edges=edges, net=net, kind=kind_, sites=sites, n=4)
    for anc in ("+", "1", "-"):
        pp = K.PP[kind_][anc]; Hp, Hm = K.model_fields(mod, 3.0); Up, Um = K.unitary(Hp, 0.4), K.unitary(Hm, 0.4); rec = pp * Up @ rho0 @ Up.conj().T + (1 - pp) * Um @ rho0 @ Um.conj().T
        kr = K.kraus_generic(rho0, mod, 3.0, 0.4, ETA[anc]); d = np.abs(rec - kr).max(); worst = max(worst, d)
        report(f"{net} network, {kind_} contact at {''.join('abcd'[s] for s in sites)}, ancilla {anc} (p_+ = {pp}): recurrence = Kraus map", d < 1e-12, f"{d:.1e}")
section("(S3b) which ancilla is coherent: |+> for the x contact, |1> for the z contact")
report("x contact: |+> gives p_+ = 1 (coherent), |1> gives p_+ = 1/2 (random sign)", K.PP["x"]["+"] == 1.0 and K.PP["x"]["1"] == 0.5)
report("z contact: |1> gives p_+ = 0 (coherent), |+> gives p_+ = 1/2", K.PP["z"]["1"] == 0.0 and K.PP["z"]["+"] == 0.5)

section("(S4) five-vertex graphs: ancilla vertex a, network = the other four vertices (XX), contact = ZZ on every edge at a")
def pairs5():
    out = []
    for gi, g in enumerate(G5):
        autos = X.automorphisms(g); seen = set()
        for a in sorted(g.nodes()):
            if a in seen: continue
            orb = sorted({p[a] for p in autos}); seen |= set(orb); out.append((gi, g, a, orb, autos))
    return out
PAIRS = pairs5()
def build(g, a):
    rest = [q for q in sorted(g.nodes()) if q != a]; idx = {q: i for i, q in enumerate(rest)}
    edges = [(idx[i], idx[j]) for i, j in sorted(g.edges()) if a not in (i, j)]; sites = [idx[j] for j in g.neighbors(a)]; return rest, idx, edges, sorted(sites)
worst = 0.0
for gi, g, a, orb, autos in PAIRS:
    rest, idx, edges, sites = build(g, a); mod = dict(edges=edges, net="xx", kind="z", sites=sites, n=4)
    # direct five-qubit evolution in the ORIGINAL vertex order
    def opq(o, q): return K.op(o, q, 5)
    H = sum(opq(K._PX, i) @ opq(K._PX, j) for i, j in g.edges() if a not in (i, j)) + 5.0 * sum(opq(K._PZ, a) @ opq(K._PZ, j) for j in g.neighbors(a)); U = K.unitary(H, 0.4)
    eta = np.array([1, 1], complex) / np.sqrt(2); st = np.array([1.0 + 0j]); 
    for q in range(5): st = np.kron(st, eta if q == a else np.array([1, 0], complex))
    r5 = np.outer(U @ st, (U @ st).conj()).reshape((2,) * 10); r5 = np.trace(r5, axis1=a, axis2=a + 5) if False else np.einsum(r5, list(range(5)) + list(range(5, 10)), [q for q in range(5) if q != a] + [q + 5 for q in range(5) if q != a] if False else None) if False else None
    psi_full = U @ st; T = psi_full.reshape((2,) * 5); T = np.moveaxis(T, a, 0).reshape(2, 16); rho_net = np.einsum("ai,aj->ij", T, T.conj())                  # trace out the ancilla, remaining qubits keep their order
    kr = K.kraus_generic(np.outer(vac(4), vac(4).conj()), mod, 5.0, 0.4, ETA["+"]); d = np.abs(rho_net - kr).max(); worst = max(worst, d)
report(f"all {len(PAIRS)} pairs: direct five-qubit graph evolution = network + ancilla model (|+> ancilla, g/J = 5)", worst < 1e-12, f"{worst:.1e}")

section("(S5) symmetry under the automorphisms fixing the ancilla vertex / contact site")
def perm_matrix(p):
    d = 16; idx = np.arange(d); bits = ((idx[:, None] >> (3 - np.arange(4))) & 1); new = np.zeros(d, int)
    for k in range(d): nb = np.zeros(4, int); nb[list(p)] = bits[k]; new[k] = int("".join(map(str, nb)), 2)
    P = np.zeros((d, d)); P[new, idx] = 1; return P
worst = 0.0; nst = 0
for gi, g, a, orb, autos in PAIRS:
    rest, idx, edges, sites = build(g, a); mod = dict(edges=edges, net="xx", kind="z", sites=sites, n=4); st = [p for p in autos if p[a] == a]
    for anc in ("+", "1"):
        kind, r = K.collide_model(vac(4), mod, 5.0, 0.4, 8, K.PP["z"][anc]); r = K.rho_of_pure(r) if kind == "pure" else r
        for p in st:
            q = tuple(idx[p[rest[i]]] for i in range(4)); P = perm_matrix(q); worst = max(worst, float(np.abs(P @ r @ P.T - r).max())); nst += 1
report(f"{len(PAIRS)} pairs x both ancillas, {nst} automorphism checks: P rho P^T = rho", worst < 1e-12, f"{worst:.1e}")
worst = 0.0
for g in G4:
    autos = X.automorphisms(g); edges = sorted(g.edges())
    for v, orb in K.contact_sites(g):
        mod = dict(edges=edges, net="ex", kind="x", sites=[v], n=4)
        for anc in ("+", "1"):
            kind, r = K.collide_model(vac(4), mod, 5.0, 0.4, 8, K.PP["x"][anc]); r = K.rho_of_pure(r) if kind == "pure" else r
            for p in K.stabilizer(autos, v): P = perm_matrix(p); worst = max(worst, float(np.abs(P @ r @ P.T - r).max()))
report("(EX, XX) setup: P rho P^T = rho for all 11 contact sites, both ancillas", worst < 1e-12, f"{worst:.1e}")

section("(S6) counts")
report("21 connected graphs on five vertices", len(G5) == 21, str(len(G5))); report("58 symmetry-inequivalent (graph, ancilla vertex) pairs", len(PAIRS) == 58, str(len(PAIRS)))

section("(S7) channel spectrum (plan Sec. 5.5), P4, contact site 2, g/J = 5, J tau = 0.4")
mod = dict(edges=P4, net="ex", kind="x", sites=[1], n=4)
sp = K.channel_spectrum(mod, 5.0, 0.4, 1.0); report("coherent drive (|+>): unitary channel, all 256 eigenvalues on the unit circle", sp["n_unit"] == 256, str(sp))
sp = K.channel_spectrum(mod, 5.0, 0.4, 0.5); report("random sign (|1>): at least one fixed point (trace) and a gap", sp["n_fixed"] >= 1 and sp["second"] < 1, str(sp))
print("\nOVERALL:", "PASS" if not fails else "FAIL: " + "; ".join(fails)); sys.exit(1 if fails else 0)
