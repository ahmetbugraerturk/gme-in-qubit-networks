"""Save / load the XX-from-|0000> results in a plain, self-describing format (no pickles), so they can be re-plotted in any style.

<outdir>/<graph>.npz        compressed arrays, all on the FULL grid  t = J*t in [0, pi/2]  (n_intervals + 1 points):
    t
    pair/<xy>/C, pair/<xy>/N                      concurrence and negativity of the pair (one entry per symmetry orbit)
    tcut/<xyz>|<v>/N                              negativity of the 1|2 cut v|rest of triple xyz
    tri/<xyz>/{L,U,cls,conv}                      GMN bracket of the triple, class (0 zero, 1 positive, 2 near-threshold), converged
    qcut/<cut>/{N,purity}                         four-qubit bipartitions, e.g. a|bcd, ab|cd
    Q, cgme2                                      Meyer-Wallach Q and C_GME^2 of the four-qubit state
    gmn4/{L,U,cls,conv}                           GMN bracket of the four-qubit state
    part/<partition>                              largest block-respecting cut negativity, all 15 partitions (e.g. ab|c|d)
    fid/pair/<xy>/<target>, fid/tri/<xyz>/<target>, fid/whole/<target>     fidelities with Bell / GHZ / W targets
<outdir>/<graph>.meta.json  edges, qubit labels, symmetry maps (every pair/cut/triple -> the orbit representative that holds its data),
                            target families and witness thresholds, class legend, parameters, provenance.
Entries exist once per symmetry orbit; `Results` resolves any equivalent label to its data.
"""
import datetime, json, os
import numpy as np
from . import xxvac as X

CLASS_LEGEND = {0: "zero: U <= 1e-6 (rigorous)", 1: "positive: L - (U - L) > 1e-6 (margin rule, not rigorous)", 2: "near threshold: report [L, U]"}


def save(res, outdir, params=None):
    os.makedirs(outdir, exist_ok=True); name = res["name"]; mir = X.mirror; a = {}
    fid = res["fid"]; a["t"] = fid["th"]
    for r, v in res["pair"].items(): a[f"pair/{X.label(r)}/C"] = mir(v["C"]); a[f"pair/{X.label(r)}/N"] = mir(v["N"])
    for (t, v), arr in res["tcut"].items(): a[f"tcut/{X.label(t)}|{X.LET[v]}/N"] = mir(arr)
    def gm(prefix, d):
        if d is None: return
        a[f"{prefix}/L"] = mir(d["L"]); a[f"{prefix}/U"] = mir(d["U"]); a[f"{prefix}/cls"] = mir(d["cls"]).astype(np.int8); a[f"{prefix}/conv"] = mir(d["conv"]).astype(np.int8)
    for t, d in res["triple_gmn"].items(): gm(f"tri/{X.label(t)}", d)
    for c, d in res["qcut"].items(): a[f"qcut/{X.cut_label(c)}/N"] = mir(d["N"]); a[f"qcut/{X.cut_label(c)}/purity"] = mir(d["purity"])
    a["Q"] = mir(res["Q"]); a["cgme2"] = mir(res["cgme2"]); gm("gmn4", res["gmn4"])
    for P, v in res["part_maxN"].items(): a[f"part/{X.part_label(P)}"] = mir(v)
    for r, d in fid["pair"].items():
        for nm, v in d.items(): a[f"fid/pair/{X.label(r)}/{nm}"] = v
    for t, d in fid["triple"].items():
        for nm, v in d.items(): a[f"fid/tri/{X.label(t)}/{nm}"] = v
    for nm, v in fid["whole"].items(): a[f"fid/whole/{nm}"] = v
    meta = dict(name=name, edges=[[int(i), int(j)] for i, j in res["edges"]], qubits="abcd", n_autos=res["n_autos"], n_intervals=res["n_intervals"],
                units="t is J*t with H = J * sum_(i,j in E) X_i X_j, J = 1; qubit a is the leftmost tensor factor; initial state |0000>",
                symmetry="all curves except fidelities satisfy f(pi/2 - t) = f(t) (computed on [0, pi/4], mirrored); fidelities are computed on the full grid",
                pair_map={X.label(p): X.label(r) for p, r in res["pair_rep"].items()},
                tcut_map={f"{X.label(t)}|{X.LET[v]}": f"{X.label(r[0])}|{X.LET[r[1]]}" for (t, v), r in res["tcut_rep"].items()},
                triple_map={X.label(t): X.label(r) for t, r in res["triple_rep"].items()},
                qcut_map={X.cut_label(c): X.cut_label(r) for c, r in res["qcut_rep"].items()},
                targets={str(k): {nm: fid["family"][(k, nm)] for nm in fid["names"][k]} for k in (2, 3, 4)},
                witness_threshold={f"{k}:{fam}": v for (k, fam), v in fid["alpha"].items()},
                class_legend={str(k): v for k, v in CLASS_LEGEND.items()}, concurrence_roundoff=res["conc_roundoff"],
                params=params or {}, created=datetime.datetime.now().isoformat(timespec="seconds"),
                notes="GMN: L = primal estimate, U = certified upper bound (also capped by min cut negativity); PPT across a cut gives exact 0.")
    np.savez_compressed(os.path.join(outdir, f"{name}.npz"), **a)
    json.dump(meta, open(os.path.join(outdir, f"{name}.meta.json"), "w"), indent=1)


class Results:
    def __init__(self, path_prefix):
        self.z = np.load(path_prefix + ".npz"); self.meta = json.load(open(path_prefix + ".meta.json"))
        self.name = self.meta["name"]; self.edges = self.meta["edges"]; self.t = self.z["t"]
        self.Q = self.z["Q"]; self.cgme2 = self.z["cgme2"]; self._orbits = {}
        cp = path_prefix + ".cert.npz"; self.certz = np.load(cp) if os.path.exists(cp) else None     # optional witness certificates (witness_certs.py)

    @staticmethod
    def _norm(s): return "".join(sorted(s))
    def _cut_key(self, c):
        a, b = c.split("|"); return "|".join(sorted([self._norm(a), self._norm(b)], key=lambda x: (len(x), x)))

    def pair(self, p, what="C"): return self.z[f"pair/{self.meta['pair_map'][self._norm(p)]}/{what}"]
    def tcut_N(self, triple, v): return self.z[f"tcut/{self.meta['tcut_map'][self._norm(triple) + '|' + v]}/N"]
    def qcut(self, cut, what="N"): return self.z[f"qcut/{self.meta['qcut_map'][self._cut_key(cut)]}/{what}"]
    TOL_CONSISTENT = 1e-7          # = the solver's gap tolerance: an overshoot L - U below it is numerical noise, not an inconsistency

    @staticmethod
    def _consistent(d, tol=1e-7):
        """U is a certified upper bound but L is only a primal estimate.  An overshoot 0 < L - U <= tol is within the solver's own tolerance:
        L is silently clipped to U and nothing else changes.  A larger overshoot (an unconverged run) is a real inconsistency: L is clipped to U,
        the sample is marked unconverged and a 'positive' class (1) is downgraded to near threshold (2).  d['bad'] flags only those samples."""
        L, U = np.array(d["L"], float), np.asarray(d["U"], float); cls = np.array(d["cls"]).astype(int); conv = np.array(d["conv"]).astype(bool)
        over = L - U; bad = over > tol; L = np.where(over > 0, U, L)
        conv[bad] = False; cls[bad & (cls == 1)] = 2
        return dict(L=L, U=U, cls=cls.astype(np.int8), conv=conv, bad=bad)
    def triple_gmn(self, triple):
        k = f"tri/{self.meta['triple_map'][self._norm(triple)]}"
        return None if f"{k}/L" not in self.z.files else self._consistent({w: self.z[f"{k}/{w}"] for w in ("L", "U", "cls", "conv")})
    def cert_triple(self, triple):
        """int8 array on the full grid: bit 1 = Guhne-Seevinck witness, bit 2 = fidelity witness certify genuine multipartite entanglement (None if witness_certs.py was not run)."""
        return None if self.certz is None else self.certz[f"tri/{self.meta['triple_map'][self._norm(triple)]}"]
    def cert_whole(self): return None if self.certz is None else self.certz["gmn4"]
    def gmn4(self): return None if "gmn4/L" not in self.z.files else self._consistent({w: self.z[f"gmn4/{w}"] for w in ("L", "U", "cls", "conv")})
    def partition(self, label):
        blocks = sorted(["".join(sorted(b)) for b in label.split("|")], key=lambda b: (-len(b), b)); return self.z["part/" + "|".join(blocks)]
    def fid(self, subset, target):
        s = self._norm(subset)
        if len(s) == 2: return self.z[f"fid/pair/{self.meta['pair_map'][s]}/{target}"]
        if len(s) == 3: return self.z[f"fid/tri/{self.meta['triple_map'][s]}/{target}"]
        return self.z[f"fid/whole/{target}"]
    def targets(self, k): return self.meta["targets"][str(k)]
    def fid_best(self, subset, family):
        k = len(self._norm(subset)); return np.max([self.fid(subset, nm) for nm, f in self.targets(k).items() if f == family], axis=0)
    def threshold(self, k, family): return self.meta["witness_threshold"][f"{k}:{family}"]
    def equivalents(self, kind, label):
        """Labels that share the data of `label` (same orbit).  kind in pair, triple, tcut, qcut."""
        m = self.meta[f"{kind}_map"]; rep = m[label if kind in ("tcut",) else (self._cut_key(label) if kind == "qcut" else self._norm(label))]
        return sorted(k for k, v in m.items() if v == rep)


def load(path_prefix): return Results(path_prefix)
def load_all(outdir):
    suf = ".meta.json"
    return {f[: -len(suf)]: Results(os.path.join(outdir, f[: -len(suf)])) for f in sorted(os.listdir(outdir)) if f.endswith(suf)}
