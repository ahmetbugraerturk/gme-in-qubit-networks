#!/usr/bin/env python3
"""Plot the saved XX-from-|0000> results (qgpu.store) in any style, one figure per graph.   python plot_xx.py --style light|dark|paper|mono
Reads ONLY the saved store (results/xx_vacuum/*.npz + .meta.json); nothing is recomputed, so a new style is a few lines in STYLES.

Colour roles (consistent in every style):
  L  = certified lower bound on GMN                   blue
  U  = ADMM certified upper bound on GMN           orange   (band between L and U shaded)
  our assumptions: decision threshold tau = 1e-6, fidelity witness thresholds    violet, dashed/dotted
  decision classes along the time axis: zero (U<=tau) aqua, positive (certified L > tau) green, near-threshold yellow hatched
  other quantities are identified by entity (orbit of equivalent subsets/cuts), colour slots in fixed order plus a dash pattern."""
import argparse, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from qgpu import store

LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
STYLES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e8e7e3", pal=LIGHT, font="DejaVu Sans"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#2d2d2b", pal=DARK, font="DejaVu Sans"),
    "paper": dict(bg="#ffffff", ink="#000000", ink2="#333333", grid="#dddddd", pal=LIGHT, font="DejaVu Serif"),
    "mono": dict(bg="#ffffff", ink="#000000", ink2="#444444", grid="#e5e5e5", font="DejaVu Sans",
                 pal=['#000000', '#6e6e6e', '#222222', '#9a9a9a', '#444444', '#7f7f7f', '#555555', '#b0b0b0']),
}
ROLE = dict(L=0, U=1, assume=6, zero=2, positive=5, near=3)
DASH = ["-", "--", "-.", ":"]
POS = {  # node positions per graph, keyed by the vertex ids of graph4c.g6
    "star": {3: (0, 0), 0: (0, 1), 1: (-.87, -.5), 2: (.87, -.5)},
    "path": {2: (0, 0), 0: (1, 0), 3: (2, 0), 1: (3, 0)},
    "paw": {3: (1, 0), 0: (0, .8), 2: (0, -.8), 1: (2, 0)},
    "cycle": {0: (0, 1), 2: (1, 1), 1: (1, 0), 3: (0, 0)},
    "diamond": {2: (-.6, 0), 3: (.6, 0), 0: (0, .9), 1: (0, -.9)},
    "K4": {0: (0, 1), 1: (1, 1), 3: (1, 0), 2: (0, 0)},
}
TITLE = {"star": "Star K1,3", "path": "Path P4", "paw": "Paw", "cycle": "Cycle C4", "diamond": "Diamond (K4 minus an edge)", "K4": "Complete K4"}
LET = "abcd"; TAU = 1e-6


def uniq(m): return sorted(set(m.values()))
def tlabel(rep):  # "abc|a" -> "a|bc"
    t, v = rep.split("|"); return f"{v}|{''.join(c for c in t if c != v)}"


def setup(st):
    plt.rcParams.update({"font.family": st["font"], "figure.facecolor": st["bg"], "axes.facecolor": st["bg"], "savefig.facecolor": st["bg"],
                         "text.color": st["ink"], "axes.labelcolor": st["ink2"], "xtick.color": st["ink2"], "ytick.color": st["ink2"],
                         "axes.edgecolor": st["grid"], "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "legend.fontsize": 7.6, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})


def axes_time(ax, st, ylabel=None, ymax=None):
    ax.set_xlim(0, 0.5); ax.set_xticks([0, .125, .25, .375, .5]); ax.set_xticklabels(["0", "π/8", "π/4", "3π/8", "π/2"])
    ax.grid(True, color=st["grid"], lw=0.8); ax.set_axisbelow(True); ax.axvline(0.25, color=st["ink2"], lw=0.8, alpha=0.6)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_xlabel("Jt", labelpad=1)
    if ymax is not None: ax.set_ylim(0, ymax)


def headroom(ax, f=1.45):
    """leave room above the curves for the legend: y from 0 to f * (largest plotted value)"""
    ys = [float(np.max(l.get_ydata())) for l in ax.lines if len(l.get_ydata()) > 2]
    if ys: ax.set_ylim(0, max(ys) * f)


def color(st, i): return st["pal"][i % len(st["pal"])]


def line(ax, st, x, y, i, label, lw=1.9, ls=None, role=None):
    ax.plot(x, y, color=color(st, i if role is None else ROLE[role]), ls=ls or DASH[(i // 3) % 4 if role is None else 0], lw=lw, label=label, solid_capstyle="round")


def draw_graph(ax, st, R):
    pos = POS[R.name]; ax.set_aspect("equal"); ax.axis("off")
    for i, j in R.edges: ax.plot(*zip(pos[i], pos[j]), color=st["ink2"], lw=2.2, zorder=1, solid_capstyle="round")
    for v, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.17, fc=st["bg"], ec=st["ink"], lw=1.8, zorder=2)); ax.text(x, y, LET[v], ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)
    xs, ys = zip(*pos.values()); ax.set_xlim(min(xs) - .4, max(xs) + .4); ax.set_ylim(min(ys) - .4, max(ys) + .4)
    ax.set_title(f"{TITLE[R.name]}   |Aut| = {R.meta['n_autos']}", pad=2)


def strip(ax, st, x, cls):
    """decision classes along the time axis; hatch = secondary encoding"""
    spec = {0: (color(st, ROLE["zero"]), ".."), 1: (color(st, ROLE["positive"]), ""), 2: (color(st, ROLE["near"]), "////")}
    i = 0; n = len(cls); dx = x[1] - x[0]
    while i < n:
        j = i
        while j + 1 < n and cls[j + 1] == cls[i]: j += 1
        fc, h = spec[int(cls[i])]; ax.axvspan(x[i] - dx / 2, x[j] + dx / 2, fc=fc, hatch=h, ec=st["bg"], lw=0, alpha=0.9); i = j + 1
    ax.set_xlim(0, 0.5); ax.set_yticks([]); ax.set_xticks([]); ax.axvline(0.25, color=st["ink"], lw=0.8, alpha=0.6)
    for s in ax.spines.values(): s.set_visible(False)


def gmn_panel(fig, ax, st, x, d, title, symlog):
    L, U, cls = d["L"], d["U"], d["cls"]; marg = np.clip(2 * L - U, 0, None)
    ax.fill_between(x, L, U, color=color(st, ROLE["U"]), alpha=0.16, lw=0, label="band L to U")
    line(ax, st, x, U, 0, "U  certified upper bound", lw=1.5, ls="--", role="U")
    line(ax, st, x, L, 0, "L  certified lower bound", lw=2.0, role="L")
    if symlog:
        line(ax, st, x, np.clip(U - L, 0, None), 0, "U-L  width of the bracket", lw=1.3, ls=":", role="assume")
        ax.axhline(TAU, color=color(st, ROLE["assume"]), lw=1.3, ls=(0, (6, 3)), label="τ = 1e-6  our threshold")
        ax.set_yscale("symlog", linthresh=1e-8, linscale=0.35); ax.set_ylim(0, max(0.6, float(np.max(U)) * 1.5))
        ax.set_yticks([0, 1e-8, 1e-6, 1e-4, 1e-2, 0.5]); ax.set_yticklabels(["0", "1e-8", "1e-6", "1e-4", "1e-2", "0.5"]); ax.minorticks_off()
    else:
        ax.set_ylim(0, max(0.05, float(np.max(U)) * 1.12))
    axes_time(ax, st); ax.set_title(title)
    sax = ax.inset_axes([0, -0.30, 1, 0.075]); strip(sax, st, x, cls)
    nn = int((cls == 2).sum()); ax.text(1.0, -0.43, f"near-threshold samples: {nn} of {len(cls)}", transform=ax.transAxes, ha="right", fontsize=7.4, color=st["ink2"])
    if symlog: ax.legend(loc="lower right", ncol=1, handlelength=2.4)


def render(R, st, path):
    setup(st); x = R.t / math.pi; m = R.meta
    fig = plt.figure(figsize=(21, 20), dpi=125); gs = fig.add_gridspec(5, 4, hspace=0.95, wspace=0.27, left=0.04, right=0.99, top=0.925, bottom=0.045)
    fig.suptitle(f"{TITLE[R.name]}: XX network from |0000>, entanglement versus time", x=0.04, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.957, "H = Σ XᵢXⱼ over the drawn edges, J = 1.  Equivalent subsets (same orbit under the graph's symmetry) share one curve and are listed together in the legends.",
             ha="left", fontsize=10, color=st["ink2"])
    draw_graph(fig.add_subplot(gs[0, 0]), st, R)

    # pairs
    ax = fig.add_subplot(gs[0, 1]); reps = uniq(m["pair_map"])
    for k, r in enumerate(reps): line(ax, st, x, R.pair(r), k, " = ".join(R.equivalents("pair", r)))
    axes_time(ax, st, "concurrence C"); headroom(ax, 1.5); ax.set_title("Pairs: concurrence (all 6 pairs)"); ax.legend(loc="upper right", ncol=2, handlelength=2.2)

    # Q and C_GME^2 on ONE axis
    ax = fig.add_subplot(gs[0, 2]); line(ax, st, x, R.Q, 0, "Q  Meyer-Wallach"); line(ax, st, x, R.cgme2, 1, "C_GME²  (min over cuts of 2(1-Tr ρ²))", ls="--")
    axes_time(ax, st); headroom(ax, 1.4); ax.set_title("Four qubits: Q and C_GME²"); ax.legend(loc="upper right", handlelength=2.4)

    # exclusion strip for the 15 partitions
    ax = fig.add_subplot(gs[0, 3]); labs = sorted([k for k in R.z.files if k.startswith("part/")], key=lambda s: (s.count("|"), s)); labs = [s[5:] for s in labs]
    for r, lab in enumerate(labs):
        exc = R.partition(lab) > 1e-9; ax.fill_between(x, r, r + 0.8, where=exc, color=color(st, 0), lw=0, step="mid")
    ax.set_yticks(np.arange(len(labs)) + 0.4); ax.set_yticklabels(["".join(l) for l in labs], fontsize=7); ax.set_ylim(0, len(labs)); ax.invert_yaxis()
    axes_time(ax, st); ax.set_title("Partitions excluded (shaded)")

    # triple cut negativities
    ax = fig.add_subplot(gs[1, 0]); reps = uniq(m["tcut_map"])
    for k, r in enumerate(reps): line(ax, st, x, R.z[f"tcut/{r}/N"], k, f"{tlabel(r)} of {r.split('|')[0]}" + (f"  (+{len(R.equivalents('tcut', r)) - 1})" if len(R.equivalents('tcut', r)) > 1 else ""))
    axes_time(ax, st, "negativity N"); headroom(ax, 1.7); ax.set_title("Triples: negativity of each 1|2 cut"); ax.legend(loc="upper right", ncol=2, fontsize=6.8, handlelength=2.0)

    # four-qubit cuts
    ax = fig.add_subplot(gs[1, 1]); reps = uniq(m["qcut_map"])
    for k, r in enumerate(reps): line(ax, st, x, R.qcut(r), k, " = ".join(R.equivalents("qcut", r)))
    axes_time(ax, st, "negativity N"); headroom(ax, 1.55); ax.set_title("Four qubits: negativity of the 7 cuts"); ax.legend(loc="upper right", ncol=2, handlelength=2.0, fontsize=7)

    # 2|1|1 partitions: largest block-respecting cut negativity
    ax = fig.add_subplot(gs[1, 2]); pl = [l for l in labs if l.count("|") == 2]; groups = []
    for l in pl:
        for g in groups:
            if np.allclose(R.partition(g[0]), R.partition(l), atol=1e-12): g.append(l); break
        else: groups.append([l])
    for k, g in enumerate(groups): line(ax, st, x, R.partition(g[0]), k, " = ".join(g))
    axes_time(ax, st, "max N over block cuts"); headroom(ax, 1.5); ax.set_title("Three-block partitions (ab|c|d ...)"); ax.legend(loc="upper right", fontsize=7, handlelength=2.0)

    # notes / legend of colour roles
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off"); ax.set_title("How to read the GMN rows")
    hs = [Line2D([], [], color=color(st, ROLE["L"]), lw=2), Line2D([], [], color=color(st, ROLE["U"]), lw=1.5, ls="--"),
          Line2D([], [], color=color(st, ROLE["assume"]), lw=1.3, ls=(0, (6, 3))), Line2D([], [], color=color(st, ROLE["assume"]), lw=1.3, ls=":"),
          Patch(fc=color(st, ROLE["zero"]), hatch="..", ec=st["bg"]), Patch(fc=color(st, ROLE["positive"]), ec=st["bg"]), Patch(fc=color(st, ROLE["near"]), hatch="////", ec=st["bg"])]
    ax.legend(hs, ["L: certified lower bound on GMN", "U: certified upper bound on GMN", "our threshold τ = 1e-6",
                   "U-L: width of the bracket (log-like rows)", "strip: zero  (U ≤ τ, rigorous)", "strip: positive  (certified L > τ)",
                   "strip: near threshold, report [L, U]"], loc="upper left", fontsize=8.4, handlelength=2.6, labelspacing=0.8, bbox_to_anchor=(0, 1.0))
    ax.text(0, 0.02, f"edges: {', '.join(LET[i] + LET[j] for i, j in R.edges)}\nstate |0000>, J t from 0 to π/2; mirror about π/4 (all but fidelity)\nconcurrence round-off {m['concurrence_roundoff']:.1e}", transform=ax.transAxes, fontsize=8, color=st["ink2"], va="bottom")

    # GMN rows: linear then symlog (so that tau and the L..U gap are visible)
    ents = [(f"GMN of {r} (triples {', '.join(R.equivalents('triple', r))})", R.triple_gmn(r)) for r in uniq(m["triple_map"])]
    if R.gmn4() is not None: ents.append(("GMN of abcd", R.gmn4()))
    for c in range(4):
        for row, sym in ((2, False), (3, True)):
            ax = fig.add_subplot(gs[row, c])
            if c < len(ents) and ents[c][1] is not None: gmn_panel(fig, ax, st, x, ents[c][1], ents[c][0] + ("  (log-like scale)" if sym else "  (linear)"), sym)
            else: ax.axis("off")

    # fidelities
    ax = fig.add_subplot(gs[4, 0]); th = R.meta["witness_threshold"]
    for k, r in enumerate(uniq(m["pair_map"])): line(ax, st, x, R.fid_best(r, "Bell"), k, f"{' = '.join(R.equivalents('pair', r))}: best Bell")
    ax.axhline(th["2:Bell"], color=color(st, ROLE["assume"]), lw=1.3, ls=(0, (6, 3)), label="witness 1/2: above proves entanglement"); axes_time(ax, st, "fidelity"); headroom(ax, 1.55)
    ax.set_title("Pairs: best fidelity with a Bell state"); ax.legend(loc="upper right", fontsize=6.8, ncol=1, handlelength=2.0)
    for c, (sub_list, k) in enumerate((([r for r in uniq(m["triple_map"])], 3), (["abcd"], 4))):
        ax = fig.add_subplot(gs[4, 1 + c])
        for si, sname in enumerate(sub_list):
            for fam, ci in (("GHZ", 0), ("W", 1)):
                line(ax, st, x, R.fid_best(sname, fam), ci, f"{sname}: best {fam}" if len(sub_list) > 1 or True else None, ls=DASH[si % 4], lw=1.7)
        ax.axhline(th[f"{k}:GHZ"], color=color(st, ROLE["assume"]), lw=1.3, ls=(0, (6, 3)), label=f"GHZ witness {th[f'{k}:GHZ']:.2g}")
        ax.axhline(th[f"{k}:W"], color=color(st, ROLE["assume"]), lw=1.3, ls=":", label=f"W witness {th[f'{k}:W']:.2g}")
        axes_time(ax, st, "fidelity"); headroom(ax, 1.75); ax.set_title(f"{'Triples' if k == 3 else 'Four qubits'}: best fidelity with GHZ and W (z, x, y bases)")
        ax.legend(loc="upper right", fontsize=6.6, ncol=2, handlelength=2.0)
    ax = fig.add_subplot(gs[4, 3]); ax.axis("off"); ax.set_title("Fidelity targets")
    ax.text(0, 0.98, "pairs: Φ±, Ψ± (4 Bell states)\ntriples, four qubits: GHZ± and W, W' (flipped)\neach in the z, x and y local bases;\ncurves show the best of each family.\n\nviolet dashed/dotted: witness thresholds\n(F above the line proves entanglement of a pair,\nor genuine multipartite entanglement of 3 or 4 qubits).\nFidelity is not mirror symmetric: full grid is used.",
            transform=ax.transAxes, fontsize=8.6, va="top", color=st["ink2"])
    os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--style", default="light", choices=list(STYLES))
    ap.add_argument("--results", default=os.path.join(HERE, "results", "xx_vacuum")); ap.add_argument("--out", default=None); ap.add_argument("--only", default=None)
    a = ap.parse_args(); out = a.out or os.path.join(HERE, "figures", a.style)
    for name, R in store.load_all(a.results).items():
        if a.only and name != a.only: continue
        p = os.path.join(out, f"xx_{name}.png"); render(R, STYLES[a.style], p); print("wrote", p, flush=True)
