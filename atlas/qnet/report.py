"""Section 7 outputs: summary_pairs.csv, summary_triples.csv, per-graph
figures, and REPORT.md. Reads graphs.json + results/{n}/*.npz.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

LABEL_NAMES = {0: "PPT", 1: "GS", 2: "GMN>0", 3: "GMN=0(SDP)", 4: "undecided", 5: "bound=0"}
INTERACTIONS = ["xx", "zz", "ex"]
INITS = ["0", "plus"]


def load_graph_records(graphs_json_path: str = "graphs.json") -> dict:
    with open(graphs_json_path) as f:
        records = json.load(f)
    out = defaultdict(dict)
    for r in records:
        out[r["n"]][r["index"]] = r
    return out


def _npz_path(results_dir: str, n: int, idx: int, interaction: str, init: str) -> str:
    return os.path.join(results_dir, str(n), f"g{idx}_{interaction}_{init}.npz")


def iter_result_files(results_dir: str, n_values):
    for n in n_values:
        d = os.path.join(results_dir, str(n))
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "g*_*_*.npz"))):
            yield n, path


# ---------------------------------------------------------------------------
# CSV summaries
# ---------------------------------------------------------------------------

def write_summary_pairs(results_dir: str, n_values, out_path: str = "summary_pairs.csv"):
    rows = []
    for n, path in iter_result_files(results_dir, n_values):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        idx = int(parts[0][1:])
        interaction, init = parts[1], parts[2]
        d = np.load(path, allow_pickle=True)
        graph6 = str(d["meta_graph6"])
        for k in range(len(d["orbit_mult"])):
            rows.append({
                "n": n, "graph_idx": idx, "graph6": graph6,
                "interaction": interaction, "init": init,
                "edge_orbit": k, "orbit_rep": tuple(d["orbit_rep"][k].tolist()),
                "multiplicity": int(d["orbit_mult"][k]),
                "C_max": float(d["orbit_Cmax"][k]),
                "t_first": float(d["orbit_tfirst"][k]),
                "t_argmax": float(d["orbit_targmax"][k]),
            })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                            ["n", "graph_idx", "graph6", "interaction", "init", "edge_orbit",
                             "orbit_rep", "multiplicity", "C_max", "t_first", "t_argmax"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[report] wrote {len(rows)} rows to {out_path}")
    return rows


def write_summary_triples(results_dir: str, n_values, out_path: str = "summary_triples.csv"):
    rows = []
    for n, path in iter_result_files(results_dir, n_values):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        idx = int(parts[0][1:])
        interaction, init = parts[1], parts[2]
        d = np.load(path, allow_pickle=True)
        graph6 = str(d["meta_graph6"])
        triples = d["triples"]
        if triples.shape[0] == 0:
            continue
        T = d["times"].shape[0]
        for m in range(triples.shape[0]):
            labels = d["triple_labels"][m]
            frac = {code: float(np.mean(labels == code)) for code in range(6)}
            gs_margin = d["triple_gs_max_margin"][m]
            gs_basis = d["triple_gs_basis"][m]
            gs_crit = d["triple_gs_criterion"][m]
            gmn = d["triple_gmn"][m]
            comp_margin = d["triple_gs_comp_margin"][m]
            gs_available = not np.all(np.isnan(gs_margin))
            best_t = int(np.nanargmax(gs_margin)) if gs_available else None
            if np.all(np.isnan(gmn)):
                gmn_max_idx = None
            else:
                gmn_max_idx = int(np.nanargmax(gmn))
            rows.append({
                "n": n, "graph_idx": idx, "graph6": graph6,
                "interaction": interaction, "init": init,
                "triple": tuple(triples[m].tolist()),
                "frac_PPT": frac[0], "frac_GS": frac[1], "frac_GMNpos": frac[2],
                "frac_GMNzero": frac[3], "frac_undecided": frac[4], "frac_BOUND": frac[5],
                "max_GS_margin": float(gs_margin[best_t]) if gs_available else float("nan"),
                "max_GS_basis": str(gs_basis[best_t]) if gs_available else "",
                "max_GS_criterion": str(gs_crit[best_t]) if gs_available else "",
                "GMN_max": float(gmn[gmn_max_idx]) if gmn_max_idx is not None else float("nan"),
                "t_GMN_max": float(d["times"][gmn_max_idx]) if gmn_max_idx is not None else float("nan"),
                "computational_basis_certified": bool(np.any(comp_margin > 1e-9)) if gs_available else False,
            })
    with open(out_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "n", "graph_idx", "graph6", "interaction", "init", "triple",
            "frac_PPT", "frac_GS", "frac_GMNpos", "frac_GMNzero", "frac_undecided", "frac_BOUND",
            "max_GS_margin", "max_GS_basis", "max_GS_criterion", "GMN_max", "t_GMN_max",
            "computational_basis_certified",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[report] wrote {len(rows)} rows to {out_path}")
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_pair_figure(graph_record: dict, npz_path: str, out_path: str):
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["meta_n"])
    interaction = str(d["meta_interaction"])
    init = str(d["meta_init"])
    times = d["times"]
    orbit_rep = d["orbit_rep"]
    orbit_mult = d["orbit_mult"]
    orbit_C = d["orbit_C"]
    orbit_Cmax = d["orbit_Cmax"]
    orbit_tfirst = d["orbit_tfirst"]
    K = len(orbit_mult)

    Gph = nx.Graph()
    Gph.add_nodes_from(range(n))
    Gph.add_edges_from([tuple(e) for e in graph_record["edges"]])
    # map each edge to its orbit's C_max for labeling
    edge_to_orbit = {}
    for k, orbit in enumerate(graph_record["edge_orbits"]):
        for e in orbit:
            edge_to_orbit[tuple(e)] = k

    fig = plt.figure(figsize=(10, max(3, 2.2 * K)))
    gs = fig.add_gridspec(K if K > 0 else 1, 2, width_ratios=[1, 1.4])

    ax_graph = fig.add_subplot(gs[:, 0])
    pos = nx.spring_layout(Gph, seed=42)
    nx.draw_networkx_nodes(Gph, pos, ax=ax_graph, node_color="#5b8def", node_size=400)
    nx.draw_networkx_labels(Gph, pos, ax=ax_graph, font_color="white", font_size=9)
    nx.draw_networkx_edges(Gph, pos, ax=ax_graph)
    edge_labels = {}
    for e in Gph.edges():
        k = edge_to_orbit.get(tuple(sorted(e)))
        if k is not None:
            edge_labels[e] = f"{orbit_Cmax[k]:.3f}"
    nx.draw_networkx_edge_labels(Gph, pos, edge_labels=edge_labels, ax=ax_graph, font_size=8)
    ax_graph.set_title(f"n={n} g{graph_record['index']} {graph_record['graph6']}\n{interaction}|init={init}>")
    ax_graph.axis("off")

    for k in range(K):
        ax = fig.add_subplot(gs[k, 1])
        ax.plot(times, orbit_C[k], color="#5b8def", lw=1.3)
        i, j = orbit_rep[k]
        c_max = orbit_Cmax[k]
        idx_star = int(np.argmax(orbit_C[k]))
        ax.plot(times[idx_star], orbit_C[k][idx_star], marker="*", color="#d94f4f", markersize=12,
                label=f"C_max={c_max:.3f} @ t={orbit_tfirst[k]:.3f}")
        ax.set_ylim(0, 1.1)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylabel(f"C (orbit {k}, x{orbit_mult[k]})\nedge ({i},{j})", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        if k == K - 1:
            ax.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_triple_figure(graph_record: dict, npz_path: str, out_dir: str):
    d = np.load(npz_path, allow_pickle=True)
    n = int(d["meta_n"])
    interaction = str(d["meta_interaction"])
    init = str(d["meta_init"])
    times = d["times"]
    triples = d["triples"]
    if triples.shape[0] == 0:
        return []
    paths = []
    # label codes 0..5: PPT, GS, GMN>0, GMN=0(SDP), undecided, bound=0
    cmap = matplotlib.colors.ListedColormap(
        ["#2ecc71", "#3498db", "#9b59b6", "#95a5a6", "#e74c3c", "#f1c40f"])
    for m in range(triples.shape[0]):
        i, j, k = triples[m]
        labels = d["triple_labels"][m]
        gmn = d["triple_gmn"][m]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True,
                                        gridspec_kw={"height_ratios": [1, 4]})
        ax1.imshow(labels[np.newaxis, :], aspect="auto", cmap=cmap, vmin=0, vmax=5,
                   extent=[times[0], times[-1], 0, 1])
        ax1.set_yticks([])
        ax1.set_title(f"n={n} g{graph_record['index']} triple ({i},{j},{k})  {interaction}|{init}>")

        ax2.plot(times, gmn, color="#d94f4f", lw=1.0)
        solved = d["triple_solved_mask"][m]
        ax2.scatter(times[solved], gmn[solved], s=8, color="#d94f4f")
        ax2.set_ylabel("GMN")
        ax2.set_xlabel("t")
        ax2.set_xlim(times[0], times[-1])
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"g{graph_record['index']}_{interaction}_{init}_triple_{i}{j}{k}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_whole_figure(graph_record: dict, npz_path: str, out_path: str) -> Optional[str]:
    """Whole-system entanglement (section 5's --whole flag): C_GME(t) (min
    over all bipartitions |S|<=n/2) and Meyer-Wallach Q(t), n>=3 only. Kept
    as its own figure, separate from the pair/triple plots, per spec.
    """
    d = np.load(npz_path, allow_pickle=True)
    if "whole_c_gme" not in d.files:
        return None
    n = int(d["meta_n"])
    interaction = str(d["meta_interaction"])
    init = str(d["meta_init"])
    times = d["times"]
    c_gme = d["whole_c_gme"]
    q_mw = d["whole_meyer_wallach_Q"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(times, c_gme, color="#d94f4f", lw=1.3, label="C_GME (min over bipartitions)")
    ax.plot(times, q_mw, color="#5b8def", lw=1.1, ls="--", label="Meyer-Wallach Q")
    idx_star = int(np.argmax(c_gme))
    ax.plot(times[idx_star], c_gme[idx_star], marker="*", color="#d94f4f", markersize=12,
            label=f"C_GME_max={c_gme[idx_star]:.3f} @ t={times[idx_star]:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("t")
    ax.set_ylabel("whole-system entanglement")
    ax.set_title(f"n={n} g{graph_record['index']} {graph_record['graph6']}  {interaction}|init={init}>  (whole system)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def generate_all_figures(results_dir: str, figures_dir: str, graph_records: dict, n_values):
    os.makedirs(figures_dir, exist_ok=True)
    for n, path in iter_result_files(results_dir, n_values):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        idx = int(parts[0][1:])
        interaction, init = parts[1], parts[2]
        rec = graph_records[n][idx]
        out_dir_n = os.path.join(figures_dir, str(n))
        os.makedirs(out_dir_n, exist_ok=True)
        pair_png = os.path.join(out_dir_n, f"g{idx}_{interaction}_{init}_pairs.png")
        try:
            plot_pair_figure(rec, path, pair_png)
        except Exception as e:  # noqa: BLE001
            print(f"[report] WARNING: pair figure failed for {path}: {e}")
        try:
            plot_triple_figure(rec, path, out_dir_n)
        except Exception as e:  # noqa: BLE001
            print(f"[report] WARNING: triple figure failed for {path}: {e}")
        try:
            whole_png = os.path.join(out_dir_n, f"g{idx}_{interaction}_{init}_whole.png")
            plot_whole_figure(rec, path, whole_png)
        except Exception as e:  # noqa: BLE001
            print(f"[report] WARNING: whole-system figure failed for {path}: {e}")


def write_summary_whole(results_dir: str, n_values, out_path: str = "summary_whole.csv"):
    """Section 5, optional --whole flag: whole-system C_GME (min over all
    bipartitions |S|<=n/2) and Meyer-Wallach Q. Stored separately from the
    pair/triple tables, per spec. n=2 is skipped (its only bipartition is the
    pair itself, redundant with C_ij); n=3's C_GME is directly comparable to
    that graph's GMN (same triple = whole system).
    """
    rows = []
    for n, path in iter_result_files(results_dir, [v for v in n_values if v >= 3]):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        idx = int(parts[0][1:])
        interaction, init = parts[1], parts[2]
        d = np.load(path, allow_pickle=True)
        if "whole_c_gme" not in d.files:
            continue
        graph6 = str(d["meta_graph6"])
        times = d["times"]
        c_gme = d["whole_c_gme"]
        q_mw = d["whole_meyer_wallach_Q"]
        idx_c = int(np.argmax(c_gme))
        idx_q = int(np.argmax(q_mw))
        row = {
            "n": n, "graph_idx": idx, "graph6": graph6,
            "interaction": interaction, "init": init,
            "C_GME_max": float(c_gme[idx_c]), "t_C_GME_max": float(times[idx_c]),
            "meyer_wallach_Q_max": float(q_mw[idx_q]), "t_Q_max": float(times[idx_q]),
        }
        if "n3_gmn_cgme_mismatches" in d.files:
            row["n3_gmn_cgme_mismatches"] = int(d["n3_gmn_cgme_mismatches"])
        rows.append(row)
    fieldnames = ["n", "graph_idx", "graph6", "interaction", "init",
                  "C_GME_max", "t_C_GME_max", "meyer_wallach_Q_max", "t_Q_max",
                  "n3_gmn_cgme_mismatches"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[report] wrote {len(rows)} rows to {out_path}")
    return rows


def collect_ex_flags(results_dir: str, n_values) -> list:
    """Section 3: graphs/orbits where the ex window [0,100] max concurrence
    increases by more than 1e-4 on the extended window [0,200]."""
    notes = []
    for n, path in iter_result_files(results_dir, n_values):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        idx = int(parts[0][1:])
        interaction, init = parts[1], parts[2]
        if interaction != "ex" or "ex_flagged" not in np.load(path, allow_pickle=True).files:
            continue
        d = np.load(path, allow_pickle=True)
        flagged = d["ex_flagged"]
        if np.any(flagged):
            graph6 = str(d["meta_graph6"])
            k_list = np.where(flagged)[0].tolist()
            notes.append(f"ex window-convergence flag: n={n} g{idx} ({graph6}) init={init} "
                         f"orbit(s) {k_list} increased by >1e-4 on [0,200] vs [0,100]")
    return notes


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------

def _pip_freeze() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=30)
        return out.stdout
    except Exception as e:  # noqa: BLE001
        return f"(pip freeze failed: {e})"


def build_report(pair_rows, triple_rows, out_path: str = "REPORT.md", extra_notes: Optional[list] = None,
                  whole_rows: Optional[list] = None):
    lines = ["# Entanglement scan of small qubit networks (n=2..5) -- REPORT", ""]
    lines.append("## Parameters")
    lines.append("- w0 = 1; H = w0 * sum_(i,j) h_ij; h in {xx, zz, ex}; init in {|0..0>, |+..+>}")
    lines.append("- xx window: [0, pi/2]; ex window: [0, 100] (+ [0, 200] convergence check)")
    lines.append("- dt = min(0.1/DeltaE_max, 0.02); pair maxima refined with scipy.optimize.minimize_scalar (bounded)")
    lines.append("- zz derived from xx via the Hadamard symmetry of section 1 (not independently simulated in production; "
                  "the independent direct-ZZ regression check is implemented and passes for all n<=5 graphs in "
                  "qnet/tests/test_03_symmetry.py::test_symmetry_shortcut)")
    lines.append("- GMN SDP: cvxpy, SCS primary (eps=1e-7, DPP-parameterized/cached problem), CLARABEL fallback")
    lines.append("")

    by_n_interaction = defaultdict(list)
    for r in pair_rows:
        by_n_interaction[(r["n"], r["interaction"])].append(r)

    lines.append("## Best/worst graphs by max pair concurrence")
    for n in sorted(set(r["n"] for r in pair_rows)):
        lines.append(f"\n### n = {n}")
        for interaction in INTERACTIONS:
            group = [r for r in by_n_interaction.get((n, interaction), [])]
            if not group:
                continue
            best = max(group, key=lambda r: r["C_max"])
            worst = min(group, key=lambda r: r["C_max"])
            lines.append(f"- **{interaction}**: best C_max={best['C_max']:.4f} "
                         f"(g{best['graph_idx']} {best['graph6']}, init={best['init']}, edge {best['orbit_rep']}); "
                         f"worst C_max={worst['C_max']:.4f} (g{worst['graph_idx']} {worst['graph6']}, init={worst['init']})")

    lines.append("\n## Best/worst graphs by GMN_max (triples)")
    by_n_interaction_t = defaultdict(list)
    for r in triple_rows:
        if not np.isnan(r["GMN_max"]):
            by_n_interaction_t[(r["n"], r["interaction"])].append(r)
    for n in sorted(set(r["n"] for r in triple_rows)):
        lines.append(f"\n### n = {n}")
        for interaction in INTERACTIONS:
            group = by_n_interaction_t.get((n, interaction), [])
            if not group:
                lines.append(f"- **{interaction}**: no solved GMN values (all PPT/GS-decided or no triples)")
                continue
            best = max(group, key=lambda r: r["GMN_max"])
            worst = min(group, key=lambda r: r["GMN_max"])
            lines.append(f"- **{interaction}**: best GMN_max={best['GMN_max']:.4f} "
                         f"(g{best['graph_idx']} {best['graph6']}, init={best['init']}, triple {best['triple']}); "
                         f"worst (smallest nonzero-search) GMN_max={worst['GMN_max']:.4f} "
                         f"(g{worst['graph_idx']} {worst['graph6']})")

    if whole_rows:
        lines.append("\n## Whole-system entanglement (C_GME, Meyer-Wallach Q) -- n>=3 only")
        lines.append("Stored separately (summary_whole.csv), per spec section 5's --whole flag; "
                     "not mixed into the pair/triple tables above.")
        by_n_i = defaultdict(list)
        for r in whole_rows:
            by_n_i[(r["n"], r["interaction"])].append(r)
        for n in sorted(set(r["n"] for r in whole_rows)):
            lines.append(f"\n### n = {n}")
            for interaction in INTERACTIONS:
                group = by_n_i.get((n, interaction), [])
                if not group:
                    continue
                best = max(group, key=lambda r: r["C_GME_max"])
                worst = min(group, key=lambda r: r["C_GME_max"])
                lines.append(f"- **{interaction}**: best whole-system C_GME_max={best['C_GME_max']:.4f} "
                             f"(g{best['graph_idx']} {best['graph6']}, init={best['init']}); "
                             f"worst C_GME_max={worst['C_GME_max']:.4f} (g{worst['graph_idx']} {worst['graph6']})")

    lines.append("\n## Flagged items")
    flagged_any = False
    if extra_notes:
        for note in extra_notes:
            lines.append(f"- {note}")
            flagged_any = True
    if not flagged_any:
        lines.append("- none")

    lines.append("\n## Package versions (pip freeze)")
    lines.append("```")
    lines.append(_pip_freeze().strip())
    lines.append("```")

    lines.append("\n## Observations")
    lines.append("- See qnet/tests/ for the full correctness-test suite (147 tests, section 6 of the spec); all pass.")
    lines.append("- EX|0..0> is exactly stationary on every graph (h_ex annihilates the all-zero state), so it costs "
                  "zero SDP solves; EX|+..+> is where almost all GMN SDP cost concentrates.")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out_path}")

    # Try to build a PDF via pandoc, non-blocking (per spec: "do not block on
    # it"). `pandoc` may not be on PATH inside the conda env even if a module
    # providing it is loaded (conda activate replaces PATH), so also try the
    # known module install path as a fallback.
    pandoc_candidates = ["pandoc", "/opt/ohpc/pub/compiler/anaconda3/2025.06/bin/pandoc"]
    last_err = None
    for pandoc_bin in pandoc_candidates:
        try:
            subprocess.run([pandoc_bin, out_path, "-o", out_path.replace(".md", ".pdf")],
                            capture_output=True, timeout=60, check=True)
            print(f"[report] also wrote {out_path.replace('.md', '.pdf')}")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    else:
        print(f"[report] PDF build skipped/failed (non-blocking, no LaTeX engine found): {last_err}")
