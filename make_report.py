#!/usr/bin/env python3
"""Generate summary_pairs.csv, summary_triples.csv, figures/, REPORT.md from
whatever is currently in results/ (per section 7 of the spec). Safe to run
repeatedly / after a partial production run.
"""
import argparse

from qnet import report as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--graphs-json", default="graphs.json")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--suffix", default="",
                     help="appended to all output filenames (summary_pairs{suffix}.csv, "
                          "REPORT{suffix}.md, figures{suffix}/) so a separate experiment "
                          "(different --results-dir) never overwrites the main report.")
    args = ap.parse_args()

    n_values = list(range(2, args.n_max + 1))
    graph_records = R.load_graph_records(args.graphs_json)
    figures_dir = args.figures_dir + args.suffix if args.suffix else args.figures_dir

    pair_rows = R.write_summary_pairs(args.results_dir, n_values, f"summary_pairs{args.suffix}.csv")
    triple_rows = R.write_summary_triples(args.results_dir, n_values, f"summary_triples{args.suffix}.csv")
    whole_rows = R.write_summary_whole(args.results_dir, n_values, f"summary_whole{args.suffix}.csv")
    if not args.no_figures:
        R.generate_all_figures(args.results_dir, figures_dir, graph_records, n_values)
    ex_notes = R.collect_ex_flags(args.results_dir, n_values)
    for r in whole_rows:
        if r.get("n3_gmn_cgme_mismatches"):
            ex_notes.append(f"n=3 GMN<=>C_GME mismatch: g{r['graph_idx']} ({r['graph6']}) "
                             f"{r['interaction']}|{r['init']}>: {r['n3_gmn_cgme_mismatches']} solved time(s) disagree")
    R.build_report(pair_rows, triple_rows, f"REPORT{args.suffix}.md", extra_notes=ex_notes, whole_rows=whole_rows)


if __name__ == "__main__":
    main()
