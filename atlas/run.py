#!/usr/bin/env python3
"""CLI entry point for the entanglement scan (n = 2..5).

Examples
--------
  # one task (job-array element):
  python run.py --n 5 --graph-index 3 --interaction ex --init plus

  # everything for n<=5 (checkpointed/resumable; uses multiprocessing):
  python run.py --all --n-max 5

  # just regenerate graphs.json:
  python run.py --build-graphs --n-max 5

This script is deliberately "job-array ready": each (n, graph_index,
interaction, init) combination is an independent, checkpointed unit of work
(qnet.pipeline.run_task), addressable by SLURM_ARRAY_TASK_ID via --task-index
into the full task list, so the exact same code path runs whether invoked
directly or from an sbatch array.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from qnet import graphs as G
from qnet import pipeline as P

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run")


def _worker(task, config: P.TaskConfig):
    n, idx, interaction, init = task
    t0 = time.time()
    try:
        path = P.run_task(n, idx, interaction, init, config)
        return (task, "ok", path, time.time() - t0, None)
    except Exception as e:  # noqa: BLE001
        return (task, "error", None, time.time() - t0, f"{e}\n{traceback.format_exc()}")


def n_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, mp.cpu_count())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="single n (2..7)")
    ap.add_argument("--n-max", type=int, default=5, help="max n for --all / --build-graphs (default 5)")
    ap.add_argument("--graph-index", type=int, help="graph index within n (required with --n unless --all-graphs)")
    ap.add_argument("--interaction", choices=["xx", "zz", "ex"], help="interaction type")
    ap.add_argument("--init", choices=["0", "plus"], help="initial state")
    ap.add_argument("--all", action="store_true", help="run every task for n=2..--n-max")
    ap.add_argument("--all-graphs", action="store_true", help="run all graph indices for the given --n")
    ap.add_argument("--build-graphs", action="store_true", help="(re)download graphs and write graphs.json, then exit")
    ap.add_argument("--task-index", type=int, help="index into the full task list for --n-max (SLURM array mode)")
    ap.add_argument("--data-dir", default="data/graphs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=None, help="override worker count (default: SLURM_CPUS_PER_TASK or all CPUs)")
    ap.add_argument("--force", action="store_true", help="recompute even if checkpoint .npz already exists")
    ap.add_argument("--no-whole", action="store_true", help="disable the --whole (C_GME/Meyer-Wallach) computation")
    ap.add_argument("--gmn-n-uniform", type=int, default=100)
    ap.add_argument("--gmn-n-refine", type=int, default=20)
    ap.add_argument("--always-gmn", action="store_true",
                     help="new experiment mode: skip the Guhne-Seevinck witness and SDP subsampling "
                          "entirely, solving the exact GMN SDP at every single remaining (non-PPT, "
                          "non-bound-zero) grid point. Per-triple checkpointed (a single triple's SDP "
                          "sweep can itself take a long time). Much more expensive than the default mode.")
    ap.add_argument("--combo", action="append", default=None, metavar="INTERACTION:INIT",
                     help="restrict --all/--all-graphs to specific (interaction,init) pairs, e.g. "
                          "--combo zz:0 --combo ex:plus (repeatable). Overrides --interaction/--init "
                          "for task-list construction. Default: the full cross product.")
    args = ap.parse_args()

    combos = None
    if args.combo:
        combos = set()
        for c in args.combo:
            try:
                interaction, init = c.split(":")
            except ValueError:
                ap.error(f"--combo must be INTERACTION:INIT, got {c!r}")
            if interaction not in ("xx", "zz", "ex") or init not in ("0", "plus"):
                ap.error(f"--combo {c!r}: interaction must be xx/zz/ex, init must be 0/plus")
            combos.add((interaction, init))

    if args.build_graphs:
        G.download_graphs(n_max=max(args.n_max, 7), data_dir=args.data_dir)
        G.save_graphs_json(list(range(2, args.n_max + 1)), data_dir=args.data_dir, out_path="graphs.json")
        return

    config = P.TaskConfig(
        data_dir=args.data_dir, results_dir=args.results_dir,
        whole=not args.no_whole, force=args.force,
        gmn_n_uniform=args.gmn_n_uniform, gmn_n_refine=args.gmn_n_refine,
        always_gmn=args.always_gmn,
    )

    if args.task_index is not None:
        tasks = P.all_tasks(range(2, args.n_max + 1))
        if not (0 <= args.task_index < len(tasks)):
            logger.error("task-index %d out of range [0, %d)", args.task_index, len(tasks))
            sys.exit(2)
        tasks = [tasks[args.task_index]]
    elif args.all:
        tasks = P.all_tasks(range(2, args.n_max + 1))
        if combos is not None:
            tasks = [t for t in tasks if (t[2], t[3]) in combos]
    elif args.all_graphs:
        if args.n is None:
            ap.error("--all-graphs requires --n")
        n_graphs = len(G.load_graphs(args.n, data_dir=args.data_dir))
        if combos is not None:
            pairs = sorted(combos)
        else:
            interactions = [args.interaction] if args.interaction else ["xx", "zz", "ex"]
            inits = [args.init] if args.init else ["0", "plus"]
            pairs = [(i, s) for i in interactions for s in inits]
        tasks = [(args.n, idx, i, s) for idx in range(n_graphs) for (i, s) in pairs]
    else:
        if args.n is None or args.graph_index is None or args.interaction is None or args.init is None:
            ap.error("single-task mode requires --n --graph-index --interaction --init (or use --all / --all-graphs / --task-index)")
        tasks = [(args.n, args.graph_index, args.interaction, args.init)]

    logger.info("Running %d task(s) with %d worker(s)", len(tasks), args.workers or n_workers())
    nw = args.workers or n_workers()

    results = []
    t0 = time.time()
    if nw == 1 or len(tasks) == 1:
        for task in tasks:
            results.append(_worker(task, config))
    else:
        # Force "fork": this cluster's scratch filesystem does not support the
        # unix-domain sockets multiprocessing's default "forkserver"/"spawn"
        # start methods need (chmod on the socket file fails with ENOTSUP).
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=nw) as pool:
            async_results = [pool.apply_async(_worker, (task, config)) for task in tasks]
            for ar in async_results:
                results.append(ar.get())

    n_ok = sum(1 for r in results if r[1] == "ok")
    n_err = sum(1 for r in results if r[1] == "error")
    logger.info("Done: %d ok, %d error, total wall time %.1fs", n_ok, n_err, time.time() - t0)
    for task, status, path, dt, err in results:
        if status == "error":
            logger.error("FAILED %s after %.1fs: %s", task, dt, err)
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
