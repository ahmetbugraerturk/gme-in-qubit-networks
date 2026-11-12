"""Graph source: download/parse connected simple graphs (graph6 format) from
Brendan McKay's collection, with a networkx fallback. Also computes edge
orbits under Aut(G), needed because the symmetric initial states |0...0>
and |+...+> make concurrences equal within an edge orbit.
"""
from __future__ import annotations

import gzip
import itertools
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx

BASE_URL = "https://users.cecs.anu.edu.au/~bdm/data/"
# Verified connected-graph counts, OEIS A001349 (also cross-checked against
# McKay's graphcounts.txt on 2026-10-03).
EXPECTED_CONNECTED_COUNTS = {1: 1, 2: 1, 3: 2, 4: 6, 5: 21, 6: 112, 7: 853}


def _candidate_urls(n: int) -> List[str]:
    """graph{n}c.g6 (small n) or graph{n}c.g6.gz (larger n); try both."""
    return [f"{BASE_URL}graph{n}c.g6", f"{BASE_URL}graph{n}c.g6.gz"]


def download_graphs(n_max: int = 7, data_dir: str = "data/graphs", timeout: float = 20.0) -> None:
    """Download connected-graph graph6 files for n = 2..n_max into data_dir,
    caching whatever is already present. Capable of n up to 7 (not used for
    the n<=5 production run, per spec).
    """
    os.makedirs(data_dir, exist_ok=True)
    for n in range(2, n_max + 1):
        dest = os.path.join(data_dir, f"graph{n}c.g6")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        last_err = None
        for url in _candidate_urls(n):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    data = resp.read()
                if url.endswith(".gz"):
                    data = gzip.decompress(data)
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"[graphs] downloaded n={n} from {url} ({len(data)} bytes)")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        else:
            print(f"[graphs] WARNING: could not download n={n}: {last_err}")


def _read_graph6_file(path: str) -> List[nx.Graph]:
    graphs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = nx.from_graph6_bytes(line.encode("ascii"))
            graphs.append(g)
    return graphs


def _fallback_atlas(n: int) -> List[nx.Graph]:
    """networkx.graph_atlas_g() fallback, valid for n<=7."""
    atlas = nx.graph_atlas_g()
    out = [g for g in atlas if g.number_of_nodes() == n and nx.is_connected(g)]
    return out


def _assert_pairwise_non_isomorphic(graphs: List[nx.Graph], n: int) -> None:
    for i, j in itertools.combinations(range(len(graphs)), 2):
        if nx.is_isomorphic(graphs[i], graphs[j]):
            raise AssertionError(
                f"n={n}: graphs at index {i} and {j} are isomorphic (should be pairwise distinct)"
            )


def load_graphs(n: int, data_dir: str = "data/graphs", allow_fallback: bool = True) -> List[nx.Graph]:
    """Load the list of connected graphs on n nodes, verified against
    EXPECTED_CONNECTED_COUNTS. Tries the downloaded graph6 file first, falls
    back to networkx.graph_atlas_g() if offline/missing (n<=7 only).
    """
    path = os.path.join(data_dir, f"graph{n}c.g6")
    graphs: List[nx.Graph] = []
    source = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            graphs = _read_graph6_file(path)
            source = f"graph6 file {path}"
        except Exception as e:  # noqa: BLE001
            print(f"[graphs] failed to parse {path}: {e}")
            graphs = []
    if not graphs:
        if not allow_fallback:
            raise RuntimeError(f"No graph6 data for n={n} and fallback disabled")
        if n > 7:
            raise RuntimeError(f"Fallback (graph_atlas_g) only valid for n<=7, got n={n}")
        graphs = _fallback_atlas(n)
        source = "networkx.graph_atlas_g() fallback"

    expected = EXPECTED_CONNECTED_COUNTS.get(n)
    if expected is not None and len(graphs) != expected:
        raise AssertionError(
            f"n={n}: expected {expected} connected graphs, got {len(graphs)} (source: {source})"
        )
    _assert_pairwise_non_isomorphic(graphs, n)

    # Canonicalize node labels to 0..n-1 in a fixed order (graph6 node order).
    out = []
    for g in graphs:
        g2 = nx.Graph()
        g2.add_nodes_from(range(n))
        g2.add_edges_from(g.edges())
        out.append(g2)
    print(f"[graphs] n={n}: loaded {len(out)} connected graphs from {source}")
    return out


def all_automorphisms(G: nx.Graph) -> List[dict]:
    """Enumerate Aut(G) by brute force (fine for n<=7: at most 7! = 5040 perms)."""
    nodes = list(G.nodes())
    edges = set(frozenset(e) for e in G.edges())
    autos = []
    for perm in itertools.permutations(nodes):
        mapping = dict(zip(nodes, perm))
        mapped_edges = set(frozenset((mapping[a], mapping[b])) for a, b in G.edges())
        if mapped_edges == edges:
            autos.append(mapping)
    return autos


def edge_orbits(G: nx.Graph) -> List[List[Tuple[int, int]]]:
    """Orbits of the edge set under Aut(G). Returns list of orbits, each a
    list of (i, j) edges with i<j, in a canonical order.
    """
    autos = all_automorphisms(G)
    edges = [tuple(sorted(e)) for e in G.edges()]
    edge_set = set(edges)
    visited = set()
    orbits = []
    for e in edges:
        if e in visited:
            continue
        orbit = set()
        for mapping in autos:
            me = tuple(sorted((mapping[e[0]], mapping[e[1]])))
            assert me in edge_set
            orbit.add(me)
        orbit = sorted(orbit)
        orbits.append(orbit)
        visited |= set(orbit)
    return orbits


@dataclass
class GraphRecord:
    n: int
    index: int
    graph6: str
    edges: List[Tuple[int, int]]
    edge_orbits: List[List[Tuple[int, int]]]


def build_graph_records(n: int, data_dir: str = "data/graphs") -> List[GraphRecord]:
    graphs = load_graphs(n, data_dir=data_dir)
    records = []
    for idx, g in enumerate(graphs):
        g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
        edges = [tuple(sorted(e)) for e in g.edges()]
        orbits = edge_orbits(g)
        records.append(GraphRecord(n=n, index=idx, graph6=g6, edges=edges, edge_orbits=orbits))
    return records


def save_graphs_json(n_values: List[int], data_dir: str = "data/graphs", out_path: str = "graphs.json") -> None:
    all_records = []
    for n in n_values:
        for rec in build_graph_records(n, data_dir=data_dir):
            all_records.append({
                "n": rec.n,
                "index": rec.index,
                "graph6": rec.graph6,
                "edges": rec.edges,
                "edge_orbits": rec.edge_orbits,
            })
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[graphs] wrote {len(all_records)} graph records to {out_path}")


def induced_connected_triples(G: nx.Graph) -> List[Tuple[int, int, int]]:
    """All (i<j<k) such that the induced subgraph on {i,j,k} is connected
    (path or triangle)."""
    nodes = sorted(G.nodes())
    out = []
    for i, j, k in itertools.combinations(nodes, 3):
        sub = G.subgraph([i, j, k])
        if nx.is_connected(sub):
            out.append((i, j, k))
    return out
