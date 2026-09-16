"""Build the full PyG graph dataset from SKEMPI and save it to disk.

The graph-shaped counterpart of ``run_ml_baseline.py``. One graph per complex; nodes are
interface residues; labels come from SKEMPI with everything untested masked out.

The build is the expensive part (a structure download + full Phase-1 pipeline per complex),
so the result is pickled once and reloaded for every training run afterwards.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\build_graphs.py                     # alanine
    .\\.venv\\Scripts\\python.exe scripts\\build_graphs.py --strategy max
    .\\.venv\\Scripts\\python.exe scripts\\build_graphs.py --limit 20          # smoke test
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hotspotter.ml.graph_dataset import (  # noqa: E402
    DDG_HOTSPOT_THRESHOLD, INTRA_SIDE_CUTOFF, build_graph_dataset,
)

DEFAULT_CSV = REPO_ROOT / "data" / "skempi_v2.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine",
                    help="how to collapse multiple substitutions onto one node (default: alanine)")
    ap.add_argument("--threshold", type=float, default=DDG_HOTSPOT_THRESHOLD,
                    help=f"ddG (kcal/mol) for a hot spot (default: {DDG_HOTSPOT_THRESHOLD})")
    ap.add_argument("--intra-cutoff", type=float, default=INTRA_SIDE_CUTOFF,
                    help=f"CB-CB cutoff (A) for same-side edges (default: {INTRA_SIDE_CUTOFF})")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap graphs (default: no cap = full run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .pt (default: data/graphs_<strategy>.pt)")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="save a partial file every N graphs (0 disables; default: 20)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing partial checkpoint instead of restarting")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: SKEMPI csv not found at {args.csv}", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError:
        print("ERROR: needs the gnn extras:  pip install -e .[gnn]", file=sys.stderr)
        return 2

    out = args.out or (REPO_ROOT / "data" / f"graphs_{args.strategy}.pt")
    scope = f"limit={args.limit}" if args.limit else "FULL DATASET"

    print("=" * 78)
    print(f"  Graph build  |  {scope}  |  strategy={args.strategy}  |  "
          f"ddG>={args.threshold}  |  intra-edge {args.intra_cutoff} A")
    print("=" * 78)

    # ---- resume support ----------------------------------------------------------------
    # A full build is long, and a machine that dies partway should not cost the whole run.
    # Partial results land in <out>.partial and can be picked back up with --resume.
    partial_path = out.with_suffix(out.suffix + ".partial")
    done: list = []
    skip: set[str] = set()
    if args.resume and partial_path.exists():
        done = torch.load(partial_path, weights_only=False)
        skip = {g.complex_group for g in done}
        print(f"\nResuming: {len(done)} graphs already built, skipping those complexes.\n")
    elif partial_path.exists():
        print(f"\nNOTE: a partial checkpoint exists at {partial_path.name}. "
              f"Ignoring it (pass --resume to continue from it).\n")

    out.parent.mkdir(parents=True, exist_ok=True)

    fresh: list = []

    def checkpoint(graph, n_kept: int) -> None:
        fresh.append(graph)
        if args.checkpoint_every and n_kept % args.checkpoint_every == 0:
            # Write to a temp file and replace, so an interrupt mid-write cannot leave a
            # truncated checkpoint behind.
            tmp = partial_path.with_suffix(".tmp")
            torch.save(done + fresh, tmp)
            tmp.replace(partial_path)
            print(f"[graph] checkpoint: {len(done) + n_kept} graphs saved", flush=True)

    new_graphs, stats = build_graph_dataset(
        args.csv,
        strategy=args.strategy,
        ddg_threshold=args.threshold,
        intra_side_cutoff=args.intra_cutoff,
        limit_complexes=args.limit,
        skip_complexes=skip,
        on_graph=checkpoint,
    )

    graphs = done + new_graphs
    if not graphs:
        print("\nRESULT: no graphs built. Nothing to save.")
        return 1

    torch.save(graphs, out)
    partial_path.unlink(missing_ok=True)   # complete run supersedes any checkpoint

    # ---- report ------------------------------------------------------------------------
    n_nodes = sum(int(g.num_nodes) for g in graphs)
    n_edges = sum(int(g.edge_index.size(1)) for g in graphs)
    cross = sum(int(g.edge_attr[:, -1].sum()) for g in graphs)
    labeled = sum(int(g.label_mask.sum()) for g in graphs)
    pos = sum(int((g.y == 1.0).sum()) for g in graphs)
    neg = labeled - pos
    per_graph = Counter()
    for g in graphs:
        per_graph[int(g.label_mask.sum())] += 1

    print("\n" + "-" * 78)
    print("  GRAPH DATASET")
    print("-" * 78)
    resumed_note = "  (this run only)" if done else ""
    print(f"  graphs (complexes)      : {len(graphs)}")
    if done:
        print(f"    from checkpoint       : {len(done)}")
        print(f"    built this run        : {len(new_graphs)}")
    print(f"  complexes failed        : {stats.complexes_failed}{resumed_note}")
    print(f"  complexes with no label : {stats.complexes_unlabeled}  (dropped){resumed_note}")
    print(f"  mutations off-interface : {stats.mutations_unmatched}{resumed_note}")
    print()
    print(f"  nodes total             : {n_nodes}")
    print(f"  nodes/graph (mean)      : {n_nodes/len(graphs):.1f}")
    print(f"  edges total (directed)  : {n_edges}")
    print(f"    cross-interface       : {cross}  ({100*cross/n_edges:.1f}%)")
    print(f"    same-side             : {n_edges-cross}  ({100*(n_edges-cross)/n_edges:.1f}%)")
    print()
    print(f"  LABELED nodes           : {labeled}  ({100*labeled/n_nodes:.1f}% of nodes)")
    print(f"    POSITIVE (hot spot)   : {pos}")
    print(f"    NEGATIVE (not)        : {neg}")
    print(f"    positive class rate   : {pos/labeled:.4f}" if labeled else "    n/a")
    print(f"  unlabeled (masked)      : {n_nodes-labeled}")
    print()
    print(f"  saved -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
