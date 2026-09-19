"""Build the flat V2 feature table from SKEMPI, using atom-level physics chemistry.

Mirrors ``hotspotter.ml.dataset.build_dataset`` exactly -- same SKEMPI parsing, same
single-mutation policy, same ddG math, same residue-matching rules -- and swaps only the
chemistry block for ``features_v2``. Everything else (SASA, topology, identity) is carried
over unchanged, so a V1-vs-V2 comparison isolates the chemistry rewrite.

THREE DELIBERATE DIFFERENCES FROM THE V1 EXTRACTION
    1. 1BRS IS KEPT. The V1 table also contained it; the exclusion happens at TRAIN time,
       not extraction time, so the complex stays available for held-out benchmarking.
    2. RAW MUTATION COLUMNS ARE PRESERVED -- ``pdb_id``, ``wt``, ``mut``, ``ddg`` -- so the
       alanine and max label strategies can both be derived later by filtering, without
       re-running a 20-minute extraction.
    3. NO LABEL THRESHOLD IS BAKED IN. ``ddg`` is written raw; ``label`` is included at the
       default 2.0 kcal/mol for convenience but can be recomputed from ``ddg`` at any time.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\build_dataset_v2.py
    .\\.venv\\Scripts\\python.exe scripts\\build_dataset_v2.py --limit 20   # smoke run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from features_v2 import build_v2_table  # noqa: E402
from hotspotter.ml.dataset import (  # noqa: E402
    SKEMPI_COLUMNS, _clean_temperature, _to_float, ddg_from_kd, load_skempi,
    parse_mutation, parse_pdb_field,
)
from hotspotter.pipeline import analyze_complex  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "skempi_v2.csv"


def find_row(table: pd.DataFrame, mutation):
    """Locate the mutated residue's V2 feature row.

    Same rules as the (fixed) V1 matcher: chain + resseq + INSERTION CODE, then the
    wild-type letter must agree. A disagreement means SKEMPI meant a different residue, so
    we refuse rather than attach features from the wrong one.
    """
    ic = mutation.icode.strip()
    icodes = (table["icode"].fillna("").astype(str).str.strip()
              if "icode" in table.columns else "")
    hit = table[(table["chain"] == mutation.chain)
                & (table["resseq"] == mutation.resseq)
                & (icodes == ic)]
    if len(hit) == 0:
        return None
    if "aa" in hit.columns:
        hit = hit[hit["aa"] == mutation.wt]
        if len(hit) == 0:
            return None
    return hit.iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_v2.csv")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: missing {args.csv}", file=sys.stderr)
        return 2

    df = load_skempi(args.csv)
    c = SKEMPI_COLUMNS
    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict] = []
    skipped = {"multi_mut": 0, "no_kd": 0, "no_structure": 0, "residue_not_found": 0}
    seen: set[str] = set()

    print("=" * 78)
    print("  V2 FLAT DATASET  |  atom-level physics chemistry")
    print("=" * 78)
    print("  1BRS is KEPT (held out at train time, not extraction time)")
    print("  raw pdb_id / wt / mut / ddg preserved for later strategy filtering\n")

    for _, r in df.iterrows():
        mut_field = str(r[c["mutation"]])
        if "," in mut_field:
            skipped["multi_mut"] += 1
            continue
        try:
            pdb_id, side_a, side_b = parse_pdb_field(str(r[c["pdb_field"]]))
        except ValueError:
            continue
        if args.limit and pdb_id not in seen and len(seen) >= args.limit:
            continue

        ddg = ddg_from_kd(_to_float(r[c["kd_wt"]]), _to_float(r[c["kd_mut"]]),
                          _clean_temperature(r[c["temperature"]]))
        if math.isnan(ddg):
            skipped["no_kd"] += 1
            continue

        if pdb_id not in cache:
            try:
                analysis = analyze_complex(pdb_id, chains=(side_a, side_b))
                cache[pdb_id] = build_v2_table(analysis)
            except Exception as exc:
                cache[pdb_id] = None
                print(f"[v2] skip {pdb_id}: {type(exc).__name__}: {exc}", flush=True)
            seen.add(pdb_id)
            if len(seen) % 25 == 0:
                print(f"[v2] ...{len(seen)} complexes analyzed, {len(rows)} rows so far",
                      flush=True)

        table = cache[pdb_id]
        if table is None:
            skipped["no_structure"] += 1
            continue

        mutation = parse_mutation(mut_field)
        feat = find_row(table, mutation)
        if feat is None:
            skipped["residue_not_found"] += 1
            continue

        rec = feat.to_dict()
        rec.update({
            "pdb_id": pdb_id,               # <- the column the trainer filters 1BRS on
            "complex_group": pdb_id,        # kept for parity with the V1 table
            "mutation": mut_field,
            "wt": mutation.wt,              # <- raw mutation columns, preserved
            "mut": mutation.mut,
            "ddg": ddg,
            "label": int(ddg >= args.threshold),
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        print("\nRESULT: zero rows extracted.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n_pos = int(out["label"].sum())
    print(f"\n  rows            : {len(out)}")
    print(f"  complexes       : {out['pdb_id'].nunique()}")
    print(f"  positive / neg  : {n_pos} / {len(out) - n_pos}   rate {n_pos/len(out):.4f}")
    print(f"  1BRS present    : {'1BRS' in set(out['pdb_id'])}"
          f"  ({int((out['pdb_id'] == '1BRS').sum())} rows)")
    print(f"  to-alanine rows : {int((out['mut'] == 'A').sum())}")
    print(f"  skipped         : {skipped}")
    print(f"  columns         : {len(out.columns)}")
    print(f"\n  saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
