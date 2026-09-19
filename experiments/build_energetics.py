"""Compute implicit-solvent binding energetics for every SKEMPI complex.

Runs pdb2pqr (AMBER charges, PROPKA protonation) then the generalized-Born decomposition
in hotspotter.energetics, and writes one row per interface residue. Checkpoints every 20
complexes so an interrupted run resumes instead of restarting.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from energetics import energetics_for_complex
from hotspotter.io import fetch_pdb
from hotspotter.ml.dataset import SKEMPI_COLUMNS, load_skempi, parse_pdb_field
from hotspotter.pipeline import analyze_complex

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "energetics.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    part = a.out.with_suffix(".partial.csv")

    sk = load_skempi(REPO / "data" / "skempi_v2.csv"); C = SKEMPI_COLUMNS
    sides = {}
    for _, r in sk.iterrows():
        try: p, x, y = parse_pdb_field(str(r[C["pdb_field"]]))
        except Exception: continue
        sides.setdefault(p, (set(x), set(y)))

    done, rows = set(), []
    if a.resume and part.exists():
        prev = pd.read_csv(part); rows = prev.to_dict("records"); done = set(prev.pdb_id)
        print(f"  resuming: {len(done)} complexes already done", flush=True)

    todo = [p for p in sides if p not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"  {len(todo)} complexes to process", flush=True)
    t0 = time.time(); fail = 0
    for n, p in enumerate(todo, 1):
        sa, sb = sides[p]
        try:
            # ddSASA per residue comes from the Phase-1 pipeline so the nonpolar term is
            # consistent with the SASA the rest of the project uses
            an = analyze_complex(p, chains=(tuple(sa), tuple(sb)))
            ds = {(r.chain, int(r.resseq), str(r.icode or "").strip()): float(r.dsasa)
                  for r in an.table.itertuples()}
            # only interface residues are ever consumed downstream
            e = energetics_for_complex(fetch_pdb(p), sa, sb, ds, keep=set(ds))
        except Exception as ex:
            fail += 1; print(f"    skip {p}: {type(ex).__name__}: {str(ex)[:80]}", flush=True); continue
        for (ch, rs, ic), v in e.items():
            rows.append({"pdb_id": p, "chain": ch, "resseq": rs, "icode": ic, **v})
        if n % 20 == 0:
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  ...{n}/{len(todo)} ({len(rows)} residues, {fail} failed)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False); part.unlink(missing_ok=True)
    print(f"\n  {len(df)} residues from {df.pdb_id.nunique()} complexes in {time.time()-t0:.0f}s")
    print(f"  failed: {fail}\n  saved -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
