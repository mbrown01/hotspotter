"""Download a structure from the RCSB PDB or AlphaFold DB into data/raw/.

    python scripts/fetch_structure.py --pdb 1BRS
    python scripts/fetch_structure.py --pdb 1BRS --cif
    python scripts/fetch_structure.py --alphafold P69905      # UniProt accession

Both sources are free and need no account or API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotspot.io import fetch_alphafold, fetch_pdb  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch a structure from RCSB or AlphaFold DB.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb", help="4-character PDB id.")
    g.add_argument("--alphafold", help="UniProt accession for an AlphaFold DB model.")
    p.add_argument("--cif", action="store_true", help="Fetch mmCIF instead of PDB format.")
    args = p.parse_args()

    if args.pdb:
        path = fetch_pdb(args.pdb, fmt="cif" if args.cif else "pdb")
    else:
        path = fetch_alphafold(args.alphafold)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
