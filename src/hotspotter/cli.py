"""Command-line entry point:  python -m hotspotter.cli --pdb 1BRS --chains A,D

Runs the full Phase-1 pipeline and writes the feature table, contact list, and report to
``outputs/`` (or a directory you choose). Designed to also be scriptable in batch for the
Phase-2 SKEMPI sweep.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from hotspotter.pipeline import analyze_complex
from hotspotter.report import save_outputs, text_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hotspotter",
        description="Rank protein-protein interface residues by predicted hot-spot importance.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", help="4-character PDB id to download (e.g. 1BRS).")
    src.add_argument("--file", help="Path to a local .pdb/.cif structure.")
    p.add_argument("--chains", default=None,
                   help="Interface sides, e.g. 'A,D' or multi-chain 'AB,CD'. Default: auto.")
    p.add_argument("--predicted", action="store_true",
                   help="Structure is an AlphaFold/ColabFold model (B-factor col = pLDDT).")
    p.add_argument("--top", type=int, default=10, help="How many candidates to print.")
    p.add_argument("--out", default=None, help="Output directory (default: ./outputs).")
    p.add_argument("--no-save", action="store_true", help="Print only; don't write files.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = args.pdb if args.pdb else args.file

    analysis = analyze_complex(source, chains=args.chains, is_predicted=args.predicted)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(text_report(analysis, top_n=args.top))

    if not args.no_save:
        paths = save_outputs(analysis, out_dir=args.out)
        print("\nWrote:")
        for name, path in paths.items():
            print(f"  {name:9s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
