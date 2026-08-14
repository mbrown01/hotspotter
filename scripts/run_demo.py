"""Barnase-barstar 'hello world' — the whole Phase-1 pipeline on a known-answer complex.

Why 1BRS: barnase (an RNase) and barstar (its inhibitor) form the most-studied,
most-measured protein interface in existence. It's small, and its interface is dominated
by charged residues and salt bridges — the exact chemistry the tool exists to catch. If
the pipeline is right anywhere, it should light up the known barnase-barstar salt-bridge
residues (e.g. barnase Arg59/Arg83/Arg87/His102, barstar Asp35/Asp39/Glu76) here.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py

The RCSB entry 1BRS contains three copies of the complex (chains A/B/C = barnase,
D/E/F = barstar). We analyze one copy: barnase A vs barstar D.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `hotspot` importable whether or not the package is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotspot.pipeline import analyze_complex   # noqa: E402
from hotspot.report import save_outputs, text_report  # noqa: E402


def main() -> int:
    print("Analyzing barnase-barstar (PDB 1BRS), chains A (barnase) vs D (barstar)...\n")
    analysis = analyze_complex("1BRS", chains="A,D")

    print(text_report(analysis, top_n=10))

    paths = save_outputs(analysis, tag="1brs_barnase_barstar")
    print("\nWrote:")
    for name, path in paths.items():
        print(f"  {name:9s} {path}")

    # A quick sanity check the user can eyeball: did we find salt bridges at all?
    n_sb = sum(1 for c in analysis.contacts if c.kind == "salt_bridge")
    print(f"\nSanity check: {n_sb} salt-bridge contacts detected across the interface.")
    if n_sb == 0:
        print("  WARNING: expected salt bridges in barnase-barstar - check cutoffs/chains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
