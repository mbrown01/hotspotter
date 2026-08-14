"""Export the analysis: model-ready feature table, a contact list, and a readable report.

Three artifacts, each with a job:
  - features CSV   the full per-residue table (this is what Phase 2 trains on).
  - contacts CSV   every detected interaction (for validating against LigPlot+/DIMPLOT).
  - report  .txt   a human summary: top hot-spot candidates + naive-vs-chemistry comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hotspotter.pipeline import ComplexAnalysis
from hotspotter.ranking import compare_rankings

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"


def contacts_to_frame(analysis: ComplexAnalysis) -> pd.DataFrame:
    """Every detected cross-interface interaction as a tidy DataFrame."""
    return pd.DataFrame(
        [
            {
                "kind": c.kind,
                "res_a": c.res_a.label,
                "atom_a": c.atom_a,
                "res_b": c.res_b.label,
                "atom_b": c.atom_b,
                "distance": round(c.distance, 2),
                "detail": c.detail,
            }
            for c in analysis.contacts
        ]
    )


def save_outputs(analysis: ComplexAnalysis, out_dir: str | Path | None = None,
                 tag: str | None = None) -> dict[str, Path]:
    """Write features CSV, contacts CSV, and a text report. Returns the paths written."""
    out_dir = Path(out_dir) if out_dir else OUTPUTS
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or (analysis.source or "complex").replace("/", "_").replace("\\", "_")

    features_csv = out_dir / f"{tag}_features.csv"
    contacts_csv = out_dir / f"{tag}_contacts.csv"
    report_txt = out_dir / f"{tag}_report.txt"

    analysis.table.to_csv(features_csv, index=False)
    contacts_to_frame(analysis).to_csv(contacts_csv, index=False)
    report_txt.write_text(text_report(analysis), encoding="utf-8")

    return {"features": features_csv, "contacts": contacts_csv, "report": report_txt}


def text_report(analysis: ComplexAnalysis, top_n: int = 10) -> str:
    """A readable summary a scientist can skim without opening a spreadsheet."""
    a = analysis
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  Hot-spot interface analysis: {a.source}")
    lines.append("=" * 78)
    lines.append(f"  Sides: {a.side_a_chains}  <->  {a.side_b_chains}")
    lines.append(f"  Interface residues: {len(a.interface)}")
    lines.append(f"  Detected interactions: {len(a.contacts)}")
    lines.append(f"  SASA backend: {a.sasa_backend}"
                 f"{'   (PREDICTED structure: B-factor column = pLDDT)' if a.is_predicted else ''}")
    lines.append("")

    # Contact-type tally.
    if a.contacts:
        tally: dict[str, int] = {}
        for c in a.contacts:
            tally[c.kind] = tally.get(c.kind, 0) + 1
        lines.append("  Interaction inventory:")
        for k in ("salt_bridge", "hydrogen_bond", "hydrophobic", "aromatic", "disulfide"):
            if k in tally:
                lines.append(f"      {k:14s} {tally[k]}")
        lines.append("")

    lines.append(f"  TOP {top_n} HOT-SPOT CANDIDATES (chemistry-aware ranking):")
    lines.append("  " + "-" * 74)
    top = a.table.nsmallest(top_n, "hotspot_rank")
    for _, r in top.iterrows():
        lines.append(f"   #{int(r['hotspot_rank']):>2}  {r['residue']:<12} "
                     f"score={r['hotspot_score']:<7} (naive rank #{int(r['naive_rank'])})")
        lines.append(f"        why: {r['reasoning']}")
    lines.append("")

    # The money comparison: where chemistry and buriedness disagree.
    lines.append("  NAIVE (most-buried) vs. CHEMISTRY-AWARE — where they disagree:")
    lines.append("  " + "-" * 74)
    cmp = compare_rankings(a.table, top_n=top_n)
    movers = cmp[cmp["hotspot_rank"] != cmp["naive_rank"]] if "naive_rank" in cmp else cmp
    if len(movers) == 0:
        lines.append("      (the two rankings agree on the top residues here)")
    else:
        for _, r in movers.iterrows():
            direction = "UP" if r["hotspot_rank"] < r["naive_rank"] else "down"
            lines.append(f"      {r['residue']:<12} buriedness #{int(r['naive_rank'])} "
                         f"-> chemistry #{int(r['hotspot_rank'])}  ({direction})")
    lines.append("")
    lines.append("  NOTE: heuristic ranking (hand-set weights), not a trained model. "
                 "Treat as hypotheses.")
    lines.append("=" * 78)
    return "\n".join(lines)
