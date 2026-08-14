"""3D interface visualization with py3Dmol (renders inline in Jupyter/Colab).

Shows the complex as a cartoon, highlights the interface residues, and paints the top
hot-spot candidates as sticks colored by rank — so the "which residue should I mutate?"
answer is something you can see and rotate, not just read off a table.

Usage (in a notebook)::

    from hotspotter.pipeline import analyze_complex
    from hotspotter.viz import show_interface
    a = analyze_complex("1BRS", chains="A,D")
    show_interface(a)          # returns a py3Dmol view; displays inline
"""

from __future__ import annotations

from pathlib import Path

from hotspotter.pipeline import ComplexAnalysis

# A small colorblind-friendly ramp for the top hot spots (best = warmest).
_HOTSPOT_COLORS = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]


def _pdb_string(analysis: ComplexAnalysis) -> str:
    """Serialize the analyzed structure back to a PDB string for the viewer."""
    from io import StringIO

    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(analysis.structure)
    buf = StringIO()
    io.save(buf)
    return buf.getvalue()


def show_interface(analysis: ComplexAnalysis, top_n: int = 5, width: int = 800,
                   height: int = 600):
    """Return a py3Dmol view of the complex with interface + top hot spots highlighted."""
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(_pdb_string(analysis), "pdb")

    # Base: whole complex as a faint cartoon, one color per side.
    for chain in analysis.side_a_chains:
        view.setStyle({"chain": chain}, {"cartoon": {"color": "#bbbbbb"}})
    for chain in analysis.side_b_chains:
        view.setStyle({"chain": chain}, {"cartoon": {"color": "#88aacc"}})

    # All interface residues: thin sticks so you can see the contact patch.
    for rid in analysis.interface.residues:
        view.addStyle(
            {"chain": rid.chain, "resi": str(rid.resseq)},
            {"stick": {"radius": 0.15, "color": "#dddddd"}},
        )

    # Top hot spots: fat sticks, warm-to-cool by rank, with labels.
    top = analysis.table.nsmallest(top_n, "hotspot_rank")
    for i, (_, r) in enumerate(top.iterrows()):
        color = _HOTSPOT_COLORS[min(i, len(_HOTSPOT_COLORS) - 1)]
        sel = {"chain": r["chain"], "resi": str(int(r["resseq"]))}
        view.addStyle(sel, {"stick": {"radius": 0.3, "color": color}})
        view.addResLabels(sel, {"fontSize": 12, "backgroundColor": color,
                                "backgroundOpacity": 0.8})

    view.zoomTo()
    return view


def save_pdb(analysis: ComplexAnalysis, path: str | Path) -> Path:
    """Write the analyzed structure to a PDB file (e.g. for PyMOL follow-up)."""
    path = Path(path)
    path.write_text(_pdb_string(analysis), encoding="utf-8")
    return path
