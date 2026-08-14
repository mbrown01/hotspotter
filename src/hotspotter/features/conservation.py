"""Evolutionary conservation — the highest-signal feature after chemistry.

BIOLOGY NOTE (why this matters so much):
    Evolution is a natural mutagenesis experiment run over billions of years. If a residue
    is *conserved* — the same amino acid across many related species — it's because
    mutations there were selected against, i.e. they broke something the organism needed.
    Functional interface residues (the load-bearing ones) tend to be strongly conserved,
    while a residue that merely happens to be buried but does no real work is free to drift.

    So conservation is an *orthogonal* line of evidence to geometry: geometry says "this
    contact looks strong," conservation says "and evolution wouldn't let it change." When
    both agree, you have a strong hot-spot candidate. This is exactly the kind of signal
    that distinguishes a functionally critical Arg95 from an incidentally-buried Y96.

HOW WE'LL COMPUTE IT (planned — this module is the seam, not yet the implementation):
    1. Get a multiple-sequence alignment (MSA) for each chain's sequence:
         - cheapest: reuse ColabFold's MSA if the structure was predicted (it already ran
           one), or download a precomputed conservation score from ConSurf-DB when the PDB
           entry is covered.
         - otherwise: run MMseqs2 or HHblits against a sequence database to build an MSA.
       None of this is GPU-heavy; it's the most *plumbing* of the feature set, which is why
       it's staged after the geometric features.
    2. Score each column of the alignment. Good options: Shannon entropy (low entropy =
       conserved), or a Jensen-Shannon divergence score (Capra & Singh 2007), or ConSurf's
       Bayesian scores if we pull them directly.
    3. Map alignment columns back to residue positions and attach a per-residue score.

    Deliberately NOT implemented on day one so Phase 1's geometric core is finished and
    runnable first. The function signature below is the contract the pipeline expects, so
    dropping in a real implementation later is a one-file change.
"""

from __future__ import annotations

from pathlib import Path

from hotspotter.interface import Interface
from hotspotter.io import ResidueId


def compute_conservation_features(
    interface: Interface,
    scores: dict[ResidueId, float] | None = None,
    consurf_grades: str | Path | None = None,
) -> dict[ResidueId, dict]:
    """Attach a per-residue conservation score to each interface residue.

    Parameters
    ----------
    scores : optional precomputed mapping ResidueId -> conservation score (e.g. from your
        own MMseqs2/HHblits + entropy pipeline). If given, it's used directly.
    consurf_grades : optional path to a ConSurf grades file to parse (not yet implemented).

    Returns
    -------
    Per residue: {"conservation": float|None, "has_conservation": 0|1}. When no source is
    supplied we return None so the column exists and models can treat it as missing rather
    than silently imputing zeros.

    STATUS: interface only. Wire in a real MSA/ConSurf source here (see module docstring).
    """
    if consurf_grades is not None:
        raise NotImplementedError(
            "ConSurf grades parsing is planned; see conservation.py module docstring for "
            "the intended pipeline (MSA -> per-column score -> map to residues)."
        )
    feats: dict[ResidueId, dict] = {}
    for rid in interface.residues:
        val = scores.get(rid) if scores else None
        feats[rid] = {
            "conservation": val,
            "has_conservation": int(val is not None),
        }
    return feats
