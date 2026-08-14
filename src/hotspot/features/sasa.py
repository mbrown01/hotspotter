"""Solvent accessibility: how buried each residue is, and how much it buries on binding.

BIOLOGY NOTE:
    SASA = solvent-accessible surface area: the area of a residue a water molecule can
    touch, in square angstroms. Rolling a water-sized probe (radius ~1.4 A) over the
    protein and measuring where it can reach.

    Two derived quantities matter:

      dSASA / BSA (buried surface area):  SASA of the residue with its partner *absent*
          minus SASA with the partner *present*. If a residue was exposed to water when the
          protein was alone, and gets covered when the partner docks, it buries area -> it's
          physically at the interface and contributes to the binding *surface*. This is the
          classic "PISA-style" interface signal. It's real and useful — but it measures
          *how much surface a residue contributes*, NOT *how much binding energy depends on
          it*. That gap is the whole reason for this project (buried Y96 vs. load-bearing
          Arg95).

      RSA (relative SASA):  a residue's SASA divided by the maximum it could have (a
          Gly-X-Gly tripeptide, Tien et al. 2013). ~0 = fully buried core, ~1 = fully
          exposed. Low RSA in the *complex* means the residue sits in the packed core of the
          interface.

BACKEND:
    Uses the native ``freesasa`` library if it's importable (fast, the field reference),
    otherwise falls back to Biopython's pure-Python ``ShrakeRupley`` — no C compiler
    required, which matters on Windows. Both implement the same Shrake-Rupley algorithm;
    absolute numbers differ by a couple of percent, which doesn't affect the ranking.
"""

from __future__ import annotations

import copy

from Bio.PDB.SASA import ShrakeRupley

from hotspot.constants import MAX_ASA_TIEN, Cutoffs
from hotspot.io import ResidueId, is_amino_acid, residue_id

try:  # optional native backend
    import freesasa  # noqa: F401
    _HAS_FREESASA = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_FREESASA = False

SASA_BACKEND = "freesasa" if _HAS_FREESASA else "biopython-shrake-rupley"


def _subset_model(model, chains):
    """Deep-copy `model` keeping only standard amino acids in `chains`.

    Isolating a side lets us compute its 'unbound' accessibility (the same atoms, minus
    the partner) so BSA = unbound - complex is a like-for-like subtraction.
    """
    chains = set(chains)
    m = copy.deepcopy(model)
    for chain in list(m.get_chains()):
        if chain.id not in chains:
            m.detach_child(chain.id)
            continue
        for res in list(chain.get_residues()):
            if not is_amino_acid(res):
                chain.detach_child(res.get_id())
    return m


def _compute_residue_sasa(model) -> dict[ResidueId, float]:
    """Per-residue SASA for exactly the residues present in `model`."""
    sr = ShrakeRupley()          # default probe radius 1.40 A, 100 sample points
    sr.compute(model, level="R")  # stores .sasa on each residue
    out: dict[ResidueId, float] = {}
    for chain in model.get_chains():
        for res in chain.get_residues():
            if is_amino_acid(res):
                out[residue_id(res)] = float(res.sasa)
    return out


def compute_sasa_features(
    model,
    side_a_chains,
    side_b_chains,
) -> dict[ResidueId, dict]:
    """Compute per-residue SASA features for both sides of an interface.

    Returns, per residue:
        sasa_complex   SASA in the bound complex (both sides present)
        sasa_unbound   SASA of that side alone (partner removed)
        dsasa          buried surface area = unbound - complex (>= 0)
        rsa_complex    relative SASA in the complex (0 buried .. 1 exposed)
        rsa_unbound    relative SASA of the isolated side
        is_interface_sasa  1 if the residue buries >= SASA_INTERFACE_MIN A^2
    """
    side_a_chains = tuple(side_a_chains)
    side_b_chains = tuple(side_b_chains)

    complex_model = _subset_model(model, set(side_a_chains) | set(side_b_chains))
    sasa_complex = _compute_residue_sasa(complex_model)

    # Unbound = each side computed in isolation, then merged.
    sasa_unbound: dict[ResidueId, float] = {}
    for side in (side_a_chains, side_b_chains):
        sasa_unbound.update(_compute_residue_sasa(_subset_model(model, side)))

    feats: dict[ResidueId, dict] = {}
    for rid, s_complex in sasa_complex.items():
        s_unbound = sasa_unbound.get(rid, s_complex)
        dsasa = max(0.0, s_unbound - s_complex)
        max_asa = MAX_ASA_TIEN.get(rid.resname, None)
        feats[rid] = {
            "sasa_complex": round(s_complex, 2),
            "sasa_unbound": round(s_unbound, 2),
            "dsasa": round(dsasa, 2),
            "rsa_complex": round(s_complex / max_asa, 3) if max_asa else None,
            "rsa_unbound": round(s_unbound / max_asa, 3) if max_asa else None,
            "is_interface_sasa": int(dsasa >= Cutoffs.SASA_INTERFACE_MIN),
        }
    return feats
