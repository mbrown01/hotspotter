"""Interface topology: where a residue sits in the contact patch, and how tightly packed.

BIOLOGY NOTE — the "O-ring" idea:
    Bogan & Thorn (1998) noticed hot spots aren't spread evenly over an interface. They
    cluster in the *center*, surrounded by a ring of less-important residues that act like
    a gasket ("O-ring"), occluding water so the central residues sit in a dry, protein-like
    environment where their interactions pay off energetically. A charge-charge or H-bond
    contact is worth much more with water excluded than out in bulk solvent.

    So *central* interface residues are better hot-spot candidates than *peripheral* ones,
    all else equal. We approximate centrality three cheap ways:

      interface_neighbors   how many OTHER interface residues surround this one (more
                            neighbors = more central, more O-ring shielding).
      n_cross_contacts      how many residues on the other side it directly touches (a
                            residue wedged into the partner makes many cross-contacts).
      packing_density       heavy atoms within ~10 A — how tightly packed the local
                            environment is, central cores being denser than rims.
"""

from __future__ import annotations

from Bio.PDB import NeighborSearch

from hotspotter.constants import Cutoffs
from hotspotter.interface import Interface
from hotspotter.io import ResidueId, is_amino_acid, residue_id
from hotspotter.features.geometry import distance


def _representative_atom(residue):
    """CB if present (side-chain direction), else CA; the residue's 'position'."""
    if "CB" in residue:
        return residue["CB"]
    if "CA" in residue:
        return residue["CA"]
    return next(residue.get_atoms())


def compute_topology_features(model, interface: Interface) -> dict[ResidueId, dict]:
    """Per-residue topology features for all interface residues."""
    # All heavy atoms in the analyzed chains, for packing density.
    analyzed = set(interface.side_a_chains) | set(interface.side_b_chains)
    heavy_atoms = []
    for chain in model.get_chains():
        if chain.id in analyzed:
            for res in chain.get_residues():
                if is_amino_acid(res):
                    heavy_atoms.extend(a for a in res.get_atoms() if a.element != "H")
    ns = NeighborSearch(heavy_atoms)

    # Representative-atom coordinates of interface residues, for neighbor counting.
    rep = {rid: _representative_atom(ir.residue) for rid, ir in interface.residues.items()}

    feats: dict[ResidueId, dict] = {}
    for rid, ir in interface.residues.items():
        rep_atom = rep[rid]

        # Packing density: heavy atoms within radius (minus self would be ~residue size).
        packing = len(ns.search(rep_atom.coord, Cutoffs.PACKING_RADIUS, level="A"))

        # Interface neighbors: other interface residues whose rep atom is within 10 A.
        neighbors = 0
        for other_rid, other_atom in rep.items():
            if other_rid == rid:
                continue
            if distance(rep_atom.coord, other_atom.coord) <= Cutoffs.PACKING_RADIUS:
                neighbors += 1

        feats[rid] = {
            "n_cross_contacts": ir.n_partner_residues,   # residues touched on other side
            "n_atom_contacts": ir.atom_contacts,          # atom-atom pairs within cutoff
            "interface_neighbors": neighbors,             # O-ring shielding proxy
            "packing_density": packing,                   # local heavy-atom count
        }

    # Normalize centrality to [0, 1] within THIS interface, so it's comparable across
    # residues of one complex (a peripheral rim residue -> low; a buried core one -> high).
    if feats:
        max_nb = max(f["interface_neighbors"] for f in feats.values()) or 1
        for f in feats.values():
            f["centrality"] = round(f["interface_neighbors"] / max_nb, 3)
    return feats
