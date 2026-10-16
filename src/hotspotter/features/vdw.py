"""Van der Waals packing: how *well* a side chain fits against the partner protein.

BIOLOGY NOTE — why a count was not enough:
    ``n_atom_contacts`` counts heavy-atom pairs inside a 5 A cutoff. A pair at 3.5 A and a
    pair at 4.9 A increment the same integer. Physically they are nothing alike: at 3.5 A
    two carbons sit near the bottom of their attractive well, at 4.9 A they barely feel
    each other, and below ~3.0 A they start to repel. Hot-spot side chains are the ones
    that fit snugly into a complementary pocket, and a step function cannot express
    "snugly".

    The Lennard-Jones potential does. It is the standard pairwise model of that curve:
    steeply repulsive when atoms overlap, attractive at the contact distance, decaying as
    r^-6 beyond it. Summing it over the side chain's atoms against the partner protein
    gives a continuous measure of packing quality.

    Measured contribution: +0.0115 PR-AUC over the 25-feature baseline, significant across
    six fold partitions. It is the only one of the four classical force-field terms that
    added anything -- hydrogen bonds, desolvation and electrostatics were already captured
    well enough by the existing counts, because those interactions are closer to
    all-or-nothing while packing is genuinely continuous.

PARAMETERS
    Element-level radii and well depths, not a full force field. That is deliberate: the
    gain is insensitive to the parameters (nine variants spanning +/-10% radii, uniform
    well depths, and three smoothing/cutoff settings all landed between +0.0073 and
    +0.0145). What matters is the SHAPE of the potential, not per-atom-type accuracy, so
    there is nothing to gain from a licensed parameter table.

    The potential is "smoothed" after Lise et al. (2009): flat-bottomed across r_ij +/- r0
    so that small coordinate errors and minor crystallographic clashes cannot produce
    enormous energies.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from hotspotter.interface import Interface
from hotspotter.io import ResidueId, is_amino_acid

#: Element van der Waals radii (A) and well depths (kcal/mol).
VDW_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
VDW_EPSILON = {"C": 0.12, "N": 0.16, "O": 0.20, "S": 0.20}

#: Flat-bottom half-width (A). Widens the minimum so the term is robust to coordinate noise.
SMOOTHING = 0.5
#: Pair interactions beyond this are ignored (A).
VDW_CUTOFF = 8.0
#: Pairs within this count toward ``n_vdw_pairs`` (A).
PAIR_CUTOFF = 4.5

#: Backbone atoms are not removed by an alanine substitution, so they are excluded:
#: the label is ddG of X->Ala, which deletes the side chain BEYOND CB.
#:
#: OPEN QUESTION: CB is kept here, and alanine keeps its CB too, so strictly the term
#: includes one atom the mutation does not remove. Excluding it would match the label
#: more literally; keeping it means the term also reports how the position is packed,
#: which is what makes a buried residue costly to truncate. Left as-is because the
#: +0.0115 PR-AUC was measured with this definition -- changing it needs a re-run.
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

VDW_COLUMNS = ("e_vdw_sc", "n_vdw_pairs", "e_vdw_rep")


def _lennard_jones(dist, r_min, epsilon):
    """Smoothed 6-12 potential. Negative is favourable.

    Flat at the well bottom across ``r_min +/- SMOOTHING``; the usual 6-12 form outside,
    with the distance shifted toward the flat region so the curve stays continuous.
    """
    out = np.zeros_like(dist)
    below = dist < r_min - SMOOTHING
    above = dist > r_min + SMOOTHING
    for mask, shift in ((below, +SMOOTHING), (above, -SMOOTHING)):
        if mask.any():
            ratio = r_min[mask] / (dist[mask] + shift)
            out[mask] = epsilon[mask] * (ratio ** 12 - 2 * ratio ** 6)
    inside = ~(below | above)
    out[inside] = -epsilon[inside]
    return out


def _side_atoms(model, chains) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-atom coordinates and element symbols for a set of chains."""
    coords, elements = [], []
    for chain in model:
        if chain.id not in chains:
            continue
        for residue in chain:
            if not is_amino_acid(residue):
                continue
            for atom in residue:
                if atom.element in VDW_RADIUS:
                    coords.append(atom.coord)
                    elements.append(atom.element)
    if not coords:
        return np.empty((0, 3)), np.empty(0, dtype=object)
    return np.asarray(coords, dtype=float), np.asarray(elements, dtype=object)


def compute_vdw_features(model, interface: Interface) -> dict[ResidueId, dict]:
    """Per-residue van der Waals packing against the *other* side of the interface.

    Returns, for every interface residue:
        e_vdw_sc      summed LJ energy, side chain against the partner protein (kcal/mol,
                      negative = well packed)
        n_vdw_pairs   side-chain / partner atom pairs within PAIR_CUTOFF
        e_vdw_rep     the repulsive part only, i.e. strain from atoms sitting too close
    """
    sides = {
        "A": _side_atoms(model, set(interface.side_a_chains)),
        "B": _side_atoms(model, set(interface.side_b_chains)),
    }
    trees = {
        s: (cKDTree(coords), elements)
        for s, (coords, elements) in sides.items() if len(coords)
    }

    feats: dict[ResidueId, dict] = {}
    for rid, ir in interface.residues.items():
        partner = "B" if ir.side == "A" else "A"
        blank = dict(e_vdw_sc=0.0, n_vdw_pairs=0, e_vdw_rep=0.0)
        if partner not in trees:
            feats[rid] = blank
            continue
        tree, elements = trees[partner]

        total = repulsive = 0.0
        n_pairs = 0
        for atom in ir.residue:
            if atom.get_name() in BACKBONE_ATOMS or atom.element not in VDW_RADIUS:
                continue
            idx = tree.query_ball_point(atom.coord, VDW_CUTOFF)
            if not idx:
                continue
            dist = np.linalg.norm(tree.data[idx] - atom.coord, axis=1)
            dist = np.clip(dist, 0.8, None)          # guard against overlapping atoms
            partner_elements = elements[idx]
            r_min = np.array([VDW_RADIUS[atom.element] + VDW_RADIUS[e]
                              for e in partner_elements])
            eps = np.array([np.sqrt(VDW_EPSILON[atom.element] * VDW_EPSILON[e])
                            for e in partner_elements])
            energy = _lennard_jones(dist, r_min, eps)
            total += float(energy.sum())
            repulsive += float(energy[energy > 0].sum())
            n_pairs += int((dist <= PAIR_CUTOFF).sum())

        feats[rid] = dict(e_vdw_sc=round(total, 4),
                          n_vdw_pairs=n_pairs,
                          e_vdw_rep=round(repulsive, 4))
    return feats
