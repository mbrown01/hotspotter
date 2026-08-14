"""Interaction chemistry across the interface — the core discriminating signal.

This is the module that would have caught **Arg95**. Buriedness said "mutate Y96";
chemistry says "Arg95 forms a salt bridge across the interface, that's the load-bearing
contact." Everything here is computed from atomic geometry, so it's transparent and
sanity-checkable against a 2D tool like LigPlot+/DIMPLOT.

We detect five interaction types between the two sides of the interface:

    salt bridge     cationic side-chain N ... anionic side-chain O within ~4.0 A.
                    A charge-charge attraction; the strongest non-covalent contact here,
                    and the one most likely to be individually load-bearing. (Arg95.)

    hydrogen bond   donor heavy atom (N/O with an H) ... acceptor heavy atom (N/O lone
                    pair) within ~3.5 A. If explicit hydrogens exist we also require a
                    reasonable D-H...A angle (>=120 deg). Directional and moderately strong.

    hydrophobic     apolar side-chain carbon ... apolar side-chain carbon within ~4.5 A.
                    Individually weak; collectively they're the "glue" that buries greasy
                    surface away from water and drives most of the binding *area*.

    aromatic / pi   two aromatic rings whose centroids are within ~6 A. Face-to-face
                    (pi-stacking) or edge-to-face (T-shaped) depending on the plane angle.

    disulfide       two cysteine SG atoms within ~2.5 A — a genuine covalent cross-link.
                    Rare across interfaces, but decisive when present.

Each detected interaction is a :class:`Contact`; we also roll them up into per-residue
counts that become feature columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from hotspot.constants import (
    ANIONIC_ATOMS,
    AROMATIC_RING_ATOMS,
    BACKBONE_ACCEPTOR_ATOMS,
    BACKBONE_DONOR_ATOMS,
    CATIONIC_ATOMS,
    Cutoffs,
    HBOND_ACCEPTORS,
    HBOND_DONORS,
    HYDROPHOBIC_CARBONS,
)
from hotspot.features.geometry import (
    angle,
    distance,
    ring_centroid_and_normal,
    ring_dihedral,
)
from hotspot.interface import Interface
from hotspot.io import ResidueId

CONTACT_TYPES = ("salt_bridge", "hydrogen_bond", "hydrophobic", "aromatic", "disulfide")


@dataclass
class Contact:
    """One detected cross-interface interaction between two residues."""

    kind: str                 # one of CONTACT_TYPES
    res_a: ResidueId
    res_b: ResidueId
    atom_a: str
    atom_b: str
    distance: float
    detail: str = ""          # e.g. ring plane angle, or "backbone" for backbone H-bonds

    def describe(self) -> str:
        d = f" ({self.detail})" if self.detail else ""
        return (
            f"{self.kind:14s} {self.res_a.label}:{self.atom_a} -- "
            f"{self.res_b.label}:{self.atom_b}  {self.distance:.2f} A{d}"
        )


def _atoms_by_name(residue, names) -> list:
    """Return the residue's atoms whose names are in `names` (silently skips missing)."""
    out = []
    for name in names:
        if name in residue:
            out.append(residue[name])
    return out


def _has_hydrogens(residue) -> bool:
    return any(a.element == "H" for a in residue.get_atoms())


def _salt_bridges(res_a, res_b, rid_a, rid_b) -> list[Contact]:
    """Cationic N (A) ... anionic O (B), and vice versa."""
    contacts = []
    for (cat_res, cat_id), (ani_res, ani_id) in (
        ((res_a, rid_a), (res_b, rid_b)),
        ((res_b, rid_b), (res_a, rid_a)),
    ):
        cat_names = CATIONIC_ATOMS.get(cat_res.get_resname(), set())
        ani_names = ANIONIC_ATOMS.get(ani_res.get_resname(), set())
        for ca, an in product(_atoms_by_name(cat_res, cat_names),
                              _atoms_by_name(ani_res, ani_names)):
            d = distance(ca.coord, an.coord)
            if d <= Cutoffs.SALT_BRIDGE:
                contacts.append(Contact("salt_bridge", cat_id, ani_id,
                                        ca.get_name(), an.get_name(), d))
    return contacts


def _hydrogen_bonds(res_a, res_b, rid_a, rid_b) -> list[Contact]:
    """Donor heavy atom ... acceptor heavy atom, both directions, backbone + side chain.

    If both residues carry explicit hydrogens, we additionally require the best D-H...A
    angle to be >= the cutoff; otherwise we fall back to a distance-only criterion (which
    is standard for X-ray structures that lack modeled hydrogens).
    """
    contacts = []
    check_angle = _has_hydrogens(res_a) and _has_hydrogens(res_b)

    for (don_res, don_id), (acc_res, acc_id) in (
        ((res_a, rid_a), (res_b, rid_b)),
        ((res_b, rid_b), (res_a, rid_a)),
    ):
        don_names = set(HBOND_DONORS.get(don_res.get_resname(), set())) | BACKBONE_DONOR_ATOMS
        acc_names = set(HBOND_ACCEPTORS.get(acc_res.get_resname(), set())) | BACKBONE_ACCEPTOR_ATOMS
        for don, acc in product(_atoms_by_name(don_res, don_names),
                                _atoms_by_name(acc_res, acc_names)):
            d = distance(don.coord, acc.coord)
            if d > Cutoffs.HBOND_DISTANCE:
                continue
            detail = ""
            if check_angle:
                # Find an H bonded to the donor (nearest H within ~1.2 A) and check angle.
                hs = [a for a in don_res.get_atoms() if a.element == "H"
                      and distance(a.coord, don.coord) < 1.2]
                if hs:
                    best = max(angle(don.coord, h.coord, acc.coord) for h in hs)
                    if best < Cutoffs.HBOND_ANGLE_MIN:
                        continue
                    detail = f"angle {best:.0f} deg"
            if don.get_name() in BACKBONE_DONOR_ATOMS or acc.get_name() in BACKBONE_ACCEPTOR_ATOMS:
                detail = (detail + "; backbone").lstrip("; ") if detail else "backbone"
            contacts.append(Contact("hydrogen_bond", don_id, acc_id,
                                    don.get_name(), acc.get_name(), d, detail))
    return contacts


def _hydrophobic(res_a, res_b, rid_a, rid_b) -> list[Contact]:
    """Closest apolar C ... apolar C pair (report one contact per residue pair, not per atom).

    Hydrophobic contacts are many and individually weak; counting every carbon pair would
    swamp the table. We report the single closest qualifying carbon-carbon pair per residue
    pair, which is what a 2D diagram like DIMPLOT effectively shows.
    """
    a_names = HYDROPHOBIC_CARBONS.get(res_a.get_resname(), set())
    b_names = HYDROPHOBIC_CARBONS.get(res_b.get_resname(), set())
    if not a_names or not b_names:
        return []
    best = None
    for ca, cb in product(_atoms_by_name(res_a, a_names), _atoms_by_name(res_b, b_names)):
        d = distance(ca.coord, cb.coord)
        if d <= Cutoffs.HYDROPHOBIC_CONTACT and (best is None or d < best[0]):
            best = (d, ca.get_name(), cb.get_name())
    if best is None:
        return []
    return [Contact("hydrophobic", rid_a, rid_b, best[1], best[2], best[0])]


def _aromatic(res_a, res_b, rid_a, rid_b) -> list[Contact]:
    """Ring centroid ... ring centroid within cutoff; report the plane angle as detail."""
    ra = AROMATIC_RING_ATOMS.get(res_a.get_resname())
    rb = AROMATIC_RING_ATOMS.get(res_b.get_resname())
    if not ra or not rb:
        return []
    a_atoms = _atoms_by_name(res_a, ra)
    b_atoms = _atoms_by_name(res_b, rb)
    if len(a_atoms) < 3 or len(b_atoms) < 3:
        return []
    ca, na = ring_centroid_and_normal([a.coord for a in a_atoms])
    cb, nb = ring_centroid_and_normal([a.coord for a in b_atoms])
    d = distance(ca, cb)
    if d > Cutoffs.AROMATIC_CENTROID:
        return []
    plane = ring_dihedral(na, nb)
    geom = "face-to-face" if plane < 30 else ("T-shaped" if plane > 60 else "tilted")
    return [Contact("aromatic", rid_a, rid_b, "ring", "ring", d,
                    f"{geom}, planes {plane:.0f} deg")]


def _disulfide(res_a, res_b, rid_a, rid_b) -> list[Contact]:
    if res_a.get_resname() != "CYS" or res_b.get_resname() != "CYS":
        return []
    if "SG" not in res_a or "SG" not in res_b:
        return []
    d = distance(res_a["SG"].coord, res_b["SG"].coord)
    if d <= Cutoffs.DISULFIDE:
        return [Contact("disulfide", rid_a, rid_b, "SG", "SG", d)]
    return []


_DETECTORS = (_salt_bridges, _hydrogen_bonds, _hydrophobic, _aromatic, _disulfide)


def detect_contacts(interface: Interface) -> list[Contact]:
    """Find every cross-interface interaction, iterating only over residues in contact.

    We use the interface contact graph (built during detection) so we only test residue
    pairs that are already near each other — cheap and exact.
    """
    contacts: list[Contact] = []
    seen_pairs: set[tuple[ResidueId, ResidueId]] = set()

    for ir in interface.side_a:  # side A vs its partners avoids double-counting pairs
        res_a = ir.residue
        rid_a = ir.res_id
        for partner_id in ir.contact_partners:
            partner = interface.residues.get(partner_id)
            if partner is None:
                continue
            key = (rid_a, partner_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            res_b = partner.residue
            for detector in _DETECTORS:
                contacts.extend(detector(res_a, res_b, rid_a, partner_id))
    return contacts


def per_residue_chemistry(
    interface: Interface, contacts: list[Contact] | None = None
) -> dict[ResidueId, dict]:
    """Roll detected contacts up into per-residue feature counts.

    Counts are of DISTINCT PARTNER RESIDUES, not raw atom pairs: "Asp39 forms a salt
    bridge with Arg83" is one salt bridge, even though Asp39's two carboxylate oxygens
    each sit within cutoff of several of Arg83's guanidinium nitrogens (which would be ~6
    atom pairs). Residue-residue counting is how structural biologists report these and
    keeps the feature from being inflated by how many polar atoms happen to be in range.
    The full atom-level detail is still available via :func:`detect_contacts`.

    Returns, for every interface residue, a dict of columns:
        n_salt_bridges, n_hydrogen_bonds, n_hydrophobic, n_aromatic, n_disulfides,
        n_chem_contacts (total distinct interactions), has_salt_bridge (0/1).

    Pass a precomputed `contacts` list to avoid recomputing it (the pipeline does this).
    """
    if contacts is None:
        contacts = detect_contacts(interface)

    key_map = {
        "salt_bridge": "n_salt_bridges",
        "hydrogen_bond": "n_hydrogen_bonds",
        "hydrophobic": "n_hydrophobic",
        "aromatic": "n_aromatic",
        "disulfide": "n_disulfides",
    }
    # Per residue, per kind: the SET of partner residues it interacts with.
    partners: dict[ResidueId, dict[str, set]] = {
        rid: {col: set() for col in key_map.values()} for rid in interface.residues
    }
    for c in contacts:
        col = key_map[c.kind]
        if c.res_a in partners:
            partners[c.res_a][col].add(c.res_b)
        if c.res_b in partners:
            partners[c.res_b][col].add(c.res_a)

    feats: dict[ResidueId, dict] = {}
    for rid, per_kind in partners.items():
        f = {col: len(s) for col, s in per_kind.items()}
        f["n_chem_contacts"] = sum(f.values())
        f["has_salt_bridge"] = int(f["n_salt_bridges"] > 0)
        feats[rid] = f
    return feats
