"""V2 feature extraction: atom-level, continuous physics instead of residue-level counts.

V1 (``hotspotter.features.chemistry``) is untouched and still the production path. This
module is a parallel implementation so the two can be compared on identical folds.

WHAT CHANGED AND WHY
    V1 asks "does this residue make a salt bridge?" and answers with a small integer. That
    framing has three failure modes we hit in practice:

    1. RESIDUE-TYPE GATING LOSES REAL CONTACTS. V1's hydrophobic detector consults a lookup
       table keyed on residue name that lists only ALA/ILE/LEU/MET/PHE/PRO/TRP/VAL, so a
       charged residue can never register a hydrophobic contact. But an arginine has a long
       aliphatic CB-CG-CD stem that stacks against aromatic rings, and that motif is real.
       Live case: ARL15 Arg95 CG sits 3.53 A from CNNM2 Phe524 CZ -- inside the 4.5 A cutoff
       -- and V1 scores it zero, while Mahbub et al. (2023) name that stacking as Arg95's
       PRIMARY interaction. V2 evaluates ATOMS, not residue types: an apolar carbon is
       apolar regardless of what its parent residue is called.

    2. THE ALANINE TRAP. The label we train on is ddG of X->Ala, which removes a side chain
       and leaves the backbone intact. A residue held by backbone hydrogen bonds should
       therefore survive alanine substitution, while one held by side-chain bonds should not.
       V1 pools both into one ``n_hydrogen_bonds`` count, so the feature cannot express the
       distinction the label is actually made of. V2 splits side/side, side/back, back/back.

    3. COUNTS DISCARD GEOMETRY. A contact at 2.6 A and one at 4.4 A both increment the same
       integer. V2 adds two continuous summaries -- an inverse-square distance sum and a
       Coulombic sum -- that vary smoothly with how good the packing actually is.

    Also: ``n_disulfides`` is dropped (identically zero across all 8,525 nodes in the
    dataset, so it is pure wasted input width), and ``bfactor`` is clipped at 100 A^2
    (unclipped it reaches 502.6 against a mean of 45.3; above ~100 the value indicates a
    poorly-ordered region and is not meaningfully comparable).

HONESTY ABOUT THE PHYSICS
    ``coulombic_sum`` uses two-value formal-ish charges (+/-0.5 on charged N/O) and no
    dielectric, distance-dependent or otherwise. It is a monotone electrostatic PROXY, not
    an energy in kcal/mol; do not present it as one. A real treatment needs partial charges
    from a force field (AMBER/CHARMM) and a solvent model. Likewise ``inverse_distance_sum``
    is a packing proxy, not a Lennard-Jones term -- there is no repulsive wall, so it rises
    monotonically as atoms approach.

    These are deliberately crude. The question V2 tests is whether CONTINUOUS geometry beats
    CATEGORICAL counts at this data scale, not whether we can do real molecular mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from hotspotter.constants import (
    ANIONIC_ATOMS,
    AROMATIC_RING_ATOMS,
    BACKBONE_ACCEPTOR_ATOMS,
    BACKBONE_DONOR_ATOMS,
    CATIONIC_ATOMS,
    Cutoffs,
    HBOND_ACCEPTORS,
    HBOND_DONORS,
)
from hotspotter.features.geometry import angle, distance, ring_centroid_and_normal
from hotspotter.interface import Interface
from hotspotter.io import ResidueId

# ---------------------------------------------------------------------------------------
# V2 cutoffs. Inherit V1 where the chemistry is unchanged so the two stay comparable; only
# the genuinely new interaction types get new numbers.
# ---------------------------------------------------------------------------------------
CATION_PI_DISTANCE = 5.0     # cationic N ... aromatic ring centroid (Gallivan & Dougherty)
PI_PI_DISTANCE = 6.0         # ring centroid ... ring centroid; same as V1's AROMATIC_CENTROID
BFACTOR_CLIP = 100.0         # A^2; above this the residue is poorly ordered, not "more mobile"

#: Atoms treated as apolar for the atom-level hydrophobic test. Carbon is apolar unless it is
#: bonded to N/O; sulfur in MET/CYS is effectively apolar too. Rather than encode bond
#: topology we exclude the carbons that sit adjacent to polar atoms, by name.
#:
#: This is the amphipathic fix: no residue-type gate. ARG's CB/CG/CD qualify, LYS's CB..CE
#: qualify, GLU/GLN's CB/CG qualify -- exactly the stems V1 could not see.
POLAR_ADJACENT_CARBONS = {
    "C",        # backbone carbonyl carbon
    "CZ",       # ARG guanidinium carbon (bonded to three N)
    "CD",       # ASN/GLN amide carbon -- excluded per-residue below, not globally
}

#: Per-residue carbons to EXCLUDE from the apolar set because they carry or neighbour a
#: heteroatom. Everything else that is a carbon (or a MET/CYS sulfur) counts.
NON_APOLAR_BY_RESIDUE = {
    "ARG": {"CZ"},                    # guanidinium carbon
    "ASN": {"CG"}, "GLN": {"CD"},     # amide carbons
    "ASP": {"CG"}, "GLU": {"CD"},     # carboxylate carbons
    "SER": {"CB"}, "THR": {"CB"},     # hydroxyl-bearing
    "TYR": {"CZ"},                    # phenol carbon
    "HIS": {"CG", "CD2", "CE1"},      # imidazole carbons flanking N
    "TRP": {"CD1", "CE2"},            # indole carbons flanking N
}

#: Rough partial charges for the Coulombic proxy. Two values only, on formally charged
#: groups; everything else contributes nothing.
PARTIAL_CHARGE = {
    **{(res, at): +0.5 for res, ats in CATIONIC_ATOMS.items() for at in ats},
    **{(res, at): -0.5 for res, ats in ANIONIC_ATOMS.items() for at in ats},
}

V2_CHEMISTRY_COLUMNS = (
    "n_salt_bridges", "n_saltbridge_side_side", "n_saltbridge_side_back",
    "n_hbond_side_side", "n_hbond_side_back", "n_hbond_back_back", "n_hydrogen_bonds",
    "n_hydrophobic", "n_aromatic", "n_cation_pi", "n_pi_pi",
    "n_chem_contacts", "has_salt_bridge",
    "inverse_distance_sum", "coulombic_sum",
)


@dataclass
class ContactV2:
    """One detected cross-interface interaction, with its geometry retained."""

    kind: str
    res_a: ResidueId
    res_b: ResidueId
    atom_a: str
    atom_b: str
    distance: float
    detail: str = ""

    def describe(self) -> str:
        d = f" ({self.detail})" if self.detail else ""
        return (f"{self.kind:22s} {self.res_a.label}:{self.atom_a} -- "
                f"{self.res_b.label}:{self.atom_b}  {self.distance:.2f} A{d}")


# ---------------------------------------------------------------------------------------
# Atom-level predicates
# ---------------------------------------------------------------------------------------
def is_apolar_atom(resname: str, atom) -> bool:
    """True for carbons (and MET/CYS sulfurs) that are not bonded to a heteroatom.

    Deliberately ignores which residue this is. The whole point of V2's hydrophobic fix is
    that an aliphatic carbon is aliphatic whether it belongs to leucine or arginine.
    """
    name = atom.get_name()
    el = atom.element
    if el == "S":
        return True                                  # MET SD, CYS SG
    if el != "C":
        return False
    if name == "C":
        return False                                 # backbone carbonyl carbon
    if name in NON_APOLAR_BY_RESIDUE.get(resname, ()):  # noqa: SIM118
        return False
    return True


def is_backbone(atom_name: str) -> bool:
    return atom_name in ("N", "O", "C", "CA", "OXT")


def _atoms(residue, names) -> list:
    return [residue[n] for n in names if n in residue]


def _ring_centroid(residue):
    """Ring centroid + normal for an aromatic residue, or None."""
    names = AROMATIC_RING_ATOMS.get(residue.get_resname())
    if not names:
        return None
    atoms = _atoms(residue, names)
    if len(atoms) < 3:
        return None
    return ring_centroid_and_normal([a.coord for a in atoms])


def _has_hydrogens(residue) -> bool:
    return any(a.element == "H" for a in residue.get_atoms())


# ---------------------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------------------
def _salt_bridges(res_a, res_b, rid_a, rid_b) -> list[ContactV2]:
    """Cationic N ... anionic O, split by whether both partners are side-chain groups.

    All formally charged groups in the 20 standard residues are on side chains, so
    side/back here means a charged side chain reaching a backbone carbonyl -- a
    charge-assisted hydrogen bond rather than a true salt bridge.
    """
    out = []
    for (cres, cid), (ares, aid) in (((res_a, rid_a), (res_b, rid_b)),
                                     ((res_b, rid_b), (res_a, rid_a))):
        cat = _atoms(cres, CATIONIC_ATOMS.get(cres.get_resname(), ()))
        ani = _atoms(ares, ANIONIC_ATOMS.get(ares.get_resname(), ()))
        for c, a in product(cat, ani):
            d = distance(c.coord, a.coord)
            if d <= Cutoffs.SALT_BRIDGE:
                out.append(ContactV2("salt_bridge_side_side", cid, aid,
                                     c.get_name(), a.get_name(), d))
        # charged side chain -> backbone carbonyl oxygen
        for c, o in product(cat, _atoms(ares, ("O",))):
            d = distance(c.coord, o.coord)
            if d <= Cutoffs.SALT_BRIDGE:
                out.append(ContactV2("salt_bridge_side_back", cid, aid,
                                     c.get_name(), "O", d, "charge-assisted"))
    return out


def _hydrogen_bonds(res_a, res_b, rid_a, rid_b) -> list[ContactV2]:
    """Donor ... acceptor, classified by which partners are backbone vs side chain.

    THE ALANINE TRAP: the training label is ddG of X->Ala, which deletes the side chain but
    keeps the backbone. A back/back hydrogen bond therefore survives the mutation and should
    NOT predict disruption; a side/side one should. V1 summed them into one number and
    destroyed exactly that signal.
    """
    out = []
    check_angle = _has_hydrogens(res_a) and _has_hydrogens(res_b)
    for (dres, did), (ares, aid) in (((res_a, rid_a), (res_b, rid_b)),
                                     ((res_b, rid_b), (res_a, rid_a))):
        dnames = set(HBOND_DONORS.get(dres.get_resname(), set())) | BACKBONE_DONOR_ATOMS
        anames = set(HBOND_ACCEPTORS.get(ares.get_resname(), set())) | BACKBONE_ACCEPTOR_ATOMS
        for don, acc in product(_atoms(dres, dnames), _atoms(ares, anames)):
            d = distance(don.coord, acc.coord)
            if d > Cutoffs.HBOND_DISTANCE:
                continue
            detail = ""
            if check_angle:
                hs = [a for a in dres.get_atoms()
                      if a.element == "H" and distance(a.coord, don.coord) < 1.2]
                if hs:
                    best = max(angle(don.coord, h.coord, acc.coord) for h in hs)
                    if best < Cutoffs.HBOND_ANGLE_MIN:
                        continue
                    detail = f"angle {best:.0f} deg"
            d_bb, a_bb = is_backbone(don.get_name()), is_backbone(acc.get_name())
            kind = ("hbond_back_back" if (d_bb and a_bb)
                    else "hbond_side_side" if not (d_bb or a_bb)
                    else "hbond_side_back")
            out.append(ContactV2(kind, did, aid, don.get_name(), acc.get_name(), d, detail))
    return out


def _hydrophobic(res_a, res_b, rid_a, rid_b) -> list[ContactV2]:
    """ATOM-LEVEL apolar contact -- the amphipathic fix.

    V1 gated on residue type and so could not see an arginine stem packing against a ring.
    Here every apolar heavy atom of either residue is eligible. One contact is reported per
    residue pair (the closest qualifying atom pair), matching V1's convention so the counts
    remain comparable; the continuous metrics below capture the rest of the geometry.
    """
    a_ap = [a for a in res_a if is_apolar_atom(res_a.get_resname(), a)]
    b_ap = [b for b in res_b if is_apolar_atom(res_b.get_resname(), b)]
    if not a_ap or not b_ap:
        return []
    best = None
    for a, b in product(a_ap, b_ap):
        d = distance(a.coord, b.coord)
        if d <= Cutoffs.HYDROPHOBIC_CONTACT and (best is None or d < best[0]):
            best = (d, a.get_name(), b.get_name())
    if best is None:
        return []
    return [ContactV2("hydrophobic", rid_a, rid_b, best[1], best[2], best[0])]


def _aromatic(res_a, res_b, rid_a, rid_b) -> list[ContactV2]:
    """Ring-centroid stacking (pi-pi), with the plane angle retained as detail."""
    ra, rb = _ring_centroid(res_a), _ring_centroid(res_b)
    if ra is None or rb is None:
        return []
    (ca, na), (cb, nb) = ra, rb
    d = distance(ca, cb)
    if d > PI_PI_DISTANCE:
        return []
    cosang = abs(float(np.dot(na, nb)) /
                 (np.linalg.norm(na) * np.linalg.norm(nb) + 1e-12))
    plane = float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
    geom = "face-to-face" if plane < 30 else ("T-shaped" if plane > 60 else "tilted")
    return [ContactV2("pi_pi", rid_a, rid_b, "ring", "ring", d,
                      f"{geom}, planes {plane:.0f} deg")]


def _cation_pi(res_a, res_b, rid_a, rid_b) -> list[ContactV2]:
    """Cationic nitrogen ... aromatic ring centroid.

    A genuinely strong interface motif (Gallivan & Dougherty 1999) that V1 has no concept
    of. Both directions are tested, since either partner may carry the cation.
    """
    out = []
    for (cres, cid), (ares, aid) in (((res_a, rid_a), (res_b, rid_b)),
                                     ((res_b, rid_b), (res_a, rid_a))):
        ring = _ring_centroid(ares)
        if ring is None:
            continue
        centroid, _normal = ring
        for n in _atoms(cres, CATIONIC_ATOMS.get(cres.get_resname(), ())):
            d = distance(n.coord, centroid)
            if d <= CATION_PI_DISTANCE:
                out.append(ContactV2("cation_pi", cid, aid, n.get_name(), "ring", d))
    return out


_DETECTORS = (_salt_bridges, _hydrogen_bonds, _hydrophobic, _aromatic, _cation_pi)


def detect_contacts_v2(interface: Interface) -> list[ContactV2]:
    """Every cross-interface interaction, V2 taxonomy."""
    contacts, seen = [], set()
    for ir in interface.side_a:
        for pid in ir.contact_partners:
            partner = interface.residues.get(pid)
            if partner is None or (ir.res_id, pid) in seen:
                continue
            seen.add((ir.res_id, pid))
            for det in _DETECTORS:
                contacts.extend(det(ir.residue, partner.residue, ir.res_id, pid))
    return contacts


# ---------------------------------------------------------------------------------------
# Continuous physics
# ---------------------------------------------------------------------------------------
def continuous_metrics(interface: Interface) -> dict[ResidueId, dict]:
    """Per-residue inverse-square packing sum and Coulombic proxy, over cross-interface pairs.

    Both are computed atom-by-atom across the interface, so they vary smoothly with geometry
    where the count features step discretely. Neither is an energy: see the module docstring.
    """
    out = {rid: {"inverse_distance_sum": 0.0, "coulombic_sum": 0.0}
           for rid in interface.residues}

    for ir in interface.side_a:
        rid_a, res_a = ir.res_id, ir.residue
        name_a = res_a.get_resname()
        heavy_a = [a for a in res_a if a.element != "H"]
        for pid in ir.contact_partners:
            partner = interface.residues.get(pid)
            if partner is None:
                continue
            res_b, name_b = partner.residue, partner.residue.get_resname()
            heavy_b = [b for b in res_b if b.element != "H"]
            inv = 0.0
            cou = 0.0
            for a, b in product(heavy_a, heavy_b):
                d = distance(a.coord, b.coord)
                if d < 0.1:                      # guard against coincident atoms
                    continue
                if d <= Cutoffs.PACKING_RADIUS:  # 10 A -- beyond this the term is negligible
                    inv += 1.0 / (d * d)
                qa = PARTIAL_CHARGE.get((name_a, a.get_name()), 0.0)
                qb = PARTIAL_CHARGE.get((name_b, b.get_name()), 0.0)
                if qa and qb:
                    cou += (qa * qb) / d
            # the pair contributes to BOTH residues
            out[rid_a]["inverse_distance_sum"] += inv
            out[pid]["inverse_distance_sum"] += inv
            out[rid_a]["coulombic_sum"] += cou
            out[pid]["coulombic_sum"] += cou
    return out


# ---------------------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------------------
_KIND_TO_COLUMN = {
    "salt_bridge_side_side": "n_saltbridge_side_side",
    "salt_bridge_side_back": "n_saltbridge_side_back",
    "hbond_side_side": "n_hbond_side_side",
    "hbond_side_back": "n_hbond_side_back",
    "hbond_back_back": "n_hbond_back_back",
    "hydrophobic": "n_hydrophobic",
    "pi_pi": "n_pi_pi",
    "cation_pi": "n_cation_pi",
}


def per_residue_chemistry_v2(
    interface: Interface, contacts: list[ContactV2] | None = None
) -> dict[ResidueId, dict]:
    """Roll V2 contacts up into per-residue columns.

    Counts are of DISTINCT PARTNER RESIDUES per interaction kind, matching V1's convention.
    """
    if contacts is None:
        contacts = detect_contacts_v2(interface)

    cols = tuple(_KIND_TO_COLUMN.values())
    partners: dict[ResidueId, dict[str, set]] = {
        rid: {c: set() for c in cols} for rid in interface.residues
    }
    for c in contacts:
        col = _KIND_TO_COLUMN[c.kind]
        if c.res_a in partners:
            partners[c.res_a][col].add(c.res_b)
        if c.res_b in partners:
            partners[c.res_b][col].add(c.res_a)

    cont = continuous_metrics(interface)

    feats: dict[ResidueId, dict] = {}
    for rid, per_kind in partners.items():
        f = {col: len(s) for col, s in per_kind.items()}
        # aggregates kept for continuity with V1's column names
        f["n_salt_bridges"] = f["n_saltbridge_side_side"] + f["n_saltbridge_side_back"]
        f["n_hydrogen_bonds"] = (f["n_hbond_side_side"] + f["n_hbond_side_back"]
                                 + f["n_hbond_back_back"])
        f["n_aromatic"] = f["n_pi_pi"]          # V1 name, pi-pi only
        f["n_chem_contacts"] = (f["n_salt_bridges"] + f["n_hydrogen_bonds"]
                                + f["n_hydrophobic"] + f["n_pi_pi"] + f["n_cation_pi"])
        f["has_salt_bridge"] = int(f["n_salt_bridges"] > 0)
        f.update(cont[rid])
        feats[rid] = {k: f[k] for k in V2_CHEMISTRY_COLUMNS}
    return feats


def clip_bfactor(value: float, cap: float = BFACTOR_CLIP) -> float:
    """Clip the B-factor tail. Above ~100 A^2 a residue is disordered, not merely mobile."""
    try:
        return min(float(value), cap)
    except (TypeError, ValueError):
        return cap


def build_v2_table(analysis) -> "object":
    """Take a V1 ComplexAnalysis and return a V2 feature table (a pandas DataFrame).

    Reuses V1's burial / topology / identity / SASA columns unchanged -- V2 only revises
    chemistry, drops ``n_disulfides`` and clips ``bfactor`` -- so any difference measured
    between V1 and V2 is attributable to the chemistry rewrite alone.
    """
    import pandas as pd

    v2 = per_residue_chemistry_v2(analysis.interface)
    rows = []
    for row in analysis.table.to_dict("records"):
        rid = next((r for r in v2
                    if r.chain == row["chain"] and r.resseq == row["resseq"]
                    and r.icode.strip() == str(row.get("icode", "")).strip()), None)
        new = dict(row)
        new.pop("n_disulfides", None)                      # dead feature
        new["bfactor"] = clip_bfactor(row.get("bfactor", 0.0))
        if rid is not None:
            new.update(v2[rid])
        rows.append(new)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from hotspotter.pipeline import analyze_complex

    pdb = sys.argv[1] if len(sys.argv) > 1 else "1BRS"
    chains = sys.argv[2] if len(sys.argv) > 2 else "A,D"
    focus = sys.argv[3] if len(sys.argv) > 3 else "A:87"      # barnase Arg87, the thesis case
    fc, fr = focus.split(":")
    fr = int(fr)

    a, b = chains.split(",")
    print("=" * 78)
    print(f"  features_v2 smoke test  |  {pdb}  {tuple(a)} vs {tuple(b)}  |  focus {focus}")
    print("=" * 78)

    analysis = analyze_complex(pdb, chains=(tuple(a), tuple(b)))
    contacts = detect_contacts_v2(analysis.interface)
    feats = per_residue_chemistry_v2(analysis.interface, contacts)
    print(f"  interface residues : {len(analysis.interface)}")
    print(f"  V2 contacts        : {len(contacts)}")

    from collections import Counter
    print(f"  by kind            : {dict(Counter(c.kind for c in contacts))}")

    target = next((r for r in feats if r.chain == fc and r.resseq == fr), None)
    if target is None:
        print(f"\n  {focus} is not an interface residue in this complex.")
        raise SystemExit(1)

    print(f"\n  FEATURE DICT for {target.label}")
    print("  " + "-" * 56)
    for k in V2_CHEMISTRY_COLUMNS:
        v = feats[target][k]
        print(f"    {k:<26}{v:>12.4f}" if isinstance(v, float) else
              f"    {k:<26}{v:>12}")

    print(f"\n  contacts involving {target.label}:")
    hit = [c for c in contacts if c.res_a == target or c.res_b == target]
    for c in hit:
        print("    ", c.describe())
    if not hit:
        print("     (none)")

    # V1 comparison on the same residue, so the delta is visible
    v1 = analysis.table
    r = v1[(v1.chain == fc) & (v1.resseq == fr)]
    if len(r):
        r = r.iloc[0]
        print(f"\n  V1 vs V2 on {target.label}:")
        print(f"    {'':<26}{'V1':>10}{'V2':>12}")
        print(f"    {'n_hydrophobic':<26}{int(r.n_hydrophobic):>10}"
              f"{feats[target]['n_hydrophobic']:>12}")
        print(f"    {'n_hydrogen_bonds':<26}{int(r.n_hydrogen_bonds):>10}"
              f"{feats[target]['n_hydrogen_bonds']:>12}"
              f"   (V2 splits: {feats[target]['n_hbond_side_side']} s/s, "
              f"{feats[target]['n_hbond_side_back']} s/b, "
              f"{feats[target]['n_hbond_back_back']} b/b)")
        print(f"    {'n_salt_bridges':<26}{int(r.n_salt_bridges):>10}"
              f"{feats[target]['n_salt_bridges']:>12}")
        print(f"    {'n_aromatic / n_pi_pi':<26}{int(r.n_aromatic):>10}"
              f"{feats[target]['n_pi_pi']:>12}")
        print(f"    {'n_cation_pi':<26}{'--':>10}{feats[target]['n_cation_pi']:>12}"
              "   (new in V2)")
        print(f"    {'bfactor':<26}{float(r.bfactor):>10.2f}"
              f"{clip_bfactor(r.bfactor):>12.2f}   (clipped at {BFACTOR_CLIP})")
    print("=" * 78)
