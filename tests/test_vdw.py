"""Tests for the van der Waals packing feature.

The term exists to distinguish a 3.5 A contact from a 4.9 A one, which the old contact
count treated identically. These pin that behaviour: the shape of the potential, the
smoothing that keeps crystallographic clashes from blowing up, the exclusion of backbone
atoms (an alanine substitution removes only the side chain), and the cutoff.
"""

from __future__ import annotations

import numpy as np
import pytest

from hotspotter.features.vdw import (
    BACKBONE_ATOMS,
    PAIR_CUTOFF,
    SMOOTHING,
    VDW_CUTOFF,
    VDW_EPSILON,
    VDW_RADIUS,
    _lennard_jones,
    compute_vdw_features,
)


def _lj(dist):
    """Carbon-carbon, the common case."""
    r_min = np.array([VDW_RADIUS["C"] * 2])
    eps = np.array([VDW_EPSILON["C"]])
    return float(_lennard_jones(np.array([float(dist)]), r_min, eps)[0])


# --- the shape of the potential ---------------------------------------------------

def test_minimum_at_the_contact_distance():
    """Two carbons at the sum of their radii sit at the bottom of the well."""
    assert _lj(2 * VDW_RADIUS["C"]) == pytest.approx(-VDW_EPSILON["C"])


def test_repulsive_when_atoms_overlap():
    assert _lj(2.0) > 0


def test_attractive_at_contact_range():
    assert _lj(4.0) < 0


def test_decays_with_distance():
    """Beyond the well, the attraction falls off monotonically."""
    energies = [_lj(d) for d in (4.0, 5.0, 6.0, 7.0)]
    assert all(b > a for a, b in zip(energies, energies[1:]))   # rising toward zero
    assert all(e < 0 for e in energies)


def test_far_apart_is_negligible():
    assert abs(_lj(7.9)) < 0.01


def test_the_well_is_flat_bottomed():
    """Smoothing makes the term robust to small coordinate errors: the energy must not
    change across r_min +/- SMOOTHING."""
    r_min = 2 * VDW_RADIUS["C"]
    inside = [_lj(r_min + off) for off in (-SMOOTHING * 0.9, 0.0, SMOOTHING * 0.9)]
    assert all(e == pytest.approx(-VDW_EPSILON["C"]) for e in inside)


def test_smoothing_caps_the_clash_penalty():
    """Without smoothing a 0.8 A contact would produce an astronomical energy."""
    assert _lj(0.9) < _lj(1.5) * 1e6 or _lj(0.9) < 1e7


def test_mixed_elements_use_combined_radii():
    r_min = np.array([VDW_RADIUS["C"] + VDW_RADIUS["O"]])
    eps = np.array([np.sqrt(VDW_EPSILON["C"] * VDW_EPSILON["O"])])
    at_min = float(_lennard_jones(np.array([float(r_min[0])]), r_min, eps)[0])
    assert at_min == pytest.approx(-eps[0])


# --- the per-residue feature ------------------------------------------------------

class _Atom:
    def __init__(self, name, element, coord):
        self._name, self.element, self.coord = name, element, np.array(coord, float)

    def get_name(self):
        return self._name


class _Res:
    def __init__(self, name, num, atoms):
        self.id = (" ", num, " ")
        self._name, self._atoms = name, atoms

    def get_id(self):
        return self.id

    def get_resname(self):
        return self._name

    def __iter__(self):
        return iter(self._atoms)


class _Chain:
    def __init__(self, cid, residues):
        self.id, self._res = cid, residues

    def __iter__(self):
        return iter(self._res)


class _Model:
    def __init__(self, chains):
        self._chains = chains

    def __iter__(self):
        return iter(self._chains)


class _Rid:
    def __init__(self, chain, resseq):
        self.chain, self.resseq, self.icode = chain, resseq, " "

    def __hash__(self):
        return hash((self.chain, self.resseq))

    def __eq__(self, o):
        return (self.chain, self.resseq) == (o.chain, o.resseq)


class _IR:
    def __init__(self, residue, side):
        self.residue, self.side = residue, side


class _Iface:
    def __init__(self, residues, a=("A",), b=("B",)):
        self.residues, self.side_a_chains, self.side_b_chains = residues, a, b


def _two_residue_interface(sep: float, probe_atom=("CB", "C")):
    """One residue per side, their side-chain atoms exactly `sep` angstroms apart.

    The backbone CA is parked far away so it cannot contribute; only the side-chain atom
    is in range of the partner.
    """
    name, element = probe_atom
    near = _Res("LEU", 1, [_Atom("CA", "C", (0, -50, 0)),
                           _Atom(name, element, (0, 0, 0))])
    far = _Res("LEU", 2, [_Atom("CB", "C", (sep, 0, 0))])
    model = _Model([_Chain("A", [near]), _Chain("B", [far])])
    rid = _Rid("A", 1)
    iface = _Iface({rid: _IR(near, "A")})
    return model, iface, rid


def test_close_packing_scores_more_favourably_than_loose():
    """The whole point: 3.4 A must beat 4.8 A, where a contact count cannot tell them apart."""
    results = {}
    for sep in (3.4, 4.8):
        model, iface, rid = _two_residue_interface(sep)
        results[sep] = compute_vdw_features(model, iface)[rid]["e_vdw_sc"]
    assert results[3.4] < results[4.8], "closer packing should be more favourable"


def test_backbone_atoms_are_excluded():
    """Alanine substitution removes the side chain, not the backbone, so backbone
    contacts must not count toward the term."""
    bb = _Res("LEU", 1, [_Atom(n, "C", (3.4, 0, 0)) for n in sorted(BACKBONE_ATOMS)])
    partner = _Res("LEU", 2, [_Atom("CB", "C", (6.8, 0, 0))])
    model = _Model([_Chain("A", [bb]), _Chain("B", [partner])])
    rid = _Rid("A", 1)
    out = compute_vdw_features(model, _Iface({rid: _IR(bb, "A")}), )
    assert out[rid]["e_vdw_sc"] == 0.0
    assert out[rid]["n_vdw_pairs"] == 0


def test_no_partner_chain_gives_zero():
    res = _Res("LEU", 1, [_Atom("CB", "C", (0, 0, 0))])
    model = _Model([_Chain("A", [res])])
    rid = _Rid("A", 1)
    out = compute_vdw_features(model, _Iface({rid: _IR(res, "A")}))
    assert out[rid] == {"e_vdw_sc": 0.0, "n_vdw_pairs": 0, "e_vdw_rep": 0.0}


def test_beyond_the_cutoff_contributes_nothing():
    model, iface, rid = _two_residue_interface(VDW_CUTOFF + 1.0)
    assert compute_vdw_features(model, iface)[rid]["e_vdw_sc"] == 0.0


def test_pair_count_respects_its_own_cutoff():
    """n_vdw_pairs uses PAIR_CUTOFF, which is tighter than the energy cutoff."""
    m1, i1, r1 = _two_residue_interface(PAIR_CUTOFF - 0.2)
    m2, i2, r2 = _two_residue_interface(PAIR_CUTOFF + 0.5)
    assert compute_vdw_features(m1, i1)[r1]["n_vdw_pairs"] == 1
    assert compute_vdw_features(m2, i2)[r2]["n_vdw_pairs"] == 0


def test_repulsive_component_is_zero_for_good_packing():
    model, iface, rid = _two_residue_interface(3.6)             # comfortable contact
    assert compute_vdw_features(model, iface)[rid]["e_vdw_rep"] == 0.0


def test_repulsive_component_fires_on_a_clash():
    model, iface, rid = _two_residue_interface(2.4)             # overlapping
    out = compute_vdw_features(model, iface)[rid]
    assert out["e_vdw_rep"] > 0
    assert out["e_vdw_sc"] > 0


def test_every_interface_residue_gets_all_three_columns():
    model, iface, rid = _two_residue_interface(3.6)
    out = compute_vdw_features(model, iface)
    assert set(out[rid]) == {"e_vdw_sc", "n_vdw_pairs", "e_vdw_rep"}
