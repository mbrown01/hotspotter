"""Integration tests on synthetic structures: interface detection + chemistry.

These assert the geometry code fires exactly when it should, using atoms placed at known
distances (see conftest.build_structure / salt_bridge_pair). No network required.
"""

from __future__ import annotations

from hotspotter.features.chemistry import detect_contacts, per_residue_chemistry
from hotspotter.interface import detect_interface
from hotspotter.io import ResidueId


def test_interface_detects_both_residues(salt_bridge_pair):
    iface = detect_interface(salt_bridge_pair, ["A"], ["B"])
    assert len(iface) == 2
    sides = {ir.side for ir in iface.residues.values()}
    assert sides == {"A", "B"}


def test_interface_contact_graph_is_symmetric(salt_bridge_pair):
    iface = detect_interface(salt_bridge_pair, ["A"], ["B"])
    arg = next(ir for ir in iface.residues.values() if ir.res_id.resname == "ARG")
    asp = next(ir for ir in iface.residues.values() if ir.res_id.resname == "ASP")
    assert asp.res_id in arg.contact_partners
    assert arg.res_id in asp.contact_partners


def test_salt_bridge_detected(salt_bridge_pair):
    iface = detect_interface(salt_bridge_pair, ["A"], ["B"])
    contacts = detect_contacts(iface)
    salt = [c for c in contacts if c.kind == "salt_bridge"]
    assert len(salt) >= 1
    # The bridge should be between the Arg cationic N and the Asp anionic O.
    c = salt[0]
    names = {c.atom_a, c.atom_b}
    assert names & {"NH1", "NH2", "NE"}      # a cationic atom
    assert names & {"OD1", "OD2"}            # an anionic atom
    assert c.distance <= 4.0


def test_per_residue_counts_are_distinct_partners(salt_bridge_pair):
    iface = detect_interface(salt_bridge_pair, ["A"], ["B"])
    feats = per_residue_chemistry(iface)
    arg_id = next(rid for rid in feats if rid.resname == "ARG")
    # One Arg bridging one Asp = exactly one salt bridge, even though multiple N...O atom
    # pairs are within cutoff (this is the distinct-partner counting we care about).
    assert feats[arg_id]["n_salt_bridges"] == 1
    assert feats[arg_id]["has_salt_bridge"] == 1


def test_no_false_salt_bridge_when_far_apart():
    from tests.conftest import build_structure

    residues = [
        ("A", "ARG", 1, {"CA": (0, 0, 0), "NH1": (1, 0, 0)}),
        ("B", "ASP", 1, {"CA": (50, 0, 0), "OD1": (49, 0, 0)}),  # far away
    ]
    model = build_structure(residues)
    # They're not even in contact, so there should be no interface at all.
    try:
        iface = detect_interface(model, ["A"], ["B"])
        assert len([c for c in detect_contacts(iface) if c.kind == "salt_bridge"]) == 0
    except ValueError:
        pass  # "no interface residues" is also an acceptable outcome here


def test_residue_id_label():
    rid = ResidueId(chain="A", resseq=95, icode=" ", resname="ARG")
    assert rid.label == "A/ARG95"
    assert rid.one_letter == "R"
