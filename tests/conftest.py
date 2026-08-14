"""Shared test fixtures: build tiny synthetic structures in memory (no network).

Constructing structures programmatically lets us place atoms at *known* distances and
assert that each detector fires (or doesn't) exactly when it should — the cleanest way to
unit-test geometry without depending on a downloaded PDB.
"""

from __future__ import annotations

import numpy as np
import pytest
from Bio.PDB.StructureBuilder import StructureBuilder


def build_structure(residues):
    """Build a Biopython Structure from a compact spec.

    Parameters
    ----------
    residues : list of (chain_id, resname, resseq, atoms) where `atoms` is a dict mapping
        atom name -> (x, y, z). Element is inferred from the atom name's first letter.

    Returns the first model of the built structure.
    """
    sb = StructureBuilder()
    sb.init_structure("synthetic")
    sb.init_model(0)
    current_chain = None
    for chain_id, resname, resseq, atoms in residues:
        if chain_id != current_chain:
            sb.init_chain(chain_id)
            current_chain = chain_id
        sb.init_seg(" ")
        sb.init_residue(resname, " ", resseq, " ")
        for serial, (name, xyz) in enumerate(atoms.items(), start=1):
            element = name[0]
            sb.init_atom(
                name, np.array(xyz, dtype=float), 20.0, 1.0, " ", name, serial, element
            )
    return sb.get_structure()[0]  # first model


@pytest.fixture
def salt_bridge_pair():
    """Arg (chain A) and Asp (chain B) with NH1...OD1 ~2.8 A apart -> one salt bridge.

    Also close enough overall to register as an interface contact.
    """
    residues = [
        ("A", "ARG", 1, {
            "CA": (0.0, 0.0, 0.0),
            "CB": (1.5, 0.0, 0.0),
            "NE": (3.0, 0.0, 0.0),
            "NH1": (4.0, 0.0, 0.0),
            "NH2": (4.0, 1.0, 0.0),
        }),
        ("B", "ASP", 1, {
            "CA": (10.0, 0.0, 0.0),
            "CB": (8.5, 0.0, 0.0),
            "OD1": (6.8, 0.0, 0.0),   # ~2.8 A from NH1 at x=4.0
            "OD2": (6.8, 1.2, 0.0),
        }),
    ]
    return build_structure(residues)
