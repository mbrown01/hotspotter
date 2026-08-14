"""Residue identity & physicochemical properties — cheap lookup-table features.

BIOLOGY NOTE:
    The 20 amino acids differ in a handful of properties that predict how they behave at
    an interface. None of these needs the structure — they're intrinsic to the residue
    type — but they contextualize the geometric features. A big aromatic residue burying a
    lot of area behaves differently from a small polar one; a charged residue is a
    salt-bridge candidate before we even look at geometry.

    charge          formal side-chain charge at pH ~7.4 (+1 Arg/Lys, -1 Asp/Glu, else 0).
    hydropathy      Kyte-Doolittle scale (+ = greasy/buries well, - = polar/likes water).
    volume          side-chain volume (A^3); mutating a big residue to a small one leaves a
                    cavity, a classic way to disrupt binding.
    is_aromatic     Phe/Tyr/Trp/His — ring-bearing, capable of pi-stacking/cation-pi.
    is_charged / is_polar   coarse class flags.
    flexibility     intrinsic backbone mobility propensity (Vihinen); rigid residues at an
                    interface are cheaper to freeze on binding.
"""

from __future__ import annotations

from hotspot.constants import (
    AROMATIC_RESIDUES,
    FLEXIBILITY,
    FORMAL_CHARGE,
    KYTE_DOOLITTLE,
    RESIDUE_VOLUME,
    THREE_TO_ONE,
)
from hotspot.interface import Interface
from hotspot.io import ResidueId

_POLAR = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "TRP"})
_CHARGED = frozenset(FORMAL_CHARGE) | {"ARG", "LYS", "ASP", "GLU"}


def compute_identity_features(interface: Interface) -> dict[ResidueId, dict]:
    """Per-residue physicochemical properties for all interface residues."""
    feats: dict[ResidueId, dict] = {}
    for rid in interface.residues:
        resname = rid.resname
        feats[rid] = {
            "aa": THREE_TO_ONE.get(resname, "X"),
            "charge": FORMAL_CHARGE.get(resname, 0),
            "hydropathy": KYTE_DOOLITTLE.get(resname),
            "volume": RESIDUE_VOLUME.get(resname),
            "flexibility": FLEXIBILITY.get(resname),
            "is_aromatic": int(resname in AROMATIC_RESIDUES),
            "is_charged": int(resname in _CHARGED),
            "is_polar": int(resname in _POLAR),
        }
    return feats
