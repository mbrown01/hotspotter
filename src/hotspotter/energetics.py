"""Implicit-solvent binding energetics per interface residue.

WHAT THIS ADDS OVER THE EXISTING FEATURES
    Everything in V1 is a geometric proxy for energy: buried area, relative accessibility,
    packing density. Those correlate with binding energy because burial and energy are
    related, not because anything was computed thermodynamically.

    This module computes three terms that are actually energetic, using real per-atom
    partial charges and Born radii assigned by pdb2pqr (AMBER force field, with PROPKA
    resolving protonation states and histidine tautomers):

        coulomb        electrostatic interaction with the partner chain, screened by a
                       distance-dependent dielectric
        desolvation    the polar cost of removing water from a charged atom as it becomes
                       buried -- literally "what it costs to strip the hydration shell"
        nonpolar       the hydrophobic-effect term, gamma * buried area

    Coulomb + polar desolvation + nonpolar is the same three-way decomposition a Poisson-
    Boltzmann/surface-area calculation reports. This is the generalized-Born approximation
    to it, which is the standard cheap substitute.

WHY NOT FULL POISSON-BOLTZMANN
    APBS solves the PB equation properly, but it returns a total electrostatic energy for a
    system, not a per-residue attribution. Decomposing it per residue needs either one
    calculation per residue per state (172 complexes x ~50 residues x 3 states is not
    tractable here) or a linear-response approximation that gives up most of the accuracy
    advantage anyway. Generalized Born captures the same physics -- charge screening and
    burial-dependent desolvation -- analytically and per atom.

WHY THIS IS WORTH TRYING WHEN EIGHT PREVIOUS FEATURE ADDITIONS FAILED
    The existing crude electrostatic term (`coulombic_sum` in features_v2) uses two-value
    +/-0.5 charges on formally charged groups only. It is therefore exactly 0.0 for 76% of
    residues -- and STILL correlates with ddG at rho = -0.162, p = 8e-08. A term that is
    three-quarters dead and still significant is the clearest signal we have that a proper
    treatment has headroom. Every other failed addition was a new signal we hoped for; this
    one is a measured signal we are un-crippling.

HONEST LIMITS
    - Generalized Born is an approximation to PB, and the simple Still-style formulation
      used here is the cheap end of GB.
    - The distance-dependent dielectric eps(r) = 4r is a convention, not a measurement.
    - Nonpolar surface tension gamma = 0.005 kcal/(mol*A^2) is the common literature value
      but varies between force fields.
    - No conformational relaxation: this is a rigid-body, single-structure calculation, so
      it cannot see strain relief or side-chain repacking on mutation. FoldX and Rosetta do
      model those, which is part of why they are slower and better.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Coulomb constant in kcal*A/(mol*e^2).
COULOMB_K = 332.0637

#: Nonpolar surface tension, kcal/(mol*A^2). Common implicit-solvent value.
GAMMA_NONPOLAR = 0.005

#: Solvent and solute dielectric constants.
EPS_SOLVENT = 78.5
EPS_SOLUTE = 4.0

ENERGY_COLUMNS = (
    "e_coulomb", "e_desolvation", "e_nonpolar", "e_total",
    "charge_buried", "n_charged_contacts",
)


@dataclass
class PQRAtom:
    chain: str
    resseq: int
    icode: str
    resname: str
    name: str
    coord: np.ndarray
    charge: float
    radius: float


def _pdb2pqr_exe() -> str:
    """Locate the pdb2pqr executable that ships with the active interpreter."""
    exe = shutil.which("pdb2pqr") or shutil.which("pdb2pqr.exe")
    if exe:
        return exe
    candidate = Path(sys.executable).parent / "pdb2pqr.exe"
    if candidate.exists():
        return str(candidate)
    candidate = Path(sys.executable).parent / "pdb2pqr"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("pdb2pqr not found; pip install pdb2pqr")


def run_pdb2pqr(pdb_path: Path, out_pqr: Path | None = None,
                forcefield: str = "AMBER", timeout: int = 300) -> Path:
    """Assign per-atom partial charges and Born radii.

    pdb2pqr also fixes protonation: it adds hydrogens, picks histidine tautomers, and
    assigns Asp/Glu/Lys protonation via PROPKA. That matters here because an electrostatic
    calculation on a structure with guessed protonation is not meaningfully better than a
    geometric proxy.
    """
    out_pqr = out_pqr or Path(tempfile.gettempdir()) / f"{pdb_path.stem}.pqr"
    # --keep-chain is not optional here: without it pdb2pqr omits the chain column
    # entirely and every atom in a multi-chain complex collapses onto one chain id,
    # which silently destroys the interface definition.
    cmd = [_pdb2pqr_exe(), f"--ff={forcefield}", "--whitespace", "--keep-chain",
           str(pdb_path), str(out_pqr)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if not out_pqr.exists() or out_pqr.stat().st_size == 0:
        raise RuntimeError(f"pdb2pqr produced no output for {pdb_path.name}: "
                           f"{res.stderr[-400:]}")
    return out_pqr


def parse_pqr(pqr_path: Path) -> list[PQRAtom]:
    """Read a whitespace-delimited PQR file into atoms.

    PQR replaces the occupancy and B-factor columns with charge and radius, and the
    --whitespace flag makes the file space-delimited rather than column-fixed, which avoids
    the field-overflow problems that plague fixed-width PDB parsing of large structures.
    """
    atoms: list[PQRAtom] = []
    for line in pqr_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        f = line.split()
        # ATOM serial name resname [chain] resseq x y z charge radius
        # pdb2pqr omits the chain field when the input has a blank chain id
        try:
            if len(f) >= 11:
                _, _, name, resname, chain, resseq, x, y, z, q, r = f[:11]
            else:
                _, _, name, resname, resseq, x, y, z, q, r = f[:10]
                chain = "A"
            icode = ""
            if not resseq.lstrip("-").isdigit():           # trailing insertion code
                icode, resseq = resseq[-1], resseq[:-1]
            atoms.append(PQRAtom(chain=chain, resseq=int(resseq), icode=icode,
                                 resname=resname, name=name,
                                 coord=np.array([float(x), float(y), float(z)]),
                                 charge=float(q), radius=float(r)))
        except (ValueError, IndexError):
            continue
    return atoms


def _born_radius(atom: PQRAtom, neighbours: np.ndarray, radii: np.ndarray) -> float:
    """Still-style effective Born radius: larger when the atom is more buried.

    A charge deep inside protein sees little solvent, so its effective Born radius grows
    and its solvation free energy shrinks in magnitude. That difference between the bound
    and unbound state IS the desolvation penalty.
    """
    if len(neighbours) == 0:
        return atom.radius
    d = np.linalg.norm(neighbours - atom.coord, axis=1)
    keep = (d > 0.1) & (d < 12.0)
    if not keep.any():
        return atom.radius
    # occlusion proxy: summed neighbour volume weighted by inverse distance^4
    occ = float(np.sum((radii[keep] ** 3) / (d[keep] ** 4)))
    return atom.radius * (1.0 + 0.6 * occ)


def _gb_self_energy(charge: float, born_r: float) -> float:
    """Generalized-Born self-energy of one charge, kcal/mol (negative = favourable)."""
    if born_r <= 0:
        return 0.0
    return -0.5 * COULOMB_K * (1.0 / EPS_SOLUTE - 1.0 / EPS_SOLVENT) * charge ** 2 / born_r


def residue_energetics(
    atoms: list[PQRAtom],
    side_a: set[str],
    side_b: set[str],
    dsasa_by_residue: dict | None = None,
    cutoff: float = 12.0,
    keep: set[tuple] | None = None,
) -> dict[tuple, dict]:
    """Per-residue Coulomb, desolvation and nonpolar terms across the interface.

    Keys are (chain, resseq, icode) to match the pipeline's residue identity.

    ``keep`` restricts the OUTPUT loop to specific residues -- normally the interface set.
    Only ~40 residues per complex are ever used downstream but a chain pair holds ~480, so
    scoring everything does twelve times the necessary work. The partner atoms must still
    all be present (they are the field the kept residues feel), so this filters the residues
    we iterate over, not the atoms we compute against.
    """
    a_atoms = [a for a in atoms if a.chain in side_a]
    b_atoms = [a for a in atoms if a.chain in side_b]
    if not a_atoms or not b_atoms:
        return {}

    a_xyz = np.array([a.coord for a in a_atoms])
    b_xyz = np.array([a.coord for a in b_atoms])
    a_q = np.array([a.charge for a in a_atoms])
    b_q = np.array([a.charge for a in b_atoms])
    a_r = np.array([a.radius for a in a_atoms])
    b_r = np.array([a.radius for a in b_atoms])

    out: dict[tuple, dict] = {}

    for own, own_xyz, own_q, own_r, other_xyz, other_q, other_r in (
        (a_atoms, a_xyz, a_q, a_r, b_xyz, b_q, b_r),
        (b_atoms, b_xyz, b_q, b_r, a_xyz, a_q, a_r),
    ):
        for i, atom in enumerate(own):
            key = (atom.chain, atom.resseq, atom.icode)
            if keep is not None and key not in keep:
                continue
            rec = out.setdefault(key, {c: 0.0 for c in ENERGY_COLUMNS})

            d = np.linalg.norm(other_xyz - atom.coord, axis=1)
            near = (d > 0.1) & (d < cutoff)
            if near.any():
                dd = d[near]
                # distance-dependent dielectric eps(r) = 4r screens long-range terms the
                # way bulk solvent would, without an explicit solvent model
                rec["e_coulomb"] += float(
                    COULOMB_K * atom.charge * np.sum(other_q[near] / (EPS_SOLUTE * dd * dd))
                )
                rec["n_charged_contacts"] += float(
                    np.sum(np.abs(other_q[near]) > 0.3) * (abs(atom.charge) > 0.3)
                )

            # desolvation: Born self-energy unbound (own chain only) vs bound (both chains)
            own_neigh = own_xyz[np.linalg.norm(own_xyz - atom.coord, axis=1) < cutoff]
            own_rad = own_r[np.linalg.norm(own_xyz - atom.coord, axis=1) < cutoff]
            r_unbound = _born_radius(atom, own_neigh, own_rad)
            all_neigh = np.vstack([own_neigh, other_xyz[near]]) if near.any() else own_neigh
            all_rad = np.concatenate([own_rad, other_r[near]]) if near.any() else own_rad
            r_bound = _born_radius(atom, all_neigh, all_rad)
            rec["e_desolvation"] += (_gb_self_energy(atom.charge, r_bound)
                                     - _gb_self_energy(atom.charge, r_unbound))
            if near.any():
                rec["charge_buried"] += abs(atom.charge)

    for key, rec in out.items():
        if dsasa_by_residue:
            rec["e_nonpolar"] = -GAMMA_NONPOLAR * float(dsasa_by_residue.get(key, 0.0))
        rec["e_total"] = rec["e_coulomb"] + rec["e_desolvation"] + rec["e_nonpolar"]
    return out


def energetics_for_complex(pdb_path: Path, side_a, side_b,
                           dsasa_by_residue: dict | None = None,
                           keep: set[tuple] | None = None) -> dict[tuple, dict]:
    """End-to-end: pdb2pqr then per-residue energies.

    Pass ``keep`` (usually the interface residue keys) to avoid scoring residues nothing
    downstream will read.
    """
    pqr = run_pdb2pqr(Path(pdb_path))
    atoms = parse_pqr(pqr)
    return residue_energetics(atoms, set(side_a), set(side_b), dsasa_by_residue, keep=keep)
