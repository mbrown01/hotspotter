"""Interface detection: which residues from each side are actually in contact.

BIOLOGY NOTE:
    An "interface" is the patch of surface where two proteins touch. There are two common,
    complementary ways to define it, and we compute both because they disagree at the
    edges and the disagreement is informative:

      1. CONTACT-based (this module): a residue is at the interface if any of its heavy
         atoms comes within a cutoff (default 5 A) of the *other* protein. Fast, purely
         geometric, and gives us per-residue *contact counts* for free.

      2. SASA-based (features/sasa.py): a residue is at the interface if it *buries* surface
         area when the two proteins come together (it was exposed to water alone, and gets
         covered upon binding). This is the classic "buried surface area" definition and is
         what tools like PISA emphasize.

    Hot spots live in this patch — but *which* residue in the patch is load-bearing is
    exactly what buriedness alone gets wrong (the Y96-vs-Arg95 lesson). So interface
    detection just *scopes* the problem; the feature modules do the discriminating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from Bio.PDB import NeighborSearch
from Bio.PDB.Residue import Residue

from hotspotter.constants import Cutoffs
from hotspotter.io import ResidueId, is_amino_acid, residue_id


@dataclass
class InterfaceResidue:
    """One residue found at the interface, with its cross-interface contacts."""

    res_id: ResidueId
    residue: Residue = field(repr=False)
    side: str                      # which group of chains this residue belongs to ("A" or "B")
    atom_contacts: int = 0         # number of heavy-atom pairs across the interface within cutoff
    contact_partners: set = field(default_factory=set)  # set[ResidueId] on the other side

    @property
    def n_partner_residues(self) -> int:
        return len(self.contact_partners)


@dataclass
class Interface:
    """The detected interface: interface residues on each side + the contact graph."""

    side_a_chains: tuple[str, ...]
    side_b_chains: tuple[str, ...]
    residues: dict[ResidueId, InterfaceResidue]  # all interface residues, both sides
    cutoff: float

    @property
    def side_a(self) -> list[InterfaceResidue]:
        return [r for r in self.residues.values() if r.side == "A"]

    @property
    def side_b(self) -> list[InterfaceResidue]:
        return [r for r in self.residues.values() if r.side == "B"]

    def __len__(self) -> int:
        return len(self.residues)


def _heavy_atoms(residue: Residue) -> list:
    """Non-hydrogen atoms of a residue (H atoms are often absent and add noise)."""
    return [a for a in residue.get_atoms() if a.element != "H"]


def detect_interface(
    model,
    side_a_chains: Iterable[str],
    side_b_chains: Iterable[str],
    cutoff: float = Cutoffs.INTERFACE_HEAVY_ATOM,
) -> Interface:
    """Find interface residues between two groups of chains via heavy-atom contacts.

    Parameters
    ----------
    model : a Biopython Model (use ``hotspotter.io.get_model(structure)``).
    side_a_chains, side_b_chains : chain ids on each side of the interface, e.g.
        ("A",) and ("D",) for barnase-barstar, or multi-chain groups for larger assemblies.
    cutoff : heavy-atom distance (A) defining a contact.

    Returns
    -------
    Interface with per-residue contact counts and the cross-interface contact graph.
    """
    side_a_chains = tuple(side_a_chains)
    side_b_chains = tuple(side_b_chains)
    a_set, b_set = set(side_a_chains), set(side_b_chains)

    # Collect heavy atoms per side, tagging each atom with its side for the search.
    atoms_a, atoms_b = [], []
    for chain in model.get_chains():
        if chain.id in a_set:
            for res in chain.get_residues():
                if is_amino_acid(res):
                    atoms_a.extend(_heavy_atoms(res))
        elif chain.id in b_set:
            for res in chain.get_residues():
                if is_amino_acid(res):
                    atoms_b.extend(_heavy_atoms(res))

    if not atoms_a or not atoms_b:
        raise ValueError(
            f"No amino-acid atoms found for one side "
            f"(side A chains {side_a_chains}: {len(atoms_a)} atoms, "
            f"side B chains {side_b_chains}: {len(atoms_b)} atoms). "
            f"Check the chain ids."
        )

    # KD-tree over side B; query each side-A atom's neighborhood. O(N log N), scales fine.
    ns = NeighborSearch(atoms_b)
    residues: dict[ResidueId, InterfaceResidue] = {}

    def _get_or_make(res: Residue, side: str) -> InterfaceResidue:
        rid = residue_id(res)
        if rid not in residues:
            residues[rid] = InterfaceResidue(res_id=rid, residue=res, side=side)
        return residues[rid]

    for atom_a in atoms_a:
        near = ns.search(atom_a.coord, cutoff, level="A")  # nearby side-B atoms
        if not near:
            continue
        res_a = atom_a.get_parent()
        ir_a = _get_or_make(res_a, "A")
        for atom_b in near:
            res_b = atom_b.get_parent()
            ir_b = _get_or_make(res_b, "B")
            ir_a.atom_contacts += 1
            ir_b.atom_contacts += 1
            ir_a.contact_partners.add(ir_b.res_id)
            ir_b.contact_partners.add(ir_a.res_id)

    return Interface(
        side_a_chains=side_a_chains,
        side_b_chains=side_b_chains,
        residues=residues,
        cutoff=cutoff,
    )


def guess_two_sides(chain_ids: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Best-effort split of chains into two interacting sides when the user doesn't specify.

    Simplest sensible default: first chain vs. the rest. For a clean 2-chain complex this
    is exactly right; for anything bigger we warn the caller to pass chains explicitly.
    """
    if len(chain_ids) < 2:
        raise ValueError(f"Need at least 2 chains to have an interface; got {chain_ids}.")
    return (chain_ids[0],), tuple(chain_ids[1:])
