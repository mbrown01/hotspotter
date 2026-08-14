"""The Phase-1 pipeline: structure in -> ranked per-residue feature table out.

    analyze_complex(source, chains) does the whole thing:
        1. load (download by PDB id, or read a local file)
        2. detect the interface (heavy-atom contacts)
        3. compute every feature group
        4. merge into one tidy per-residue table
        5. rank with the transparent hot-spot heuristic

The returned table is deliberately "model-ready": one row per interface residue, one
column per feature. That's exactly the shape Phase 2 trains on — run this across SKEMPI's
complexes, join the ΔΔG labels, and you have a training set. Same engine, two uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from hotspotter.features.chemistry import Contact, detect_contacts, per_residue_chemistry
from hotspotter.features.confidence import compute_confidence_features
from hotspotter.features.conservation import compute_conservation_features
from hotspotter.features.identity import compute_identity_features
from hotspotter.features.sasa import SASA_BACKEND, compute_sasa_features
from hotspotter.features.topology import compute_topology_features
from hotspotter.interface import Interface, detect_interface, guess_two_sides
from hotspotter.io import (
    ResidueId,
    fetch_pdb,
    get_chain_ids,
    get_model,
    load_structure,
)
from hotspotter.ranking import Weights, hotspot_score


@dataclass
class ComplexAnalysis:
    """Everything Phase 1 produces for one complex."""

    table: pd.DataFrame                 # ranked, one row per interface residue
    contacts: list[Contact]             # every detected cross-interface interaction
    interface: Interface = field(repr=False)
    structure: object = field(repr=False)
    source: str = ""
    side_a_chains: tuple[str, ...] = ()
    side_b_chains: tuple[str, ...] = ()
    is_predicted: bool = False
    sasa_backend: str = SASA_BACKEND

    def top(self, n: int = 10) -> pd.DataFrame:
        """The n highest-ranked hot-spot candidates, key columns only."""
        cols = [c for c in ("hotspot_rank", "naive_rank", "residue", "aa",
                            "hotspot_score", "dsasa", "n_salt_bridges",
                            "n_hydrogen_bonds", "reasoning")
                if c in self.table.columns]
        return self.table.nsmallest(n, "hotspot_rank")[cols]


def _parse_chains(chains, chain_ids):
    """Normalize the `chains` argument into (side_a, side_b) chain-id tuples.

    Accepts:
      None                       -> auto: first chain vs. the rest (warns for >2 chains)
      "A,D" or "A/D" or "AD"     -> two single-chain sides A and D
      "AB,CD"                    -> multi-chain sides {A,B} and {C,D}
      (("A",), ("D",))           -> already-split groups, passed through
    """
    if chains is None:
        a, b = guess_two_sides(chain_ids)
        if len(chain_ids) > 2:
            print(f"[hotspotter] >2 chains {chain_ids}; defaulting to {a} vs {b}. "
                  f"Pass chains=('X','Y') to be explicit.")
        return a, b
    if isinstance(chains, str):
        sep = "," if "," in chains else ("/" if "/" in chains else None)
        if sep:
            left, right = chains.split(sep, 1)
            return tuple(left), tuple(right)
        if len(chains) == 2:            # "AD" -> A vs D
            return (chains[0],), (chains[1],)
        raise ValueError(f"Ambiguous chains string {chains!r}; use 'A,D' or 'AB,CD'.")
    # assume a 2-tuple of iterables
    a, b = chains
    return tuple(a), tuple(b)


def analyze_complex(
    source: str | Path,
    chains=None,
    is_predicted: bool = False,
    conservation_scores: dict[ResidueId, float] | None = None,
    weights: Weights | None = None,
) -> ComplexAnalysis:
    """Run the full Phase-1 pipeline on one complex.

    Parameters
    ----------
    source : a 4-character PDB id (downloaded and cached) OR a path to a local .pdb/.cif.
    chains : which chains form the two sides (see :func:`_parse_chains`). None = auto.
    is_predicted : True if this is an AlphaFold/ColabFold model (so the B-factor column is
        pLDDT, and confidence features are labeled accordingly).
    conservation_scores : optional precomputed per-residue conservation (Phase-1.5).
    weights : optional custom ranking weights.
    """
    # 1. Load ------------------------------------------------------------------------
    source_str = str(source)
    is_pdb_id = (
        not Path(source_str).exists()
        and len(source_str) == 4
        and source_str.isalnum()
    )
    path = fetch_pdb(source_str) if is_pdb_id else Path(source_str)
    structure = load_structure(path, structure_id=source_str)
    model = get_model(structure)

    chain_ids = get_chain_ids(structure)
    side_a, side_b = _parse_chains(chains, chain_ids)

    # 2. Interface -------------------------------------------------------------------
    interface = detect_interface(model, side_a, side_b)
    if len(interface) == 0:
        raise ValueError(
            f"No interface residues found between {side_a} and {side_b}. "
            f"Are these chains actually in contact? Available chains: {chain_ids}."
        )

    # 3. Features --------------------------------------------------------------------
    contacts = detect_contacts(interface)
    chem = per_residue_chemistry(interface, contacts=contacts)
    sasa = compute_sasa_features(model, side_a, side_b)
    topo = compute_topology_features(model, interface)
    ident = compute_identity_features(interface)
    conf = compute_confidence_features(interface, is_predicted=is_predicted)
    cons = compute_conservation_features(interface, scores=conservation_scores)

    # 4. Merge into one tidy table ---------------------------------------------------
    rows = []
    for rid, ir in interface.residues.items():
        row = {
            "residue": rid.label,
            "chain": rid.chain,
            "resseq": rid.resseq,
            "icode": rid.icode.strip(),
            "resname": rid.resname,
            "side": ir.side,
        }
        for group in (chem, sasa, topo, ident, conf, cons):
            row.update(group.get(rid, {}))
        rows.append(row)
    df = pd.DataFrame(rows)

    # 5. Rank ------------------------------------------------------------------------
    ranked = hotspot_score(df, weights=weights)

    return ComplexAnalysis(
        table=ranked,
        contacts=contacts,
        interface=interface,
        structure=structure,
        source=source_str,
        side_a_chains=side_a,
        side_b_chains=side_b,
        is_predicted=is_predicted,
    )
