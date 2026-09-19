"""Feature column definitions and label construction, in one place.

These lists are the contract between the feature table, the model, and every evaluation
script. They previously lived inside a benchmark script, which meant the production
inference path imported its column order from an experiment. Anything that trains, scores
or explains a model should import from here.

THE TWO FEATURE SETS, AND WHY THEY DIFFER BY ONE COLUMN
    ``XGB_FEATURES`` (25) is what the accepted model uses.
    ``NODE_FEATURES`` (26) is the graph node matrix, which additionally carries
    ``n_disulfides``. Interface disulfides are rare enough in SKEMPI that the column is
    almost always zero, so the tabular model drops it; the graph builder keeps it because
    the node matrix is also used for inspection, not only for fitting.
"""

from __future__ import annotations

import pandas as pd

#: ddG (kcal/mol) at or above which a mutation counts as disruptive.
#: 2.0 is the Bogan & Thorn (1998) / Clackson & Wells (1995) hot-spot definition.
DDG_HOTSPOT_THRESHOLD = 2.0

#: Burial, topology, identity and confidence columns. Identical across feature-set versions.
SHARED_NON_CHEMISTRY: tuple[str, ...] = (
    # burial / accessibility
    "sasa_complex", "sasa_unbound", "dsasa", "rsa_complex", "rsa_unbound",
    "is_interface_sasa",
    # interface topology
    "n_cross_contacts", "n_atom_contacts", "interface_neighbors", "packing_density",
    "centrality",
    # residue identity / physicochemistry
    "charge", "hydropathy", "volume", "flexibility", "is_aromatic", "is_charged",
    "is_polar",
    # structure confidence
    "bfactor",
)

#: Interaction-chemistry counts. Counts are DISTINCT PARTNER RESIDUES, not atom pairs.
V1_CHEMISTRY: tuple[str, ...] = (
    "n_salt_bridges", "n_hydrogen_bonds", "n_hydrophobic", "n_aromatic",
    "n_chem_contacts", "has_salt_bridge",
)

#: The 25 inputs to the accepted XGBoost model.
XGB_FEATURES: tuple[str, ...] = SHARED_NON_CHEMISTRY + V1_CHEMISTRY

#: The 26 columns of the graph node matrix, in a pinned order a saved model depends on.
NODE_FEATURES: tuple[str, ...] = (
    # interaction chemistry
    "n_salt_bridges", "n_hydrogen_bonds", "n_hydrophobic", "n_aromatic", "n_disulfides",
    "n_chem_contacts", "has_salt_bridge",
    # burial / accessibility
    "sasa_complex", "sasa_unbound", "dsasa", "rsa_complex", "rsa_unbound",
    "is_interface_sasa",
    # interface topology
    "n_cross_contacts", "n_atom_contacts", "interface_neighbors", "packing_density",
    "centrality",
    # residue identity / physicochemistry
    "charge", "hydropathy", "volume", "flexibility", "is_aromatic", "is_charged",
    "is_polar",
    # structure confidence
    "bfactor",
)


def collapse(df: pd.DataFrame, strategy: str = "alanine",
             threshold: float = DDG_HOTSPOT_THRESHOLD) -> pd.DataFrame:
    """One row per residue POSITION, from a table with one row per MUTATION.

    SKEMPI labels mutations; the model predicts residues. 3,752 mutations cover far fewer
    positions, and a position can carry both a disruptive and a neutral substitution, so the
    two have to be reconciled explicitly rather than by whichever row happens to sort first.

    strategy="alanine"  keep only X->Ala substitutions. This is the literature hot-spot
                        definition, so the label means one specific thing. Costs data.
    strategy="max"      take the largest ddG at the position: "load-bearing under any
                        substitution?" Keeps more positions but conflates interface hot
                        spots with proline and charge-reversal backbone effects.

    Requires a ``mut`` column (the substituted residue letter) and ``pdb_id``.
    """
    sub = df if strategy == "max" else df[df["mut"] == "A"]
    sub = sub.copy()
    sub["icode"] = sub["icode"].fillna("").astype(str).str.strip()
    key = ["pdb_id", "chain", "resseq", "icode"]
    idx = sub.groupby(key)["ddg"].idxmax()
    out = sub.loc[idx].copy()
    out["label"] = (out["ddg"] >= threshold).astype(int)
    return out


def load_labeled_table(csv_path, strategy: str = "alanine",
                       threshold: float = DDG_HOTSPOT_THRESHOLD,
                       exclude: set[str] | None = None) -> pd.DataFrame:
    """Read the SKEMPI feature table and collapse it to one labeled row per residue.

    ``exclude`` drops complexes by PDB id before collapsing -- used to hold a target complex
    out of its own training set.
    """
    from hotspotter.ml.dataset import parse_mutation

    df = pd.read_csv(csv_path)
    df["mut"] = df["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    df["pdb_id"] = df["complex_group"]
    if exclude:
        df = df[~df["pdb_id"].str.upper().isin({e.upper() for e in exclude})]
    return collapse(df, strategy, threshold).reset_index(drop=True)
