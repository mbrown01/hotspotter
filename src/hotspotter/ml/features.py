"""Feature column definitions and label construction, in one place.

These lists are the contract between the feature table, the model, and every evaluation
script. They previously lived inside a benchmark script, which meant the production
inference path imported its column order from an experiment. Anything that trains, scores
or explains a model should import from here.

THE TWO FEATURE SETS
    ``XGB_FEATURES`` (30) is what the shipped model uses.
    ``NODE_FEATURES`` (26) is the graph node matrix. It is not a subset: it carries
    ``n_disulfides``, which the tabular model drops because interface disulfides are
    rare enough in SKEMPI that the column is almost always zero, and it lacks the five
    columns added after the graphs were built (van der Waals packing, conservation,
    side-chain plasticity). Its order is pinned because saved checkpoints depend on it.
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
CHEMISTRY: tuple[str, ...] = (
    "n_salt_bridges", "n_hydrogen_bonds", "n_hydrophobic", "n_aromatic",
    "n_chem_contacts", "has_salt_bridge",
)

#: Van der Waals packing against the partner protein. The only one of the four classical
#: force-field terms that added anything the counts above were not already capturing:
#: +0.0115 PR-AUC, significant across six fold partitions. See features/vdw.py.
VDW: tuple[str, ...] = ("e_vdw_sc", "n_vdw_pairs", "e_vdw_rep")

#: Evolutionary conservation from ConSurf-DB: +0.0085 PR-AUC on top of the above. The only
#: feature not derived from the wild-type structure. MAY BE NaN -- roughly 4% of chains
#: have no ConSurf entry, and median-imputing those rows costs 78% of the benefit, so the
#: missing values are passed to XGBoost untouched. See features/conservation.py.
CONSERVATION: tuple[str, ...] = ("cons_score",)
#: Structure-quality context: properties of the EXPERIMENT, not of the residue.
#:
#: COMPUTED AND STORED BUT **NOT** IN THE MODEL. Both measured as significantly harmful:
#: `resolution` -0.0137 (p=0.011), both together -0.0160 (p=0.003). The reason is
#: structural rather than statistical -- each takes exactly ONE value per complex, so it
#: cannot reorder residues within an interface, which is the only thing that helps a
#: user. It can only shift whole complexes against each other in the pooled metric, and
#: that does damage. Kept as columns because they are the right way to filter or
#: stratify rows by how trustworthy they are.
QUALITY: tuple[str, ...] = ("resolution", "is_xray")

#: Side-chain plasticity relative to the residue's own backbone: +0.0161 PR-AUC, the
#: largest single-feature gain here. The raw side-chain B-factor is NOT shipped and is
#: worth nothing on its own; `features.confidence._sidechain_bfactor_ratio` says why.
PLASTICITY: tuple[str, ...] = ("bfactor_sc_bb",)

#: The inputs to the accepted XGBoost model.
XGB_FEATURES: tuple[str, ...] = (
    SHARED_NON_CHEMISTRY + CHEMISTRY + VDW + CONSERVATION + PLASTICITY
)

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

    The key carries ``pdb_tag`` when the table has it. ``pdb_id`` is the PDB entry, but
    SKEMPI keys a measurement by entry AND partner chains, so one entry alanine-scanned
    against two different partners is two interfaces with two sets of features. Grouping
    on the entry alone would merge them and keep whichever ddG happened to be larger.
    No row in the current table hits this; the key is correct anyway.
    """
    sub = df if strategy == "max" else df[df["mut"] == "A"]
    sub = sub.copy()
    sub["icode"] = sub["icode"].fillna("").astype(str).str.strip()
    key = (["pdb_id"] + (["pdb_tag"] if "pdb_tag" in sub.columns else [])
           + ["chain", "resseq", "icode"])
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


# ---------------------------------------------------------------------------------------
# Missing values, per model family
# ---------------------------------------------------------------------------------------
#: Features that can legitimately be absent. Adding a column here is the ONLY change
#: needed for every model that cannot take NaN to receive a missing-indicator for it.
#:
#: The missingness is informative and must survive imputation: ConSurf-DB has no entry
#: for ~4% of chains, and those chains are not a random sample -- they are short
#: peptides, 50.9% hot against a 23.4% base rate. The B-factor columns are nullable
#: because a non-X-ray entry has no crystallographic B-factor to report.
NULLABLE: tuple[str, ...] = ("cons_score", "bfactor", "resolution", "bfactor_sc_bb")

#: Model families that learn a default branch direction for NaN and need no imputation.
NAN_NATIVE: frozenset[str] = frozenset(
    {"xgboost", "lightgbm", "catboost", "hist_gradient_boosting"}
)


def missing_indicator_name(column: str) -> str:
    return f"{column}_is_missing"


def prepare_matrix(df, features=XGB_FEATURES, model="xgboost", stats=None):
    """Build the model matrix, handling missing values the way `model` requires.

    Models in NAN_NATIVE get the values untouched, because learning a default branch per
    split beats substituting a constant: median-filling conservation cost 78% of its
    measured benefit. Everything else gets each NULLABLE column imputed PLUS a binary
    ``<column>_is_missing`` indicator, so the imputation does not destroy the information
    in the missingness itself.

    Parameters
    ----------
    stats : None to compute imputation values from ``df`` -- do this on the TRAINING fold
        only -- or the dict returned by such a call, to apply the same values to a test
        fold. Computing medians over the full dataset would leak test information into
        training, which is subtle and would not show up as an error.

    Returns
    -------
    (X, stats) : the matrix, and the imputation values to reuse on held-out data.
    """
    features = list(features)
    X = df[features].copy()
    if model in NAN_NATIVE:
        return X, {}

    nullable = [c for c in NULLABLE if c in features]
    if stats is None:
        stats = {c: float(X[c].median()) for c in nullable}
        stats = {c: (0.0 if v != v else v) for c, v in stats.items()}   # all-NaN column
    for c in nullable:
        X[missing_indicator_name(c)] = X[c].isna().astype(float)
        X[c] = X[c].fillna(stats[c])
    # any remaining NaN is unexpected; fail loudly rather than silently filling
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"unexpected NaN in {bad} for model={model!r}; add them to NULLABLE "
            f"if they can legitimately be missing")
    return X, stats
