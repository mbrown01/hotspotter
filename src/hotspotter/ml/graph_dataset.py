"""Graph data: turn a ComplexAnalysis into a PyTorch Geometric ``Data`` object.

This is the graph-shaped sibling of :mod:`hotspotter.ml.dataset`. That module flattens each
mutated residue into an independent row for XGBoost; this one keeps the interface as what it
actually is — a graph — so a GNN can use a residue's neighborhood, not just its own features.
Both read the SAME feature-pipeline output, so the two paths stay comparable and neither disturbs the
other.

WHAT ONE GRAPH IS
    One graph = one protein complex's interface.
      nodes      interface residues (both sides)
      node feats the 26 numeric feature-table columns, in a pinned order (NODE_FEATURES)
      edges      residue-residue contacts, of TWO classes (see below)
      edge feats which interaction types the pair makes, how far apart, and which class
      labels     per-node hot-spot label from SKEMPI, with a mask for untested residues

TWO EDGE CLASSES, AND WHY THIS MATTERS
    ``Interface.contact_partners`` is populated by a KD-tree built over side B and queried
    with side-A atoms, so it contains ONLY cross-interface pairs — the contact graph is
    strictly bipartite (verified on 1BRS: 110 cross edges, 0 same-side). Handing that to a
    GNN unmodified means a residue can never see its own-side neighbors, which is exactly
    the local environment the O-ring topology features are about.

    So we add a second edge class here: same-side residue pairs whose representative atoms
    (CB, or CA for glycine) fall within ``intra_side_cutoff``. This is computed in THIS
    module rather than in ``interface.py`` so locked feature-extraction behaviour is untouched. The
    ``is_cross_interface`` flag in ``edge_attr`` lets the model treat the two classes
    differently.

    NOTE: the 8 A CB-CB default for same-side edges is a graph-construction
    convention (common in protein GNNs), not one of the chemistry cutoffs. It is a
    knob, not a measurement.

LABELS, AND THE COLLAPSE PROBLEM
    A node is a RESIDUE; a SKEMPI label is a MUTATION. They are not 1:1 — 3,788 mutations
    cover only 1,975 residue positions, and 7% of those positions carry both a disruptive
    and a neutral substitution. Two strategies, selected by ``strategy``:

      "alanine" (default)  Use only X->Ala substitutions. This is the literature definition
                           of a hot spot (Bogan & Thorn 1998; Clackson & Wells 1995):
                           ddG(X->Ala) >= 2 kcal/mol. Cleanest biology, and the label means
                           one specific thing. Costs data: ~1,547 positions / 172 complexes.
      "max"                Take the max ddG over ALL substitutions at the position: "is this
                           position load-bearing under any substitution?" Keeps ~1,975
                           positions / 294 complexes, but conflates true interface hot spots
                           with proline/charge-reversal effects that disrupt the backbone.

    Untested residues get ``y = nan`` AND ``label_mask = False`` — both, so a loss function
    can select whichever it prefers. Roughly 85-90% of nodes are unlabeled (about 6.7 labeled
    positions per complex against interfaces of 40-80 residues), which is normal
    semi-supervised node classification but does mean the effective supervised set is small.

torch and torch_geometric are imported LAZILY inside the functions that need them, mirroring
how ``train.py`` defers sklearn/xgboost — so ``import hotspotter.ml`` never requires a 2 GB
deep-learning stack. Install them with ``pip install -e .[experiments]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hotspotter.features.geometry import distance
from hotspotter.io import ResidueId
from hotspotter.ml.dataset import (
    SKEMPI_COLUMNS,
    _clean_temperature,
    _to_float,
    ddg_from_kd,
    load_skempi,
    parse_mutation,
    parse_pdb_field,
)
from hotspotter.ml.features import DDG_HOTSPOT_THRESHOLD, NODE_FEATURES
from hotspotter.pipeline import ComplexAnalysis, analyze_complex

# ---------------------------------------------------------------------------------------
# Feature layout — pinned so every graph in the dataset has identical column semantics.
# ---------------------------------------------------------------------------------------

#: ``NODE_FEATURES`` and ``DDG_HOTSPOT_THRESHOLD`` are defined in ``hotspotter.ml.features``
#: and re-exported here, because the node matrix and the tabular model must never disagree
#: about column order or the hot-spot cutoff. Heuristic ranking outputs
#: (naive_score/hotspot_score/*_rank) are deliberately EXCLUDED from the node features: they
#: are derived from these same columns by hand-set weights, so feeding them in would leak
#: the heuristic the model is supposed to replace.

#: Interaction types that become multi-hot dimensions of ``edge_attr``.
EDGE_TYPES: tuple[str, ...] = (
    "salt_bridge", "hydrogen_bond", "hydrophobic", "aromatic", "disulfide",
)

#: edge_attr column order: the 5 type flags, then geometry, then the edge-class flag.
EDGE_ATTR_NAMES: tuple[str, ...] = EDGE_TYPES + ("distance", "is_cross_interface")

#: Default CB-CB cutoff (A) for same-side edges. Graph-construction convention, not a
#: chemistry cutoff.
INTRA_SIDE_CUTOFF = 8.0


def node_key(chain: str, resseq, icode: str = " ") -> tuple[str, int, str]:
    """The key that identifies one residue node: chain, number, and insertion code.

    Insertion codes are normalized to "" so that " ", "", and NaN all agree — the feature pipeline's
    table stores the code stripped while ``ResidueId`` stores it as " ".
    """
    ic = "" if icode is None else str(icode).strip()
    return (str(chain), int(resseq), ic)


@dataclass
class GraphBuildStats:
    """Counters from a dataset build, so a silent drop never looks like a clean run."""

    complexes_built: int = 0
    complexes_failed: int = 0
    complexes_unlabeled: int = 0     # built fine but no SKEMPI residue landed on the interface
    nodes_total: int = 0
    nodes_labeled: int = 0
    nodes_positive: int = 0
    mutations_unmatched: int = 0     # mutation not at the interface (expected; see docs)

    def summary(self) -> str:
        pos_rate = (self.nodes_positive / self.nodes_labeled) if self.nodes_labeled else 0.0
        return (
            f"graphs built {self.complexes_built} (failed {self.complexes_failed}, "
            f"unlabeled {self.complexes_unlabeled}) | nodes {self.nodes_total} "
            f"({self.nodes_labeled} labeled, {self.nodes_positive} positive, "
            f"rate {pos_rate:.4f}) | mutations off-interface {self.mutations_unmatched}"
        )


# ---------------------------------------------------------------------------------------
# SKEMPI -> tidy mutation frame
# ---------------------------------------------------------------------------------------
def prepare_mutation_frame(
    skempi: pd.DataFrame | str | Path,
    only_single_mutations: bool = True,
) -> pd.DataFrame:
    """Flatten SKEMPI into one tidy row per usable point mutation.

    Returns columns: pdb_id, side_a, side_b, chain, resseq, icode, wt, mut, ddg, mutation.

    ``only_single_mutations`` stays True by design: a row mutating three residues reports one
    combined ddG, so the change cannot be attributed to any single residue — and a GNN node
    label needs exactly that attribution.
    """
    df = load_skempi(skempi) if isinstance(skempi, (str, Path)) else skempi
    c = SKEMPI_COLUMNS
    rows = []
    for _, r in df.iterrows():
        mut_field = str(r[c["mutation"]])
        if only_single_mutations and ("," in mut_field):
            continue
        try:
            pdb_id, side_a, side_b = parse_pdb_field(str(r[c["pdb_field"]]))
        except ValueError:
            continue
        ddg = ddg_from_kd(
            _to_float(r[c["kd_wt"]]),
            _to_float(r[c["kd_mut"]]),
            _clean_temperature(r[c["temperature"]]),
        )
        if math.isnan(ddg):
            continue
        try:
            m = parse_mutation(mut_field)
        except Exception:
            continue
        rows.append({
            "pdb_id": pdb_id, "side_a": side_a, "side_b": side_b,
            "chain": m.chain, "resseq": m.resseq, "icode": m.icode,
            "wt": m.wt, "mut": m.mut, "ddg": ddg, "mutation": mut_field,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------------------
def build_node_matrix(analysis: ComplexAnalysis):
    """Return (x, residue_ids, index_of) for one complex.

    x           float tensor [n_nodes, len(NODE_FEATURES)]
    residue_ids list[ResidueId], row-aligned with x
    index_of    dict[(chain, resseq, icode)] -> row index, for joining SKEMPI mutations

    The key INCLUDES the insertion code. Kabat-numbered antibodies and proteases put several
    distinct residues at one number (1EAW chain A has 60, 60a, 60b, 60c, 60e, 60f, 60g); a
    (chain, resseq) key would collapse them onto a single node and mislabel the rest.
    """
    import torch

    t = analysis.table
    missing = [c for c in NODE_FEATURES if c not in t.columns]
    if missing:
        raise KeyError(
            f"feature table is missing expected node features {missing}. "
            f"Available: {list(t.columns)}"
        )

    # Row order is the table's order; everything downstream is aligned to it.
    feats = t[list(NODE_FEATURES)].astype(float)
    # A NaN here would silently poison training. Median-fill within the complex, matching
    # how train.py handles it, and 0.0 if a whole column is absent for this complex.
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0.0)
    x = torch.tensor(feats.to_numpy(), dtype=torch.float)

    residue_ids = [
        ResidueId(chain=str(r.chain), resseq=int(r.resseq),
                  icode=(str(r.icode) if str(r.icode).strip() else " "),
                  resname=str(r.resname))
        for r in t.itertuples()
    ]
    index_of = {node_key(rid.chain, rid.resseq, rid.icode): i
                for i, rid in enumerate(residue_ids)}
    if len(index_of) != len(residue_ids):
        raise AssertionError(
            f"node key collision in {analysis.source}: {len(residue_ids)} residues "
            f"collapsed to {len(index_of)} keys — labels would be misassigned"
        )
    return x, residue_ids, index_of


# ---------------------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------------------
def _representative_atom(residue):
    """CB if present (points along the side chain), else CA. Mirrors topology.py."""
    if "CB" in residue:
        return residue["CB"]
    if "CA" in residue:
        return residue["CA"]
    return next(residue.get_atoms())


def build_edges(
    analysis: ComplexAnalysis,
    residue_ids: list[ResidueId],
    index_of: dict,
    intra_side_cutoff: float = INTRA_SIDE_CUTOFF,
):
    """Build (edge_index, edge_attr) with BOTH cross-interface and same-side edges.

    Edges are undirected and stored in both directions, as PyG expects.

    edge_attr columns are EDGE_ATTR_NAMES:
        5 multi-hot interaction-type flags (cross-interface pairs only; a same-side pair or
        an untyped proximity pair is all-zero there), the representative-atom distance, and
        is_cross_interface.

    Using representative-atom distance for EVERY edge keeps one geometric feature that means
    the same thing on both edge classes — rather than a per-contact atom distance that would
    need a sentinel value on edges that have no typed contact.
    """
    import torch

    iface = analysis.interface
    pos = {rid: _representative_atom(ir.residue).coord
           for rid, ir in iface.residues.items()}
    side_of = {rid: ir.side for rid, ir in iface.residues.items()}

    # --- 1. chemistry multi-hot, keyed by unordered residue pair ------------------------
    chem: dict[tuple[int, int], list[float]] = {}
    for con in analysis.contacts:
        ia = index_of.get(node_key(con.res_a.chain, con.res_a.resseq, con.res_a.icode))
        ib = index_of.get(node_key(con.res_b.chain, con.res_b.resseq, con.res_b.icode))
        if ia is None or ib is None or ia == ib:
            continue
        key = (min(ia, ib), max(ia, ib))
        vec = chem.setdefault(key, [0.0] * len(EDGE_TYPES))
        if con.kind in EDGE_TYPES:
            vec[EDGE_TYPES.index(con.kind)] = 1.0

    # --- 2. the pair set: cross-interface contacts + same-side proximity ----------------
    pairs: dict[tuple[int, int], bool] = {}   # (i, j) -> is_cross_interface

    for rid, ir in iface.residues.items():
        i = index_of.get(node_key(rid.chain, rid.resseq, rid.icode))
        if i is None:
            continue
        for partner in ir.contact_partners:          # cross-interface by construction
            j = index_of.get(node_key(partner.chain, partner.resseq, partner.icode))
            if j is None or i == j:
                continue
            pairs[(min(i, j), max(i, j))] = True

    # Same-side edges: the piece contact_partners cannot give us. Without these the graph is
    # bipartite and a residue never sees its own-side neighborhood.
    ids = list(iface.residues.keys())
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            ra, rb = ids[a_i], ids[b_i]
            if side_of[ra] != side_of[rb]:
                continue
            i = index_of.get(node_key(ra.chain, ra.resseq, ra.icode))
            j = index_of.get(node_key(rb.chain, rb.resseq, rb.icode))
            if i is None or j is None or i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in pairs:
                continue
            if distance(pos[ra], pos[rb]) <= intra_side_cutoff:
                pairs[key] = False

    if not pairs:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, len(EDGE_ATTR_NAMES)), dtype=torch.float))

    # --- 3. materialize, both directions ------------------------------------------------
    src, dst, attrs = [], [], []
    coord = {index_of[node_key(rid.chain, rid.resseq, rid.icode)]: pos[rid]
             for rid in iface.residues
             if node_key(rid.chain, rid.resseq, rid.icode) in index_of}
    for (i, j), is_cross in sorted(pairs.items()):
        d = float(distance(coord[i], coord[j])) if i in coord and j in coord else 0.0
        vec = chem.get((i, j), [0.0] * len(EDGE_TYPES)) + [d, 1.0 if is_cross else 0.0]
        for a, b in ((i, j), (j, i)):
            src.append(a)
            dst.append(b)
            attrs.append(vec)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------
def build_labels(
    mutations: pd.DataFrame,
    index_of: dict,
    n_nodes: int,
    strategy: str = "alanine",
    ddg_threshold: float = DDG_HOTSPOT_THRESHOLD,
    residue_ids: list | None = None,
):
    """Collapse per-mutation ddG onto per-node labels.

    Parameters
    ----------
    mutations : tidy rows for THIS complex only (see :func:`prepare_mutation_frame`).
    strategy  : "alanine" -> only X->Ala substitutions (literature hot-spot definition).
                "max"     -> max ddG over all substitutions at the position.
    residue_ids : row-aligned ResidueIds. When given, a mutation whose wild-type letter
                disagrees with the residue actually at that position is REFUSED rather than
                used — a disagreement means we are looking at a different residue than
                SKEMPI meant, and a wrong label is worse than a missing one.

    Returns (y, label_mask, y_ddg, n_muts, n_unmatched):
        y           float tensor [n_nodes]; 1.0 / 0.0, and nan where untested
        label_mask  bool tensor [n_nodes]; True where a label exists
        y_ddg       float tensor [n_nodes]; the collapsed ddG (nan where untested), kept so
                    the same graphs can later be used for regression
        n_muts      long tensor [n_nodes]; how many substitutions backed each label
        n_unmatched number of mutation rows that hit no interface node (expected: SKEMPI
                    includes mutations away from the interface, which have no node)
    """
    import torch

    if strategy not in ("alanine", "max"):
        raise ValueError(f"strategy must be 'alanine' or 'max', got {strategy!r}")

    y = torch.full((n_nodes,), float("nan"), dtype=torch.float)
    y_ddg = torch.full((n_nodes,), float("nan"), dtype=torch.float)
    n_muts = torch.zeros((n_nodes,), dtype=torch.long)
    label_mask = torch.zeros((n_nodes,), dtype=torch.bool)

    subset = mutations if strategy == "max" else mutations[mutations["mut"] == "A"]

    n_unmatched = 0
    best: dict[int, float] = {}
    counts: dict[int, int] = {}
    for r in subset.itertuples():
        idx = index_of.get(node_key(r.chain, r.resseq, getattr(r, 'icode', ' ')))
        if idx is None:
            n_unmatched += 1          # mutation is not at the interface -> no node for it
            continue
        if residue_ids is not None and residue_ids[idx].one_letter != r.wt:
            n_unmatched += 1          # wrong residue at that position -> refuse the join
            continue
        counts[idx] = counts.get(idx, 0) + 1
        # Both strategies take the max: under "alanine" there is normally one Ala
        # substitution per position, but SKEMPI can hold repeat measurements of the same
        # mutation from different papers, and max keeps that deterministic.
        if idx not in best or r.ddg > best[idx]:
            best[idx] = float(r.ddg)

    for idx, ddg in best.items():
        y_ddg[idx] = ddg
        y[idx] = 1.0 if ddg >= ddg_threshold else 0.0
        label_mask[idx] = True
        n_muts[idx] = counts[idx]

    return y, label_mask, y_ddg, n_muts, n_unmatched


# ---------------------------------------------------------------------------------------
# One complex -> one Data
# ---------------------------------------------------------------------------------------
def complex_to_data(
    analysis: ComplexAnalysis,
    mutations: pd.DataFrame,
    strategy: str = "alanine",
    ddg_threshold: float = DDG_HOTSPOT_THRESHOLD,
    intra_side_cutoff: float = INTRA_SIDE_CUTOFF,
):
    """Build one PyG ``Data`` object for a complex's interface.

    Parameters
    ----------
    analysis  : feature-pipeline output for this complex.
    mutations : tidy SKEMPI rows for THIS complex (from :func:`prepare_mutation_frame`,
                filtered to one pdb_id). Pass an empty frame for an unlabeled graph.

    The returned Data carries, besides x/edge_index/edge_attr/y:
        label_mask     bool [n_nodes], True where y is defined (use this in the loss)
        y_ddg          float [n_nodes], collapsed ddG, nan where untested
        n_mutations    long [n_nodes], substitutions behind each label
        complex_group  pdb id — the grouping key for split-by-complex, same as dataset.py
        residue_labels list[str] like 'A/ARG87', for interpreting predictions
    """
    from torch_geometric.data import Data

    x, residue_ids, index_of = build_node_matrix(analysis)
    edge_index, edge_attr = build_edges(
        analysis, residue_ids, index_of, intra_side_cutoff=intra_side_cutoff
    )
    y, label_mask, y_ddg, n_muts, n_unmatched = build_labels(
        mutations, index_of, n_nodes=x.size(0),
        strategy=strategy, ddg_threshold=ddg_threshold,
        residue_ids=residue_ids,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.label_mask = label_mask
    data.y_ddg = y_ddg
    data.n_mutations = n_muts
    data.complex_group = analysis.source
    data.residue_labels = [rid.label for rid in residue_ids]
    data.side = [analysis.interface.residues[rid].side for rid in residue_ids]
    data.n_unmatched_mutations = n_unmatched
    data.node_feature_names = list(NODE_FEATURES)
    data.edge_attr_names = list(EDGE_ATTR_NAMES)
    return data


# ---------------------------------------------------------------------------------------
# Whole dataset
# ---------------------------------------------------------------------------------------
def build_graph_dataset(
    skempi_csv: str | Path,
    strategy: str = "alanine",
    ddg_threshold: float = DDG_HOTSPOT_THRESHOLD,
    intra_side_cutoff: float = INTRA_SIDE_CUTOFF,
    limit_complexes: int | None = None,
    require_labels: bool = True,
    verbose: bool = True,
    skip_complexes: set[str] | None = None,
    on_graph=None,
):
    """Build one PyG ``Data`` per SKEMPI complex.

    Mirrors ``dataset.build_dataset``: same source, same split-by-complex key, same
    single-mutation policy — just graph-shaped output. Structures are downloaded and
    analyzed once each, so the first full run costs the same as the tabular build.

    ``require_labels`` drops complexes where no SKEMPI mutation landed on an interface
    residue; such a graph contributes no supervised signal.

    ``skip_complexes`` are pdb ids to pass over — used to resume a build that died partway
    without redoing the complexes already on disk.

    ``on_graph(graph, n_kept)`` is called after each graph is appended, so a caller can
    checkpoint incrementally. A full build is long enough that losing it to a crash is a
    real cost, so the persistence policy lives with the caller rather than being hardcoded.

    Returns (list[Data], GraphBuildStats).
    """
    muts = prepare_mutation_frame(skempi_csv)
    stats = GraphBuildStats()
    graphs = []
    skip = skip_complexes or set()

    by_complex = muts.groupby("pdb_id", sort=False)
    for n_seen, (pdb_id, rows) in enumerate(by_complex, start=1):
        if limit_complexes and len(graphs) >= limit_complexes:
            break
        if pdb_id in skip:
            continue
        side_a, side_b = rows.iloc[0]["side_a"], rows.iloc[0]["side_b"]
        try:
            analysis = analyze_complex(pdb_id, chains=(side_a, side_b))
            data = complex_to_data(
                analysis, rows, strategy=strategy, ddg_threshold=ddg_threshold,
                intra_side_cutoff=intra_side_cutoff,
            )
        except Exception as exc:
            stats.complexes_failed += 1
            if verbose:
                print(f"[graph] skip {pdb_id}: {type(exc).__name__}: {exc}", flush=True)
            continue

        n_labeled = int(data.label_mask.sum())
        stats.mutations_unmatched += int(data.n_unmatched_mutations)
        if require_labels and n_labeled == 0:
            stats.complexes_unlabeled += 1
            continue

        stats.complexes_built += 1
        stats.nodes_total += int(data.num_nodes)
        stats.nodes_labeled += n_labeled
        stats.nodes_positive += int((data.y == 1.0).sum())
        graphs.append(data)
        if on_graph is not None:
            on_graph(data, len(graphs))

        if verbose and n_seen % 25 == 0:
            print(f"[graph] ...{n_seen} complexes seen, {len(graphs)} graphs kept",
                  flush=True)

    if verbose:
        print(f"[graph] {stats.summary()}", flush=True)
    return graphs, stats
