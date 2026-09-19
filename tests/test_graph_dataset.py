"""Unit tests for the PyG graph builder.

Same philosophy as the other tests: build a tiny synthetic complex with atoms at KNOWN
distances, so every assertion is about logic we control rather than about a downloaded
structure. Nothing here touches the network.

The torch-dependent tests skip cleanly when the `gnn` extra isn't installed, matching how
the ml extras are optional elsewhere in the project.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hotspotter.features.chemistry import detect_contacts
from hotspotter.interface import detect_interface
from hotspotter.ml.graph_dataset import (
    EDGE_ATTR_NAMES,
    EDGE_TYPES,
    NODE_FEATURES,
    node_key,
    prepare_mutation_frame,
)
from hotspotter.pipeline import ComplexAnalysis

from conftest import build_structure


# ---------------------------------------------------------------------------------------
# A 4-residue synthetic complex: 2 residues per side, laid out so the edge set is known.
#
#   chain A            chain B
#   ARG 1  <-- salt bridge (2.8 A) -->  ASP 1
#     |  7 A CB-CB (same side)            |  7 A CB-CB (same side)
#   LEU 2  <-- hydrophobic (2.3 A) -->  ILE 2
#
#   The diagonal pairs must NOT form: closest approach is ARG1 NH2 -> ILE2 CD1 at 6.50 A
#   and LEU2 CD1 -> ASP1 OD2 at 6.36 A, both clear of the 5 A heavy-atom cutoff. (Watch
#   the off-axis atoms here: at a 5 A row separation those two pairs fall to 4.72 and
#   4.60 A and the diagonals silently become contacts.)
#
#   Expected: 2 cross-interface edges + 2 same-side edges = 4 undirected -> 8 directed.
# ---------------------------------------------------------------------------------------
@pytest.fixture
def mini_complex():
    model = build_structure([
        ("A", "ARG", 1, {"CA": (0.0, 0.0, 0.0), "CB": (1.5, 0.0, 0.0),
                         "NE": (3.0, 0.0, 0.0), "NH1": (4.0, 0.0, 0.0),
                         "NH2": (4.0, 1.0, 0.0)}),
        ("A", "LEU", 2, {"CA": (0.0, 7.0, 0.0), "CB": (1.5, 7.0, 0.0),
                         "CD1": (4.2, 7.0, 0.0)}),
        ("B", "ASP", 1, {"CA": (10.0, 0.0, 0.0), "CB": (8.5, 0.0, 0.0),
                         "OD1": (6.8, 0.0, 0.0), "OD2": (6.8, 1.2, 0.0)}),
        ("B", "ILE", 2, {"CA": (10.0, 7.0, 0.0), "CB": (8.5, 7.0, 0.0),
                         "CD1": (6.5, 7.0, 0.0)}),
    ])
    interface = detect_interface(model, ("A",), ("B",))
    contacts = detect_contacts(interface)

    # Build a pipeline-shaped table from the interface, one row per residue, with every
    # NODE_FEATURES column present. Values are distinct per row so column/row ordering
    # errors show up as wrong numbers rather than passing silently.
    rows = []
    for n, (rid, ir) in enumerate(interface.residues.items()):
        row = {"residue": rid.label, "chain": rid.chain, "resseq": rid.resseq,
               "icode": rid.icode.strip(), "resname": rid.resname, "side": ir.side}
        for k, col in enumerate(NODE_FEATURES):
            row[col] = float(n * 100 + k)
        rows.append(row)
    table = pd.DataFrame(rows)

    return ComplexAnalysis(
        table=table, contacts=contacts, interface=interface, structure=model,
        source="MINI", side_a_chains=("A",), side_b_chains=("B",),
    )


def _mutation_frame(records):
    """Build the tidy frame shape that build_labels consumes."""
    return pd.DataFrame(records, columns=["pdb_id", "chain", "resseq", "icode",
                                          "wt", "mut", "ddg", "mutation"])


# ---------------------------------------------------------------------------------------
# Layout constants (no torch needed)
# ---------------------------------------------------------------------------------------
def test_feature_name_lists_are_well_formed():
    assert len(NODE_FEATURES) == len(set(NODE_FEATURES)), "duplicate node feature"
    assert len(EDGE_ATTR_NAMES) == len(EDGE_TYPES) + 2
    assert EDGE_ATTR_NAMES[-1] == "is_cross_interface"
    # Heuristic ranking outputs must never become model inputs (they'd leak the heuristic).
    for leaky in ("hotspot_score", "hotspot_rank", "naive_score", "naive_rank"):
        assert leaky not in NODE_FEATURES


def test_prepare_mutation_frame_parses_and_filters():
    raw = pd.DataFrame({
        "#Pdb": ["1ABC_A_B", "1ABC_A_B", "1ABC_A_B"],
        "Mutation(s)_PDB": ["RA95A", "TB17G,SB19G", "YA96A"],  # middle one is multi
        "Affinity_wt_parsed": [1e-9, 1e-9, 1e-9],
        "Affinity_mut_parsed": [1e-6, 1e-6, 1e-9],
        "Temperature": [298, 298, "298(assumed)"],
    })
    out = prepare_mutation_frame(raw)
    assert len(out) == 2, "multi-mutation row should be dropped"
    assert set(out["mutation"]) == {"RA95A", "YA96A"}
    first = out[out["mutation"] == "RA95A"].iloc[0]
    assert (first["chain"], first["resseq"], first["wt"], first["mut"]) == ("A", 95, "R", "A")
    assert first["ddg"] > 0     # weaker binding -> positive ddG
    # the junk temperature string must not crash parsing
    assert abs(out[out["mutation"] == "YA96A"].iloc[0]["ddg"]) < 1e-9


def test_prepare_mutation_frame_drops_rows_without_affinity():
    raw = pd.DataFrame({
        "#Pdb": ["1ABC_A_B"],
        "Mutation(s)_PDB": ["RA95A"],
        "Affinity_wt_parsed": [None],
        "Affinity_mut_parsed": [1e-6],
        "Temperature": [298],
    })
    assert prepare_mutation_frame(raw).empty


# ---------------------------------------------------------------------------------------
# Node matrix
# ---------------------------------------------------------------------------------------
def test_node_matrix_shape_and_row_alignment(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_node_matrix

    x, residue_ids, index_of = build_node_matrix(mini_complex)
    assert x.shape == (4, len(NODE_FEATURES))
    assert len(residue_ids) == 4
    # Row n was filled with n*100 + k, so row order and column order are both checkable.
    for n in range(4):
        assert x[n, 0].item() == pytest.approx(n * 100)
        assert x[n, 3].item() == pytest.approx(n * 100 + 3)
    # index_of must point back at the right rows
    for i, rid in enumerate(residue_ids):
        assert index_of[node_key(rid.chain, rid.resseq, rid.icode)] == i


def test_node_matrix_rejects_missing_feature_column(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_node_matrix

    mini_complex.table = mini_complex.table.drop(columns=["dsasa"])
    with pytest.raises(KeyError, match="dsasa"):
        build_node_matrix(mini_complex)


def test_node_matrix_fills_nan(mini_complex):
    pytest.importorskip("torch")
    import torch
    from hotspotter.ml.graph_dataset import build_node_matrix

    mini_complex.table.loc[0, "dsasa"] = float("nan")
    x, _, _ = build_node_matrix(mini_complex)
    assert not torch.isnan(x).any(), "NaN in node features would poison training"


# ---------------------------------------------------------------------------------------
# Edges — the part the bipartite blocker was about
# ---------------------------------------------------------------------------------------
def test_edges_include_both_cross_and_same_side(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_edges, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(mini_complex)
    edge_index, edge_attr = build_edges(mini_complex, residue_ids, index_of)

    assert edge_index.shape[0] == 2
    assert edge_index.size(1) == edge_attr.size(0)
    is_cross = edge_attr[:, EDGE_ATTR_NAMES.index("is_cross_interface")]
    n_cross, n_same = int(is_cross.sum()), int((is_cross == 0).sum())
    # 2 cross + 2 same-side undirected pairs, each stored both ways
    assert n_cross == 4, f"expected 4 directed cross edges, got {n_cross}"
    assert n_same == 4, f"expected 4 directed same-side edges, got {n_same}"


def test_graph_is_not_bipartite(mini_complex):
    """Regression guard: contact_partners alone gives a bipartite graph. Same-side edges
    must exist, or a GNN can never see a residue's own-side neighborhood."""
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_edges, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(mini_complex)
    edge_index, _ = build_edges(mini_complex, residue_ids, index_of)
    side = {index_of[node_key(r.chain, r.resseq, r.icode)]: mini_complex.interface.residues[r].side
            for r in residue_ids}
    same_side = [1 for a, b in zip(*edge_index.tolist()) if side[a] == side[b]]
    assert same_side, "no same-side edges: the graph is still bipartite"


def test_edges_are_undirected(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_edges, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(mini_complex)
    edge_index, _ = build_edges(mini_complex, residue_ids, index_of)
    pairs = set(zip(*edge_index.tolist()))
    for a, b in pairs:
        assert (b, a) in pairs, f"edge {a}->{b} has no reverse"


def test_edge_attr_marks_salt_bridge_on_the_right_pair(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_edges, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(mini_complex)
    edge_index, edge_attr = build_edges(mini_complex, residue_ids, index_of)

    sb_col = EDGE_ATTR_NAMES.index("salt_bridge")
    arg = index_of[node_key("A", 1)]
    asp = index_of[node_key("B", 1)]
    flagged = {
        tuple(sorted((a, b)))
        for (a, b), v in zip(zip(*edge_index.tolist()), edge_attr[:, sb_col].tolist())
        if v == 1.0
    }
    assert flagged == {tuple(sorted((arg, asp)))}, \
        "salt bridge should be flagged on exactly the ARG-ASP edge"


def test_intra_side_cutoff_controls_same_side_edges(mini_complex):
    """The same-side CB-CB cutoff is a knob; shrinking it below 5 A drops those edges."""
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_edges, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(mini_complex)
    _, wide = build_edges(mini_complex, residue_ids, index_of, intra_side_cutoff=8.0)
    _, tight = build_edges(mini_complex, residue_ids, index_of, intra_side_cutoff=2.0)
    col = EDGE_ATTR_NAMES.index("is_cross_interface")
    assert int((wide[:, col] == 0).sum()) == 4
    assert int((tight[:, col] == 0).sum()) == 0, "2 A cutoff should admit no same-side edge"


# ---------------------------------------------------------------------------------------
# Labels — the collapse problem
# ---------------------------------------------------------------------------------------
def test_alanine_strategy_ignores_non_alanine_substitutions(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, _, index_of = build_node_matrix(mini_complex)
    muts = _mutation_frame([
        ("MINI", "A", 1, " ", "R", "A", 5.0, "RA1A"),   # alanine, disruptive
        ("MINI", "B", 1, " ", "D", "G", 9.0, "DB1G"),   # NOT alanine -> ignored
    ])
    y, mask, y_ddg, n_muts, unmatched = build_labels(muts, index_of, n_nodes=4,
                                                     strategy="alanine")
    arg, asp = index_of[node_key("A", 1)], index_of[node_key("B", 1)]
    assert bool(mask[arg]) and y[arg].item() == 1.0
    assert not bool(mask[asp]), "non-alanine substitution must not label a node"
    assert unmatched == 0


def test_max_strategy_takes_the_largest_ddg(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, _, index_of = build_node_matrix(mini_complex)
    muts = _mutation_frame([
        ("MINI", "A", 1, " ", "R", "G", 0.1, "RA1G"),
        ("MINI", "A", 1, " ", "R", "D", 4.4, "RA1D"),   # the max
        ("MINI", "A", 1, " ", "R", "S", 1.0, "RA1S"),
    ])
    y, mask, y_ddg, n_muts, _ = build_labels(muts, index_of, n_nodes=4, strategy="max")
    arg = index_of[node_key("A", 1)]
    assert y_ddg[arg].item() == pytest.approx(4.4)
    assert y[arg].item() == 1.0
    assert int(n_muts[arg]) == 3, "should record how many substitutions backed the label"


def test_threshold_is_inclusive(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, _, index_of = build_node_matrix(mini_complex)
    at = _mutation_frame([("MINI", "A", 1, " ", "R", "A", 2.0, "RA1A")])
    just_under = _mutation_frame([("MINI", "A", 1, " ", "R", "A", 1.999, "RA1A")])
    arg = index_of[node_key("A", 1)]
    assert build_labels(at, index_of, 4)[0][arg].item() == 1.0
    assert build_labels(just_under, index_of, 4)[0][arg].item() == 0.0


def test_untested_nodes_are_nan_and_masked_out(mini_complex):
    pytest.importorskip("torch")
    import torch
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, _, index_of = build_node_matrix(mini_complex)
    muts = _mutation_frame([("MINI", "A", 1, " ", "R", "A", 5.0, "RA1A")])
    y, mask, y_ddg, _, _ = build_labels(muts, index_of, n_nodes=4, strategy="alanine")
    assert int(mask.sum()) == 1
    untested = ~mask
    assert torch.isnan(y[untested]).all(), "untested nodes must be nan, not 0"
    assert torch.isnan(y_ddg[untested]).all()


def test_off_interface_mutation_is_counted_not_dropped_silently(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, _, index_of = build_node_matrix(mini_complex)
    muts = _mutation_frame([("MINI", "A", 999, " ", "K", "A", 5.0, "KA999A")])
    y, mask, _, _, unmatched = build_labels(muts, index_of, n_nodes=4, strategy="alanine")
    assert int(mask.sum()) == 0
    assert unmatched == 1, "a mutation off the interface must be counted"


def test_unknown_strategy_raises(mini_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, _, index_of = build_node_matrix(mini_complex)
    with pytest.raises(ValueError, match="strategy"):
        build_labels(_mutation_frame([]), index_of, 4, strategy="median")


# ---------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------
def test_complex_to_data_produces_a_usable_graph(mini_complex):
    pytest.importorskip("torch_geometric")
    from hotspotter.ml.graph_dataset import complex_to_data

    muts = _mutation_frame([("MINI", "A", 1, " ", "R", "A", 5.0, "RA1A")])
    data = complex_to_data(mini_complex, muts, strategy="alanine")

    assert data.num_nodes == 4
    assert data.x.shape == (4, len(NODE_FEATURES))
    assert data.edge_attr.shape == (data.edge_index.size(1), len(EDGE_ATTR_NAMES))
    assert data.complex_group == "MINI"
    assert len(data.residue_labels) == 4
    assert int(data.label_mask.sum()) == 1
    # PyG's own structural validation
    data.validate(raise_on_error=True)


def test_empty_mutation_frame_gives_an_unlabeled_graph(mini_complex):
    pytest.importorskip("torch_geometric")
    from hotspotter.ml.graph_dataset import complex_to_data

    data = complex_to_data(mini_complex, _mutation_frame([]), strategy="alanine")
    assert int(data.label_mask.sum()) == 0
    assert data.num_nodes == 4, "an unlabeled complex should still produce a graph"


# ---------------------------------------------------------------------------------------
# Insertion codes — regression tests for a real mis-join found on SKEMPI.
#
# Kabat-numbered antibodies and proteases put several DISTINCT residues at one number
# (1EAW chain A has 60, 60a, 60b, 60c, 60e, 60f, 60g). Keying nodes on (chain, resseq)
# collapsed them onto one node and joined every mutation there, silently mislabeling rows.
# ---------------------------------------------------------------------------------------
@pytest.fixture
def icode_complex():
    """Chain A carries residues 60 (ASP) and 60A (PHE) — same number, different residues."""
    model = build_structure([
        ("A", "ASP", 60, {"CA": (0.0, 0.0, 0.0), "CB": (1.5, 0.0, 0.0),
                          "OD1": (4.0, 0.0, 0.0), "OD2": (4.0, 1.0, 0.0)}),
        ("B", "ARG", 1, {"CA": (10.0, 0.0, 0.0), "CB": (8.5, 0.0, 0.0),
                         "NE": (6.8, 0.0, 0.0), "NH1": (6.8, 1.2, 0.0)}),
    ])
    # Insert residue 60A into chain A, positioned to also contact chain B.
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    import numpy as np

    res = Residue((" ", 60, "A"), "PHE", " ")
    for n, (name, xyz) in enumerate({"CA": (0.0, 4.0, 0.0), "CB": (1.5, 4.0, 0.0),
                                     "CZ": (4.2, 3.0, 0.0)}.items(), start=1):
        res.add(Atom(name, np.array(xyz, dtype=float), 20.0, 1.0, " ", name, n, name[0]))
    model["A"].add(res)

    interface = detect_interface(model, ("A",), ("B",))
    contacts = detect_contacts(interface)
    rows = []
    for n, (rid, ir) in enumerate(interface.residues.items()):
        row = {"residue": rid.label, "chain": rid.chain, "resseq": rid.resseq,
               "icode": rid.icode.strip(), "resname": rid.resname, "side": ir.side}
        for k, col in enumerate(NODE_FEATURES):
            row[col] = float(n * 100 + k)
        rows.append(row)
    return ComplexAnalysis(
        table=pd.DataFrame(rows), contacts=contacts, interface=interface, structure=model,
        source="ICODE", side_a_chains=("A",), side_b_chains=("B",),
    )


def test_insertion_code_residues_get_separate_nodes(icode_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_node_matrix

    _, residue_ids, index_of = build_node_matrix(icode_complex)
    keys = set(index_of)
    assert node_key("A", 60, "") in keys, "residue 60 missing"
    assert node_key("A", 60, "A") in keys, "residue 60A missing — icode was dropped"
    assert index_of[node_key("A", 60, "")] != index_of[node_key("A", 60, "A")], \
        "60 and 60A collapsed onto the same node"


def test_mutation_labels_land_on_the_icode_residue_not_its_neighbour(icode_complex):
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(icode_complex)
    # 'FA60aA' = Phe, chain A, residue 60 insertion-code a, to Ala.
    muts = _mutation_frame([("ICODE", "A", 60, "A", "F", "A", 6.0, "FA60aA")])
    y, mask, y_ddg, _, unmatched = build_labels(
        muts, index_of, n_nodes=len(residue_ids), strategy="alanine",
        residue_ids=residue_ids,
    )
    at_60a = index_of[node_key("A", 60, "A")]
    at_60 = index_of[node_key("A", 60, "")]
    assert bool(mask[at_60a]) and y[at_60a].item() == 1.0
    assert not bool(mask[at_60]), "label leaked onto residue 60 instead of 60A"
    assert unmatched == 0


def test_wildtype_letter_mismatch_is_refused_not_mislabelled(icode_complex):
    """A wt letter that disagrees means SKEMPI meant a different residue. Refuse the join —
    a wrong label is worse than a missing one."""
    pytest.importorskip("torch")
    from hotspotter.ml.graph_dataset import build_labels, build_node_matrix

    _, residue_ids, index_of = build_node_matrix(icode_complex)
    # Claims Ile at A60, but A60 is Asp. This is the 1EAW 'IA60A' case.
    muts = _mutation_frame([("ICODE", "A", 60, " ", "I", "A", 6.0, "IA60A")])
    y, mask, _, _, unmatched = build_labels(
        muts, index_of, n_nodes=len(residue_ids), strategy="alanine",
        residue_ids=residue_ids,
    )
    assert int(mask.sum()) == 0, "joined a mutation onto the wrong residue"
    assert unmatched == 1
