"""Regression tests for the joins between SKEMPI, the structure, and the feature table.

Every bug pinned here was silent. Nothing raised, nothing logged; rows simply went
missing or carried the wrong numbers, and the only symptom was a score that looked
plausible. That is the whole reason these are tests rather than comments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotspotter.features.chemistry import backbone_donor_atoms
from hotspotter.ml.dataset import Mutation, parse_mutation
from hotspotter.ranking import hotspot_score
from hotspotter.report import _source_tag


# --- SKEMPI insertion codes -----------------------------------------------------------

def test_insertion_code_is_upper_cased():
    """SKEMPI writes the code lowercase, the PDB writes it uppercase.

    'DA60aA' is Asp, chain A, residue 60A -> Ala. Comparing 'a' to the structure's 'A'
    never matches, which dropped all 35 insertion-code mutations in the database --
    including the 1EAW 60a-60g series the residue lookup was written to handle.
    """
    m = parse_mutation("DA60aA")
    assert (m.wt, m.chain, m.resseq, m.mut) == ("D", "A", 60, "A")
    assert m.icode == "A", "insertion code must be normalised to the PDB's case"


def test_every_insertion_code_letter_normalises():
    for letter in "abcdefg":
        assert parse_mutation(f"RD100{letter}A").icode == letter.upper()


def test_a_plain_mutation_still_has_no_insertion_code():
    m = parse_mutation("TI17A")
    assert m.resseq == 17 and m.icode == " "


def test_mutation_matches_an_upper_cased_structure_residue():
    class _Rid:
        chain, resseq, icode, one_letter = "A", 60, "A", "D"

    assert Mutation(wt="D", chain="A", resseq=60, mut="A", icode="A").matches(_Rid())


# --- conservation in the heuristic ranking --------------------------------------------

def _ranking_frame(cons):
    return pd.DataFrame({
        "residue": [f"A/X{i}" for i in range(len(cons))],
        "dsasa": [50.0] * len(cons),
        "centrality": [0.5] * len(cons),
        "naive_score": [50.0] * len(cons),
        "cons_score": cons,
    })


def test_conservation_term_reads_the_column_the_module_emits():
    """The feature is `cons_score`; the ranking used to look for `conservation`, so the
    second-heaviest weight in the whole heuristic was always multiplied by zero."""
    out = hotspot_score(_ranking_frame([-1.5, 0.0, 1.5]))
    assert out["hotspot_score"].nunique() > 1, "conservation is not reaching the score"


def test_conserved_residues_rank_above_variable_ones():
    """ConSurf's scale runs BACKWARDS: conserved is negative. Without the negation the
    term rewards exactly the wrong residues."""
    out = hotspot_score(_ranking_frame([-1.5, 1.5]))
    top = out.iloc[0]
    assert top["cons_score"] == pytest.approx(-1.5), \
        "the conserved residue (negative score) must rank first"


def test_missing_conservation_does_not_crash_or_penalise():
    out = hotspot_score(_ranking_frame([-1.5, np.nan]))
    assert out["hotspot_score"].notna().all()


def test_conservation_reasoning_string_appears():
    out = hotspot_score(_ranking_frame([-1.5, 0.0, 0.0, 1.5]))
    assert any("evolutionarily conserved" in r for r in out["reasoning"])


# --- proline has no backbone N-H ------------------------------------------------------

def test_proline_donates_no_backbone_hydrogen_bond():
    """Proline's side chain rings back onto its own backbone nitrogen, leaving no
    hydrogen to donate -- the one residue of the twenty that cannot. Checked through
    the donor lookup rather than by reading the constant back to itself, so it fails if
    the set stops being consulted."""
    assert backbone_donor_atoms("ALA") == {"N"}
    assert backbone_donor_atoms("PRO") == set()


# --- output paths ---------------------------------------------------------------------

def test_source_tag_collapses_a_path_to_its_stem():
    """Keeping the path produced a tag like 'C:_x_y.pdb', and joining that onto an output
    directory on Windows reads as drive-qualified, so --out was silently discarded."""
    assert _source_tag(r"C:\data\raw\1brs.pdb") == "1brs"
    assert _source_tag("data/raw/1brs.pdb") == "1brs"


def test_source_tag_leaves_a_pdb_id_alone():
    assert _source_tag("1BRS") == "1BRS"


def test_source_tag_has_no_path_separators_or_drive_colon():
    for bad in ("/", "\\", ":"):
        assert bad not in _source_tag(r"C:\data\raw\1brs.pdb")


# --- one PDB, two chain pairings ------------------------------------------------------

_SKEMPI_HEADER = ("#Pdb;Mutation(s)_PDB;Affinity_wt_parsed;Affinity_mut_parsed;Temperature")


def _tiny_skempi(tmp_path):
    """Two rows for the same PDB under DIFFERENT pairings -- the 3SE4 case."""
    rows = [
        "3SE4_B_A;TB10A;1e-9;1e-7;298",
        "3SE4_B_C;TB10A;1e-9;1e-6;298",
    ]
    p = tmp_path / "skempi.csv"
    p.write_text("\n".join([_SKEMPI_HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def test_each_chain_pairing_is_analysed_separately(tmp_path, monkeypatch):
    """THE IMPORTANT ONE.

    SKEMPI lists 2C5D, 3SE3 and 3SE4 twice, with different binding partners. Caching the
    pipeline result on the PDB id alone meant the second pairing silently reused the
    first one's interface, so those rows carried burial and contact features describing
    binding to the wrong chain. Nothing raised; the numbers were simply about a different
    interaction.
    """
    import hotspotter.ml.dataset as ds

    calls = []

    class _Analysis:
        def __init__(self, partner):
            self.table = pd.DataFrame([{
                "chain": "B", "resseq": 10, "icode": "", "aa": "T",
                "dsasa": 100.0 if partner == ("A",) else 999.0,
            }])

    def _fake(pdb_id, chains=None, **kw):
        calls.append((pdb_id, chains))
        return _Analysis(chains[1])

    monkeypatch.setattr(ds, "analyze_complex", _fake)
    out = ds.build_dataset(_tiny_skempi(tmp_path))

    assert len(calls) == 2, f"expected one analysis per pairing, got {calls}"
    assert {c[1] for c in calls} == {(("B",), ("A",)), (("B",), ("C",))}
    assert sorted(out["dsasa"]) == [100.0, 999.0], \
        "both pairings must keep their own features, not share one cached interface"


def test_the_pairing_is_recorded_on_every_row(tmp_path, monkeypatch):
    """complex_group stays the PDB id so folds keep a structure together; pdb_tag keeps
    the pairing recoverable, which is what makes the bug above detectable at all."""
    import hotspotter.ml.dataset as ds

    class _Analysis:
        table = pd.DataFrame([{"chain": "B", "resseq": 10, "icode": "", "aa": "T"}])

    monkeypatch.setattr(ds, "analyze_complex", lambda *a, **k: _Analysis())
    out = ds.build_dataset(_tiny_skempi(tmp_path))

    assert set(out["complex_group"]) == {"3SE4"}
    assert set(out["pdb_tag"]) == {"3SE4_B_A", "3SE4_B_C"}


# --- side-chain plasticity ------------------------------------------------------------

class _BAtom:
    def __init__(self, name, b):
        self._name, self._b = name, b
    def get_name(self): return self._name
    def get_bfactor(self): return self._b


class _BRes:
    def __init__(self, atoms): self._atoms = atoms
    def get_atoms(self): return iter(self._atoms)


def test_sidechain_ratio_is_relative_to_its_own_backbone():
    """THE POINT OF THE FEATURE.

    Two residues with identical side-chain mobility RELATIVE to their backbone must
    score the same even when their absolute B-factors differ by a factor of ten --
    that per-structure scale is set by resolution and refinement, not by biology.
    """
    from hotspotter.features.confidence import _sidechain_bfactor_ratio
    low = _BRes([_BAtom("N", 10.0), _BAtom("CA", 10.0), _BAtom("CB", 20.0)])
    high = _BRes([_BAtom("N", 100.0), _BAtom("CA", 100.0), _BAtom("CB", 200.0)])
    assert _sidechain_bfactor_ratio(low)[1] == pytest.approx(2.0)
    assert _sidechain_bfactor_ratio(high)[1] == pytest.approx(2.0)


def test_a_rigid_sidechain_scores_below_one():
    from hotspotter.features.confidence import _sidechain_bfactor_ratio
    res = _BRes([_BAtom("N", 40.0), _BAtom("CA", 40.0), _BAtom("CB", 20.0)])
    assert _sidechain_bfactor_ratio(res)[1] < 1.0


def test_glycine_has_no_sidechain_so_the_ratio_is_nan():
    """Not 0.0, not 1.0 -- unknown. NULLABLE handles it from there."""
    from hotspotter.features.confidence import _sidechain_bfactor_ratio
    gly = _BRes([_BAtom("N", 30.0), _BAtom("CA", 30.0), _BAtom("C", 30.0)])
    assert _sidechain_bfactor_ratio(gly)[1] != _sidechain_bfactor_ratio(gly)[1]


def test_zero_backbone_bfactor_does_not_divide_by_zero():
    from hotspotter.features.confidence import _sidechain_bfactor_ratio
    res = _BRes([_BAtom("N", 0.0), _BAtom("CA", 0.0), _BAtom("CB", 5.0)])
    assert _sidechain_bfactor_ratio(res)[1] != _sidechain_bfactor_ratio(res)[1]


def test_non_xray_entries_get_nan_not_a_fabricated_constant():
    """An NMR file carries a flat placeholder in the B-factor column (1KBH uses 10.00).
    Reporting that as flexibility invents a measurement out of a file convention."""
    from hotspotter.features.confidence import _is_xray
    assert _is_xray("X-RAY DIFFRACTION")
    assert not _is_xray("SOLUTION NMR")
    assert not _is_xray("ELECTRON MICROSCOPY")


def test_plasticity_is_registered_and_nullable():
    from hotspotter.ml.features import NULLABLE, PLASTICITY, XGB_FEATURES
    assert "bfactor_sc_bb" in PLASTICITY
    assert "bfactor_sc_bb" in XGB_FEATURES
    assert "bfactor_sc_bb" in NULLABLE          # glycine, alanine, non-X-ray
    assert "bfactor_sc" not in XGB_FEATURES     # raw value measured null; ratio only


def test_a_header_with_no_method_still_reports_b_factors():
    """A PDB with no EXPDTA record parses as method "unknown", not None.

    Treating that as "not a crystal" silently blanked bfactor, bfactor_sc and
    bfactor_sc_bb for every residue -- three model inputs gone -- on exactly the files
    the tool is meant for: anything re-saved by PyMOL, PDBFixer or a docking program.
    """
    from hotspotter.features.confidence import _method_stated

    for unstated in (None, "", "   ", "unknown", "UNKNOWN", "N/A"):
        assert not _method_stated(unstated), f"{unstated!r} should mean 'not stated'"
    for stated in ("X-RAY DIFFRACTION", "SOLUTION NMR", "ELECTRON MICROSCOPY"):
        assert _method_stated(stated)


def test_insertion_codes_survive_the_residue_collapse():
    """Kabat numbering puts several distinct residues at one number (1EAW chain A has
    60, 60A, 60B, 60C). A key without the insertion code merges them and keeps whichever
    ddG is larger, so three residues inherit a fourth's label."""
    from hotspotter.ml.features import collapse

    df = pd.DataFrame({
        "pdb_id": ["1EAW"] * 4, "chain": ["A"] * 4, "resseq": [60] * 4,
        "icode": ["", "A", "B", "C"], "mut": ["A"] * 4,
        "ddg": [0.1, 3.0, 0.2, 0.3],
    })
    out = collapse(df, "alanine", 2.0).sort_values("icode")
    assert len(out) == 4, "the four residues at position 60 collapsed into fewer rows"
    assert out["label"].tolist() == [0, 1, 0, 0]
