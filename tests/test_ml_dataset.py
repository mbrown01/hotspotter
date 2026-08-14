"""Unit tests for the pure Phase-2 helpers (no SKEMPI file, no xgboost needed)."""

from __future__ import annotations

import math

from hotspotter.io import ResidueId
from hotspotter.ml.dataset import (
    ddg_from_kd,
    parse_mutation,
    parse_pdb_field,
)


def test_parse_mutation_simple():
    m = parse_mutation("TI17A")   # Thr, chain I, residue 17, to Ala
    assert (m.wt, m.chain, m.resseq, m.mut) == ("T", "I", 17, "A")
    assert m.icode.strip() == ""


def test_parse_mutation_with_insertion_code():
    m = parse_mutation("YI96AG")  # ...96 with insertion code 'A', to Gly
    assert (m.wt, m.chain, m.resseq, m.mut, m.icode) == ("Y", "I", 96, "G", "A")


def test_mutation_matches_residue_id():
    m = parse_mutation("RA95A")   # Arg, chain A, 95, to Ala  (the Arg95 case!)
    rid = ResidueId(chain="A", resseq=95, icode=" ", resname="ARG")
    assert m.matches(rid)
    # wrong wild-type letter should not match (guards against numbering mismatch)
    rid_wrong = ResidueId(chain="A", resseq=95, icode=" ", resname="TYR")
    assert not m.matches(rid_wrong)


def test_parse_pdb_field_single_and_multi_chain():
    assert parse_pdb_field("1CSE_E_I") == ("1CSE", ("E",), ("I",))
    assert parse_pdb_field("3BDY_HL_A") == ("3BDY", ("H", "L"), ("A",))


def test_ddg_sign_convention():
    # Mutant binds WEAKER (higher Kd) -> positive ΔΔG -> residue mattered.
    assert ddg_from_kd(kd_wt=1e-9, kd_mut=1e-6, temperature_k=298) > 0
    # Mutant binds tighter -> negative ΔΔG.
    assert ddg_from_kd(kd_wt=1e-6, kd_mut=1e-9, temperature_k=298) < 0
    # No change -> ~0.
    assert abs(ddg_from_kd(1e-9, 1e-9, 298)) < 1e-9


def test_ddg_missing_kd_is_nan():
    assert math.isnan(ddg_from_kd(None, 1e-9))
    assert math.isnan(ddg_from_kd(0.0, 1e-9))


def test_ddg_magnitude_reasonable():
    # A 1000-fold Kd increase at 298 K is ~4.09 kcal/mol.
    val = ddg_from_kd(1e-9, 1e-6, 298)
    assert 3.9 < val < 4.3
