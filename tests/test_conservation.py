"""Tests for the ConSurf conservation feature.

All offline. The module's job is to attach an evolutionary conservation score to each
interface residue, and the thing that must never happen is attaching the WRONG one: the
scores arrive in a different structure's residue numbering, so a careless join produces
plausible-looking garbage and raises nothing. These tests pin the safeguard that stops
that, plus the parser and the failure modes.
"""

from __future__ import annotations

import pytest

from hotspotter.features.conservation import (
    MIN_SEQUENCE_MATCH,
    _align,
    compute_conservation_features,
    parse_grades,
)

# A few real rows from a ConSurf-DB grades file, header included so the parser has to
# skip it the way it does in production.
GRADES = """\t Amino Acid Conservation Scores
\t=======================================

- POS: The position of the acid in the sequence.
- SCORE: The normalized conservation scores.

POS\tSEQ\t     ATOM\t SCORE\tCOLOR\tCONFIDENCE INTERVAL\tB/E\tF/S\tMSA DATA\tRESIDUE VARIETY
  1\t  A\t  ALA:1:B\t-0.282\t   6*\t-1.190,  0.373  8,4\t  e\t   \t    1/16\tA
  2\t  G\t  GLY:2:B\t-0.744\t   7*\t-1.488, -0.332  9,6\t  e\t   \t    4/16\tG
  3\t  V\t  VAL:3:B\t 0.443\t   4*\t-0.512,  1.213  6,3\t  e\t   \t    4/16\tT 50%, V 25%
  4\t  M\t  MET:4:B\t-0.767\t   7 \t-1.262, -0.425  9,6\t  e\t   \t    8/16\tM 75%, F 12%
"""


def test_parse_grades_reads_sequence_and_scores():
    seq, scores = parse_grades(GRADES)
    assert seq == "AGVM"
    assert scores == pytest.approx([-0.282, -0.744, 0.443, -0.767])


def test_parse_grades_ignores_the_header():
    """The header mentions POS and SCORE; only the table rows are data."""
    seq, scores = parse_grades(GRADES)
    assert len(seq) == len(scores) == 4


def test_parse_grades_tolerates_junk():
    assert parse_grades("") == ("", [])
    assert parse_grades("nothing parseable here\n") == ("", [])


def test_align_maps_identical_sequences_one_to_one():
    m = _align("ACDEF", "ACDEF")
    assert m == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_align_handles_an_offset():
    """The stored structure often starts a few residues earlier than ours."""
    m = _align("CDEF", "ACDEF")
    assert [m[i] for i in sorted(m)] == [1, 2, 3, 4]


def test_align_handles_a_gap():
    """Unresolved loops mean our chain can be missing residues theirs has."""
    m = _align("ACDKLM", "ACDEFGKLM")
    assert m[0] == 0 and m[2] == 2          # ACD lines up at the start
    assert m[3] == 6                        # KLM lines up after their EFG


class _FakeResidue:
    def __init__(self, name, num, icode=" "):
        self.id = (" ", num, icode)
        self._name = name

    def get_id(self):
        return self.id

    def get_resname(self):
        return self._name

    def __iter__(self):
        return iter(())


class _FakeChain:
    def __init__(self, cid, residues):
        self.id = cid
        self._residues = residues

    def __iter__(self):
        return iter(self._residues)


class _FakeModel:
    def __init__(self, chains):
        self._chains = chains

    def __iter__(self):
        return iter(self._chains)


class _FakeInterface:
    def __init__(self, residues):
        self.residues = residues


class _Rid:
    """Minimal stand-in for ResidueId: hashable, with the attributes used."""

    def __init__(self, chain, resseq, icode=" "):
        self.chain, self.resseq, self.icode = chain, resseq, icode

    def __hash__(self):
        return hash((self.chain, self.resseq, self.icode))

    def __eq__(self, other):
        return (self.chain, self.resseq, self.icode) == \
               (other.chain, other.resseq, other.icode)


def test_no_pdb_id_gives_nan_and_does_not_crash():
    """A local structure file is not a PDB entry, so there is nothing to look up."""
    rid = _Rid("A", 1)
    feats = compute_conservation_features(_FakeModel([]), _FakeInterface({rid: None}),
                                          pdb_id=None)
    assert feats[rid]["cons_score"] != feats[rid]["cons_score"]      # NaN


def test_bad_pdb_id_gives_nan():
    rid = _Rid("A", 1)
    feats = compute_conservation_features(_FakeModel([]), _FakeInterface({rid: None}),
                                          pdb_id="not-a-pdb-id")
    assert feats[rid]["cons_score"] != feats[rid]["cons_score"]


def test_offline_with_no_cache_gives_nan_without_network(tmp_path, monkeypatch):
    """offline=True must never reach the network, even for a real PDB id."""
    import hotspotter.features.conservation as cons

    monkeypatch.setattr(cons, "_cache_dir", lambda: tmp_path)

    def _boom(*a, **k):
        raise AssertionError("offline mode attempted a network call")

    monkeypatch.setattr(cons, "_fetch", _boom)
    rid = _Rid("A", 1)
    feats = cons.compute_conservation_features(
        _FakeModel([]), _FakeInterface({rid: None}), pdb_id="1BRS", offline=True)
    assert feats[rid]["cons_score"] != feats[rid]["cons_score"]


def test_mismatched_sequence_is_refused(tmp_path, monkeypatch):
    """THE IMPORTANT ONE.

    Conservation scores arrive in a different structure's numbering. If the sequences do
    not actually correspond, the join must produce NOTHING rather than attach wrong
    scores -- a silent mis-join is the failure mode this whole design exists to prevent.
    """
    import hotspotter.features.conservation as cons

    monkeypatch.setattr(cons, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cons, "fetch_grades", lambda pdb, chain: GRADES)  # seq "AGVM"

    # Our chain is WWWWWW -- nothing in common with the grades file's AGVM.
    residues = [_FakeResidue("TRP", i) for i in range(1, 7)]
    model = _FakeModel([_FakeChain("A", residues)])
    rids = {_Rid("A", i): None for i in range(1, 7)}

    feats = cons.compute_conservation_features(model, _FakeInterface(rids), pdb_id="1BRS")
    assert all(v["cons_score"] != v["cons_score"] for v in feats.values()), \
        "a non-matching sequence must yield NaN, never a mapped score"


def test_matching_sequence_is_accepted(tmp_path, monkeypatch):
    """The converse: when the sequences agree, scores land on the right residues."""
    import hotspotter.features.conservation as cons

    monkeypatch.setattr(cons, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cons, "fetch_grades", lambda pdb, chain: GRADES)

    residues = [_FakeResidue(n, i) for i, n in
                enumerate(["ALA", "GLY", "VAL", "MET"], start=1)]
    model = _FakeModel([_FakeChain("A", residues)])
    rids = {_Rid("A", i): None for i in range(1, 5)}

    feats = cons.compute_conservation_features(model, _FakeInterface(rids), pdb_id="1BRS")
    assert feats[_Rid("A", 1)]["cons_score"] == pytest.approx(-0.282)
    assert feats[_Rid("A", 4)]["cons_score"] == pytest.approx(-0.767)


def test_match_threshold_is_strict_enough_to_matter():
    assert 0.5 < MIN_SEQUENCE_MATCH <= 1.0
