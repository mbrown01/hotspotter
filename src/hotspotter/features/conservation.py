"""Evolutionary conservation, from ConSurf-DB.

BIOLOGY NOTE — why this is worth a network call:
    Every other feature in this package is computed from one frozen wild-type structure.
    That turns out to be a real limit: burial, packing, contact counts and force-field
    energy terms all end up measuring the same thing from different angles, and adding
    more of them stops helping. Conservation is the first signal from outside that
    structure.

    It asks a different question: across the homologues of this protein in other species,
    how much does evolution let this position vary? A position that is identical from
    bacteria to humans has been under selection -- organisms that mutated it did not
    survive. A position that is leucine here, serine there and glutamate elsewhere is
    tolerant. Functional sites are conserved; incidental surface is not.

    Measured contribution: +0.0085 PR-AUC on top of the 25 features plus van der Waals,
    positive across five fold partitions.

MISSING VALUES ARE LEFT MISSING
    ConSurf-DB covers most but not all of the PDB; roughly 4% of chains have no entry, and
    the server can be down. Those residues get NaN, never a filled-in value. This matters
    more than it sounds: median-imputing the missing 3.7% of our training rows destroyed
    78% of the feature's benefit (+0.0019 imputed, +0.0085 with NaN preserved). XGBoost
    learns a default branch direction for missing values at every split, which is strictly
    better than any constant we could substitute. Downstream models that cannot accept NaN
    must handle it themselves.

NUMBERING IS NOT SHARED, SO WE ALIGN
    ConSurf-DB stores one profile per unique sequence, so a query usually redirects to a
    different structure that is 100% sequence identical -- 1BRS chain A's scores arrive
    numbered as 1A2P chain B. PDB numbering is chosen by the depositor, so the two
    disagree: different offsets, different resolved loops, and antibodies numbered by
    Kabat or Chothia with insertion codes. Joining on residue number would attach
    plausible-looking wrong scores to most residues and raise no error.

    So we align the sequences and additionally require the amino acid to match at every
    mapped position. A mis-alignment then shows up as missing data rather than as quiet
    corruption.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hotspotter.constants import THREE_TO_ONE
from hotspotter.interface import Interface
from hotspotter.io import DATA_RAW, ResidueId, is_amino_acid

CONSURF_BASE = "https://consurfdb.tau.ac.il"
#: Below this fraction of residues matching by amino acid, the alignment is not trusted
#: and the whole chain is left unscored.
MIN_SEQUENCE_MATCH = 0.80

CONSERVATION_COLUMNS = ("cons_score",)

#: One parsed row of a ConSurf grades table.
_ROW = re.compile(
    r"^\s*(\d+)\s+([A-Z-])\s+(\S+)\s+(-?[\d.]+)\s+(\d+)(\*?)\s"
)


def _cache_dir() -> Path:
    """Anchored to the repo, NOT the working directory.

    A relative path here means the 694 cached grade files are invisible to anything run
    from ``scripts/`` or a notebook, and with ``offline=True`` that failure is silent --
    every residue just comes back NaN.
    """
    d = DATA_RAW.parent / "consurf" / "grades"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(url: str, timeout: int = 45) -> str | None:
    """GET a URL, or None on any failure. Never raises."""
    from hotspotter.io import _ensure_trust

    try:
        import requests

        _ensure_trust()
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None


def _representative(pdb_id: str, chain: str) -> tuple[str, str] | None:
    """ConSurf-DB keeps one profile per unique sequence; find which entry holds ours."""
    raw = _fetch(f"{CONSURF_BASE}/get_identical_chain/"
                 f"?unique_pdb={pdb_id}&unique_chain={chain}")
    if not raw:
        return None
    try:
        j = json.loads(raw)
        return j["identical_pdb"], j["identical_chain"]
    except Exception:
        return None


def fetch_grades(pdb_id: str, chain: str) -> str | None:
    """Raw ConSurf grades text for one chain, cached on disk. None if unavailable."""
    pdb_id = pdb_id.upper()
    cached = _cache_dir() / f"{pdb_id}_{chain}.txt"
    if cached.exists() and cached.stat().st_size > 2000:
        return cached.read_text(encoding="utf-8", errors="replace")
    # A chain with no ConSurf entry costs up to two 45 s timeouts. Remember the miss so
    # the second run is instant; a stale miss is cleared by deleting the marker.
    missing = _cache_dir() / f"{pdb_id}_{chain}.missing"
    if missing.exists():
        return None

    rep = _representative(pdb_id, chain)
    if rep is None:
        missing.touch()
        return None
    rp, rc = rep
    text = _fetch(f"{CONSURF_BASE}/results/{rp}/{rc}/{rp}_{rc}_consurf_grades.txt")
    if not text or "POS" not in text or len(text) < 2000:
        missing.touch()
        return None
    cached.write_text(text, encoding="utf-8")
    return text


def parse_grades(text: str) -> tuple[str, list[float]]:
    """-> (sequence, per-position conservation score). Score is negative when conserved."""
    seq, scores = [], []
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        _pos, aa, _atom, score, _colour, _star = m.groups()
        seq.append(aa)
        scores.append(float(score))
    return "".join(seq), scores


def _align(our_seq: str, their_seq: str) -> dict[int, int]:
    """Global alignment -> {index in ours: index in theirs}."""
    try:
        from Bio import Align
    except Exception:  # pragma: no cover
        return {}
    aligner = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                    open_gap_score=-10, extend_gap_score=-0.5)
    try:
        alignment = aligner.align(our_seq, their_seq)[0]
    except Exception:
        return {}
    out = {}
    for (a0, a1), (b0, b1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for k in range(a1 - a0):
            out[a0 + k] = b0 + k
    return out


def _chain_sequence(model, chain_id: str) -> tuple[str, list]:
    """One-letter sequence and the matching residue ids, in chain order."""
    seq, ids = [], []
    for chain in model:
        if chain.id != chain_id:
            continue
        for residue in chain:
            if not is_amino_acid(residue):
                continue
            one = THREE_TO_ONE.get(residue.get_resname())
            if one:
                seq.append(one)
                ids.append((residue.id[1], residue.id[2]))
    return "".join(seq), ids


def compute_conservation_features(
    model, interface: Interface, pdb_id: str | None = None, offline: bool = False
) -> dict[ResidueId, dict]:
    """Per-residue ConSurf conservation score, or NaN where unavailable.

    ``pdb_id`` is needed to look the structure up; without it (a local file that is not a
    PDB entry) every residue is left unscored. ``offline=True`` uses only cached files.
    """
    feats: dict[ResidueId, dict] = {rid: {"cons_score": float("nan")}
                                    for rid in interface.residues}
    if not pdb_id or len(pdb_id) != 4:
        return feats

    for chain_id in {rid.chain for rid in interface.residues}:
        cached = _cache_dir() / f"{pdb_id.upper()}_{chain_id}.txt"
        if offline and not cached.exists():
            continue
        text = fetch_grades(pdb_id, chain_id)
        if not text:
            continue
        their_seq, scores = parse_grades(text)
        our_seq, ids = _chain_sequence(model, chain_id)
        if not our_seq or not their_seq:
            continue

        mapping = _align(our_seq, their_seq)
        # SAFEGUARD: only keep positions where the amino acid itself agrees.
        good = {i: j for i, j in mapping.items()
                if j < len(their_seq) and our_seq[i] == their_seq[j]}
        if len(good) / max(len(our_seq), 1) < MIN_SEQUENCE_MATCH:
            continue

        by_residue = {ids[i]: scores[good[i]] for i in good
                      if i < len(ids) and good[i] < len(scores)}
        for rid in interface.residues:
            if rid.chain != chain_id:
                continue
            # Exact key only. Falling back from 52A to 52 would attach a DIFFERENT
            # residue's score -- precisely the quiet corruption this module exists to
            # prevent. Both keys come from the same model, so the exact one is present
            # whenever the residue aligned at all.
            score = by_residue.get((rid.resseq, rid.icode))
            if score is not None:
                feats[rid] = {"cons_score": score}
    return feats
