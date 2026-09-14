"""Categorize SKEMPI rows that build_dataset drops as 'residue_not_found'.

WHY THIS EXISTS
    build_dataset reports one lumped `residue_not_found` count, but that number mixes two
    completely different things:

      * mutations at positions that are simply NOT AT THE INTERFACE. The Phase-1 pipeline
        only emits rows for interface residues, so these correctly have no feature row.
        Dropping them is right, and they are not recoverable data.
      * mutations we FAILED TO LOCATE because of a chain-id, numbering, or insertion-code
        mismatch between SKEMPI and the PDB file. These are lost training data and worth
        rescuing.

    Treating the lump as one number either understates a real bug or overstates a
    non-problem. This script splits them.

HOW
    `_find_residue_row` fails exactly when (chain, resseq) is missing from the interface
    table. So the question "was this a real failure?" reduces to "does that residue exist
    in the structure at all?" — which only needs the parsed structure, not the full
    pipeline. That makes this run in a couple of minutes instead of ~20.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\diagnose_unmatched.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hotspotter.io import (  # noqa: E402
    fetch_pdb, get_chain_ids, get_model, is_amino_acid, load_structure,
)
from hotspotter.ml.dataset import (  # noqa: E402
    SKEMPI_COLUMNS, _clean_temperature, _to_float, ddg_from_kd, load_skempi,
    parse_mutation, parse_pdb_field,
)
from hotspotter.constants import THREE_TO_ONE  # noqa: E402

CSV = REPO_ROOT / "data" / "skempi_v2.csv"


def structure_index(pdb_id: str):
    """Return maps describing which residues actually exist in the structure.

    by_chain_resseq : (chain, resseq) -> set of icodes present
    by_chain        : chain -> set of resseq
    aa_at           : (chain, resseq, icode) -> one-letter aa
    """
    path = fetch_pdb(pdb_id)
    structure = load_structure(path, structure_id=pdb_id)
    model = get_model(structure)
    by_chain_resseq = defaultdict(set)
    by_chain = defaultdict(set)
    aa_at = {}
    for chain in model.get_chains():
        for res in chain.get_residues():
            if not is_amino_acid(res):
                continue
            _, resseq, icode = res.get_id()
            by_chain_resseq[(chain.id, resseq)].add(icode)
            by_chain[chain.id].add(resseq)
            aa_at[(chain.id, resseq, icode)] = THREE_TO_ONE.get(res.get_resname(), "X")
    return by_chain_resseq, by_chain, aa_at, set(get_chain_ids(structure))


def main() -> int:
    df = load_skempi(CSV)
    c = SKEMPI_COLUMNS

    # Which (pdb, mutation) pairs actually made it into the built feature table? Matching is
    # deterministic per pair, so a pair with zero rows in the CSV is one that failed.
    built = REPO_ROOT / "data" / "skempi_features_full.csv"
    if not built.exists():
        print(f"ERROR: need {built} first (run run_ml_baseline.py).", file=sys.stderr)
        return 2
    import pandas as pd
    bdf = pd.read_csv(built)
    matched_keys = set(zip(bdf["complex_group"].astype(str), bdf["mutation"].astype(str)))
    print(f"Built table: {len(bdf)} rows, {len(matched_keys)} distinct (pdb, mutation) pairs.")

    cats = Counter()
    examples = defaultdict(list)
    cache = {}

    for _, r in df.iterrows():
        mut_field = str(r[c["mutation"]])
        if "," in mut_field:
            continue  # multi-mutation: excluded by design, not a failure
        try:
            pdb_id, side_a, side_b = parse_pdb_field(str(r[c["pdb_field"]]))
        except ValueError:
            continue

        ddg = ddg_from_kd(
            _to_float(r[c["kd_wt"]]), _to_float(r[c["kd_mut"]]),
            _clean_temperature(r[c["temperature"]]),
        )
        if math.isnan(ddg):
            continue  # counted separately as no_kd

        # ONLY analyze rows that failed to match. Rows already in the built table are, by
        # definition, interface residues we located fine.
        if (pdb_id, mut_field) in matched_keys:
            continue

        if pdb_id not in cache:
            try:
                cache[pdb_id] = structure_index(pdb_id)
            except Exception as exc:
                cache[pdb_id] = None
                print(f"[diag] cannot load {pdb_id}: {exc}", flush=True)
            if len(cache) % 50 == 0:
                print(f"[diag] ...{len(cache)} structures indexed", flush=True)
        idx = cache[pdb_id]
        if idx is None:
            cats["structure_unavailable"] += 1
            continue
        by_chain_resseq, by_chain, aa_at, chain_ids = idx

        try:
            m = parse_mutation(mut_field)
        except Exception:
            cats["mutation_unparseable"] += 1
            examples["mutation_unparseable"].append(f"{pdb_id}:{mut_field}")
            continue

        # --- categorize -------------------------------------------------------------
        if m.chain not in chain_ids:
            cats["chain_absent"] += 1
            examples["chain_absent"].append(
                f"{pdb_id} wants chain {m.chain!r}, has {sorted(chain_ids)}")
            continue

        icodes = by_chain_resseq.get((m.chain, m.resseq))
        if icodes is None:
            cats["resseq_absent"] += 1
            near = sorted(by_chain[m.chain])
            span = f"{near[0]}..{near[-1]}" if near else "empty"
            examples["resseq_absent"].append(
                f"{pdb_id} chain {m.chain} res {m.resseq} absent (chain spans {span})")
            continue

        # Residue exists. Does the wild-type letter agree?
        letters = {aa_at.get((m.chain, m.resseq, ic)) for ic in icodes}
        if m.wt not in letters:
            cats["wt_letter_mismatch"] += 1
            examples["wt_letter_mismatch"].append(
                f"{pdb_id} chain {m.chain} res {m.resseq}: SKEMPI says {m.wt}, PDB has {letters}")
            continue

        if icodes != {" "}:
            cats["exists_with_icode"] += 1
            examples["exists_with_icode"].append(
                f"{pdb_id} chain {m.chain} res {m.resseq} icodes={icodes}")
            continue

        cats["exists_in_structure"] += 1

    total = sum(cats.values())
    print("\n" + "=" * 78)
    print("  WHY ROWS FAIL TO MATCH  (single-mutation rows with a usable ddG)")
    print("=" * 78)
    print(f"  {'category':<26s} {'count':>7s}  {'%':>6s}   verdict")
    print("  " + "-" * 74)
    verdicts = {
        "exists_in_structure":  "residue IS in structure -> not at interface, CORRECT DROP",
        "exists_with_icode":    "exists but carries an insertion code -> RESCUABLE",
        "wt_letter_mismatch":   "numbering points at the wrong residue -> investigate",
        "resseq_absent":        "residue number not in that chain -> RESCUABLE?",
        "chain_absent":         "chain id not in structure -> RESCUABLE (chain mapping)",
        "mutation_unparseable": "mutation code did not parse -> RESCUABLE",
        "structure_unavailable": "structure would not load",
    }
    for k, v in cats.most_common():
        print(f"  {k:<26s} {v:>7d}  {100*v/total:>5.1f}%   {verdicts.get(k,'')}")
    print("  " + "-" * 74)
    print(f"  {'TOTAL':<26s} {total:>7d}")

    print("\n  These are ONLY the rows build_dataset dropped (matched rows excluded),")
    print("  so this total should equal the build's residue_not_found count.")

    for k in ("chain_absent", "resseq_absent", "wt_letter_mismatch",
              "exists_with_icode", "mutation_unparseable"):
        if cats.get(k):
            print(f"\n  --- {k} examples ---")
            for ex in examples[k][:8]:
                print(f"      {ex}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
