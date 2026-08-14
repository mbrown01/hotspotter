"""Build the Phase-2 training table from SKEMPI 2.0 + the Phase-1 pipeline.

BIOLOGY / DATA NOTE — what SKEMPI is:
    SKEMPI 2.0 (Jankauskaite et al. 2019) tabulates ~7,000 mutations in protein complexes
    that have *solved structures*, each with the *measured* binding affinity (Kd) of the
    wild-type and the mutant. From the two Kd's we compute ΔΔG — how much the mutation
    changed binding free energy. That ΔΔG is the label: a big positive ΔΔG means "mutating
    this residue really disrupted binding," i.e. it was a hot spot. This is the ground
    truth that lets us replace Phase 1's hand-set weights with a trained model.

THE PLAN (this module):
    1. parse SKEMPI rows -> (pdb_id, side_a_chains, side_b_chains, mutations, Kd_wt, Kd_mut, T)
    2. compute ΔΔG per row from the Kd's (real math, implemented below)
    3. for each complex, run the Phase-1 pipeline ONCE and cache its per-residue table
    4. for each mutated residue, pull its Phase-1 feature row and attach the ΔΔG label
    5. hand back a tidy (features + label + group) DataFrame for train.py

THE TWO THINGS WE MUST NOT GET WRONG (both handled here):
    - SPLIT BY COMPLEX: the `complex_group` column groups rows so train/test never share a
      complex (see train.py). Skip this and the model memorizes complexes and scores look
      great and mean nothing.
    - CLASS IMBALANCE: most single mutations barely change binding, so the "disruptive"
      class is rare. We expose a binary label via a ΔΔG threshold AND keep the continuous
      ΔΔG so you can do regression or weighted classification (train.py handles the weight).

HONESTY: the ΔΔG math and mutation/PDB parsing below are real and unit-testable. The
`build_dataset` orchestration is written but not yet run against the real file — SKEMPI's
exact column names have changed between releases, so verify `SKEMPI_COLUMNS` on first run
(one print of `df.columns` tells you). Structure-download failures for a few PDBs are
expected; we skip and log them rather than crashing the whole sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hotspot.io import ResidueId
from hotspot.pipeline import ComplexAnalysis, analyze_complex

# Gas constant in kcal/(mol*K); ΔG = R*T*ln(Kd).
R_KCAL = 1.987204259e-3

# SKEMPI 2.0 is a ';'-separated CSV (skempi_v2.csv). These are the columns we rely on;
# CONFIRM against the real header on first use (releases have renamed a few).
SKEMPI_COLUMNS = {
    "pdb_field": "#Pdb",                 # e.g. "1CSE_E_I": pdbid _ sideA-chains _ sideB-chains
    "mutation": "Mutation(s)_cleaned",    # e.g. "TI17A" or "TI17A,SI19G" (comma-separated)
    "kd_wt": "Affinity_wt_parsed",        # molar
    "kd_mut": "Affinity_mut_parsed",      # molar
    "temperature": "Temperature",         # Kelvin (sometimes with junk like "298(assumed)")
}


@dataclass
class Mutation:
    """One point mutation parsed from a SKEMPI mutation code, e.g. 'TI17A'."""

    wt: str          # wild-type one-letter aa
    chain: str
    resseq: int
    mut: str         # mutant one-letter aa
    icode: str = " "

    def matches(self, rid: ResidueId) -> bool:
        """True if a Phase-1 ResidueId is the residue this mutation refers to."""
        return (rid.chain == self.chain and rid.resseq == self.resseq
                and rid.icode.strip() == self.icode.strip()
                and rid.one_letter == self.wt)


def parse_mutation(code: str) -> Mutation:
    """Parse a SKEMPI mutation code 'X<chain><resnum>[icode]<Y>'.

    Format: first char = WT aa (1-letter), second char = chain id, then the residue number
    (optionally trailed by an insertion-code letter), last char = mutant aa. Example:
    'TI17A' -> Thr, chain I, residue 17, to Ala.
    """
    code = code.strip()
    wt, chain, mut = code[0], code[1], code[-1]
    middle = code[2:-1]
    icode = " "
    if middle and middle[-1].isalpha():   # trailing insertion code, e.g. '17A' style pos
        icode = middle[-1]
        middle = middle[:-1]
    return Mutation(wt=wt, chain=chain, resseq=int(middle), mut=mut, icode=icode)


def parse_pdb_field(field: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Parse SKEMPI '#Pdb' like '1CSE_E_I' -> ('1CSE', ('E',), ('I',)).

    Multi-chain sides appear as concatenated letters, e.g. '3BDY_HL_A' -> HL vs A.
    """
    parts = field.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected #Pdb field {field!r}; expected 'PDB_sideA_sideB'.")
    pdb_id, a, b = parts
    return pdb_id, tuple(a), tuple(b)


def ddg_from_kd(kd_wt: float, kd_mut: float, temperature_k: float = 298.0) -> float:
    """ΔΔG (kcal/mol) = R*T*ln(Kd_mut / Kd_wt).

    Sign convention: POSITIVE ΔΔG = mutant binds WEAKER (higher Kd) = the residue mattered.
    (Some papers flip this; we state it explicitly so labels are unambiguous.)
    """
    if not (kd_wt and kd_mut) or kd_wt <= 0 or kd_mut <= 0:
        return math.nan
    return R_KCAL * temperature_k * math.log(kd_mut / kd_wt)


def _clean_temperature(value) -> float:
    """SKEMPI temperatures sometimes read like '298(assumed)'. Extract the number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if (ch.isdigit() or ch == "."))
        return float(digits) if digits else 298.0


def load_skempi(csv_path: str | Path) -> pd.DataFrame:
    """Load skempi_v2.csv (';'-separated) into a DataFrame. Verify columns on first use."""
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    missing = [c for c in SKEMPI_COLUMNS.values() if c not in df.columns]
    if missing:
        raise KeyError(
            f"SKEMPI columns not found: {missing}. Actual columns: {list(df.columns)}. "
            f"Update SKEMPI_COLUMNS in dataset.py to match this release."
        )
    return df


def build_dataset(
    skempi_csv: str | Path,
    ddg_disruptive_threshold: float = 1.0,
    only_single_mutations: bool = True,
    cache_dir: str | Path | None = None,
    limit_complexes: int | None = None,
) -> pd.DataFrame:
    """Build the labeled per-mutation feature table (the Phase-2 training set).

    For each SKEMPI row we compute ΔΔG, run the Phase-1 pipeline on that complex (cached so
    each structure is analyzed once), find the mutated residue's feature row, and attach the
    label. Rows whose structure won't download, whose residue can't be located, or whose
    Kd's are missing are skipped and counted (not fatal).

    Parameters
    ----------
    ddg_disruptive_threshold : ΔΔG (kcal/mol) above which we call a mutation "disruptive"
        for the binary label. 1.0 is a common, mild cutoff; ~2.0 is the classic hot-spot
        definition. Kept as a knob because it directly shapes the class balance.
    only_single_mutations : restrict to single point mutations (the clean case) — multi-
        mutation rows conflate several residues' effects.
    limit_complexes : cap the number of distinct complexes (handy for a quick smoke run).

    Returns
    -------
    DataFrame: one row per usable mutation = Phase-1 features + ['ddg', 'label',
    'complex_group', 'mutation']. Feed straight to train.train_baseline.

    STATUS: written, not yet run against real SKEMPI. Expect to confirm column names and a
    handful of chain-mapping edge cases on the first real pass.
    """
    df = load_skempi(skempi_csv)
    cols = SKEMPI_COLUMNS
    cache: dict[str, ComplexAnalysis] = {}
    rows: list[dict] = []
    skipped = {"multi_mut": 0, "no_kd": 0, "no_structure": 0, "residue_not_found": 0}
    seen_complexes: set[str] = set()

    for _, r in df.iterrows():
        mut_field = str(r[cols["mutation"]])
        if only_single_mutations and ("," in mut_field):
            skipped["multi_mut"] += 1
            continue
        try:
            pdb_id, side_a, side_b = parse_pdb_field(str(r[cols["pdb_field"]]))
        except ValueError:
            continue

        if limit_complexes and pdb_id not in seen_complexes and len(seen_complexes) >= limit_complexes:
            continue

        ddg = ddg_from_kd(
            _to_float(r[cols["kd_wt"]]),
            _to_float(r[cols["kd_mut"]]),
            _clean_temperature(r[cols["temperature"]]),
        )
        if math.isnan(ddg):
            skipped["no_kd"] += 1
            continue

        # Analyze the complex once (cached).
        if pdb_id not in cache:
            try:
                cache[pdb_id] = analyze_complex(pdb_id, chains=(side_a, side_b))
            except Exception as exc:  # download/parse/interface failure -> skip this complex
                cache[pdb_id] = None
                print(f"[skempi] skip {pdb_id}: {exc}")
            seen_complexes.add(pdb_id)
        analysis = cache[pdb_id]
        if analysis is None:
            skipped["no_structure"] += 1
            continue

        mutation = parse_mutation(mut_field)
        feat_row = _find_residue_row(analysis, mutation)
        if feat_row is None:
            skipped["residue_not_found"] += 1
            continue

        record = feat_row.to_dict()
        record.update({
            "ddg": ddg,
            "label": int(ddg >= ddg_disruptive_threshold),
            "complex_group": pdb_id,      # <-- the group key that enforces split-by-complex
            "mutation": mut_field,
        })
        rows.append(record)

    out = pd.DataFrame(rows)
    n_pos = int(out["label"].sum()) if len(out) else 0
    print(f"[skempi] built {len(out)} labeled rows from {len(seen_complexes)} complexes "
          f"({n_pos} disruptive / {len(out) - n_pos} neutral). Skipped: {skipped}")

    if cache_dir and len(out):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_dir / "skempi_features.parquet", index=False)
    return out


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_residue_row(analysis: ComplexAnalysis, mutation: Mutation):
    """Return the Phase-1 feature row (a Series) for the mutated residue, or None.

    Matches on chain + residue number + (if the wild-type letter is given) identity, so a
    numbering mismatch between SKEMPI and the PDB surfaces as 'not found' rather than a
    silently wrong join.
    """
    t = analysis.table
    hit = t[(t["chain"] == mutation.chain) & (t["resseq"] == mutation.resseq)]
    if len(hit) == 0:
        return None
    if "aa" in hit.columns:
        typed = hit[hit["aa"] == mutation.wt]
        if len(typed):
            hit = typed
    return hit.iloc[0]
