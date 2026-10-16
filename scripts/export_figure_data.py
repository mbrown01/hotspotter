"""Export everything needed to draw the README figures, so the plotting can live
anywhere.

Two case studies, chosen because the measurements are dense enough to score, not because
they are famous:

    3M62  Ufd2-Rad23, 17 measured residues, 4 hot spots.
          Buried surface area finds 0 of 4 in its top 4. HotSpotter finds 3.
          Tyr97 is the least-buried hot spot: burial ranks it 14th, the model 1st.
    1EMV  colicin E9 DNase-Im9, 32 measured residues, 9 hot spots. The best-measured
          interface available, so very little is unknown.
          Burial finds 4 of 9, HotSpotter 6 of 9. Asn75: burial 15th, model 3rd.

PROTOCOL, which matters for anyone quoting these numbers
    The complex being shown is removed from training, AND SO IS EVERY COMPLEX IN ITS
    >=70% CHAIN-IDENTITY CLUSTER. Dropping the exact PDB id alone is not enough: SKEMPI
    carries 2VLN, 2VLO, 2VLQ and 2WPT for 1EMV and 3M63 for 3M62, so a model that had
    only lost the one entry would still have been trained on the same interface.

    Ranks are computed over the MEASURED residues only, because an unmeasured residue
    sitting in a method's top k is neither a hit nor a miss; ranking over all interface
    residues instead lets the model spend slots on plausible residues nobody ever
    tested and then counts that against it.

Writes one CSV per complex plus a summary, into assets/figure_data/.

Usage::
    .\\.venv\\Scripts\\python.exe scripts\\export_figure_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from audit_splits import chain_sequences, cluster_complexes           # noqa: E402
from hotspotter.ml.features import XGB_FEATURES, load_labeled_table   # noqa: E402
from hotspotter.ml.train import accepted_scores                       # noqa: E402
from hotspotter.pipeline import analyze_complex                       # noqa: E402

HOT = 2.0
IDENTITY = 0.70
F = list(XGB_FEATURES)
CASES = [("3M62", ("A",), ("B",), "Ufd2-Rad23"),
         ("1EMV", ("A",), ("B",), "colicin E9 DNase-Im9")]


def ca_coords(pdb: str, chains) -> dict:
    from Bio.PDB import PDBParser

    model = next(PDBParser(QUIET=True)
                 .get_structure(pdb, REPO_ROOT / "data" / "raw" / f"{pdb.lower()}.pdb")
                 .get_models())
    return {(c, int(r.id[1]), r.id[2].strip()): r["CA"].coord
            for c in chains for r in model[c]
            if r.id[0] == " " and "CA" in r}


def res_key(chain, resseq, icode) -> tuple:
    """Join key for one residue. The insertion code is part of it: Kabat-numbered
    chains put several distinct residues at one number, and dropping it hands one
    residue's features and labels to another."""
    return (str(chain), int(resseq), str(icode or "").strip())


def main() -> int:
    out_dir = REPO_ROOT / "assets" / "figure_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = load_labeled_table(REPO_ROOT / "data" / "skempi_features.csv",
                              "alanine", HOT)

    # Hold out each case study's whole identity cluster, not just its PDB id. Built
    # once because the clustering is over every complex in the table.
    print(f"  clustering {full.pdb_id.nunique()} complexes at >={IDENTITY:.0%} "
          f"chain identity...", flush=True)
    seqs = chain_sequences(sorted(full.pdb_id.unique()),
                           REPO_ROOT / "data" / "raw",
                           REPO_ROOT / "data" / "chain_sequences.json")
    clusters = cluster_complexes(seqs, IDENTITY)
    summary = []

    for pdb, a, b, label in CASES:
        mates = {p for p, c in clusters.items() if c == clusters.get(pdb, pdb)}
        train = full[~full.pdb_id.str.upper().isin({m.upper() for m in mates})]
        print(f"  {pdb}: held out {len(mates)} complexes "
              f"({', '.join(sorted(mates))}) -> {train.pdb_id.nunique()} left to train on")

        res = analyze_complex(str(REPO_ROOT / "data" / "raw" / f"{pdb.lower()}.pdb"),
                              chains=(a, b), pdb_id=pdb)
        t = res.table.copy()
        t["pred"] = accepted_scores(train, t, F)
        t_keys = [res_key(c, r, i) for c, r, i in zip(t.chain, t.resseq, t.icode)]

        lab = full[full.pdb_id.str.upper() == pdb.upper()]
        ddg = {res_key(r.chain, r.resseq, r.icode): r.ddg for r in lab.itertuples()}
        t["ddg"] = [ddg.get(k, np.nan) for k in t_keys]
        t["measured"] = t.ddg.notna()
        t["is_hot"] = (t.ddg >= HOT).fillna(False)

        xyz = ca_coords(pdb, list(a) + list(b))
        for i, ax in enumerate("xyz"):
            t[f"ca_{ax}"] = [xyz.get(k, (np.nan,) * 3)[i] for k in t_keys]

        # ranks over MEASURED residues only -- the set either method can be scored on
        m = t.measured
        t["burial_rank"] = np.nan
        t["model_rank"] = np.nan
        t.loc[m, "burial_rank"] = t.loc[m, "dsasa"].rank(ascending=False, method="min")
        t.loc[m, "model_rank"] = t.loc[m, "pred"].rank(ascending=False, method="min")
        # ranks over every interface residue, for a structure figure that shows them all
        t["burial_rank_all"] = t.dsasa.rank(ascending=False, method="min")
        t["model_rank_all"] = t.pred.rank(ascending=False, method="min")

        cols = ["chain", "resseq", "icode", "resname", "ca_x", "ca_y", "ca_z",
                "dsasa", "pred", "ddg", "measured", "is_hot",
                "burial_rank", "model_rank", "burial_rank_all", "model_rank_all"]
        dest = out_dir / f"{pdb.lower()}_residues.csv"
        t[[c for c in cols if c in t.columns]].sort_values(
            "model_rank_all").to_csv(dest, index=False)

        hot = t[t.is_hot]
        k = len(hot)
        star = hot.loc[hot.burial_rank.idxmax()]
        row = dict(
            pdb_id=pdb, name=label, interface_residues=len(t),
            measured=int(m.sum()), hot_spots=k,
            burial_finds=int((hot.burial_rank <= k).sum()),
            hotspotter_finds=int((hot.model_rank <= k).sum()),
            star_residue=f"{star.resname}{int(star.resseq)}",
            star_ddg=round(float(star.ddg), 2),
            star_burial_rank=int(star.burial_rank),
            star_model_rank=int(star.model_rank),
            star_dsasa=round(float(star.dsasa), 1),
        )
        summary.append(row)
        print(f"  {pdb}: {len(t)} interface residues, {int(m.sum())} measured, "
              f"{k} hot | burial {row['burial_finds']}/{k} | "
              f"HotSpotter {row['hotspotter_finds']}/{k} | "
              f"{row['star_residue']} burial {row['star_burial_rank']} -> "
              f"model {row['star_model_rank']}")
        print(f"      -> {dest.relative_to(REPO_ROOT)}")

    s = pd.DataFrame(summary)
    s.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n      -> {(out_dir / 'summary.csv').relative_to(REPO_ROOT)}")
    print("\n" + s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
