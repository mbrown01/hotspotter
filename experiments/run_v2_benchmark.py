"""XGBoost benchmark on V2 physics features, against the V1 baseline on identical folds.

THE CONFOUND THIS CONTROLS FOR
    The V1 matched baseline scored PR-AUC 0.5020 +/- 0.0805 over 172 alanine complexes. This
    run holds out 1BRS, leaving 171. Comparing "V1 on 172" against "V2 on 171" would mix the
    feature change with a change in the complex set, and the difference could not be
    attributed to either.

    So this script runs BOTH feature sets over the SAME 171 complexes, the same GroupKFold
    folds, the same hyperparameters and the same seed. The only thing that differs between
    the two arms is which columns go in.

MATCHING THE V1 PROTOCOL EXACTLY
    hyperparameters : n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=neg/pos, eval_metric=aucpr
    CV              : GroupKFold(n_splits=5) grouped on complex
    seed            : 0
    label strategy  : alanine (X->Ala only), collapsed per residue by max ddG, threshold 2.0
    naive baseline  : rank by dsasa on the same test rows

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\run_v2_benchmark.py
    .\\.venv\\Scripts\\python.exe scripts\\run_v2_benchmark.py --strategy max
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features_v2 import V2_CHEMISTRY_COLUMNS  # noqa: E402
from hotspotter.ml.graph_dataset import NODE_FEATURES  # noqa: E402

#: V1's 26 inputs minus the two V2 removes (n_disulfides is gone from the V2 table).
V1_FEATURES = [f for f in NODE_FEATURES if f != "n_disulfides"]

#: Non-chemistry columns are identical in both tables; V2 only revises chemistry.
SHARED_NON_CHEMISTRY = [
    "sasa_complex", "sasa_unbound", "dsasa", "rsa_complex", "rsa_unbound",
    "is_interface_sasa", "n_cross_contacts", "n_atom_contacts", "interface_neighbors",
    "packing_density", "centrality", "charge", "hydropathy", "volume", "flexibility",
    "is_aromatic", "is_charged", "is_polar", "bfactor",
]
V1_CHEMISTRY = ["n_salt_bridges", "n_hydrogen_bonds", "n_hydrophobic", "n_aromatic",
                "n_chem_contacts", "has_salt_bridge"]


def collapse(df: pd.DataFrame, strategy: str, threshold: float) -> pd.DataFrame:
    """One row per residue position, label from the max ddG at that position."""
    sub = df if strategy == "max" else df[df["mut"] == "A"]
    sub = sub.copy()
    sub["icode"] = sub["icode"].fillna("").astype(str).str.strip()
    key = ["pdb_id", "chain", "resseq", "icode"]
    idx = sub.groupby(key)["ddg"].idxmax()
    out = sub.loc[idx].copy()
    out["label"] = (out["ddg"] >= threshold).astype(int)
    return out


def run_cv(df: pd.DataFrame, features: list[str], seed: int, n_splits: int):
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    X = df[features].fillna(df[features].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    groups = df["pdb_id"].values

    scores, naive, gains = [], [], {}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pos = max(1, int(y[tr].sum()))
        m = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(tr) - y[tr].sum()) / pos,
            eval_metric="aucpr", random_state=seed, n_jobs=-1,
        )
        m.fit(X.iloc[tr], y[tr])
        scores.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
        naive.append(average_precision_score(y[te], df.iloc[te]["dsasa"].fillna(0).values))
        for k, v in m.get_booster().get_score(importance_type="gain").items():
            gains[k] = gains.get(k, 0.0) + v / n_splits
    return np.array(scores), np.array(naive), gains


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", type=Path, default=REPO_ROOT / "data" / "skempi_features_v2.csv")
    ap.add_argument("--v1", type=Path, default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--holdout", nargs="*", default=["1BRS"])
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    for p in (args.v2, args.v1):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    hold = {h.upper() for h in args.holdout}
    v2 = pd.read_csv(args.v2)
    v2 = v2[~v2["pdb_id"].str.upper().isin(hold)]        # <- 1BRS held out AFTER loading
    v2 = collapse(v2, args.strategy, args.threshold)

    # V1 arm, restricted to exactly the same complexes so the comparison is clean
    v1 = pd.read_csv(args.v1)
    from hotspotter.ml.dataset import parse_mutation
    v1["mut"] = v1["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    v1["pdb_id"] = v1["complex_group"]
    v1 = v1[~v1["pdb_id"].str.upper().isin(hold)]
    v1 = collapse(v1, args.strategy, args.threshold)

    shared = sorted(set(v1["pdb_id"]) & set(v2["pdb_id"]))
    v1 = v1[v1["pdb_id"].isin(shared)]
    v2 = v2[v2["pdb_id"].isin(shared)]

    print("=" * 78)
    print(f"  V2 PHYSICS BENCHMARK  |  strategy={args.strategy}  |  held out: {sorted(hold)}")
    print("=" * 78)
    print(f"  complexes in both arms : {len(shared)}")
    print(f"  V1 rows {len(v1)}  |  V2 rows {len(v2)}")
    if len(v1) != len(v2):
        print("  NOTE: row counts differ -- V2's stricter/looser chemistry can change which")
        print("        residues resolve. Comparison is still per-complex matched.")
    print(f"  positives: V1 {int(v1.label.sum())}  V2 {int(v2.label.sum())}  "
          f"(rate {v2.label.mean():.4f})")

    v2_feats = [c for c in SHARED_NON_CHEMISTRY + list(V2_CHEMISTRY_COLUMNS)
                if c in v2.columns]
    v1_feats = [c for c in SHARED_NON_CHEMISTRY + V1_CHEMISTRY if c in v1.columns]
    print(f"  V1 features {len(v1_feats)}  ->  V2 features {len(v2_feats)}")

    s1, n1, _ = run_cv(v1, v1_feats, args.seed, args.folds)
    s2, n2, gains = run_cv(v2, v2_feats, args.seed, args.folds)

    print("\n" + "-" * 78)
    print(f"  {args.folds}-FOLD GROUPED CV  (same folds, same seed, same hyperparameters)")
    print("-" * 78)
    print(f"  naive SASA (V1 rows)   : {n1.mean():.4f} +/- {n1.std():.4f}")
    print(f"  naive SASA (V2 rows)   : {n2.mean():.4f} +/- {n2.std():.4f}")
    print(f"  V1 chemistry PR-AUC    : {s1.mean():.4f} +/- {s1.std():.4f}   "
          f"folds {[round(float(x),4) for x in s1]}")
    print(f"  V2 physics   PR-AUC    : {s2.mean():.4f} +/- {s2.std():.4f}   "
          f"folds {[round(float(x),4) for x in s2]}")
    d = s2 - s1
    print(f"\n  V2 - V1 (paired by fold): {d.mean():+.4f}   "
          f"per-fold {[round(float(x),4) for x in d]}   wins {int((d>0).sum())}/{args.folds}")
    from scipy import stats
    if args.folds > 2:
        t, p = stats.ttest_rel(s2, s1)
        print(f"  paired t over folds     : t={t:.3f}  p={p:.4f}  (n={args.folds}, "
              f"underpowered -- treat as directional)")
    print("\n  REFERENCE: V1 matched baseline over 172 complexes was 0.5020 +/- 0.0805")
    print("             (this run holds out 1BRS, so the V1 arm above is the fair anchor)")

    print("\n" + "-" * 78)
    print(f"  TOP {args.top} V2 FEATURES BY GAIN (mean over folds)")
    print("-" * 78)
    new = set(V2_CHEMISTRY_COLUMNS) - set(V1_CHEMISTRY)
    for i, (k, v) in enumerate(sorted(gains.items(), key=lambda x: -x[1])[:args.top], 1):
        tag = "  <- NEW in V2" if k in new else ""
        print(f"    {i:>2}. {k:<26}{v:>10.2f}{tag}")
    tot = sum(gains.values())
    newshare = sum(v for k, v in gains.items() if k in new) / tot if tot else 0
    print(f"\n  new-in-V2 features carry {100*newshare:.1f}% of total gain")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
