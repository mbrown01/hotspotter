"""XGBoost baseline on EXACTLY the data the GNN sees — the apples-to-apples target.

WHY THIS EXISTS
    The headline XGBoost baseline (PR-AUC 0.5495) was trained on 3,788 mutation rows across
    294 complexes. The GNN trains on 1,547 labeled nodes across 172 complexes, because the
    alanine-only label strategy drops every complex with no Ala mutation on its interface.
    Comparing the two directly would be meaningless: different rows, different complexes,
    different class balance.

    So this script rebuilds the tabular baseline under the GNN's exact constraints:

      1. restrict to the 172 complexes present in the graph dataset (read from the .pt, so
         the graph build is the single source of truth rather than a re-derived filter)
      2. keep only X->Ala substitutions
      3. collapse repeat measurements of the same residue to one row, taking max ddG —
         identical to what graph_dataset.build_labels does per node
      4. use the SAME 26 node features, in the same order
      5. split by complex, and write the split out so the GNN can reuse it verbatim

    It asserts the resulting row/complex/positive counts equal the graph dataset's. If they
    ever diverge, the comparison is not matched and the script fails loudly rather than
    quietly reporting a number that means something subtly different.

Usage::

    .\\.venv\\Scripts\\python.exe experiments\\run_matched_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotspotter.ml.dataset import parse_mutation  # noqa: E402
from hotspotter.ml.graph_dataset import DDG_HOTSPOT_THRESHOLD, NODE_FEATURES  # noqa: E402

DEFAULT_FEATURES = REPO_ROOT / "data" / "skempi_features_full.csv"
DEFAULT_GRAPHS = REPO_ROOT / "data" / "graphs_alanine.pt"


def build_matched_frame(features_csv: Path, graphs_pt: Path, threshold: float,
                        strategy: str = "alanine"):
    """Filter + collapse the tabular table down to the GNN's exact labeled set.

    ``strategy`` must match the graph dataset's:
      "alanine" -> keep only X->Ala rows, the literature hot-spot definition
      "max"     -> keep every substitution and take the max ddG at each position
    """
    import torch

    graphs = torch.load(graphs_pt, weights_only=False)
    keep_complexes = {g.complex_group for g in graphs}
    gnn_nodes = sum(int(g.label_mask.sum()) for g in graphs)
    gnn_pos = sum(int((g.y == 1.0).sum()) for g in graphs)

    df = pd.read_csv(features_csv)
    df["to_aa"] = df["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)

    sub = df[df["complex_group"].isin(keep_complexes)].copy()
    if strategy == "alanine":
        sub = sub[sub["to_aa"] == "A"]

    # Features are properties of the STRUCTURE, not of the substitution, so every row at a
    # given position must carry identical features. Verify rather than assume.
    #
    # The key MUST include the insertion code: Kabat-numbered chains put several distinct
    # residues at one number (1EAW chain A: 60, 60a, 60b, ...), and grouping without it
    # would merge different residues and then "collapse" across them.
    sub["icode"] = sub["icode"].fillna("").astype(str).str.strip()
    key = ["complex_group", "chain", "resseq", "icode"]
    spread = sub.groupby(key)["dsasa"].nunique(dropna=False)
    if (spread > 1).any():
        raise AssertionError(
            "feature values differ between rows at the same residue position; "
            "collapsing would silently pick one arbitrarily"
        )

    # Collapse to one row per position, taking max ddG — matching build_labels.
    idx = sub.groupby(key)["ddg"].idxmax()   # max ddG at each position
    matched = sub.loc[idx].copy()
    matched["label"] = (matched["ddg"] >= threshold).astype(int)

    return matched, {
        "gnn_graphs": len(graphs),
        "gnn_labeled_nodes": gnn_nodes,
        "gnn_positive_nodes": gnn_pos,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--graphs", type=Path, default=None,
                    help="default: data/graphs_<strategy>.pt")
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine",
                    help="must match the graph dataset's label strategy")
    ap.add_argument("--threshold", type=float, default=DDG_HOTSPOT_THRESHOLD)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-out", type=Path, default=None,
                    help="default: data/split_<strategy>.json")
    args = ap.parse_args()
    if args.graphs is None:
        args.graphs = REPO_ROOT / "data" / f"graphs_{args.strategy}.pt"
    if args.split_out is None:
        args.split_out = REPO_ROOT / "data" / f"split_{args.strategy}.json"

    for p in (args.features, args.graphs):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from xgboost import XGBClassifier

    print("=" * 78)
    print(f"  MATCHED BASELINE  |  strategy={args.strategy}, "
          f"GNN's exact complexes and residues")
    print("=" * 78)

    df, gnn = build_matched_frame(args.features, args.graphs, args.threshold,
                                  strategy=args.strategy)

    n_complexes = df["complex_group"].nunique()
    n_pos = int(df["label"].sum())
    print(f"\n  graph dataset : {gnn['gnn_graphs']} graphs, "
          f"{gnn['gnn_labeled_nodes']} labeled nodes, {gnn['gnn_positive_nodes']} positive")
    print(f"  matched table : {n_complexes} complexes, {len(df)} rows, {n_pos} positive")

    mismatches = []
    if n_complexes != gnn["gnn_graphs"]:
        mismatches.append(f"complexes {n_complexes} != {gnn['gnn_graphs']}")
    if len(df) != gnn["gnn_labeled_nodes"]:
        mismatches.append(f"rows {len(df)} != {gnn['gnn_labeled_nodes']}")
    if n_pos != gnn["gnn_positive_nodes"]:
        mismatches.append(f"positives {n_pos} != {gnn['gnn_positive_nodes']}")
    if mismatches:
        print("\n  MISMATCH — this is NOT an apples-to-apples baseline:")
        for m in mismatches:
            print(f"    {m}")
        print("  Refusing to report metrics that would be quietly incomparable.")
        return 1
    print("  MATCH CONFIRMED: identical complexes, residues, and labels.\n")

    # ---- train -------------------------------------------------------------------------
    features = list(NODE_FEATURES)   # the same 26 inputs the GNN nodes carry
    X = df[features].fillna(df[features].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    groups = df["complex_group"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                 random_state=args.seed)
    tr, te = next(splitter.split(X, y, groups))

    pos = max(1, int(y[tr].sum()))
    spw = int(len(tr) - y[tr].sum()) / pos
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", random_state=args.seed, n_jobs=-1,
    )
    model.fit(X.iloc[tr], y[tr])
    proba = model.predict_proba(X.iloc[te])[:, 1]

    naive = df.iloc[te]["dsasa"].fillna(0.0).values
    pr_auc = float(average_precision_score(y[te], proba))
    naive_pr = float(average_precision_score(y[te], naive))
    roc = float(roc_auc_score(y[te], proba)) if len(set(y[te])) > 1 else float("nan")

    # ---- cross-validated, for stability on a smaller set --------------------------------
    cv, cv_naive, cv_folds = [], [], []
    for ctr, cte in GroupKFold(n_splits=5).split(X, y, groups):
        # Record fold membership BY COMPLEX so the GNN can reproduce these exact folds.
        # It cannot just re-run GroupKFold: sklearn balances folds by group size, and the
        # GNN's unit is a graph (172 items) while the baseline's is a residue (1,540), so
        # an independently-computed 5-fold split would not agree.
        cv_folds.append({
            "train_complexes": sorted(set(groups[ctr])),
            "test_complexes": sorted(set(groups[cte])),
        })
        p = max(1, int(y[ctr].sum()))
        m = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=int(len(ctr) - y[ctr].sum()) / p,
            eval_metric="aucpr", random_state=args.seed, n_jobs=-1,
        )
        m.fit(X.iloc[ctr], y[ctr])
        cv.append(average_precision_score(y[cte], m.predict_proba(X.iloc[cte])[:, 1]))
        cv_naive.append(average_precision_score(y[cte], df.iloc[cte]["dsasa"].fillna(0).values))

    # ---- persist the split so the GNN trains and tests on the same complexes ------------
    split = {
        "seed": args.seed, "test_size": args.test_size,
        "threshold": args.threshold, "strategy": args.strategy,
        "train_complexes": sorted(set(groups[tr])),
        "test_complexes": sorted(set(groups[te])),
        "cv_folds": cv_folds,
        "baseline": {
            "single_split": {"xgboost_pr_auc": pr_auc, "naive_pr_auc": naive_pr,
                             "roc_auc": roc},
            "cv": {"xgboost_pr_auc_mean": float(np.mean(cv)),
                   "xgboost_pr_auc_std": float(np.std(cv)),
                   "naive_pr_auc_mean": float(np.mean(cv_naive)),
                   "fold_scores": [float(s) for s in cv]},
        },
        "positive_class_rate": n_pos / len(df),
        "n_rows": len(df), "n_complexes": n_complexes, "n_positive": n_pos,
    }
    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    args.split_out.write_text(json.dumps(split, indent=2))

    # ---- report --------------------------------------------------------------------------
    print("-" * 78)
    print("  CLASS BALANCE (matched)")
    print("-" * 78)
    print(f"  rows (residue positions): {len(df)}")
    print(f"  complexes               : {n_complexes}")
    print(f"  positive / negative     : {n_pos} / {len(df) - n_pos}")
    print(f"  POSITIVE CLASS RATE     : {n_pos/len(df):.4f}   <-- PR-AUC chance floor")

    print("\n" + "-" * 78)
    print("  HELD-OUT (single split by complex)")
    print("-" * 78)
    print(f"  train / test rows       : {len(tr)} / {len(te)}")
    print(f"  train / test complexes  : {len(set(groups[tr]))} / {len(set(groups[te]))}")
    print(f"  test positive rate      : {y[te].mean():.4f}")
    print()
    print(f"  NAIVE SASA PR-AUC       : {naive_pr:.4f}")
    print(f"  XGBOOST PR-AUC          : {pr_auc:.4f}")
    print(f"  (xgboost ROC-AUC        : {roc:.4f})")
    print(f"\n  Model beats naive by      {pr_auc - naive_pr:+.4f} PR-AUC")

    print("\n" + "-" * 78)
    print("  5-FOLD GROUPED CV (more honest on a set this size)")
    print("-" * 78)
    print(f"  NAIVE SASA PR-AUC       : {np.mean(cv_naive):.4f} +/- {np.std(cv_naive):.4f}")
    print(f"  XGBOOST PR-AUC          : {np.mean(cv):.4f} +/- {np.std(cv):.4f}")
    print(f"  fold scores             : {[round(float(s),4) for s in cv]}")

    print(f"\n  split written -> {args.split_out}")
    print("  The GNN MUST load this split, or the comparison is not matched.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
