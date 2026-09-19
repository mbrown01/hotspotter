"""L1-regularized V2 benchmark: nested alpha selection, 25-seed paired protocol.

THE SELECTION-BIAS TRAP THIS AVOIDS
    The obvious way to "sweep reg_alpha" is to run each value over the 5 CV folds and report
    the best. That number is not an estimate of generalization -- it is the maximum of five
    noisy draws, and with a fold spread of +/-0.11 the maximum is inflated by roughly the
    size of the effect we are looking for.

    Here alpha is chosen by an INNER GroupKFold over each outer fold's TRAINING complexes
    only. The outer test fold never influences the choice. Selection happens once per outer
    fold (not per seed) because alpha is a property of the data, not of XGBoost's subsample
    draw -- re-selecting per seed would cost 25x the compute for no statistical gain.

WHAT reg_alpha ACTUALLY DOES HERE -- AN IMPORTANT CLARIFICATION
    In XGBoost, ``reg_alpha`` is an L1 penalty on LEAF WEIGHTS, not on feature coefficients.
    It does NOT zero out features the way Lasso zeroes regression coefficients. What it does
    is make marginal splits unprofitable, so weak features stop being selected at all and
    vanish from the booster.

    So "crushed to exactly 0.0" is measured here as FEATURES NEVER USED IN ANY SPLIT --
    absent from the booster's gain dictionary. That is the honest operationalization; a
    feature with a tiny non-zero gain was still used and is not "crushed".

FOUR ARMS, ALL PAIRED ON IDENTICAL FOLDS
    V1            25 V1-era features, original hyperparameters (the anchor)
    V1 + L1       same features, nested-selected alpha
    V2            34 physics features, original hyperparameters
    V2 + L1       same features, nested-selected alpha
    Running V1+L1 too answers the obvious objection: maybe L1 helps any feature set, and the
    V2 gain (if any) is really an L1 gain.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\run_v2_l1_sweep.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v2_benchmark import (  # noqa: E402
    SHARED_NON_CHEMISTRY, V1_CHEMISTRY, collapse,
)
from features_v2 import V2_CHEMISTRY_COLUMNS  # noqa: E402

#: Aggregates whose components now exist as separate columns. Keeping both the sum and its
#: parts is textbook collinearity: n_chem_contacts is the sum of five other columns, and
#: n_hydrogen_bonds / n_salt_bridges are the sums of their own side/back splits. L1 on leaf
#: weights cannot fix this -- it damps predictions, it does not choose between correlated
#: inputs -- so the only remedy is to remove the redundancy by hand.
REDUNDANT_AGGREGATES = ["n_chem_contacts", "n_hydrogen_bonds", "n_salt_bridges",
                        "n_aromatic"]

# 10.0 was selected at the boundary on 9/10 folds in a 3-seed probe, so the grid is
# extended upward -- an optimum sitting at the edge of the search usually means the
# search was too small.
ALPHAS = [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
BASE = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", n_jobs=-1)


def fit_score(Xtr, ytr, Xte, yte, alpha, seed):
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    pos = max(1, int(ytr.sum()))
    m = XGBClassifier(**BASE, reg_alpha=alpha, random_state=seed,
                      scale_pos_weight=(len(ytr) - ytr.sum()) / pos)
    m.fit(Xtr, ytr)
    return average_precision_score(yte, m.predict_proba(Xte)[:, 1]), m


def select_alpha(X, y, groups, tr_idx, inner_folds, seed):
    """Choose alpha by inner GroupKFold over the TRAINING complexes only."""
    from sklearn.model_selection import GroupKFold

    Xtr, ytr, gtr = X.iloc[tr_idx], y[tr_idx], groups[tr_idx]
    best, best_a = -1.0, 0.0
    for a in ALPHAS:
        sc = []
        for itr, ite in GroupKFold(n_splits=inner_folds).split(Xtr, ytr, gtr):
            if len(set(ytr[ite])) < 2:
                continue
            s, _ = fit_score(Xtr.iloc[itr], ytr[itr], Xtr.iloc[ite], ytr[ite], a, seed)
            sc.append(s)
        m = float(np.mean(sc)) if sc else -1.0
        if m > best:
            best, best_a = m, a
    return best_a, best


def run_arm(df, feats, seeds, folds, inner_folds, use_l1):
    """Return (scores[seed, fold], chosen_alpha_per_fold, mean_gain, unused_counts)."""
    from sklearn.model_selection import GroupKFold

    X = df[feats].fillna(df[feats].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    groups = df["pdb_id"].values
    splits = list(GroupKFold(n_splits=folds).split(X, y, groups))

    alphas = []
    for tr, _te in splits:
        if use_l1:
            a, _ = select_alpha(X, y, groups, tr, inner_folds, seed=0)
        else:
            a = 0.0
        alphas.append(a)

    scores = np.zeros((len(seeds), folds))
    gains: Counter = Counter()
    unused = []
    for si, seed in enumerate(seeds):
        for fi, (tr, te) in enumerate(splits):
            s, model = fit_score(X.iloc[tr], y[tr], X.iloc[te], y[te], alphas[fi], seed)
            scores[si, fi] = s
            g = model.get_booster().get_score(importance_type="gain")
            for k, v in g.items():
                gains[k] += v / (len(seeds) * folds)
            unused.append(len(feats) - len(g))
    return scores, alphas, gains, np.array(unused)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", type=Path, default=REPO_ROOT / "data" / "skempi_features_v2.csv")
    ap.add_argument("--v1", type=Path, default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--holdout", nargs="*", default=["1BRS"])
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    from hotspotter.ml.dataset import parse_mutation
    from scipy import stats

    hold = {h.upper() for h in args.holdout}
    v2 = pd.read_csv(args.v2)
    v2 = v2[~v2["pdb_id"].str.upper().isin(hold)]
    v2 = collapse(v2, args.strategy, args.threshold)

    v1 = pd.read_csv(args.v1)
    v1["mut"] = v1["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    v1["pdb_id"] = v1["complex_group"]
    v1 = v1[~v1["pdb_id"].str.upper().isin(hold)]
    v1 = collapse(v1, args.strategy, args.threshold)

    shared = sorted(set(v1["pdb_id"]) & set(v2["pdb_id"]))
    v1, v2 = v1[v1["pdb_id"].isin(shared)], v2[v2["pdb_id"].isin(shared)]

    v1_feats = [c for c in SHARED_NON_CHEMISTRY + V1_CHEMISTRY if c in v1.columns]
    v2_feats = [c for c in SHARED_NON_CHEMISTRY + list(V2_CHEMISTRY_COLUMNS)
                if c in v2.columns]
    v2p_feats = [c for c in v2_feats if c not in REDUNDANT_AGGREGATES]
    seeds = list(range(args.seeds))

    # persist the pruned table so the exact inputs are reproducible
    pruned_path = REPO_ROOT / "data" / "skempi_features_v2_pruned.csv"
    keep_meta = [c for c in ("pdb_id", "complex_group", "chain", "resseq", "icode",
                             "resname", "aa", "side", "residue", "mutation", "wt", "mut",
                             "ddg", "label") if c in v2.columns]
    v2[keep_meta + v2p_feats].to_csv(pruned_path, index=False)

    print("=" * 78)
    print(f"  L1 SWEEP + {args.seeds}-SEED PAIRED PROTOCOL  |  held out {sorted(hold)}")
    print("=" * 78)
    print(f"  complexes {len(shared)} | rows {len(v2)} | positives {int(v2.label.sum())} "
          f"(rate {v2.label.mean():.4f})")
    print(f"  V1 features {len(v1_feats)} | V2 features {len(v2_feats)} | "
          f"V2-pruned {len(v2p_feats)}  (dropped {REDUNDANT_AGGREGATES})")
    print(f"  alpha grid {ALPHAS}, chosen by inner GroupKFold({args.inner_folds}) on "
          f"TRAINING complexes only")
    print(f"  colsample_bytree {BASE['colsample_bytree']}, subsample {BASE['subsample']}\n")

    arms = {}
    for name, df, feats, l1 in (("V1", v1, v1_feats, False),
                                ("V1+L1", v1, v1_feats, True),
                                ("V2", v2, v2_feats, False),
                                ("V2+L1", v2, v2_feats, True),
                                ("V2p", v2, v2p_feats, False),
                                ("V2p+L1", v2, v2p_feats, True)):
        sc, al, gains, unused = run_arm(df, feats, seeds, args.folds, args.inner_folds, l1)
        arms[name] = dict(scores=sc, alphas=al, gains=gains, unused=unused,
                          n_feats=len(feats))
        print(f"  {name:<6} done: {sc.mean():.4f} +/- {sc.std():.4f}"
              + (f"   alpha per fold {al}" if l1 else ""))

    print("\n" + "-" * 78)
    print(f"  RESULTS  ({args.seeds} seeds x {args.folds} folds = "
          f"{args.seeds * args.folds} paired runs per arm)")
    print("-" * 78)
    print(f"    {'arm':<8}{'PR-AUC':>9}{'fold+/-':>10}{'seed+/-':>10}   alpha")
    for name, a in arms.items():
        s = a["scores"]
        al = a["alphas"]
        print(f"    {name:<8}{s.mean():>9.4f}{s.std():>10.4f}{s.mean(1).std():>10.4f}"
              f"   {al if any(al) else '-'}")

    print("\n  PAIRED COMPARISONS (same fold, same seed)")
    for a, b in (("V2", "V1"), ("V2+L1", "V1"), ("V2+L1", "V2"), ("V1+L1", "V1"),
                 ("V2p+L1", "V1+L1"), ("V2p+L1", "V2+L1"), ("V2p", "V2")):
        d = (arms[a]["scores"] - arms[b]["scores"]).ravel()
        t, p = stats.ttest_rel(arms[a]["scores"].ravel(), arms[b]["scores"].ravel())
        w, pw = stats.wilcoxon(arms[a]["scores"].ravel(), arms[b]["scores"].ravel())
        print(f"    {a:<6} - {b:<6}: {d.mean():+.4f}  median {np.median(d):+.4f}  "
              f"wins {int((d>0).sum()):>3}/{d.size}  t-p={p:.4f}  wilcoxon-p={pw:.4f}")

    print("\n" + "-" * 78)
    print(f"  TOP {args.top} V2p+L1 FEATURES BY GAIN (pruned: aggregates removed)")
    print("-" * 78)
    new = set(V2_CHEMISTRY_COLUMNS) - set(V1_CHEMISTRY)
    g = arms["V2p+L1"]["gains"]
    for i, (k, v) in enumerate(sorted(g.items(), key=lambda x: -x[1])[:args.top], 1):
        print(f"    {i:>2}. {k:<26}{v:>10.2f}" + ("  <- NEW in V2" if k in new else ""))
    tot = sum(g.values())
    print(f"\n  new-in-V2 share of total gain: "
          f"{100*sum(v for k,v in g.items() if k in new)/tot:.1f}%")

    print("\n  FEATURES NEVER USED IN ANY SPLIT (the honest read of 'crushed to 0')")
    for name in ("V2", "V2+L1", "V2p", "V2p+L1"):
        u = arms[name]["unused"]
        n = arms[name]["n_feats"]
        print(f"    {name:<6}: mean {u.mean():.2f} of {n} features unused "
              f"(min {u.min()}, max {u.max()})")
    never = [f for f in v2p_feats if f not in g]
    print(f"    V2p+L1 features with zero gain in EVERY run: "
          f"{never if never else 'none -- every feature was used somewhere'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
