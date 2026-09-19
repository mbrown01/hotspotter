"""Reproduce the reported metrics for the accepted model.

Everything in the README comes out of this script. One protocol, stated once:

    labels      SKEMPI 2.0 single mutations, X->Ala only, hot spot = ddG >= 2.0 kcal/mol
    splitting   5 folds grouped BY COMPLEX, so no residue is scored by a model that has
                seen another residue from the same interface. Random residue splits leak
                badly here -- interface residues share burial and partner context -- and
                are the main reason published numbers vary so much.
    folds       read from data/split_alanine.json when present, so these numbers are
                directly comparable to every other experiment in the repo; otherwise a
                fresh GroupKFold is used and the output says so.
    seeds       5, averaged, with the spread reported

Both the regularized model and the same model without L1 are run, because the L1 term is
the single largest accepted improvement and its contribution should be reproducible rather
than asserted.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --strategy max --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from hotspotter.ml.features import XGB_FEATURES, load_labeled_table  # noqa: E402

FEATURES = list(XGB_FEATURES)

#: Accepted configuration. reg_alpha=50 is the L1 term selected by a sweep; everything else
#: matches the baseline the graph model was compared against, so the two stay commensurable.
XGB_KWARGS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", n_jobs=-1)


def get_folds(df, split_path: Path, n_splits: int):
    """(name, list of (train_mask, test_mask)) — reuse the saved split when it fits."""
    pdb = df["pdb_id"].values
    if split_path.exists():
        saved = json.loads(split_path.read_text())["cv_folds"]
        covered = set()
        for f in saved:
            covered |= set(f["test_complexes"])
        if set(pdb) <= covered:
            return f"{split_path.name} ({len(saved)} folds)", [
                (np.isin(pdb, f["train_complexes"]), np.isin(pdb, f["test_complexes"]))
                for f in saved
            ]
    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(df))
    out = []
    for tr, te in GroupKFold(n_splits=n_splits).split(idx, df["label"].values, pdb):
        m_tr = np.zeros(len(df), bool); m_tr[tr] = True
        m_te = np.zeros(len(df), bool); m_te[te] = True
        out.append((m_tr, m_te))
    return f"fresh GroupKFold ({n_splits} folds)", out


def cross_validate(df, folds, alpha: float, seeds):
    """Returns (per-fold PR-AUC matrix [seed, fold], mean out-of-fold probability)."""
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    pf = np.zeros((len(seeds), len(folds)))
    oof = np.zeros((len(seeds), len(df)))
    for si, seed in enumerate(seeds):
        for fi, (tr, te) in enumerate(folds):
            pos = max(1, int(y[tr].sum()))
            m = XGBClassifier(reg_alpha=alpha, random_state=seed,
                              scale_pos_weight=(tr.sum() - y[tr].sum()) / pos,
                              **XGB_KWARGS).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            oof[si, te] = p
            pf[si, fi] = average_precision_score(y[te], p)
    return pf, oof.mean(0)


def ranking_table(df, score, ks=(1, 3, 5)):
    """Per-complex hit rate and precision@k.

    Hit rate answers "would a biologist picking k residues to mutate hit a real hot spot?",
    which is the question the tool is actually for. Precision@k answers "what fraction of
    those k were right?". They are different numbers and are reported separately because
    conflating them overstates the result.
    """
    y = df["label"].values
    out = {}
    for k in ks:
        rows = []
        for _, idx in df.groupby("pdb_id").indices.items():
            if y[idx].sum() == 0 or len(idx) < k:
                continue
            top = idx[np.argsort(-score[idx])[:k]]
            rows.append((float(y[top].sum() > 0), y[top].mean()))
        a = np.array(rows)
        out[k] = (a[:, 0].mean(), a[:, 1].mean(), len(a))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "data" / "split_alanine.json")
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--nonhot", type=float, default=0.4,
                    help="literature definition: ddG below this is a confident non-hot-spot")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: missing {args.csv}\n"
              f"Build it first:  python scripts/build_features.py", file=sys.stderr)
        return 2

    from sklearn.metrics import average_precision_score, roc_auc_score

    df = load_labeled_table(args.csv, args.strategy, args.threshold)
    y = df["label"].astype(int).values
    naive = df["dsasa"].fillna(0).values
    seeds = list(range(args.seeds))
    fold_name, folds = get_folds(df, args.split, args.folds)

    print("=" * 78)
    print(f"  HOTSPOTTER EVALUATION  |  strategy={args.strategy}  |  hot spot: "
          f"ddG >= {args.threshold} kcal/mol")
    print("=" * 78)
    print(f"  {len(df)} residues | {df.pdb_id.nunique()} complexes | "
          f"{int(y.sum())} hot ({y.mean():.1%}) | {len(FEATURES)} features")
    print(f"  splits: {fold_name}, grouped by complex | {args.seeds} seeds")

    naive_pf = np.array([average_precision_score(y[te], naive[te]) for _, te in folds])
    results = {}
    for alpha, tag in ((0.0, "XGBoost (no L1)"), (50.0, "XGBoost + L1 [accepted]")):
        results[tag] = cross_validate(df, folds, alpha, seeds)

    print("\n" + "-" * 78)
    print(f"  {'model':<26}{'PR-AUC':>10}{'sd':>9}{'ROC-AUC':>10}")
    print("-" * 78)
    print(f"  {'naive buried area':<26}{naive_pf.mean():>10.4f}{naive_pf.std():>9.4f}"
          f"{roc_auc_score(y, naive):>10.4f}")
    for tag, (pf, oof) in results.items():
        print(f"  {tag:<26}{pf.mean():>10.4f}{pf.std(axis=1).mean():>9.4f}"
              f"{roc_auc_score(y, oof):>10.4f}")
    a0 = results["XGBoost (no L1)"][0].mean()
    a1 = results["XGBoost + L1 [accepted]"][0].mean()
    print(f"\n  L1 contribution: {a1 - a0:+.4f} PR-AUC")
    print(f"  lift over naive: {a1 - naive_pf.mean():+.4f} PR-AUC "
          f"({100 * (a1 / naive_pf.mean() - 1):.0f}% relative)")

    best = results["XGBoost + L1 [accepted]"][1]
    print("\n" + "-" * 78)
    print("  PER-COMPLEX RANKING  (can a biologist pick k residues and hit a hot spot?)")
    print("-" * 78)
    hp, hn = ranking_table(df, best), ranking_table(df, naive)
    print(f"  {'k':>3}{'hit-rate':>12}{'precision@k':>14}{'naive hit-rate':>17}"
          f"{'naive prec@k':>15}{'n':>6}")
    for k in (1, 3, 5):
        print(f"  {k:>3}{hp[k][0]:>12.1%}{hp[k][1]:>14.1%}{hn[k][0]:>17.1%}"
              f"{hn[k][1]:>15.1%}{hp[k][2]:>6}")

    # The literature comparison drops the grey zone, because published AUCs are computed
    # this way; reporting our all-rows AUC against their filtered AUC would flatter us.
    lit = (df["ddg"] > args.threshold) | (df["ddg"] < args.nonhot)
    yl = (df["ddg"] > args.threshold).astype(int).values[lit.values]
    print("\n" + "-" * 78)
    print(f"  LITERATURE DEFINITION  (hot > {args.threshold}, non-hot < {args.nonhot}, "
          f"grey zone dropped)")
    print("-" * 78)
    print(f"  {int(lit.sum())} residues: {int(yl.sum())} hot, {int((yl == 0).sum())} non-hot")
    print(f"  {'model':<26}{'ROC-AUC':>10}{'PR-AUC':>10}")
    print(f"  {'naive buried area':<26}{roc_auc_score(yl, naive[lit.values]):>10.4f}"
          f"{average_precision_score(yl, naive[lit.values]):>10.4f}")
    print(f"  {'XGBoost + L1 [accepted]':<26}{roc_auc_score(yl, best[lit.values]):>10.4f}"
          f"{average_precision_score(yl, best[lit.values]):>10.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
