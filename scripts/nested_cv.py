"""Measure how much of the headline PR-AUC is the hyperparameter sweep finding noise.

``reg_alpha=50`` was originally chosen by sweeping eight values over the same five folds
``scripts/evaluate.py`` reports on. Picking the best of eight on your evaluation folds and
then reporting that value's score is selection bias, and the size of it is measurable
rather than arguable.

NESTED CROSS-VALIDATION IS THE MEASUREMENT. The outer loop is the reporting split, folds
grouped by complex, exactly as evaluate.py uses them. Inside each outer fold, a second
GroupKFold over that fold's TRAINING complexes only picks alpha; the outer test fold is
never consulted. The gap between the two numbers is the optimism.

The chosen alpha is treated as a property of the data, not of the seed, so the inner loop
runs once per outer fold and the result is reused across seeds. Running it per seed would
multiply the cost by five and measure the same thing.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\nested_cv.py
    .\\.venv\\Scripts\\python.exe scripts\\nested_cv.py --seeds 2    # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402

from evaluate import XGB_KWARGS, get_folds                            # noqa: E402
from hotspotter.ml.features import XGB_FEATURES, load_labeled_table   # noqa: E402

#: The grid the original sweep used. Reported in full at the end, because the point of
#: this script is to show what a sweep does, not to hide it.
ALPHAS = (0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SHIPPED_ALPHA = 50.0


def fit_score(X, y, tr, te, alpha: float, seed: int) -> float:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    pos = max(1, int(y[tr].sum()))
    m = XGBClassifier(reg_alpha=alpha, random_state=seed,
                      scale_pos_weight=(y[tr].size - y[tr].sum()) / pos,
                      **XGB_KWARGS).fit(X[tr], y[tr])
    return float(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))


def pick_alpha(X, y, groups, tr, inner_splits: int) -> float:
    """The best alpha on an inner GroupKFold of the outer fold's TRAINING complexes."""
    from sklearn.model_selection import GroupKFold

    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    best_score, best_alpha = -1.0, SHIPPED_ALPHA
    for a in ALPHAS:
        scores = []
        for itr, ite in GroupKFold(n_splits=inner_splits).split(Xtr, ytr, gtr):
            if len(set(ytr[ite])) < 2:      # a fold with one class scores nothing
                continue
            m_tr = np.zeros(len(ytr), bool); m_tr[itr] = True
            m_te = np.zeros(len(ytr), bool); m_te[ite] = True
            scores.append(fit_score(Xtr, ytr, m_tr, m_te, a, 0))
        if scores and np.mean(scores) > best_score:
            best_score, best_alpha = float(np.mean(scores)), a
    return best_alpha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features.csv")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "data" / "split_alanine.json")
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: missing {args.csv}\n"
              f"Build it first:  python scripts/build_features.py", file=sys.stderr)
        return 2

    from scipy import stats

    df = load_labeled_table(args.csv, args.strategy, args.threshold)
    X = df[list(XGB_FEATURES)]
    y = df["label"].astype(int).values
    groups = df["pdb_id"].values
    fold_name, folds = get_folds(df, args.split, args.folds)

    print("=" * 78)
    print("  NESTED CROSS-VALIDATION  |  what tuning on the reporting folds was worth")
    print("=" * 78)
    print(f"  {len(df)} residues | {df.pdb_id.nunique()} complexes | "
          f"{int(y.sum())} hot | {len(XGB_FEATURES)} features")
    print(f"  outer: {fold_name}, grouped by complex | {args.seeds} seeds")
    print(f"  inner: GroupKFold({args.inner_folds}) over each outer fold's training "
          f"complexes | grid {list(ALPHAS)}")

    print("\n  picking alpha inside each outer fold...", flush=True)
    chosen = [pick_alpha(X, y, groups, tr, args.inner_folds) for tr, _ in folds]
    print(f"  inner loop chose: {chosen}")

    fixed, nested = [], []
    for seed in range(args.seeds):
        for fi, (tr, te) in enumerate(folds):
            fixed.append(fit_score(X, y, tr, te, SHIPPED_ALPHA, seed))
            nested.append(fit_score(X, y, tr, te, chosen[fi], seed))
    fixed, nested = np.array(fixed), np.array(nested)

    gap = fixed.mean() - nested.mean()
    p = stats.ttest_rel(fixed, nested).pvalue
    print("\n" + "-" * 78)
    print(f"  {'estimate':<44}{'PR-AUC':>10}{'sd':>9}")
    print("-" * 78)
    print(f"  {f'fixed reg_alpha={SHIPPED_ALPHA:g} (what evaluate.py reports)':<44}"
          f"{fixed.mean():>10.4f}{fixed.std():>9.4f}")
    print(f"  {'nested, alpha picked inside the training folds':<44}"
          f"{nested.mean():>10.4f}{nested.std():>9.4f}")
    print(f"\n  optimism from tuning on the reporting folds: {gap:+.4f} PR-AUC "
          f"(paired p = {p:.4f})")
    print("  " + ("negligible -- the reported number stands as is" if abs(gap) < 0.005
                  else "material -- disclose the nested number alongside the reported one"))

    # The sweep itself, so the reader can see what was being chosen between.
    print("\n" + "-" * 78)
    print("  THE SWEEP THAT CAUSED THE PROBLEM  (scored on the reporting folds, 1 seed)")
    print("-" * 78)
    for a in ALPHAS:
        s = np.mean([fit_score(X, y, tr, te, a, 0) for tr, te in folds])
        mark = "  <- shipped" if a == SHIPPED_ALPHA else ""
        print(f"    reg_alpha {a:>6g}: {s:.4f}{mark}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
