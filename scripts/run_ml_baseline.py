"""Smoke-run the Phase-2 baseline: SKEMPI -> features -> XGBoost, on a few complexes.

This is the FIRST real run of the Phase-2 path, so it is deliberately small
(``--limit`` complexes) and deliberately loud: the point is to surface column-mapping,
chain-mapping, and residue-matching bugs cheaply, not to produce a headline number.

Reported metrics, and why all three matter together:

    positive class rate   the fraction of rows labeled disruptive. PR-AUC's random-chance
                          floor EQUALS this number, so a PR-AUC is uninterpretable without
                          it. Changing the ddG threshold changes this, and therefore
                          changes what a "good" PR-AUC even looks like.
    naive SASA PR-AUC     rank by buried surface area alone. This is the strawman the whole
                          project exists to beat.
    XGBoost PR-AUC        the trained model on the same held-out complexes.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\run_ml_baseline.py
    .\\.venv\\Scripts\\python.exe scripts\\run_ml_baseline.py --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hotspotter.ml.dataset import build_dataset          # noqa: E402
from hotspotter.ml.train import select_features, train_baseline  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "skempi_v2.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help=f"SKEMPI 2.0 csv (default: {DEFAULT_CSV})")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap distinct complexes (default: no cap = full run). Use e.g. 20 to smoke-test.")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="ddG (kcal/mol) above which a mutation is 'disruptive' (default: 2.0)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "skempi_features_full.csv",
                    help="where to save the built feature table")
    ap.add_argument("--reuse", action="store_true",
                    help="load the feature table from --out instead of rebuilding (skips hours of downloads)")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: SKEMPI csv not found at {args.csv}", file=sys.stderr)
        return 2

    scope = f"limit={args.limit} complexes" if args.limit else "FULL DATASET (no cap)"
    print("=" * 78)
    print(f"  Phase-2 run  |  {scope}  |  ddG threshold={args.threshold}")
    print("=" * 78)

    # ---- 1. Build the labeled feature table -----------------------------------------
    # Each new complex is downloaded from RCSB and run through the Phase-1 pipeline,
    # so this is the slow step (hours for the full set). build_dataset prints progress
    # and a final skip/keep summary.
    if args.reuse and args.out.exists():
        import pandas as pd
        df = pd.read_csv(args.out)
        print(f"\nLoaded cached feature table from {args.out} ({len(df)} rows).")
    else:
        df = build_dataset(
            args.csv,
            ddg_disruptive_threshold=args.threshold,
            only_single_mutations=True,   # clean 1-to-1 residue->ddG labels; see docs
            limit_complexes=args.limit,
        )

        # Save IMMEDIATELY — this table costs hours to rebuild, and we don't want a
        # downstream training error to throw it away.
        if not df.empty:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"\nSaved feature table -> {args.out}  ({len(df)} rows)")
            print(f"  Re-run training without rebuilding:  --reuse")

    if df.empty:
        print("\nRESULT: build_dataset returned ZERO rows. Nothing to train on.")
        print("Check the skip counts above to see where every row was lost.")
        return 1

    # ---- 2. Class balance -------------------------------------------------------------
    n_pos = int(df["label"].sum())
    n_neg = int(len(df) - n_pos)
    pos_rate = n_pos / len(df)
    n_complexes = df["complex_group"].nunique()

    print("\n" + "-" * 78)
    print("  CLASS BALANCE")
    print("-" * 78)
    print(f"  rows (usable mutations) : {len(df)}")
    print(f"  distinct complexes      : {n_complexes}")
    print(f"  positive (hot spot)     : {n_pos}")
    print(f"  negative (not hot spot) : {n_neg}")
    print(f"  POSITIVE CLASS RATE     : {pos_rate:.4f}   <-- PR-AUC chance floor")
    print(f"  features used           : {len(select_features(df))}")

    if n_pos == 0 or n_neg == 0:
        print("\nRESULT: only one class present. Cannot train a classifier.")
        print(f"Try a lower --threshold (currently {args.threshold}) or a larger --limit.")
        return 1

    # ---- 3. Train with split-by-complex ----------------------------------------------
    try:
        result = train_baseline(df)
    except Exception as exc:
        print(f"\nRESULT: training FAILED: {type(exc).__name__}: {exc}")
        return 1

    m = result.metrics
    print("\n" + "-" * 78)
    print("  HELD-OUT RESULTS  (split by complex — no PDB in both train and test)")
    print("-" * 78)
    print(f"  train rows / complexes  : {m['n_train']} / {m['n_train_complexes']}")
    print(f"  test  rows / complexes  : {m['n_test']} / {m['n_test_complexes']}")
    print(f"  test positive rate      : {m['test_positive_rate']:.4f}   <-- chance floor for the scores below")
    print()
    naive = m.get("naive_pr_auc")
    print(f"  NAIVE SASA PR-AUC       : {naive:.4f}" if naive is not None
          else "  NAIVE SASA PR-AUC       : n/a (no 'dsasa' column)")
    print(f"  XGBOOST PR-AUC          : {m['pr_auc']:.4f}")
    print(f"  (xgboost ROC-AUC        : {m['roc_auc']:.4f})")

    if naive is not None:
        delta = m["pr_auc"] - naive
        verdict = "BEATS" if delta > 0 else "DOES NOT BEAT"
        print(f"\n  Model {verdict} the naive burial baseline by {delta:+.4f} PR-AUC.")

    print("\n  Top 10 features by importance:")
    for _, r in result.importances.head(10).iterrows():
        print(f"    {r['feature']:<24s} {r['importance']:.4f}")

    # ---- 4. Honesty guard -------------------------------------------------------------
    if m["n_test_complexes"] < 5 or m["n_test"] < 30:
        print("\n  WARNING: the test set is tiny. These numbers are a smoke-test signal that")
        print("  the plumbing works, NOT a baseline worth recording. Re-run without --limit")
        print("  (or with a much larger one) before quoting any of these figures.")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
