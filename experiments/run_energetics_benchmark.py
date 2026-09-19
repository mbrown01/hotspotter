"""Does implicit-solvent energetics beat the geometric proxies it is meant to replace?

THE QUESTION
    Every V1 feature is a geometric stand-in for energy: buried area, relative accessibility,
    packing density. They work because burial and binding energy are correlated, not because
    anything thermodynamic was computed. data/energetics.csv adds terms that ARE energetic --
    Coulomb interaction, Born desolvation, and a nonpolar surface term -- from real per-atom
    AMBER charges and PROPKA protonation states.

    Eight previous feature additions failed to beat V1+L1. This asks whether real physics
    does what more geometry could not.

WHY THERE ARE THREE ARMS AND NOT TWO
    e_nonpolar is defined as -gamma * dsasa. It is a linear rescale of a column the model
    already has, so a tree model gains exactly nothing from it -- and e_total contains it.
    Handing the model all six terms and then reporting a win would let two redundant columns
    stand next to four real ones and share the credit.

        baseline    V1 + L1
        +energy     all six energetics columns
        +novel      only the four that are not functions of dsasa:
                    e_coulomb, e_desolvation, charge_buried, n_charged_contacts

    If +energy and +novel score the same, the nonpolar terms contributed nothing, which is
    what should happen and is worth showing rather than assuming.

HOW THE COMPARISON IS KEPT HONEST
    - Identical rows in every arm: the 96.8% of feature rows that joined to energetics,
      intersected across arms, so no arm is scored on an easier subset.
    - Identical GroupKFold folds grouped on complex, identical seeds, identical
      hyperparameters including reg_alpha=50 (the L1 setting that is part of the baseline).
    - Paired by (fold, seed), because fold-to-fold variance is far larger than the effect
      being measured and unpaired means would drown it.
    - Bootstrap over COMPLEXES, not residues: residues in one interface are correlated, so
      resampling residues would understate the interval.

    Both the project's own label definition (ddG >= 2.0 is hot, everything else is not) and
    the literature definition used for external comparison (hot > 2.0, non-hot < 0.4, the
    grey zone between them dropped) are reported, since the published AUC numbers this
    project is measured against all use the second.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\run_energetics_benchmark.py
    .\\.venv\\Scripts\\python.exe scripts\\run_energetics_benchmark.py --seeds 5
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

from energetics import ENERGY_COLUMNS  # noqa: E402
from hotspotter.ml.dataset import parse_mutation  # noqa: E402
from hotspotter.ml.features import XGB_FEATURES, collapse  # noqa: E402

BASE_FEATURES = list(XGB_FEATURES)

#: The energetics columns that are NOT a deterministic function of dsasa.
NOVEL_ENERGY = ["e_coulomb", "e_desolvation", "charge_buried", "n_charged_contacts"]

XGB_KWARGS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=50,
                  eval_metric="aucpr", n_jobs=-1)


def _norm_icode(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace("nan", "")


def oof_predictions(df: pd.DataFrame, features: list[str], seeds: list[int],
                    n_splits: int) -> np.ndarray:
    """Out-of-fold probabilities, shape (n_seeds, n_rows).

    Out-of-fold rather than per-fold scores because the two evaluations below (project
    threshold and literature definition) need to slice the same predictions differently,
    and a per-fold mean cannot be re-sliced after the fact.
    """
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    X = df[features].fillna(df[features].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    groups = df["pdb_id"].values
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    oof = np.zeros((len(seeds), len(df)))
    for si, seed in enumerate(seeds):
        for tr, te in folds:
            pos = max(1, int(y[tr].sum()))
            m = XGBClassifier(scale_pos_weight=(len(tr) - y[tr].sum()) / pos,
                              random_state=seed, **XGB_KWARGS)
            m.fit(X.iloc[tr], y[tr])
            oof[si, te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof, folds


def per_fold_scores(y: np.ndarray, oof: np.ndarray, folds) -> np.ndarray:
    """PR-AUC per (seed, fold) so arms can be paired on identical test sets."""
    from sklearn.metrics import average_precision_score
    out = np.zeros((oof.shape[0], len(folds)))
    for si in range(oof.shape[0]):
        for fi, (_, te) in enumerate(folds):
            out[si, fi] = average_precision_score(y[te], oof[si, te])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--energetics", type=Path, default=REPO_ROOT / "data" / "energetics.csv")
    ap.add_argument("--holdout", nargs="*", default=["1BRS"])
    ap.add_argument("--strategy", choices=("alanine", "max"), default="alanine")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--nonhot", type=float, default=0.4,
                    help="literature definition: ddG below this is a confident non-hot-spot")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    from scipy import stats
    from sklearn.metrics import average_precision_score, roc_auc_score

    for p in (args.csv, args.energetics):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    hold = {h.upper() for h in args.holdout}
    df = pd.read_csv(args.csv)
    df["mut"] = df["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    df["pdb_id"] = df["complex_group"]
    df = df[~df["pdb_id"].str.upper().isin(hold)]
    df = collapse(df, args.strategy, args.threshold)

    en = pd.read_csv(args.energetics)
    en["icode"] = _norm_icode(en["icode"])
    df["icode"] = _norm_icode(df["icode"])
    key = ["pdb_id", "chain", "resseq", "icode"]
    en["resseq"] = en["resseq"].astype(int)
    df["resseq"] = df["resseq"].astype(int)
    en = en.drop_duplicates(subset=key)

    before = len(df)
    df = df.merge(en[key + list(ENERGY_COLUMNS)], on=key, how="inner")

    print("=" * 78)
    print(f"  ENERGETICS BENCHMARK  |  strategy={args.strategy}  |  held out {sorted(hold)}")
    print("=" * 78)
    print(f"  rows {before} -> {len(df)} after joining energetics "
          f"({len(df)/before:.1%} kept; unjoined rows dropped from ALL arms)")
    print(f"  complexes {df.pdb_id.nunique()} | positives {int(df.label.sum())} "
          f"({df.label.mean():.1%}) | seeds {args.seeds} | folds {args.folds}")

    arms = {
        "V1 + L1  (baseline)": BASE_FEATURES,
        "+ energetics (all 6)": BASE_FEATURES + list(ENERGY_COLUMNS),
        "+ novel energy (4)": BASE_FEATURES + NOVEL_ENERGY,
    }
    seeds = list(range(args.seeds))
    y = df["label"].astype(int).values

    results = {}
    folds = None
    for name, feats in arms.items():
        feats = [f for f in feats if f in df.columns]
        oof, folds = oof_predictions(df, feats, seeds, args.folds)
        results[name] = (oof, per_fold_scores(y, oof, folds), len(feats))

    print("\n" + "-" * 78)
    print(f"  PROJECT LABEL DEFINITION  (hot: ddG >= {args.threshold}; all other rows negative)")
    print("-" * 78)
    print(f"    {'arm':<24}{'n feat':>8}{'PR-AUC':>10}{'sd':>9}{'ROC-AUC':>10}")
    naive = average_precision_score(y, df["dsasa"].fillna(0).values)
    print(f"    {'naive dsasa ranking':<24}{'-':>8}{naive:>10.4f}{'-':>9}"
          f"{roc_auc_score(y, df['dsasa'].fillna(0).values):>10.4f}")
    for name, (oof, pf, nf) in results.items():
        print(f"    {name:<24}{nf:>8}{pf.mean():>10.4f}{pf.std():>9.4f}"
              f"{roc_auc_score(y, oof.mean(0)):>10.4f}")

    base_pf = results["V1 + L1  (baseline)"][1]
    print("\n  PAIRED BY (seed, fold) -- the only comparison that controls fold variance")
    for name, (_, pf, _) in results.items():
        if name.startswith("V1"):
            continue
        d = (pf - base_pf).ravel()
        t, p = stats.ttest_rel(pf.ravel(), base_pf.ravel())
        print(f"    {name:<24} delta {d.mean():+.4f}   wins {int((d>0).sum())}/{d.size}"
              f"   t={t:+.3f}  p={p:.4g}")

    # ---- bootstrap over complexes: the honest interval -----------------------------------
    print("\n  BOOTSTRAP OVER COMPLEXES (n={}), delta PR-AUC vs baseline".format(args.boot))
    pdbs = df["pdb_id"].values
    uniq = np.unique(pdbs)
    idx_by_pdb = {c: np.flatnonzero(pdbs == c) for c in uniq}
    rng = np.random.default_rng(0)
    picks = [np.concatenate([idx_by_pdb[c]
                             for c in rng.choice(uniq, size=len(uniq), replace=True)])
             for _ in range(args.boot)]
    base_oof = results["V1 + L1  (baseline)"][0].mean(0)
    for name, (oof, _, _) in results.items():
        if name.startswith("V1"):
            continue
        p_arm = oof.mean(0)
        deltas = []
        for idx in picks:
            if len(set(y[idx])) < 2:
                continue
            deltas.append(average_precision_score(y[idx], p_arm[idx])
                          - average_precision_score(y[idx], base_oof[idx]))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"    {name:<24}{deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   wins {100*(deltas>0).mean():.1f}%")

    # ---- literature definition: drop the grey zone ---------------------------------------
    lit = (df["ddg"] > args.threshold) | (df["ddg"] < args.nonhot)
    ylit = (df["ddg"] > args.threshold).astype(int).values[lit.values]
    print("\n" + "-" * 78)
    print(f"  LITERATURE DEFINITION  (hot > {args.threshold}, non-hot < {args.nonhot}, "
          f"grey zone dropped)")
    print("-" * 78)
    print(f"  {int(lit.sum())} of {len(df)} rows survive: {int(ylit.sum())} hot, "
          f"{int((ylit==0).sum())} non-hot")
    print(f"    {'arm':<24}{'ROC-AUC':>10}{'PR-AUC':>10}")
    dn = df.loc[lit.values, "dsasa"].fillna(0).values
    print(f"    {'naive dsasa ranking':<24}{roc_auc_score(ylit, dn):>10.4f}"
          f"{average_precision_score(ylit, dn):>10.4f}")
    for name, (oof, _, _) in results.items():
        p = oof.mean(0)[lit.values]
        print(f"    {name:<24}{roc_auc_score(ylit, p):>10.4f}"
              f"{average_precision_score(ylit, p):>10.4f}")
    print("\n  REFERENCE: V1+L1 previously scored AUC 0.8392 here; published methods "
          "report ~0.9468")

    # ---- where did the model actually put the energy terms? ------------------------------
    from xgboost import XGBClassifier
    feats = [f for f in BASE_FEATURES + NOVEL_ENERGY if f in df.columns]
    X = df[feats].fillna(df[feats].median(numeric_only=True)).fillna(0.0)
    m = XGBClassifier(scale_pos_weight=(len(y) - y.sum()) / max(1, y.sum()),
                      random_state=0, **XGB_KWARGS).fit(X, y)
    gains = m.get_booster().get_score(importance_type="gain")
    print("\n" + "-" * 78)
    print("  FEATURE GAIN, +novel arm fitted on all rows (ranking only, not a score)")
    print("-" * 78)
    for i, (k, v) in enumerate(sorted(gains.items(), key=lambda x: -x[1])[:12], 1):
        tag = "  <- energetics" if k in NOVEL_ENERGY else ""
        print(f"    {i:>2}. {k:<24}{v:>10.2f}{tag}")
    tot = sum(gains.values()) or 1.0
    share = sum(v for k, v in gains.items() if k in NOVEL_ENERGY) / tot
    print(f"\n  energetics carry {100*share:.1f}% of total gain")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
