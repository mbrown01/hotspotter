"""Why does nothing improve this model? Four measurements that separate the explanations.

THE OBSERVATION THIS EXISTS TO EXPLAIN
    Ten interventions were tried against the V1+L1 tabular baseline. Two helped: a Wide &
    Deep skip connection (+0.033) and L1 regularization (+0.015). Both are changes to the
    MODEL. Eight failed: feature cleanup, XGBoost stacking, ESM-2 320-d embeddings, an ESM
    conservation scalar at 8M and at 650M, atom-level V2 chemistry, ESM-650M embeddings
    under every PCA variant, a GNN/XGBoost ensemble, and implicit-solvent energetics. Every
    one of those eight adds INFORMATION. None of them moved the score.

    A pattern that clean is not eight unrelated accidents. This script tests the four
    explanations that would each produce it, because they imply different next moves:

      1 DATA-LIMITED     163 complexes is too few to fit anything richer.
                         -> learning curve over complexes. If it is still climbing at 100%,
                            more labels are the fix and no feature will substitute.

      2 REDUNDANCY       every feature is a restatement of "how buried is this residue", so
                         column 30 is a linear combination of columns 1-29.
                         -> effective dimensionality, and how much of the score survives
                            when the feature matrix is crushed to its first few components.

      3 LABEL NOISE      SKEMPI ddG values come from different laboratories, assays and
                         temperatures. A hard threshold at 2.0 kcal/mol on a quantity with
                         experimental spread turns borderline residues into coin flips.
                         -> measure the spread directly on mutations SKEMPI records more
                            than once from independent references, and check whether the
                            model's errors concentrate where that spread matters.

      4 POWER            the improvements are real but around +0.005, and this protocol
                         cannot resolve an effect that small.
                         -> minimum detectable effect from the observed fold-to-fold spread.

    Explanations are not exclusive. The output is meant to show which ones are large.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\diagnose_ceiling.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotspotter.ml.dataset import parse_mutation  # noqa: E402
from run_v2_benchmark import SHARED_NON_CHEMISTRY, V1_CHEMISTRY, collapse  # noqa: E402

FEATURES = SHARED_NON_CHEMISTRY + V1_CHEMISTRY
XGB_KWARGS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=50, eval_metric="aucpr", n_jobs=-1)

#: Gas constant in kcal/(mol*K), for ddG = RT ln(Kd_mut / Kd_wt).
R_KCAL = 0.0019872041


def fit_score(Xtr, ytr, Xte, yte, seed):
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier
    pos = max(1, int(ytr.sum()))
    m = XGBClassifier(scale_pos_weight=(len(ytr) - ytr.sum()) / pos,
                      random_state=seed, **XGB_KWARGS)
    m.fit(Xtr, ytr)
    return average_precision_score(yte, m.predict_proba(Xte)[:, 1])


# ---------------------------------------------------------------------------------------
# 1. LEARNING CURVE
# ---------------------------------------------------------------------------------------
def learning_curve(df, X, y, seeds, n_splits, fractions):
    """PR-AUC against training-set size, with the TEST folds held fixed.

    Subsampling both train and test would move the target while measuring it. Here the five
    grouped test folds are fixed once, and only the training complexes are thinned, so every
    point on the curve is scored on identical rows.
    """
    from sklearn.model_selection import GroupKFold
    groups = df["pdb_id"].values
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    print("\n" + "=" * 78)
    print("  1. IS IT DATA-LIMITED?   learning curve, test folds held fixed")
    print("=" * 78)
    print(f"    {'train complexes':>17}{'PR-AUC':>10}{'sd':>9}{'vs full':>10}")
    out = {}
    for frac in fractions:
        scores, ntrain = [], []
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(1000 + seed)
            for tr, te in folds:
                tr_groups = np.unique(groups[tr])
                k = max(2, int(round(frac * len(tr_groups))))
                keep = set(rng.choice(tr_groups, size=k, replace=False))
                sub = tr[np.isin(groups[tr], list(keep))]
                if y[sub].sum() < 2 or len(set(y[sub])) < 2:
                    continue
                ntrain.append(k)
                scores.append(fit_score(X.iloc[sub], y[sub], X.iloc[te], y[te], seed))
        out[frac] = np.array(scores)
    full = out[max(fractions)].mean()
    for frac in fractions:
        s = out[frac]
        approx = int(round(frac * len(np.unique(groups)) * (n_splits - 1) / n_splits))
        print(f"    {approx:>17}{s.mean():>10.4f}{s.std():>9.4f}{s.mean()-full:>+10.4f}")

    # Is the curve flat at the top? Compare the last two points on paired folds.
    a, b = out[fractions[-2]], out[fractions[-1]]
    n = min(len(a), len(b))
    from scipy import stats
    t, p = stats.ttest_rel(b[:n], a[:n])
    print(f"\n    last step ({fractions[-2]:.0%} -> {fractions[-1]:.0%}): "
          f"{b.mean()-a.mean():+.4f}  t={t:+.3f}  p={p:.4g}")
    print("    a curve still climbing at 100% means labels are the binding constraint;")
    print("    a flat top means the model has already extracted what the data contains.")
    return out


# ---------------------------------------------------------------------------------------
# 2. FEATURE REDUNDANCY
# ---------------------------------------------------------------------------------------
def redundancy(df, X, y, seeds, n_splits):
    """How many independent things do 25 features actually measure?"""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 78)
    print("  2. IS IT REDUNDANCY?   effective dimensionality of the feature matrix")
    print("=" * 78)
    Z = StandardScaler().fit_transform(X.values)
    ev = PCA().fit(Z).explained_variance_ratio_
    cum = np.cumsum(ev)
    for thresh in (0.50, 0.80, 0.90, 0.95, 0.99):
        print(f"    components for {thresh:.0%} of variance: "
              f"{int(np.searchsorted(cum, thresh) + 1):>3} of {len(ev)}")
    print(f"    first component alone explains {ev[0]:.1%}")

    # correlation of every feature with dsasa: the "everything is burial" check
    corr = X.corrwith(df["dsasa"]).abs().sort_values(ascending=False)
    strong = corr[corr > 0.7].drop(labels=["dsasa"], errors="ignore")
    print(f"\n    features correlated |r| > 0.7 with dsasa: {len(strong)} of {len(X.columns)}")
    for k, v in strong.items():
        print(f"      {k:<24}{v:>7.3f}")

    # does the score survive compression? if 3 components reproduce it, the other 22
    # features were carrying nothing the trees could use.
    groups = df["pdb_id"].values
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    print(f"\n    {'inputs':<28}{'PR-AUC':>10}{'sd':>9}")
    for k in (1, 2, 3, 5, 10, len(X.columns)):
        scores = []
        for seed in seeds:
            for tr, te in folds:
                sc = StandardScaler().fit(X.iloc[tr].values)
                if k == len(X.columns):
                    Ptr, Pte = X.iloc[tr].values, X.iloc[te].values
                else:
                    pca = PCA(n_components=k).fit(sc.transform(X.iloc[tr].values))
                    Ptr = pca.transform(sc.transform(X.iloc[tr].values))
                    Pte = pca.transform(sc.transform(X.iloc[te].values))
                scores.append(fit_score(pd.DataFrame(Ptr), y[tr], pd.DataFrame(Pte), y[te], seed))
        s = np.array(scores)
        tag = "all raw features" if k == len(X.columns) else f"top {k} principal component" + ("s" if k > 1 else "")
        print(f"    {tag:<28}{s.mean():>10.4f}{s.std():>9.4f}")
    print("    if a handful of components matches the full set, the extra columns are")
    print("    restatements and a new column has nowhere orthogonal to land.")


# ---------------------------------------------------------------------------------------
# 3. LABEL NOISE
# ---------------------------------------------------------------------------------------
def _temp(v):
    m = re.search(r"\d+", str(v))
    return float(m.group()) if m else 298.0


def label_noise(skempi_path, df, X, y, seeds, n_splits, verbose=True):
    """Measure experimental spread from SKEMPI's own repeated measurements.

    Also returns out-of-fold predictions, which section 5 reuses rather than refitting.
    """
    import builtins
    print = builtins.print if verbose else (lambda *a, **k: None)
    print("\n" + "=" * 78)
    print("  3. IS IT LABEL NOISE?   SKEMPI's disagreement with itself")
    print("=" * 78)
    sk = pd.read_csv(skempi_path, sep=";", low_memory=False)
    sk = sk[sk["Affinity_mut_parsed"].notna() & sk["Affinity_wt_parsed"].notna()]
    sk = sk[~sk["Mutation(s)_PDB"].astype(str).str.contains(",")]     # single mutations only
    T = sk["Temperature"].map(_temp)
    sk = sk.assign(ddg=R_KCAL * T * np.log(sk["Affinity_mut_parsed"].astype(float)
                                           / sk["Affinity_wt_parsed"].astype(float)))
    sk["pdb"] = sk["#Pdb"].astype(str).str.split("_").str[0].str.upper()

    g = sk.groupby(["pdb", "Mutation(s)_PDB"])
    rep = g.agg(n=("ddg", "size"), refs=("Reference", "nunique"),
                spread=("ddg", lambda s: s.max() - s.min()),
                sd=("ddg", "std"), mean=("ddg", "mean")).reset_index()
    multi = rep[(rep["n"] > 1) & (rep["refs"] > 1)]
    print(f"    single mutations measured more than once by independent references: "
          f"{len(multi)}")
    if len(multi):
        print(f"    median disagreement (max - min) : {multi['spread'].median():.2f} kcal/mol")
        print(f"    mean   disagreement             : {multi['spread'].mean():.2f} kcal/mol")
        print(f"    90th percentile                 : {multi['spread'].quantile(0.9):.2f} kcal/mol")
        flip = ((multi["mean"] - multi["spread"] / 2 < 2.0)
                & (multi["mean"] + multi["spread"] / 2 > 2.0)).mean()
        print(f"    fraction whose spread straddles the 2.0 threshold: {flip:.1%}")
        print("    (those are mutations SKEMPI itself cannot consistently call hot or not)")

    # how much of the dataset sits close enough to 2.0 to be flipped by that spread?
    noise = float(multi["spread"].median()) / 2 if len(multi) else 0.5
    near = ((df["ddg"] - 2.0).abs() < noise).mean()
    print(f"\n    rows within +/-{noise:.2f} kcal/mol of the threshold: {near:.1%} of the set")

    # do the model's errors actually live there?
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier
    groups = df["pdb_id"].values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pos = max(1, int(y[tr].sum()))
        m = XGBClassifier(scale_pos_weight=(len(tr) - y[tr].sum()) / pos,
                          random_state=0, **XGB_KWARGS).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    band = pd.cut((df["ddg"] - 2.0).abs(),
                  [-0.01, 0.5, 1.0, 2.0, 100], labels=["<0.5", "0.5-1", "1-2", ">2"])
    print(f"\n    {'|ddG - 2.0|':<14}{'n':>7}{'hot rate':>11}{'model AUC':>12}")
    from sklearn.metrics import roc_auc_score
    for b in band.cat.categories:
        s = band == b
        if s.sum() < 20 or len(set(y[s.values])) < 2:
            continue
        print(f"    {b:<14}{int(s.sum()):>7}{y[s.values].mean():>11.1%}"
              f"{roc_auc_score(y[s.values], oof[s.values]):>12.4f}")
    print("    near-chance AUC in the innermost band with good AUC outside means the")
    print("    model is right wherever the label is trustworthy.")
    return oof


# ---------------------------------------------------------------------------------------
# 4. STATISTICAL POWER
# ---------------------------------------------------------------------------------------
def power(df, X, y, seeds, n_splits):
    """What size of improvement could this protocol actually have detected?"""
    from scipy import stats
    from sklearn.model_selection import GroupKFold
    groups = df["pdb_id"].values
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    # paired seed-to-seed differences give the noise floor of the protocol itself:
    # two runs of the SAME model on the SAME folds, differing only by seed.
    per = np.array([[fit_score(X.iloc[tr], y[tr], X.iloc[te], y[te], seed)
                     for tr, te in folds] for seed in seeds])
    within = per.std(axis=0).mean()
    across = per.mean(axis=0).std()
    n = per.size
    sd_paired = float(np.std(per - per.mean(axis=0), ddof=1)) or 1e-9
    mde = 2.8 * sd_paired / np.sqrt(n)          # ~80% power, two-sided alpha 0.05

    print("\n" + "=" * 78)
    print("  4. IS IT POWER?   what this protocol can resolve")
    print("=" * 78)
    print(f"    fold-to-fold spread of PR-AUC        : {across:.4f}")
    print(f"    seed-to-seed spread on the same fold : {within:.4f}   <- pure protocol noise")
    print(f"    paired comparisons available         : {n}  ({len(seeds)} seeds x {n_splits} folds)")
    print(f"    smallest detectable gain at 80% power: {mde:+.4f} PR-AUC")
    print(f"\n    measured gains that failed: ensemble +0.0034, energetics +0.0057")
    print(f"    a real improvement below {mde:.4f} cannot be distinguished from seed noise")
    print("    here, so 'failed' means 'too small to prove', not necessarily 'zero'.")
    return mde


# ---------------------------------------------------------------------------------------
# 5. THE CEILING IMPLIED BY THAT NOISE
# ---------------------------------------------------------------------------------------
def noise_ceiling(skempi_path, df, y, oof, threshold, n_sim=400):
    """How well could a PERFECT model score against labels this noisy?

    Section 3 measures how much SKEMPI disagrees with itself. This converts that into the
    number that actually matters: the maximum achievable score. An oracle that knew every
    residue's true ddG exactly would still be graded against thresholded noisy measurements,
    and would still get some of them wrong. That is the ceiling, and no feature can raise it.

    The simulation treats the recorded ddG as truth, perturbs it by the experimental spread
    SKEMPI's own repeated measurements exhibit, re-thresholds, and scores the unperturbed
    ddG against the perturbed labels.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    sk = pd.read_csv(skempi_path, sep=";", low_memory=False)
    sk = sk[sk["Affinity_mut_parsed"].notna() & sk["Affinity_wt_parsed"].notna()]
    sk = sk[~sk["Mutation(s)_PDB"].astype(str).str.contains(",")]
    T = sk["Temperature"].map(_temp)
    sk = sk.assign(ddg=R_KCAL * T * np.log(sk["Affinity_mut_parsed"].astype(float)
                                           / sk["Affinity_wt_parsed"].astype(float)))
    sk["pdb"] = sk["#Pdb"].astype(str).str.split("_").str[0].str.upper()
    grp = sk.groupby(["pdb", "Mutation(s)_PDB"])["ddg"]

    # Pooled within-mutation standard deviation: the spread of independent measurements of
    # the same quantity, which is exactly the experimental error term.
    ss, dof = 0.0, 0
    for _, s in grp:
        if len(s) > 1:
            ss += float(((s - s.mean()) ** 2).sum())
            dof += len(s) - 1
    sigma = float(np.sqrt(ss / dof)) if dof else 0.5

    print("\n" + "=" * 78)
    print("  5. WHAT IS THE CEILING?   best possible score against labels this noisy")
    print("=" * 78)
    print(f"    pooled experimental sd of ddG : {sigma:.3f} kcal/mol  ({dof} d.o.f.)")

    truth = df["ddg"].values
    rng = np.random.default_rng(0)
    pr, roc, flip = [], [], []
    for _ in range(n_sim):
        noisy = truth + rng.normal(0, sigma, size=len(truth))
        ylab = (noisy >= threshold).astype(int)
        if len(set(ylab)) < 2:
            continue
        pr.append(average_precision_score(ylab, truth))    # oracle = the true ddG itself
        roc.append(roc_auc_score(ylab, truth))
        flip.append((ylab != y).mean())
    pr, roc, flip = np.array(pr), np.array(roc), np.array(flip)
    print(f"    labels that flip under that noise : {flip.mean():.1%}")
    print(f"\n    {'':<34}{'PR-AUC':>10}{'ROC-AUC':>10}")
    print(f"    {'perfect oracle (ceiling)':<34}{pr.mean():>10.4f}{roc.mean():>10.4f}")
    cur_pr = average_precision_score(y, oof)
    cur_roc = roc_auc_score(y, oof)
    print(f"    {'this model (out-of-fold)':<34}{cur_pr:>10.4f}{cur_roc:>10.4f}")
    print(f"    {'naive burial':<34}{average_precision_score(y, df['dsasa'].fillna(0)):>10.4f}"
          f"{roc_auc_score(y, df['dsasa'].fillna(0)):>10.4f}")
    gap = pr.mean() - cur_pr
    got = (cur_pr - average_precision_score(y, df["dsasa"].fillna(0)))
    tot = pr.mean() - average_precision_score(y, df["dsasa"].fillna(0))
    print(f"\n    headroom remaining to the ceiling : {gap:+.4f} PR-AUC")
    print(f"    share of the attainable gap closed: {100*got/tot:.0f}%"
          if tot > 0 else "")
    print("    the ceiling is not 1.0 because the target is a measurement, not a constant.")


# ---------------------------------------------------------------------------------------
# 6. DO THE FEATURES DETERMINE THE ANSWER AT ALL?
# ---------------------------------------------------------------------------------------
def information_present(df, X, y, oof, n_splits, k=5):
    """Model-free test: do residues that LOOK identical BEHAVE identically?

    Sections 1-5 rule out too little data, redundant columns, and noisy labels. What is left
    is the possibility that the features simply do not contain the answer -- that two
    residues can be indistinguishable in all 25 columns and still differ by kilocalories.

    This measures that without fitting anything. For each residue, find its nearest
    neighbours in standardized feature space, restricted to OTHER complexes so the neighbour
    is not just an adjacent residue in the same interface, and ask how far apart their ddG
    values are. If look-alikes disagree by far more than the 0.5 kcal/mol experimental error,
    no model of any kind can separate them, because the inputs are identical.

    The train-versus-out-of-fold gap is reported alongside, to distinguish "the information
    is absent" from "the information is there and the model fails to generalize it".
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    print("\n" + "=" * 78)
    print("  6. IS THE ANSWER IN THE FEATURES?   look-alike residues, no model involved")
    print("=" * 78)

    Z = StandardScaler().fit_transform(X.values)
    ddg = np.asarray(df["ddg"], dtype=float)
    # .values on a pandas string column is a StringArray, which does not broadcast
    pdb = np.asarray(df["pdb_id"].astype(str), dtype=object)
    # squared euclidean distance, then mask out same-complex neighbours
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1) if len(Z) < 2500 else None
    if d2 is None:
        raise RuntimeError("dataset too large for the dense neighbour matrix")
    same = pdb[:, None] == pdb[None, :]
    d2[same] = np.inf

    nn = np.argsort(d2, axis=1)[:, :k]
    gap = np.abs(ddg[nn] - ddg[:, None]).mean(1)
    agree = (y[nn] == y[:, None]).mean(1)
    dist = np.sqrt(np.take_along_axis(d2, nn, 1)).mean(1)

    print(f"    each residue matched to its {k} nearest look-alikes in other complexes")
    print(f"    mean feature distance to them   : {dist.mean():.2f} sd-units "
          f"across {X.shape[1]} standardized columns")
    print(f"\n    mean |ddG difference| to look-alikes : {gap.mean():.2f} kcal/mol")
    print(f"    experimental error on ddG            : 0.50 kcal/mol")
    print(f"    ratio                                : {gap.mean()/0.498:.1f}x the "
          f"measurement error")
    print(f"\n    look-alikes sharing the same hot/not label : {agree.mean():.1%}")
    base = y.mean() ** 2 + (1 - y.mean()) ** 2
    print(f"    agreement expected from the base rate alone: {base:.1%}")
    print(f"    agreement a perfect feature set would give : 100.0%")

    # restrict to the very closest look-alikes: does agreement improve as they get closer?
    print(f"\n    {'feature distance to nearest look-alike':<42}{'n':>6}{'|dddG|':>9}{'agree':>8}")
    q = pd.Series(pd.qcut(dist, 4, labels=["closest 25%", "2nd", "3rd", "farthest 25%"]))
    for b in q.cat.categories:
        s = (q == b).to_numpy()
        print(f"    {b:<42}{int(s.sum()):>6}{gap[s].mean():>9.2f}{agree[s].mean():>8.1%}")
    print("    if the closest look-alikes still disagree, closeness in these features")
    print("    does not imply closeness in energy, and the inputs are the limit.")

    # ---- fit versus generalization -------------------------------------------------------
    groups = df["pdb_id"].values
    tr_scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pos = max(1, int(y[tr].sum()))
        m = XGBClassifier(scale_pos_weight=(len(tr) - y[tr].sum()) / pos,
                          random_state=0, **XGB_KWARGS).fit(X.iloc[tr], y[tr])
        tr_scores.append(average_precision_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1]))
    oof_pr = average_precision_score(y, oof)
    print(f"\n    PR-AUC on its own training rows : {np.mean(tr_scores):.4f}")
    print(f"    PR-AUC out of fold              : {oof_pr:.4f}")
    print(f"    generalization gap              : {np.mean(tr_scores)-oof_pr:.4f}")
    print("    a large gap with a flat learning curve means the model memorizes complex-")
    print("    specific quirks that do not transfer, not that it is short of examples.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "data" / "skempi_features_full.csv")
    ap.add_argument("--skempi", type=Path, default=REPO_ROOT / "data" / "skempi_v2.csv")
    ap.add_argument("--holdout", nargs="*", default=["1BRS"])
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sections", default="1,2,3,4,5",
                    help="comma-separated subset to run, so a rerun need not redo everything")
    args = ap.parse_args()
    want = {s.strip() for s in args.sections.split(",")}

    df = pd.read_csv(args.csv)
    df["mut"] = df["mutation"].astype(str).map(lambda m: parse_mutation(m).mut)
    df["pdb_id"] = df["complex_group"]
    df = df[~df["pdb_id"].str.upper().isin({h.upper() for h in args.holdout})]
    df = collapse(df, "alanine", args.threshold).reset_index(drop=True)

    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].fillna(df[feats].median(numeric_only=True)).fillna(0.0)
    y = df["label"].astype(int).values
    seeds = list(range(args.seeds))

    print("=" * 78)
    print("  WHY DOES NOTHING IMPROVE THIS MODEL?")
    print("=" * 78)
    print(f"  {len(df)} residues | {df.pdb_id.nunique()} complexes | "
          f"{int(y.sum())} hot ({y.mean():.1%}) | {len(feats)} features")

    if "1" in want:
        learning_curve(df, X, y, seeds, args.folds, [0.15, 0.3, 0.5, 0.7, 0.85, 1.0])
    if "2" in want:
        redundancy(df, X, y, seeds, args.folds)
    oof = None
    if want & {"3", "5", "6"}:
        oof = label_noise(args.skempi, df, X, y, seeds, args.folds, verbose="3" in want)
    if "4" in want:
        power(df, X, y, seeds, args.folds)
    if "5" in want:
        noise_ceiling(args.skempi, df, y, oof, args.threshold)
    if "6" in want:
        information_present(df, X, y, oof, args.folds)
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
