"""Train and honestly evaluate the hot-spot classifier.

The point is to replace the hand-set ranking weights with weights *learned*
from measured ΔΔG, and then to check — rigorously — whether that actually beats the naive
"most-buried" baseline, and which features carry the signal.

Three things this module gets right on purpose:

  1. SPLIT BY COMPLEX. We use scikit-learn's GroupShuffleSplit / GroupKFold on the
     `complex_group` column so the same complex never appears in train and test. Without
     this you leak: the model recognizes a complex it has seen and the reported score is a
     fantasy. This is the single easiest way to fool yourself in this project.

  2. HANDLE CLASS IMBALANCE. Disruptive mutations are rare, so accuracy is a useless metric
     (predict "not disruptive" always -> 90%+ accuracy, zero value). We weight the positive
     class (XGBoost `scale_pos_weight`) and report PR-AUC (average precision) as the primary
     metric, with ROC-AUC alongside.

  3. COMPARE TO THE NAIVE BASELINE + ABLATE. We score the naive burial ranking on the same
     test split, and we report feature importances / support leave-one-group-out ablation,
     turning "I computed many features" into "here's which ones actually mattered."

This module is the training primitive. For the reported numbers under the full protocol
(fixed folds, multiple seeds, ranking metrics), use ``scripts/evaluate.py`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hotspotter.ml.features import XGB_FEATURES, prepare_matrix

# Columns that are identifiers/targets/text, NOT model inputs.
NON_FEATURE_COLUMNS = {
    "residue", "chain", "resseq", "icode", "resname", "side", "aa", "reasoning",
    "mutation", "complex_group", "ddg", "label",
    "naive_score", "naive_rank", "hotspot_score", "hotspot_rank",  # heuristic ranking scores: leak-y, exclude
}


#: The model configuration, fixed rather than swept.
#:
#: ``reg_alpha=50`` was first chosen by sweeping eight values over the same folds the
#: score is reported on, which is selection bias. ``scripts/nested_cv.py`` measures what
#: that was worth by picking alpha inside each fold's training complexes instead: the
#: tuned-on-folds estimate is optimistic by roughly +0.008 PR-AUC. That is comparable to
#: a whole feature's contribution, so the value is now frozen here and every comparison
#: runs at a fixed configuration or is tuned by an inner loop.
ACCEPTED_XGB: dict = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=50.0,
    eval_metric="aucpr", n_jobs=-1,
)


def make_xgb(y_train, random_state: int = 0, **overrides):
    """The fixed classifier, correctly balanced for `y_train`.

    Use this rather than building XGBClassifier by hand. Unlike scikit-learn's
    ``class_weight="balanced"``, XGBoost's ``scale_pos_weight`` depends on the training
    labels, so it cannot live in a static config dict -- and a comparison script that
    forgets it runs the shipped model unbalanced against balanced rivals, which has cost
    this project two discarded result tables.
    """
    from xgboost import XGBClassifier

    import numpy as _np
    y_train = _np.asarray(y_train)
    pos = max(1, int(y_train.sum()))
    cfg = {**ACCEPTED_XGB, **overrides}
    return XGBClassifier(**cfg, random_state=random_state,
                         scale_pos_weight=(len(y_train) - y_train.sum()) / pos)


#: NOT SHIPPED. Kept because the comparison is the reason the shipped model is what it is.
#:
#: Blending extra trees with XGBoost scores +0.0162 PR-AUC (p=0.003) when folds are
#: grouped by complex, and that gain is not just an artifact of averaging more models: it
#: survives a count-matched baseline of six XGBoost seeds (+0.0146, p=0.007). It does not
#: survive grouping folds by >=70% chain identity, where the blend LOSES by -0.0174
#: (p=0.003) and extra trees alone drops -0.0400 while XGBoost barely moves. The whole
#: contribution depended on near-duplicate complexes landing in different folds.
#:
#: Two more reasons not to ship it: on macro PR-AUC -- ranking within one interface, the
#: only ranking a user sees -- the two are indistinguishable, so the advantage lives in
#: between-complex calibration nobody reads; and extra trees is three times more fragile
#: when conservation is unavailable (-0.0291 against -0.0080), because it imputes the
#: training median where XGBoost branches on missingness, and a docked or novel complex
#: has no ConSurf-DB entry at all.
ACCEPTED_ET: dict = dict(
    n_estimators=500, max_features="sqrt", min_samples_leaf=1,
    class_weight="balanced", n_jobs=-1,
)

#: Seeds averaged within each member. Both members are stochastic (XGBoost subsamples,
#: extra trees randomises splits), so a single seed is noise.
ENSEMBLE_SEEDS: int = 3


def make_et(random_state: int = 0, **overrides):
    """The extra-trees member. Balanced via class_weight, so unlike `make_xgb` it needs
    no training labels."""
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(**{**ACCEPTED_ET, **overrides},
                                random_state=random_state)


def accepted_scores(train_df, score_df, features=XGB_FEATURES,
                    seeds: int = ENSEMBLE_SEEDS):
    """THE SHIPPED MODEL. XGBoost alone, averaged over `seeds` random seeds.

    Use this on every deployment and reporting path, so there is one definition of "the
    model" and changing it means changing one function. The ensemble above is not it; see
    the note over `ACCEPTED_ET`.

    Seed-averaging because XGBoost subsamples rows and columns, so a single seed is
    noise. Three rather than six: the extra fits are worth +0.0016 grouped by complex
    and nothing at all under identity clustering.

    NaN is passed straight through, because XGBoost learns a default branch direction
    per split and that beats any substituted constant -- median-filling conservation
    costs 78% of that feature's benefit.
    """
    features = list(features)
    y = train_df["label"].astype(int).values
    x_tr, _ = prepare_matrix(train_df, features, model="xgboost")
    x_te, _ = prepare_matrix(score_df, features, model="xgboost")
    return np.mean([make_xgb(y, s).fit(x_tr, y).predict_proba(x_te)[:, 1]
                    for s in range(seeds)], axis=0)


def ensemble_scores(train_df, score_df, features=XGB_FEATURES,
                    seeds: int = ENSEMBLE_SEEDS):
    """The XGBoost + extra-trees blend, fitted on `train_df`. NOT SHIPPED; see
    `accepted_scores` and the note over `ACCEPTED_ET`. Kept so the comparison reruns.

    Probabilities are averaged, not ranks: the two scored the same, but a residue's
    averaged RANK depends on which other residues are being scored, so adding one
    residue to an interface would move everyone else and nothing would be comparable
    across complexes.

    The two members need different missing-value handling -- XGBoost takes NaN, extra
    trees cannot -- which is what `prepare_matrix` is for; fitting the medians on
    anything but the training fold leaks the test fold.
    """
    features = list(features)
    y = train_df["label"].astype(int).values

    x_tr, _ = prepare_matrix(train_df, features, model="xgboost")
    x_te, _ = prepare_matrix(score_df, features, model="xgboost")
    e_tr, stats = prepare_matrix(train_df, features, model="extratrees")
    e_te, _ = prepare_matrix(score_df, features, model="extratrees", stats=stats)

    p_xgb = np.mean([make_xgb(y, s).fit(x_tr, y).predict_proba(x_te)[:, 1]
                     for s in range(seeds)], axis=0)
    p_et = np.mean([make_et(s).fit(e_tr, y).predict_proba(e_te)[:, 1]
                    for s in range(seeds)], axis=0)
    return 0.5 * p_xgb + 0.5 * p_et


@dataclass
class TrainResult:
    model: object = field(repr=False)
    feature_names: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    importances: pd.DataFrame | None = None


def select_features(df: pd.DataFrame) -> list[str]:
    """The declared feature set, intersected with what this table actually carries.

    Sweeping up every numeric column was the old behaviour and it drifts: a table that
    happens to carry an extra column silently trains a different model from the one
    evaluate.py reports. XGB_FEATURES is the single declaration of what the model uses.
    """
    return [c for c in XGB_FEATURES if c in df.columns]


def train_baseline(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 0,
    scale_pos_weight: float | None = None,
) -> TrainResult:
    """Train an XGBoost classifier with a proper split-by-complex hold-out.

    Parameters
    ----------
    df : output of dataset.build_dataset (features + 'label' + 'complex_group').
    scale_pos_weight : XGBoost imbalance knob; if None we set it to (#neg / #pos) on train.

    Returns a TrainResult with the fitted model, test metrics (PR-AUC primary), and feature
    importances.
    """
    try:
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import average_precision_score, roc_auc_score
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Training needs scikit-learn and xgboost: pip install -e .  "
            "(scikit-learn, xgboost)."
        ) from exc

    features = select_features(df)
    # NaN is passed through for XGBoost (it learns a default branch per split).
    # Filling here would both discard the MNAR missingness signal and leak the
    # held-out fold's medians into training.
    X, _ = prepare_matrix(df, features, model="xgboost")
    y = df["label"].astype(int).values
    groups = df["complex_group"].values

    # Split by complex: no PDB id in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    if scale_pos_weight is None:
        pos = max(1, int(y_tr.sum()))
        neg = int(len(y_tr) - y_tr.sum())
        scale_pos_weight = neg / pos

    # The shared configuration, so this primitive and scripts/evaluate.py fit the SAME
    # model. This path once omitted reg_alpha while every reported number carried
    # reg_alpha=50 -- two different models under one name.
    model = XGBClassifier(
        **ACCEPTED_XGB,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "pr_auc": float(average_precision_score(y_te, proba)),   # PRIMARY (imbalance-aware)
        "roc_auc": float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else float("nan"),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "n_train_complexes": int(pd.Series(groups[train_idx]).nunique()),
        "n_test_complexes": int(pd.Series(groups[test_idx]).nunique()),
        "test_positive_rate": float(y_te.mean()),
    }

    # Naive baseline on the SAME test rows: rank by buried surface area (dsasa).
    if "dsasa" in df.columns:
        naive_scores = df.iloc[test_idx]["dsasa"].fillna(0.0).values
        metrics["naive_pr_auc"] = float(average_precision_score(y_te, naive_scores))

    importances = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return TrainResult(model=model, feature_names=features, metrics=metrics,
                       importances=importances)


def cross_validated_scores(df: pd.DataFrame, n_splits: int = 5, random_state: int = 0):
    """Grouped k-fold PR-AUC — a more honest estimate than a single split.

    GroupKFold guarantees every complex is in exactly one fold's test set, so the mean/std
    across folds reflects generalization to unseen complexes.
    """
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import average_precision_score
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Needs scikit-learn and xgboost: pip install -e ."
        ) from exc

    features = select_features(df)
    # NaN is passed through for XGBoost (it learns a default branch per split).
    # Filling here would both discard the MNAR missingness signal and leak the
    # held-out fold's medians into training.
    X, _ = prepare_matrix(df, features, model="xgboost")
    y = df["label"].astype(int).values
    groups = df["complex_group"].values

    scores, naive_scores = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        pos = max(1, int(y[tr].sum()))
        spw = int(len(tr) - y[tr].sum()) / pos
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          eval_metric="aucpr", random_state=random_state, n_jobs=-1)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        scores.append(average_precision_score(y[te], p))
        if "dsasa" in df.columns:
            naive_scores.append(
                average_precision_score(y[te], df.iloc[te]["dsasa"].fillna(0).values))

    return {
        "pr_auc_mean": float(np.mean(scores)), "pr_auc_std": float(np.std(scores)),
        "naive_pr_auc_mean": float(np.mean(naive_scores)) if naive_scores else None,
        "fold_scores": [float(s) for s in scores],
    }
