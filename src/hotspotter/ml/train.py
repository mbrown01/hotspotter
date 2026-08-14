"""Train and honestly evaluate the Phase-2 hot-spot classifier.

The whole point of Phase 2 is to replace Phase 1's hand-set weights with weights *learned*
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

STATUS: standard sklearn/xgboost code, written but not yet run (needs `pip install -e .[ml]`
and a dataset from dataset.build_dataset). No fabricated results here — run it to get real
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columns that are identifiers/targets/text, NOT model inputs.
NON_FEATURE_COLUMNS = {
    "residue", "chain", "resseq", "icode", "resname", "side", "aa", "reasoning",
    "mutation", "complex_group", "ddg", "label",
    "naive_score", "naive_rank", "hotspot_score", "hotspot_rank",  # Phase-1 scores: leak-y, exclude
}


@dataclass
class TrainResult:
    model: object = field(repr=False)
    feature_names: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    importances: pd.DataFrame | None = None


def select_features(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns only, excluding identifiers/targets/Phase-1 scores."""
    feats = []
    for c in df.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


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
            "Phase-2 training needs the ml extras: pip install -e .[ml]  "
            "(scikit-learn, xgboost)."
        ) from exc

    features = select_features(df)
    X = df[features].fillna(df[features].median(numeric_only=True))
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

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=random_state, n_jobs=-1,
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
        raise ImportError("Needs ml extras: pip install -e .[ml]") from exc

    features = select_features(df)
    X = df[features].fillna(df[features].median(numeric_only=True))
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
