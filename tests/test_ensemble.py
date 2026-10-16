"""Tests for the scoring functions in hotspotter.ml.train.

`accepted_scores` is the shipped model and `ensemble_scores` is the blend that was
measured against it and not kept. Both are pinned here, because the blend is still
reported as a comparison and a silent swap between the two would change every number in
the README without changing a result table's heading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotspotter.ml.features import XGB_FEATURES
from hotspotter.ml.train import (
    accepted_scores,
    ensemble_scores,
    make_et,
    make_xgb,
    prepare_matrix,
)


def _frame(n=120, seed=0):
    """A small separable-ish table with a NaN in a NULLABLE column."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in XGB_FEATURES})
    df["label"] = (df["dsasa"] + 0.5 * rng.normal(size=n) > 0).astype(int)
    df.loc[: n // 10, "cons_score"] = np.nan
    return df


# --- the balancing bug this helper exists to prevent -----------------------------------

def test_make_xgb_sets_scale_pos_weight_from_the_training_labels():
    """THE IMPORTANT ONE.

    A comparison harness that gives every competitor class_weight="balanced" and builds
    XGBoost without scale_pos_weight runs the shipped model unbalanced against balanced
    rivals. That has cost this project two result tables; nothing raised, the numbers
    were just wrong.
    """
    y = np.array([0] * 90 + [1] * 10)
    assert make_xgb(y).scale_pos_weight == pytest.approx(9.0)


def test_scale_pos_weight_tracks_a_different_class_balance():
    y = np.array([0] * 60 + [1] * 20)
    assert make_xgb(y).scale_pos_weight == pytest.approx(3.0)


def test_make_xgb_never_divides_by_zero_on_an_all_negative_fold():
    assert np.isfinite(make_xgb(np.zeros(50, dtype=int)).scale_pos_weight)


def test_make_xgb_carries_the_frozen_configuration():
    """Literal values, not a re-read of ACCEPTED_XGB: the point is to notice if the
    frozen configuration is edited, and comparing the dict to itself cannot."""
    m = make_xgb(np.array([0, 1]))
    assert m.reg_alpha == 50.0
    assert m.max_depth == 4
    assert m.n_estimators == 300
    assert m.learning_rate == 0.05


def test_the_extra_trees_member_is_balanced_too():
    """Its balancing is static, which is exactly why it was never the one forgotten."""
    assert make_et().class_weight == "balanced"


# --- the two members need different missing-value handling -----------------------------

def test_extra_trees_gets_imputed_values_and_xgboost_does_not():
    """XGBoost learns a default branch for NaN; extra trees raises on one. prepare_matrix
    is what keeps them apart, so check the matrices it hands each of them."""
    df = _frame()
    x, _ = prepare_matrix(df, list(XGB_FEATURES), model="xgboost")
    e, stats = prepare_matrix(df, list(XGB_FEATURES), model="extratrees")

    assert x["cons_score"].isna().any(), "xgboost should receive NaN untouched"
    assert not e.isna().any().any(), "extra trees cannot be handed a NaN"
    assert "cons_score_is_missing" in e.columns, \
        "imputing without an indicator throws away the missingness, which is informative"
    assert e.loc[df.cons_score.isna(), "cons_score"].eq(stats["cons_score"]).all()


def test_ensemble_runs_with_missing_values_present():
    df = _frame()
    s = ensemble_scores(df.iloc[:80], df.iloc[80:], seeds=1)
    assert len(s) == 40
    assert np.isfinite(s).all()


# --- the blend itself ------------------------------------------------------------------

def test_scores_are_probabilities():
    df = _frame()
    s = ensemble_scores(df.iloc[:80], df.iloc[80:], seeds=1)
    assert s.min() >= 0.0 and s.max() <= 1.0


class _Constant:
    """A stand-in model that predicts one fixed probability, so the test can check how
    the two members are combined without re-deriving what they predict."""

    def __init__(self, p):
        self.p = p

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def test_the_blend_weights_its_two_members_equally(monkeypatch):
    """50/50, fixed rather than swept. Substituting constant members makes the weighting
    visible; recomputing the real members here would only restate the function body."""
    import hotspotter.ml.train as T

    monkeypatch.setattr(T, "make_xgb", lambda y, s=0, **kw: _Constant(0.2))
    monkeypatch.setattr(T, "make_et", lambda s=0, **kw: _Constant(0.8))

    df = _frame()
    s = T.ensemble_scores(df.iloc[:80], df.iloc[80:], seeds=1)
    assert s == pytest.approx(np.full(40, 0.5))


def test_imputation_comes_from_the_training_rows_only():
    """The extra-trees member median-fills. Fitting that on the scored rows would leak,
    which is the bug prepare_matrix's `stats` argument exists to prevent."""
    df = _frame()
    tr = df.iloc[:80].copy()
    te_a = df.iloc[80:].copy()
    te_b = te_a.copy()
    te_b.loc[te_b.index[:5], "cons_score"] = 999.0   # wild values in the SCORED rows
    s_a = ensemble_scores(tr, te_a, seeds=1)
    s_b = ensemble_scores(tr, te_b, seeds=1)
    assert s_a[5:] == pytest.approx(s_b[5:]), \
        "rows we did not touch changed, so the scored fold is influencing its own fill"


def test_a_scored_frame_of_one_row_works():
    """Inference on a tiny interface must not depend on batch statistics."""
    df = _frame()
    s = ensemble_scores(df.iloc[:80], df.iloc[80:81], seeds=1)
    assert s.shape == (1,) and np.isfinite(s).all()

def test_accepted_scores_never_touches_extra_trees(monkeypatch):
    """The shipped model is XGBoost alone, and nothing may quietly put the blend back.

    The blend wins when folds are grouped by complex and LOSES by -0.0174 (p=0.003) once
    they are grouped by >=70% chain identity, so its gain was near-duplicate complexes
    leaking across folds. Making the extra-trees member explode proves which function
    uses it, without this test restating either function's body.
    """
    import hotspotter.ml.train as T

    def exploding(*a, **kw):
        raise AssertionError("the shipped model must not build an extra-trees member")

    monkeypatch.setattr(T, "make_et", exploding)

    df = _frame()
    tr, te = df.iloc[:80], df.iloc[80:]
    s = T.accepted_scores(tr, te, seeds=2)
    assert len(s) == 40 and np.isfinite(s).all()

    with pytest.raises(AssertionError):
        T.ensemble_scores(tr, te, seeds=1)


def test_accepted_scores_averages_over_every_seed(monkeypatch):
    """Each seed must be fitted and all of them averaged.

    Checked with a stub whose prediction IS its seed, so `seeds=4` has to come back as
    mean(0,1,2,3) = 1.5. On the real model this is invisible: at reg_alpha=50 a small
    synthetic frame has no split worth making, so every seed returns the base rate and
    a test comparing seed counts passes whatever the loop does.
    """
    import hotspotter.ml.train as T

    monkeypatch.setattr(T, "make_xgb", lambda y, s=0, **kw: _Constant(float(s)))

    df = _frame()
    s = T.accepted_scores(df.iloc[:80], df.iloc[80:], seeds=4)
    assert s == pytest.approx(np.full(40, 1.5))


def test_accepted_scores_passes_nan_straight_through():
    """A docked complex has no ConSurf entry, so cons_score is NaN for every residue.

    XGBoost must take that natively rather than seeing an imputed value: blanking
    conservation costs the shipped model -0.0080 PR-AUC, against -0.0291 for a model
    that has to impute. If this ever starts raising or filling, that gap reopens.
    """
    df = _frame()
    tr = df.iloc[:80]
    te = df.iloc[80:].copy()
    te["cons_score"] = np.nan
    s = accepted_scores(tr, te, seeds=1)
    assert len(s) == len(te)
    assert np.isfinite(s).all()
