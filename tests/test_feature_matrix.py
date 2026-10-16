"""Tests for per-model missing-value handling.

XGBoost learns a default branch for NaN; SVMs, Gaussian processes, random forests and
linear models cannot take NaN at all. The matrix builder has to serve both without
silently throwing away the information carried by the missingness itself -- which matters
here because the data is MNAR: the residues lacking conservation are 50.9% hot against a
23.4% base rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotspotter.ml.features import (
    missing_indicator_name,
    prepare_matrix,
)

FEATURES = ["dsasa", "cons_score"]


def _frame():
    return pd.DataFrame({
        "dsasa": [10.0, 20.0, 30.0, 40.0],
        "cons_score": [-1.0, np.nan, 1.0, 3.0],
    })


def test_nan_native_models_get_the_values_untouched():
    X, stats = prepare_matrix(_frame(), FEATURES, model="xgboost")
    assert list(X.columns) == FEATURES
    assert X["cons_score"].isna().sum() == 1
    assert stats == {}


def test_other_models_get_an_indicator_and_no_nan():
    X, _ = prepare_matrix(_frame(), FEATURES, model="svm")
    col = missing_indicator_name("cons_score")
    assert col in X.columns
    assert X.isna().sum().sum() == 0
    assert X[col].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_the_imputed_value_is_the_median_of_the_observed():
    X, stats = prepare_matrix(_frame(), FEATURES, model="svm")
    assert stats["cons_score"] == pytest.approx(1.0)       # median of -1, 1, 3
    assert X["cons_score"].tolist() == pytest.approx([-1.0, 1.0, 1.0, 3.0])


def test_stats_can_be_reused_so_test_folds_do_not_leak():
    """THE IMPORTANT ONE.

    Imputation values must come from the training fold. Recomputing the median on the
    test fold leaks test information into the features, which raises no error and would
    quietly inflate every score.
    """
    train = _frame()
    _, stats = prepare_matrix(train, FEATURES, model="svm")

    test = pd.DataFrame({"dsasa": [50.0, 60.0], "cons_score": [np.nan, 99.0]})
    X, _ = prepare_matrix(test, FEATURES, model="svm", stats=stats)
    # filled with the TRAINING median, not the test fold's own
    assert X["cons_score"].tolist() == pytest.approx([1.0, 99.0])


def test_a_column_with_no_missing_values_still_gets_an_indicator():
    """Shape must not depend on whether this particular fold happens to have a gap."""
    df = pd.DataFrame({"dsasa": [1.0, 2.0], "cons_score": [0.5, 0.6]})
    X, _ = prepare_matrix(df, FEATURES, model="svm")
    assert missing_indicator_name("cons_score") in X.columns
    assert X[missing_indicator_name("cons_score")].sum() == 0


def test_unexpected_nan_raises_rather_than_being_filled():
    """A NaN in a feature not declared NULLABLE is a bug upstream, not something to
    paper over silently."""
    df = pd.DataFrame({"dsasa": [1.0, np.nan], "cons_score": [0.5, 0.6]})
    with pytest.raises(ValueError, match="unexpected NaN"):
        prepare_matrix(df, FEATURES, model="svm")


def test_an_all_missing_column_does_not_produce_a_nan_fill():
    df = pd.DataFrame({"dsasa": [1.0, 2.0], "cons_score": [np.nan, np.nan]})
    X, stats = prepare_matrix(df, FEATURES, model="svm")
    assert stats["cons_score"] == 0.0
    assert X.isna().sum().sum() == 0


def test_the_registry_decides_which_model_gets_imputed_values():
    """Asserting the registry's own contents against itself proves nothing. What matters
    is that membership changes the matrix a model is handed."""
    df = pd.DataFrame({"dsasa": [1.0, 2.0], "cons_score": [np.nan, 3.0]})
    x, _ = prepare_matrix(df, FEATURES, model="xgboost")
    s, _ = prepare_matrix(df, FEATURES, model="svm")

    assert x["cons_score"].isna().any(), "a NaN-native model keeps its NaN"
    assert not s.isna().any().any(), "a model that cannot take NaN must not receive one"
    assert "cons_score_is_missing" in s.columns and "cons_score_is_missing" not in x
