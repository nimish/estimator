# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for trend term in predict.

Covers the dimension mismatch when predicting on a time window that spans
fewer periods than the training data (n_periods_pred < n_periods_fit), plus
the strict no-gap validation for prediction timestamps.
"""

import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamSolverConfig,
    TsgamTrendConfig,
)


def _make_trend_data(n_samples=720, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="1h")
    X = pd.DataFrame({"x0": rng.standard_normal(n_samples)}, index=timestamps)
    hours = np.arange(n_samples, dtype=float)
    y = 5.0 + 0.01 * hours + 0.3 * X["x0"].values + rng.standard_normal(n_samples) * 0.1
    return X, y


SOLVER = TsgamSolverConfig(solver="CLARABEL", verbose=False)


@pytest.fixture(params=[TrendType.LINEAR, TrendType.NONLINEAR], ids=["linear", "nonlinear"])
def fitted(request):
    X, y = _make_trend_data()
    config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=None,
        trend_config=TsgamTrendConfig(trend_type=request.param),
        solver_config=SOLVER,
    )
    est = TsgamEstimator(config=config)
    est.fit(X, y)
    return est, X


def test_predict_full_range(fitted):
    est, X = fitted
    preds = est.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_predict_subset_before_end(fitted):
    """Predict on first half: n_periods_pred < n_periods_fit."""
    est, X = fitted
    full_preds = est.predict(X)
    half = len(X) // 2
    subset_preds = est.predict(X.iloc[:half])
    np.testing.assert_allclose(subset_preds, full_preds[:half])


def test_predict_gapped_subset(fitted):
    """Predict should reject gapped timestamps even within training range."""
    est, X = fitted
    every_5th = np.arange(0, len(X), 5)
    with pytest.raises(ValueError, match="does not match expected frequency"):
        est.predict(X.iloc[every_5th])


def test_predict_beyond_training(fitted):
    """Predict on timestamps past the training window."""
    est, X = fitted
    future = pd.date_range(X.index[-1] + pd.Timedelta("1h"), periods=48, freq="1h")
    X_future = pd.DataFrame({"x0": np.zeros(48)}, index=future)
    preds = est.predict(X_future)
    assert preds.shape == (48,)
    assert np.all(np.isfinite(preds))
