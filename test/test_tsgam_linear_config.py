# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Test TsgamLinearConfig support in TsgamEstimator.

Tests that TsgamLinearConfig works correctly for fit, predict, and sample,
both alone and mixed with TsgamSplineConfig.
"""

import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TsgamArConfig,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
)


def _make_data(n_samples=500, n_exog=1, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="1h")
    exog = {f"x{i}": rng.standard_normal(n_samples) for i in range(n_exog)}
    X = pd.DataFrame(exog, index=timestamps)
    y = 3.0 + 0.5 * X.iloc[:, 0].values + rng.standard_normal(n_samples) * 0.1
    return X, y


@pytest.fixture
def solver_config():
    return TsgamSolverConfig(solver="CLARABEL", verbose=False)


class TestLinearConfigFitPredict:
    def test_single_linear_exog(self, solver_config):
        X, y = _make_data()
        config = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[TsgamLinearConfig(lags=[0])],
            solver_config=solver_config,
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_linear_exog_with_lags(self, solver_config):
        X, y = _make_data()
        config = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[TsgamLinearConfig(lags=[-1, 0, 1])],
            solver_config=solver_config,
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_linear_with_fourier(self, solver_config):
        X, y = _make_data()
        config = TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[3], periods=[24]
            ),
            exog_config=[TsgamLinearConfig(lags=[0])],
            solver_config=solver_config,
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_mixed_spline_and_linear(self, solver_config):
        X, y = _make_data(n_exog=2)
        config = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[
                TsgamSplineConfig(n_knots=5, lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            solver_config=solver_config,
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_linear_with_ar(self, solver_config):
        X, y = _make_data()
        config = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[TsgamLinearConfig(lags=[0])],
            ar_config=TsgamArConfig(lags=[1]),
            solver_config=solver_config,
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_linear_sample(self, solver_config):
        X, y = _make_data()
        config = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[TsgamLinearConfig(lags=[0])],
            ar_config=TsgamArConfig(lags=[1]),
            solver_config=solver_config,
            random_state=np.random.RandomState(0),
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        samples = est.sample(X, n_samples=5)
        assert samples.shape == (5, len(X))
        assert not np.any(np.isnan(samples))

    def test_linear_coef_shape(self, solver_config):
        """Linear config should produce (1, num_lags) coefficient matrix."""
        X, y = _make_data()
        lags = [-1, 0, 1]
        config = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[TsgamLinearConfig(lags=lags)],
            solver_config=solver_config,
        )
        est = TsgamEstimator(config=config)
        est.fit(X, y)
        coef = est.variables_["exog_coef_0"].value
        assert coef.shape == (1, len(lags))
