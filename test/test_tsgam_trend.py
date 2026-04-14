# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd

from tsgam_estimator import (
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamSolverConfig,
    TsgamTrendConfig,
)


SOLVER = TsgamSolverConfig(solver="CLARABEL", verbose=False)
TREND_TOL = 1.0e-6


def _make_trend_data(direction: str, n_periods: int = 30, samples_per_period: int = 24) -> tuple[pd.DataFrame, np.ndarray]:
    n_samples = n_periods * samples_per_period
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="1h")
    X = pd.DataFrame({"x0": np.zeros(n_samples)}, index=timestamps)

    if direction == "increasing":
        period_values = np.linspace(0.0, 3.0, n_periods)
    elif direction == "decreasing":
        period_values = np.linspace(0.0, -3.0, n_periods)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    y = 10.0 + np.repeat(period_values, samples_per_period)
    return X, y


def _fit_trend(trend_type: TrendType, direction: str) -> np.ndarray:
    X, y = _make_trend_data(direction)
    config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=None,
        trend_config=TsgamTrendConfig(
            trend_type=trend_type,
            grouping=24.0,
            reg_weight=1.0,
        ),
        solver_config=SOLVER,
    )
    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)
    trend = estimator.variables_["trend"].value
    assert trend is not None
    return np.asarray(trend, dtype=float)


def test_legacy_nonlinear_alias_remains_nonincreasing():
    trend = _fit_trend(TrendType.NONLINEAR, "decreasing")
    assert np.all(np.diff(trend) <= TREND_TOL)


def test_explicit_nonlinear_decreasing_is_nonincreasing():
    trend = _fit_trend(TrendType.NONLINEAR_DECREASING, "decreasing")
    assert np.all(np.diff(trend) <= TREND_TOL)


def test_explicit_nonlinear_increasing_is_nondecreasing():
    trend = _fit_trend(TrendType.NONLINEAR_INCREASING, "increasing")
    assert np.all(np.diff(trend) >= -TREND_TOL)
