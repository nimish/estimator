# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Test NaN handling in TsgamEstimator.

Tests that:
1. NaN's in X are rejected by check_X_y
2. NaN's in y are rejected by check_X_y
3. NaN's in y are masked out during fit (defensive programming, though check_X_y should prevent them)
"""

import pytest
import numpy as np
import pandas as pd
from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
)


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return TsgamEstimatorConfig(
        multi_periodic_config=TsgamMultiPeriodicConfig(
            num_harmonics=[6, 4, 3],
            periods=[365.2425 * 24, 7 * 24, 24]
        ),
        exog_config=None,
        ar_config=None,
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        debug=False
    )


@pytest.fixture
def basic_data():
    """Basic data for testing."""
    n_samples = 100
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='h')
    X = pd.DataFrame({'temp': np.random.randn(n_samples)}, index=timestamps)
    y = np.random.randn(n_samples)
    return X, y


def test_timestamp_gap_is_missing_on_exogenous_offset_grid():
    timestamps = pd.date_range("2024-01-01", periods=8, freq="1h")
    keep = np.arange(8) != 3
    X = pd.DataFrame({"driver": np.arange(8.0)}, index=timestamps).loc[keep]
    estimator = TsgamEstimator(
        TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[TsgamLinearConfig(lags=[-1])],
        )
    ).fit(X, np.arange(8.0)[keep])

    observed_indices = estimator.time_indices_.astype(int)
    after_gap = int(np.flatnonzero(observed_indices == 4)[0])
    assert not estimator.decomposition_["fit_mask"][observed_indices][after_gap]
    assert estimator.decomposition_["fit_mask"].shape == (8,)


def test_fit_rejects_nan_in_X(basic_config, basic_data):
    """Test that fit rejects NaN values in X via check_X_y."""
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Add NaN to X
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match=".*NaN.*"):
        estimator.fit(X_with_nan, y)


def test_fit_rejects_nan_in_y(basic_config, basic_data):
    """Test that fit rejects NaN values in y via check_X_y."""
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Add NaN to y
    y_with_nan = y.copy()
    y_with_nan[0] = np.nan

    with pytest.raises(ValueError, match=".*NaN.*"):
        estimator.fit(X, y_with_nan)


def test_fit_rejects_multiple_nans_in_X(basic_config, basic_data):
    """Test that fit rejects multiple NaN values in X via check_X_y."""
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Add multiple NaN's to X
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan
    X_with_nan.iloc[5, 0] = np.nan
    X_with_nan.iloc[10, 0] = np.nan

    with pytest.raises(ValueError, match=".*NaN.*"):
        estimator.fit(X_with_nan, y)


def test_fit_rejects_multiple_nans_in_y(basic_config, basic_data):
    """Test that fit rejects multiple NaN values in y via check_X_y."""
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Add multiple NaN's to y
    y_with_nan = y.copy()
    y_with_nan[0] = np.nan
    y_with_nan[5] = np.nan
    y_with_nan[10] = np.nan

    with pytest.raises(ValueError, match=".*NaN.*"):
        estimator.fit(X, y_with_nan)


def test_fit_rejects_nan_in_X_with_exog(basic_config, basic_data):
    """Test that fit rejects NaN values in X when exogenous variables are present via check_X_y."""
    # Add exog config
    from tsgam_estimator import TsgamSplineConfig
    basic_config.exog_config = [
        TsgamSplineConfig(n_knots=10, lags=[0])
    ]

    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Add NaN to X
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match=".*NaN.*"):
        estimator.fit(X_with_nan, y)


def test_fit_works_without_nans(basic_config, basic_data):
    """Test that fit works correctly when there are no NaN's."""
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Should not raise any errors
    estimator.fit(X, y)

    # Verify that model was fitted
    assert hasattr(estimator, 'decomposition_')
    assert hasattr(estimator, 'time_reference_')
    assert hasattr(estimator, 'freq_')


def test_fit_mask_includes_all_valid_rows(basic_config, basic_data):
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Fit with valid data
    estimator.fit(X, y)

    fit_mask = estimator.decomposition_["fit_mask"][estimator.time_indices_.astype(int)]
    assert np.all(fit_mask)
    assert len(fit_mask) == len(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
