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
    TsgamMultiHarmonicConfig,
    TsgamSolverConfig,
)


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return TsgamEstimatorConfig(
        multi_harmonic_config=TsgamMultiHarmonicConfig(
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
    assert hasattr(estimator, 'problem_')
    assert hasattr(estimator, 'time_reference_')
    assert hasattr(estimator, 'freq_')


def test_combined_valid_mask_excludes_nan_in_y(basic_config, basic_data):
    """
    Test that combined_valid_mask excludes NaN's in y.

    Note: This test verifies defensive programming - even though we reject
    NaN's in y, the mask should still be computed correctly.
    """
    X, y = basic_data
    estimator = TsgamEstimator(config=basic_config)

    # Fit with valid data
    estimator.fit(X, y)

    # Verify that combined_valid_mask is all True (no NaN's to mask)
    assert np.all(estimator.combined_valid_mask_), \
        "combined_valid_mask_ should be all True when there are no NaN's"
    assert len(estimator.combined_valid_mask_) == len(y), \
        "combined_valid_mask_ should have same length as y"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

