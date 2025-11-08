# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Test that tsgam_estimator matches notebook behavior for timestamp handling.

This test verifies that the estimator produces the same results as the notebook
when using timestamps correctly.
"""

from numpy import dtype, float64, ndarray


from numpy._typing._shape import _AnyShape


import pytest
import numpy as np
import pandas as pd
from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSplineConfig,
    TsgamSolverConfig,
)
from spcqe import make_basis_matrix


def test_matches_notebook_pattern():
    """
    Test that our implementation matches the notebook pattern:
    - Notebook: length=max(time_idxs) + 1, then F[time_idxs]
    - Our code: length=max(time_indices) + 1, then F[time_indices]
    """
    # Create timestamps matching notebook pattern
    # Training: 0 to 99 (100 samples)
    train_timestamps = pd.date_range('2020-01-01', periods=100, freq='h')

    # Prediction: 100 to 149 (50 samples, continuing from training)
    # This matches notebook: new_idx = np.arange(np.sum(df['year'] == 2022)) + np.sum(df['year'] != 2022) - 1
    pred_timestamps = pd.date_range('2020-01-01', periods=150, freq='h')[100:]

    # Manually create what notebook does
    train_indices_notebook = np.arange(100)  # 0 to 99
    pred_indices_notebook = np.arange(50) + 100  # 100 to 149

    # Notebook pattern for prediction
    F_notebook = make_basis_matrix(
        num_harmonics=[6, 4, 3],
        length=int(np.max(pred_indices_notebook)) + 1,
        periods=[365.2425 * 24, 7 * 24, 24]
    )
    F_notebook_pred: ndarray[_AnyShape, dtype[float64]] = F_notebook[pred_indices_notebook]

    # Our implementation
    config = TsgamEstimatorConfig(
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

    estimator = TsgamEstimator(config=config)

    # Simulate fit (just to set up reference)
    X_train = pd.DataFrame({'temp': np.random.randn(100)}, index=train_timestamps)
    y_train = np.random.randn(100)
    estimator.fit(X_train, y_train)

    # Get time indices that our implementation would use
    pred_indices_ours = estimator._timestamps_to_indices(pred_timestamps, estimator.time_reference_)

    # Our implementation pattern
    max_idx = int(np.max(pred_indices_ours))
    F_ours = make_basis_matrix(
        num_harmonics=[6, 4, 3],
        length=max_idx + 1,
        periods=[365.2425 * 24, 7 * 24, 24]
    )
    F_ours_pred = F_ours[pred_indices_ours.astype(int)]

    # Should match (excluding constant column)
    np.testing.assert_array_equal(
        F_notebook_pred[:, 1:],  # Notebook keeps constant, we drop it
        F_ours_pred[:, 1:],  # We drop constant column
        err_msg="Basis matrix should match notebook pattern"
    )


def test_continuous_indices_match_notebook():
    """
    Test that continuous indices (training then prediction) match notebook behavior.

    Notebook pattern:
    - Training: indices 0 to len(y)-1
    - Prediction: indices continue from training (e.g., len(y) to len(y)+len(pred)-1)
    """
    # Training data: first 100 hours
    train_timestamps = pd.date_range('2020-01-01', periods=100, freq='h')

    # Prediction data: next 50 hours (continuous)
    pred_timestamps = pd.date_range('2020-01-01', periods=150, freq='h')[100:]

    config = TsgamEstimatorConfig(
        multi_harmonic_config=TsgamMultiHarmonicConfig(
            num_harmonics=[2, 1],
            periods=[24, 7 * 24]
        ),
        exog_config=None,
        ar_config=None,
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        debug=False
    )

    estimator = TsgamEstimator(config=config)
    X_train = pd.DataFrame({'temp': np.random.randn(100)}, index=train_timestamps)
    y_train = np.random.randn(100)
    estimator.fit(X_train, y_train)

    # Get indices
    train_indices = estimator.time_indices_
    pred_indices = estimator._timestamps_to_indices(pred_timestamps, estimator.time_reference_)

    # Should be continuous
    assert train_indices[0] == 0, "Training should start at index 0"
    assert train_indices[-1] == 99, "Training should end at index 99"
    assert pred_indices[0] == 100, "Prediction should start at index 100 (continuous)"
    assert pred_indices[-1] == 149, "Prediction should end at index 149"

    # This matches notebook: new_idx = np.arange(np.sum(df['year'] == 2022)) + np.sum(df['year'] != 2022) - 1
    # Where training data has len(df['year'] != 2022) samples, so prediction starts at that index


def test_phase_alignment_matches_notebook():
    """
    Test that phase alignment matches notebook behavior.

    The notebook uses:
    - length=max(time_idxs) + 1 for basis matrix generation
    - F[time_idxs] for indexing

    We do the same with timestamp-based indices.
    """
    # Create data with known phase (daily cycle)
    n_train = 100
    n_pred = 50

    train_timestamps = pd.date_range('2020-01-01', periods=n_train, freq='h')
    pred_timestamps = pd.date_range('2020-01-01', periods=n_train + n_pred, freq='h')[n_train:]

    # Manual notebook-style calculation
    train_indices_manual = np.arange(n_train)
    pred_indices_manual = np.arange(n_pred) + n_train

    # Notebook pattern
    F_full_notebook = make_basis_matrix(
        num_harmonics=[1],
        periods=[24],  # Daily cycle
        length=int(np.max(pred_indices_manual)) + 1
    )
    F_train_notebook = F_full_notebook[train_indices_manual]
    F_pred_notebook = F_full_notebook[pred_indices_manual]

    # Our implementation
    config = TsgamEstimatorConfig(
        multi_harmonic_config=TsgamMultiHarmonicConfig(
            num_harmonics=[1],
            periods=[24]
        ),
        exog_config=None,
        ar_config=None,
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        debug=False
    )

    estimator = TsgamEstimator(config=config)
    X_train = pd.DataFrame({'temp': np.random.randn(n_train)}, index=train_timestamps)
    y_train = np.random.randn(n_train)
    estimator.fit(X_train, y_train)

    # Get our indices
    train_indices_ours = estimator.time_indices_
    pred_indices_ours = estimator._timestamps_to_indices(pred_timestamps, estimator.time_reference_)

    # Our pattern
    max_idx = int(np.max(pred_indices_ours))
    F_full_ours = make_basis_matrix(
        num_harmonics=[1],
        periods=[24],
        length=max_idx + 1
    )
    F_train_ours = F_full_ours[train_indices_ours.astype(int)]
    F_pred_ours = F_full_ours[pred_indices_ours.astype(int)]

    # Should match (excluding constant column)
    np.testing.assert_allclose(
        F_train_notebook[:, 1:],
        F_train_ours[:, 1:],
        rtol=1e-10,
        err_msg="Training basis matrix should match notebook"
    )

    np.testing.assert_allclose(
        F_pred_notebook[:, 1:],
        F_pred_ours[:, 1:],
        rtol=1e-10,
        err_msg="Prediction basis matrix should match notebook"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

