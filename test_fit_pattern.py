"""
Test that fit method matches notebook pattern during training.
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
from spcqe import make_basis_matrix


def test_fit_uses_correct_pattern():
    """
    Test that during fit, we use the same pattern as notebook.

    Notebook during training:
    - length=len(y)
    - Uses indices 0 to len(y)-1 implicitly

    Our implementation:
    - length=max(time_indices) + 1
    - time_indices should be [0, 1, 2, ..., len(y)-1]
    - So max(time_indices) + 1 = len(y) ✓
    """
    n_samples = 100
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='h')

    config = TsgamEstimatorConfig(
        multi_harmonic_config=TsgamMultiHarmonicConfig(
            num_harmonics=[6, 4, 3],
            periods=[365.2425 * 24, 7 * 24, 24]
        ),
        exog_config=None,
        ar_config=None,
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        freq='h',
        debug=False
    )

    estimator = TsgamEstimator(config=config)
    X = pd.DataFrame({'temp': np.random.randn(n_samples)}, index=timestamps)
    y = np.random.randn(n_samples)

    # Manually create notebook pattern
    F_notebook = make_basis_matrix(
        num_harmonics=[6, 4, 3],
        length=len(y),
        periods=[365.2425 * 24, 7 * 24, 24]
    )

    # Fit our estimator
    estimator.fit(X, y)

    # Verify time_indices
    assert estimator.time_indices_[0] == 0, "First index should be 0"
    assert estimator.time_indices_[-1] == n_samples - 1, f"Last index should be {n_samples - 1}"
    assert len(estimator.time_indices_) == n_samples, "Should have n_samples indices"

    # Verify that max(time_indices) + 1 equals len(y)
    max_idx = int(np.max(estimator.time_indices_))
    assert max_idx + 1 == len(y), "max(time_indices) + 1 should equal len(y)"

    # Verify basis matrix generation matches notebook pattern
    # Notebook: F_notebook with length=len(y), uses all rows implicitly
    # Our code: F_full with length=max(time_indices) + 1, then F_full[time_indices]
    # Reconstruct what our estimator generates
    F_ours_full = make_basis_matrix(
        num_harmonics=[6, 4, 3],
        length=max_idx + 1,
        periods=[365.2425 * 24, 7 * 24, 24]
    )
    F_ours = F_ours_full[estimator.time_indices_.astype(int), :]

    # Should match (notebook uses all rows, we use time_indices which are [0, 1, ..., n_samples-1])
    np.testing.assert_array_equal(
        F_notebook,
        F_ours,
        err_msg="Basis matrix should match notebook pattern during fit"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

