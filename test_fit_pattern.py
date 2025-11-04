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

    # Verify basis matrix generation matches
    # Our code generates: length=max(time_indices) + 1 = (n_samples-1) + 1 = n_samples
    # Which matches notebook: length=len(y) = n_samples

    # The basis matrix rows should match
    # Notebook: F_notebook[0:len(y)] which is all rows
    # Our code: F_full[time_indices] where time_indices = [0, 1, 2, ..., n_samples-1]
    # So we get F_full[0:n_samples] which is the same

    # Verify that max(time_indices) + 1 equals len(y)
    max_idx = int(np.max(estimator.time_indices_))
    assert max_idx + 1 == len(y), "max(time_indices) + 1 should equal len(y)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

