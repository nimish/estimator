# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for repeated fit(): successive fits rebuild the CVXPY problem each time.

The estimator does not retain a compiled parametric problem; changing sample count
between fits is supported without an explicit invalidation step.
"""

import numpy as np
import pandas as pd
from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSplineConfig,
    TsgamSolverConfig,
)


def _make_config():
    return TsgamEstimatorConfig(
        multi_periodic_config=TsgamMultiPeriodicConfig(
            num_harmonics=[2, 2],
            periods=[24.0, 168.0],
        ),
        exog_config=[TsgamSplineConfig(n_knots=5, lags=[0])],
        ar_config=None,
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )


def test_parametric_fit_correctness_equivalence():
    """Two consecutive fit() calls on the same data recover the same coefficients."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": np.random.randn(n)}, index=dates)
    y = np.random.randn(n) * 0.5 + 1.0

    config = _make_config()
    est = TsgamEstimator(config=config)
    est.fit(X, y)
    coef_first = {k: np.copy(v.value) for k, v in est.variables_.items() if v.value is not None}

    est.fit(X, y)
    for k, v in est.variables_.items():
        if v.value is None:
            continue
        np.testing.assert_allclose(
            coef_first[k],
            v.value,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Repeated fit changed coefficient {k}",
        )


def test_second_fit_different_sample_count_succeeds():
    """fit() after a prior fit with a different n_samples rebuilds and succeeds."""
    np.random.seed(43)
    config = _make_config()
    est = TsgamEstimator(config=config)

    n1 = 80
    dates1 = pd.date_range("2020-01-01", periods=n1, freq="h")
    X1 = pd.DataFrame({"x": np.random.randn(n1)}, index=dates1)
    y1 = np.random.randn(n1)
    est.fit(X1, y1)
    assert est.problem_.status in ("optimal", "optimal_inaccurate")

    n2 = 100
    dates2 = pd.date_range("2020-01-01", periods=n2, freq="h")
    X2 = pd.DataFrame({"x": np.random.randn(n2)}, index=dates2)
    y2 = np.random.randn(n2)
    est.fit(X2, y2)

    assert est.problem_.status in ("optimal", "optimal_inaccurate")
    c = est.variables_["constant"].value
    assert c is not None and np.isfinite(np.asarray(c)).all()


def test_repeated_fit_same_shape_succeeds():
    """Repeated fit() with the same X, y completes optimally each time."""
    np.random.seed(45)
    n = 70
    config = _make_config()
    est = TsgamEstimator(config=config)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": np.random.randn(n)}, index=dates)
    y = np.random.randn(n)

    est.fit(X, y)
    assert est.problem_.status in ("optimal", "optimal_inaccurate")

    est.fit(X, y)
    assert est.problem_.status in ("optimal", "optimal_inaccurate")
