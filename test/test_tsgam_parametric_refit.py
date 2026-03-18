# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for CVXPY parametric re-fitting: shape enforcement, invalidation, reuse.
"""

import pytest
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
    """Parametric fit() produces identical coefficients to single fit (same data)."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": np.random.randn(n)}, index=dates)
    y = np.random.randn(n) * 0.5 + 1.0

    config = _make_config()
    est = TsgamEstimator(config=config)
    est.fit(X, y)
    coef_first = {k: np.copy(v.value) for k, v in est.variables_.items() if v.value is not None}

    # Second fit with same shape (reuse compiled problem)
    est.fit(X, y)
    for k, v in est.variables_.items():
        if v.value is None:
            continue
        np.testing.assert_allclose(
            coef_first[k],
            v.value,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Parametric refit changed coefficient {k}",
        )


def test_shape_mismatch_raises_value_error():
    """fit() with mismatched shape raises ValueError with descriptive dimension diff."""
    np.random.seed(43)
    config = _make_config()
    est = TsgamEstimator(config=config)

    n1 = 80
    dates1 = pd.date_range("2020-01-01", periods=n1, freq="h")
    X1 = pd.DataFrame({"x": np.random.randn(n1)}, index=dates1)
    y1 = np.random.randn(n1)
    est.fit(X1, y1)

    n2 = 100
    dates2 = pd.date_range("2020-01-01", periods=n2, freq="h")
    X2 = pd.DataFrame({"x": np.random.randn(n2)}, index=dates2)
    y2 = np.random.randn(n2)

    with pytest.raises(ValueError) as exc_info:
        est.fit(X2, y2)
    msg = str(exc_info.value)
    assert "shape" in msg.lower() or "n:" in msg or "expected" in msg
    assert "invalidate_compiled_problem" in msg


def test_invalidate_then_refit_new_shape_succeeds():
    """invalidate_compiled_problem() then fit() with new shape succeeds."""
    np.random.seed(44)
    config = _make_config()
    est = TsgamEstimator(config=config)

    n1 = 60
    dates1 = pd.date_range("2020-01-01", periods=n1, freq="h")
    X1 = pd.DataFrame({"x": np.random.randn(n1)}, index=dates1)
    y1 = np.random.randn(n1)
    est.fit(X1, y1)

    est.invalidate_compiled_problem()
    n2 = 90
    dates2 = pd.date_range("2020-01-01", periods=n2, freq="h")
    X2 = pd.DataFrame({"x": np.random.randn(n2)}, index=dates2)
    y2 = np.random.randn(n2)
    est.fit(X2, y2)

    assert est.problem_.status in ("optimal", "optimal_inaccurate")
    c = est.variables_["constant"].value
    assert c is not None and np.isfinite(np.asarray(c)).all()


def test_rolling_window_reuse_timing():
    """Repeated fit() with same shape reuses compiled problem (second fit completes)."""
    np.random.seed(45)
    n = 70
    config = _make_config()
    est = TsgamEstimator(config=config)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": np.random.randn(n)}, index=dates)
    y = np.random.randn(n)

    est.fit(X, y)
    assert hasattr(est, "_parametric_problem") and est._parametric_problem is not None
    assert est._problem_shape_sig[0] == n

    # Second fit with same shape: should reuse (no recompile) and succeed
    est.fit(X, y)
    assert est.problem_.status in ("optimal", "optimal_inaccurate")
