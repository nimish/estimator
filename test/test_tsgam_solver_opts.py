# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for TsgamSolverConfig solver_opts and warm_start pass-through.

Verifies that solver_opts are forwarded as **kwargs to cvxpy Problem.solve()
and that warm_start can be toggled without breaking the fit.
"""

import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamSolverConfig,
)


def _make_simple_data(n_samples=200, seed=0):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="1h")
    X = pd.DataFrame({"x0": rng.standard_normal(n_samples)}, index=timestamps)
    y = 5.0 + 0.3 * X["x0"].values + rng.standard_normal(n_samples) * 0.1
    return X, y


@pytest.mark.parametrize("warm_start", [True, False], ids=["warm", "cold"])
def test_warm_start_flag(warm_start):
    X, y = _make_simple_data()
    config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=None,
        solver_config=TsgamSolverConfig(
            solver="CLARABEL", verbose=False, warm_start=warm_start,
        ),
    )
    est = TsgamEstimator(config=config)
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_solver_opts_forwarded():
    """solver_opts like max_iter are forwarded to the solver."""
    X, y = _make_simple_data()
    config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=None,
        solver_config=TsgamSolverConfig(
            solver="CLARABEL",
            verbose=False,
            solver_opts={"max_iter": 200, "time_limit": 120.0},
        ),
    )
    est = TsgamEstimator(config=config)
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_solver_opts_bad_option_raises():
    """An unrecognized solver option should surface an error from the solver."""
    X, y = _make_simple_data()
    config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=None,
        solver_config=TsgamSolverConfig(
            solver="CLARABEL",
            verbose=False,
            solver_opts={"not_a_real_option": 42},
        ),
    )
    est = TsgamEstimator(config=config)
    with pytest.raises(Exception, match="not_a_real_option"):
        est.fit(X, y)


@pytest.mark.parametrize("key", ["solver", "verbose", "warm_start"])
def test_solver_opts_rejects_reserved_keys(key):
    """Reserved keys in solver_opts raise ValueError before calling the solver."""
    config = TsgamSolverConfig(solver_opts={key: "ignored"})
    with pytest.raises(ValueError, match="passed explicitly"):
        config._solve_kwargs()


def test_solver_opts_defaults_to_none():
    config = TsgamSolverConfig()
    assert config.solver_opts is None
    assert config.warm_start is True
