# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd

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


def test_sample_uses_config_random_state_when_argument_is_none():
    X, y = _make_simple_data()
    config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=None,
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
        random_state=123,
    )
    est = TsgamEstimator(config=config)
    est.fit(X, y)

    samples1 = est.sample(X, n_samples=3)
    samples2 = est.sample(X, n_samples=3)

    np.testing.assert_array_equal(
        samples1,
        samples2,
        err_msg="sample() should fall back to config.random_state when no method seed is provided",
    )
