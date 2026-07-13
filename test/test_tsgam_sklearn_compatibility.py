# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
)


def _config(*, harmonics: int = 1) -> TsgamEstimatorConfig:
    return TsgamEstimatorConfig(
        multi_periodic_config=TsgamMultiPeriodicConfig(
            num_harmonics=[harmonics],
            periods=[24.0],
            reg_weight=1.0e-6,
        ),
        exog_config=[
            TsgamLinearConfig(
                lags=[0],
                reg_weight=1.0e-6,
                diff_reg_weight=0.0,
            )
        ],
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )


def _data(n_samples: int = 192) -> tuple[pd.DataFrame, np.ndarray]:
    index = pd.date_range("2025-01-01", periods=n_samples, freq="1h")
    sample = np.arange(n_samples, dtype=float)
    X = pd.DataFrame({"driver": np.zeros(n_samples)}, index=index)
    y = (
        2.0
        + np.sin(2.0 * np.pi * sample / 24.0)
        + 0.5 * np.cos(4.0 * np.pi * sample / 24.0)
    )
    return X, y


def test_estimator_is_a_cloneable_regressor():
    estimator = TsgamEstimator(_config())

    cloned = clone(estimator)

    assert is_regressor(estimator)
    assert isinstance(cloned, TsgamEstimator)
    assert cloned.config is not estimator.config
    assert cloned.config.multi_periodic_config is not (
        estimator.config.multi_periodic_config
    )


def test_nested_configs_follow_sklearn_parameter_protocol():
    estimator = TsgamEstimator(_config())

    params = estimator.get_params(deep=True)

    assert params["config__multi_periodic_config__reg_weight"] == 1.0e-6
    assert params["config__exog_config__0__reg_weight"] == 1.0e-6
    assert params["config__solver_config__verbose"] is False

    returned = estimator.set_params(
        config__multi_periodic_config__reg_weight=0.25,
        config__exog_config__0__reg_weight=0.5,
    )

    assert returned is estimator
    assert estimator.config.multi_periodic_config is not None
    assert estimator.config.multi_periodic_config.reg_weight == 0.25
    assert estimator.config.exog_config is not None
    assert estimator.config.exog_config[0].reg_weight == 0.5

    with pytest.raises(ValueError, match="Invalid parameter"):
        estimator.set_params(config__not_a_parameter=1)


def test_fit_records_sklearn_feature_metadata():
    X, y = _data()

    estimator = TsgamEstimator(_config()).fit(X, y)

    assert estimator.n_features_in_ == 1
    np.testing.assert_array_equal(estimator.feature_names_in_, ["driver"])


def test_grid_search_tunes_nested_tsgam_config():
    X, y = _data()
    search = GridSearchCV(
        TsgamEstimator(_config()),
        param_grid={
            "config__multi_periodic_config__num_harmonics": [[1], [2]],
        },
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        error_score="raise",
    )

    search.fit(X, y)

    assert search.best_params_ == {
        "config__multi_periodic_config__num_harmonics": [2]
    }
    assert is_regressor(search.best_estimator_)
    assert search.best_estimator_.n_features_in_ == 1
