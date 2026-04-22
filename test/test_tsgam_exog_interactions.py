# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for exact-pair exogenous interaction terms."""

import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TsgamArConfig,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
)


SOLVER = TsgamSolverConfig(solver="CLARABEL", verbose=False)


def _make_additive_plus_interaction_data(
    n_samples: int = 240,
    freq: str = "1h",
    seed: int = 42,
    noise_scale: float = 0.03,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=n_samples, freq=freq)
    x0 = rng.standard_normal(n_samples)
    x1 = rng.standard_normal(n_samples)
    x2 = rng.standard_normal(n_samples)
    noise = rng.normal(scale=noise_scale, size=n_samples)
    y = 1.0 + 0.4 * x0 - 0.3 * x1 + 0.2 * x2 + 1.2 * x0 * x1 + noise
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2}, index=index)
    return X, y


def _make_pure_interaction_data(
    n_samples: int = 240,
    freq: str = "1h",
    seed: int = 123,
    noise_scale: float = 0.03,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-06-01", periods=n_samples, freq=freq)
    x0 = rng.standard_normal(n_samples)
    x1 = rng.standard_normal(n_samples)
    noise = rng.normal(scale=noise_scale, size=n_samples)
    y = 0.5 + 1.8 * x0 * x1 + noise
    X = pd.DataFrame({"x0": x0, "x1": x1}, index=index)
    return X, y


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_interaction_contribution_matches_explicit_design_matrix():
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=None,
        )
    )
    left_H = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    right_H = np.array(
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    interaction_coef = np.arange(6, dtype=float) - 1.5

    expected = est._outer_column_product(left_H, right_H) @ interaction_coef
    actual = est._interaction_contribution_from_blocks(
        left_H,
        right_H,
        interaction_coef,
    )

    np.testing.assert_allclose(actual, expected)


def test_interaction_pairs_default_matches_explicit_none():
    X, y = _make_additive_plus_interaction_data(n_samples=160)
    base_kwargs = {
        "multi_periodic_config": None,
        "exog_config": [
            TsgamLinearConfig(lags=[0]),
            TsgamLinearConfig(lags=[0]),
        ],
        "solver_config": SOLVER,
    }

    est_default = TsgamEstimator(config=TsgamEstimatorConfig(**base_kwargs))
    est_explicit_none = TsgamEstimator(
        config=TsgamEstimatorConfig(
            **base_kwargs,
            interaction_pairs=None,
        )
    )

    est_default.fit(X[["x0", "x1"]], y)
    est_explicit_none.fit(X[["x0", "x1"]], y)

    np.testing.assert_allclose(
        est_default.predict(X[["x0", "x1"]]),
        est_explicit_none.predict(X[["x0", "x1"]]),
    )


def test_empty_interaction_pairs_are_noop_without_exog():
    X, y = _make_additive_plus_interaction_data(n_samples=160)
    base_kwargs = {
        "multi_periodic_config": None,
        "exog_config": None,
        "solver_config": SOLVER,
    }

    est_none = TsgamEstimator(
        config=TsgamEstimatorConfig(
            **base_kwargs,
            interaction_pairs=None,
        )
    )
    est_empty = TsgamEstimator(
        config=TsgamEstimatorConfig(
            **base_kwargs,
            interaction_pairs=[],
        )
    )

    est_none.fit(X[["x0"]], y)
    est_empty.fit(X[["x0"]], y)

    np.testing.assert_allclose(
        est_none.predict(X[["x0"]]),
        est_empty.predict(X[["x0"]]),
    )


def test_interaction_pairs_require_zero_lag_on_both_terms():
    X, y = _make_additive_plus_interaction_data(n_samples=160)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[
                TsgamLinearConfig(lags=[-1, 1]),
                TsgamLinearConfig(lags=[0]),
            ],
            interaction_pairs=[(0, 1)],
            solver_config=SOLVER,
        )
    )

    with pytest.raises(ValueError, match="lag=0"):
        est.fit(X[["x0", "x1"]], y)


@pytest.mark.parametrize("exog_config", [None, []], ids=["none", "empty"])
def test_interaction_pairs_require_nonempty_exog_config(exog_config):
    X, y = _make_additive_plus_interaction_data(n_samples=120)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=exog_config,
            interaction_pairs=[(0, 1)],
            solver_config=SOLVER,
        )
    )

    with pytest.raises(ValueError, match="requires a non-empty exog_config"):
        est.fit(X[["x0", "x1"]], y)


def test_interaction_pairs_require_integer_indices():
    X, y = _make_additive_plus_interaction_data(n_samples=160)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[
                TsgamLinearConfig(lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            interaction_pairs=[(0.9, 1)],
            solver_config=SOLVER,
        )
    )

    with pytest.raises(ValueError, match="integer"):
        est.fit(X[["x0", "x1"]], y)


@pytest.mark.parametrize(
    ("interaction_pairs", "message"),
    [
        ([(0, 0)], "self-pairs"),
        ([(0, 2)], "out of range"),
        ([(0, 1), (1, 0)], "Duplicate"),
        ([(0, 1, 2)], "exactly two"),
    ],
)
def test_interaction_pairs_validate_pair_structure(interaction_pairs, message):
    X, y = _make_additive_plus_interaction_data(n_samples=160)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[
                TsgamLinearConfig(lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            interaction_pairs=interaction_pairs,
            solver_config=SOLVER,
        )
    )

    with pytest.raises(ValueError, match=message):
        est.fit(X[["x0", "x1"]], y)


@pytest.mark.parametrize(
    ("exog_config", "expected_shape"),
    [
        pytest.param(
            [
                TsgamLinearConfig(lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            (1,),
            id="linear-linear",
        ),
        pytest.param(
            [
                TsgamLinearConfig(lags=[0]),
                TsgamSplineConfig(n_knots=5, lags=[0]),
            ],
            (4,),
            id="linear-spline",
        ),
        pytest.param(
            [
                TsgamSplineConfig(n_knots=5, lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            (4,),
            id="spline-linear",
        ),
        pytest.param(
            [
                TsgamSplineConfig(n_knots=5, lags=[0]),
                TsgamSplineConfig(n_knots=5, lags=[0]),
            ],
            (16,),
            id="spline-spline",
        ),
    ],
)
def test_interaction_pairings_have_expected_coefficient_shapes(exog_config, expected_shape):
    X, y = _make_additive_plus_interaction_data(n_samples=180)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=exog_config,
            interaction_pairs=[(0, 1)],
            solver_config=SOLVER,
        )
    )

    est.fit(X[["x0", "x1"]], y)

    coef = est.variables_["interaction_coef_0"].value
    assert coef is not None
    assert coef.shape == expected_shape


def test_interactions_use_only_current_index_when_main_effects_have_lags():
    X, y = _make_additive_plus_interaction_data(n_samples=220)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[
                TsgamLinearConfig(lags=[-1, 0, 1]),
                TsgamLinearConfig(lags=[-1, 0, 1]),
            ],
            interaction_pairs=[(0, 1)],
            solver_config=SOLVER,
        )
    )

    est.fit(X[["x0", "x1"]], y)

    assert est.variables_["exog_coef_0"].value.shape == (1, 3)
    assert est.variables_["exog_coef_1"].value.shape == (1, 3)
    assert est.variables_["interaction_coef_0"].value.shape == (1,)


def test_interactions_improve_held_out_predictions():
    X, y = _make_pure_interaction_data()
    split = 160
    X_train = X.iloc[:split]
    y_train = y[:split]
    X_test = X.iloc[split:]
    y_test = y[split:]

    base_config = dict(
        multi_periodic_config=None,
        exog_config=[
            TsgamLinearConfig(lags=[0]),
            TsgamLinearConfig(lags=[0]),
        ],
        solver_config=SOLVER,
    )
    additive_est = TsgamEstimator(config=TsgamEstimatorConfig(**base_config))
    interaction_est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            **base_config,
            interaction_pairs=[(0, 1)],
        )
    )

    additive_est.fit(X_train, y_train)
    interaction_est.fit(X_train, y_train)

    additive_rmse = _rmse(y_test, additive_est.predict(X_test))
    interaction_rmse = _rmse(y_test, interaction_est.predict(X_test))

    assert interaction_rmse < 0.4 * additive_rmse


def test_interactions_work_with_ar_baseline_and_sampling():
    X, y = _make_additive_plus_interaction_data(n_samples=220)
    est = TsgamEstimator(
        config=TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=[
                TsgamLinearConfig(lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            interaction_pairs=[(0, 1)],
            ar_config=TsgamArConfig(lags=[1]),
            solver_config=SOLVER,
            random_state=np.random.RandomState(0),
        )
    )

    est.fit(X[["x0", "x1"]], y)

    samples = est.sample(X[["x0", "x1"]], n_samples=4)
    assert samples.shape == (4, len(X))
    assert np.all(np.isfinite(samples))
