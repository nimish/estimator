from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from tsgam_estimator import TsgamLinearConfig, TsgamSplineConfig, TrendType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.synthetic_problem import (
    SyntheticPeriodicComponent,
    SyntheticProblemConfig,
    SyntheticRegressorRelationship,
    SyntheticRegressorSpec,
    SyntheticTrendKind,
    SyntheticTrendSpec,
    build_estimator_config,
    generate_synthetic_problem,
    split_problem_frames,
    synthetic_metrics,
)


def _make_problem_config() -> SyntheticProblemConfig:
    return SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=24 * 14,
        freq="1h",
        train_fraction=0.75,
        seed=7,
        noise_scale=0.05,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.2,
            ),
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=24.0 * 7.0,
                harmonics=1,
                amplitude=0.5,
            ),
        ),
        regressors=(
            SyntheticRegressorSpec(
                name="temp",
                relationship=SyntheticRegressorRelationship.LINEAR,
                effect_scale=0.8,
                driver_period_hours=24.0,
                driver_noise_scale=0.2,
                lags=(0,),
            ),
            SyntheticRegressorSpec(
                name="wind",
                relationship=SyntheticRegressorRelationship.NONLINEAR,
                effect_scale=0.6,
                driver_period_hours=12.0,
                driver_noise_scale=0.1,
                lags=(0,),
                n_knots=7,
            ),
        ),
        trend=SyntheticTrendSpec(
            kind=SyntheticTrendKind.LINEAR,
            amplitude=0.4,
            grouping_hours=24.0,
        ),
    )


def test_generate_synthetic_problem_is_deterministic():
    config = _make_problem_config()

    first = generate_synthetic_problem(config)
    second = generate_synthetic_problem(config)

    assert_frame_equal(first.X, second.X)
    assert_series_equal(first.y, second.y)
    assert_frame_equal(first.truth_components, second.truth_components)
    assert_series_equal(first.signal, second.signal)
    assert_series_equal(first.noise, second.noise)


def test_generate_synthetic_problem_tracks_component_breakdown():
    config = _make_problem_config()

    problem = generate_synthetic_problem(config)

    expected_columns = {
        "periodic:daily",
        "periodic:weekly",
        "regressor:temp",
        "regressor:wind",
        "trend",
    }
    assert expected_columns.issubset(problem.truth_components.columns)
    assert list(problem.X.columns) == ["temp", "wind"]
    assert math.isclose(problem.X["temp"].mean(), 0.0, abs_tol=1e-10)
    assert math.isclose(problem.X["wind"].mean(), 0.0, abs_tol=1e-10)
    assert math.isclose(problem.X["temp"].std(ddof=0), 1.0, rel_tol=1e-6)
    assert math.isclose(problem.X["wind"].std(ddof=0), 1.0, rel_tol=1e-6)

    reconstructed = problem.truth_components.sum(axis=1) + problem.noise
    np.testing.assert_allclose(problem.y.values, reconstructed.values)


def test_build_estimator_config_mixes_linear_and_spline_terms():
    config = _make_problem_config()

    estimator_config = build_estimator_config(
        config,
        solver_name="CLARABEL",
        fourier_reg_weight=1.0e-5,
        linear_reg_weight=2.0e-4,
        spline_reg_weight=3.0e-4,
        spline_diff_reg_weight=0.7,
        trend_reg_weight=5.0,
    )

    assert estimator_config.multi_periodic_config is not None
    assert estimator_config.multi_periodic_config.periods == [24.0, 24.0 * 7.0]
    assert estimator_config.multi_periodic_config.num_harmonics == [2, 1]
    assert estimator_config.multi_periodic_config.reg_weight == 1.0e-5
    assert estimator_config.exog_config is not None
    assert isinstance(estimator_config.exog_config[0], TsgamLinearConfig)
    assert estimator_config.exog_config[0].lags == [0]
    assert estimator_config.exog_config[0].reg_weight == 2.0e-4
    assert isinstance(estimator_config.exog_config[1], TsgamSplineConfig)
    assert estimator_config.exog_config[1].lags == [0]
    assert estimator_config.exog_config[1].n_knots == 7
    assert estimator_config.exog_config[1].reg_weight == 3.0e-4
    assert estimator_config.exog_config[1].diff_reg_weight == 0.7
    assert estimator_config.trend_config is not None
    assert estimator_config.trend_config.trend_type == TrendType.LINEAR
    assert estimator_config.trend_config.grouping == 24.0
    assert estimator_config.trend_config.reg_weight == 5.0


def test_split_problem_frames_and_metrics_smoke():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)

    split = split_problem_frames(problem)

    assert len(split.X_train) == int(config.n_samples * config.train_fraction)
    assert len(split.X_test) == config.n_samples - len(split.X_train)
    assert split.X_train.index[-1] < split.X_test.index[0]
    assert split.y_train.index.equals(split.X_train.index)
    assert split.y_test.index.equals(split.X_test.index)

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.5, 2.5, 4.5])
    metrics = synthetic_metrics(y_true, y_pred)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert math.isclose(metrics["mae"], 0.5)
    assert math.isclose(metrics["rmse"], 0.5)
    assert math.isclose(metrics["r2"], 0.8)
