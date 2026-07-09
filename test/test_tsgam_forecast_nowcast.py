# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the h=0 nowcast diagnostic baseline."""

import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TsgamEstimatorConfig,
    TsgamForecastConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
)

_N_SAMPLES = 216
_TRAIN_SAMPLES = 144
_HORIZON = 4


def _make_deterministic_problem() -> tuple[pd.DataFrame, pd.Series]:
    timestamps = pd.date_range("2024-01-01", periods=_N_SAMPLES, freq="1h")
    sample_ix = np.arange(_N_SAMPLES, dtype=float)

    # This modular sequence is deterministic but not exactly extrapolatable by
    # the direct forecast models' linear origin features.
    driver = (((np.arange(_N_SAMPLES) * 37) % 101) - 50) / 50.0
    previous_driver = np.concatenate(([0.0], driver[:-1]))
    periodic = (
        1.2 * np.sin(2.0 * np.pi * sample_ix / 24.0)
        + 0.4 * np.cos(4.0 * np.pi * sample_ix / 24.0)
    )
    target = 3.0 + periodic + 1.5 * driver + 0.6 * previous_driver

    X = pd.DataFrame({"driver": driver}, index=timestamps)
    y = pd.Series(target, index=timestamps, name="target")
    return X, y


def _base_config() -> TsgamEstimatorConfig:
    return TsgamEstimatorConfig(
        multi_periodic_config=TsgamMultiPeriodicConfig(
            num_harmonics=[2],
            periods=[24],
            reg_weight=1.0e-10,
        ),
        exog_config=[
            TsgamLinearConfig(
                lags=[-1, 0],
                reg_weight=1.0e-10,
                diff_reg_weight=0.0,
            )
        ],
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )


@pytest.fixture(scope="module")
def nowcast_evaluation() -> tuple[
    pd.DataFrame,
    pd.DatetimeIndex,
    pd.Series,
    list[str],
]:
    X, y = _make_deterministic_problem()
    X_train = X.iloc[:_TRAIN_SAMPLES]
    y_train = y.iloc[:_TRAIN_SAMPLES].to_numpy()

    forecast_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=_HORIZON,
            base_config=_base_config(),
            mode="independent",
        )
    ).fit(X_train, y_train)

    # Every horizon is scored on this exact origin set. The preceding row is
    # prediction history for lag=-1, not an additional scored origin.
    common_origins = X.index[_TRAIN_SAMPLES : _N_SAMPLES - _HORIZON]
    prediction_X = X.iloc[_TRAIN_SAMPLES - 1 : _N_SAMPLES - _HORIZON]
    forecast_predictions = forecast_model.predict(prediction_X).loc[common_origins]

    rows = []
    for horizon in range(_HORIZON + 1):
        target_times = common_origins + horizon * pd.Timedelta(hours=1)
        predictions = forecast_predictions[f"horizon_{horizon}"].to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "origin_time": common_origins,
                    "target_time": target_times,
                    "horizon": horizon,
                    "prediction": predictions,
                    "actual": y.reindex(target_times).to_numpy(),
                }
            )
        )

    evaluation = pd.concat(rows, ignore_index=True)
    evaluation["error"] = evaluation["prediction"] - evaluation["actual"]
    return evaluation, common_origins, y, list(forecast_predictions.columns)


def test_nowcast_and_forecasts_use_identical_valid_origins(nowcast_evaluation):
    evaluation, common_origins, target, forecast_columns = nowcast_evaluation

    assert forecast_columns == [f"horizon_{h}" for h in range(_HORIZON + 1)]

    for horizon, horizon_rows in evaluation.groupby("horizon", sort=True):
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(horizon_rows["origin_time"]),
            common_origins.rename("origin_time"),
        )
        expected_targets = common_origins + int(horizon) * pd.Timedelta(hours=1)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(horizon_rows["target_time"]),
            expected_targets.rename("target_time"),
        )
        np.testing.assert_allclose(
            horizon_rows["actual"].to_numpy(),
            target.reindex(expected_targets).to_numpy(),
        )


def test_h_zero_is_an_exact_diagnostic_baseline(nowcast_evaluation):
    evaluation, _, _, _ = nowcast_evaluation
    metrics = evaluation.groupby("horizon")["error"].agg(
        rmse=lambda error: float(np.sqrt(np.mean(np.square(error)))),
        mae=lambda error: float(np.mean(np.abs(error))),
    )

    # The noiseless h=0 relationship is exactly in the model basis. Future
    # modular driver values are unavailable at each origin, leaving a wide and
    # deterministic gap rather than a fragile ordering of random errors.
    assert metrics.loc[0, "rmse"] < 1.0e-6
    assert metrics.loc[0, "mae"] < 1.0e-6
    assert (metrics.loc[1:, "rmse"] > 0.25).all()
    assert (metrics.loc[1:, "mae"] > 0.20).all()
