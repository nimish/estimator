# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for direct multi-horizon forecast mode."""

import numpy as np
import pandas as pd
import pytest
import tsgam_estimator._forecast as forecast_module

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
)
from tsgam_estimator.tsgam_estimator import (
    TsgamForecastEstimator as ShimForecastEstimator,
)


def _make_data(n_samples: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="1h")
    step = np.arange(n_samples, dtype=float)
    x = np.sin(step / 6.0) + 0.1 * np.cos(step / 3.0)
    X = pd.DataFrame({"x": x}, index=timestamps)
    y = 2.0 + 1.5 * x + 0.05 * step
    return X, y


def _base_config(reg_weight: float = 1.0e-8) -> TsgamEstimatorConfig:
    return TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=[TsgamLinearConfig(lags=[0], reg_weight=reg_weight)],
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )


def _forecast_config(
    horizon: int,
    *,
    mode: str = "independent",
    roughness_weight: float = 0.0,
    base_config: TsgamEstimatorConfig | None = None,
) -> TsgamForecastConfig:
    coupling_config = None
    if mode == "coupled":
        coupling_config = TsgamForecastCouplingConfig(
            roughness_weight=roughness_weight,
        )
    return TsgamForecastConfig(
        horizon=horizon,
        base_config=base_config or _base_config(),
        mode=mode,
        coupling_config=coupling_config,
    )


def _manual_shifted_fit(
    X: pd.DataFrame,
    y: np.ndarray,
    horizon: int,
) -> TsgamEstimator:
    shifted_X = X.iloc[:-horizon].copy()
    shifted_X.index = shifted_X.index + pd.Timedelta(hours=horizon)
    estimator = TsgamEstimator(config=_base_config())
    estimator.fit(shifted_X, y[horizon:])
    return estimator


def _predict_from_origin(
    estimator: TsgamEstimator,
    X: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    shifted_X = X.copy()
    shifted_X.index = shifted_X.index + pd.Timedelta(hours=horizon)
    return estimator.predict(shifted_X)


def test_independent_forecast_predict_returns_dataframe_columns():
    X, y = _make_data()
    estimator = TsgamForecastEstimator(config=_forecast_config(horizon=3))

    estimator.fit(X, y)
    predictions = estimator.predict(X.iloc[-10:])

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (10, 3)
    assert predictions.index.equals(X.index[-10:])
    assert list(predictions.columns) == ["horizon_1", "horizon_2", "horizon_3"]


def test_independent_forecast_aligns_child_models_to_target_time():
    X, y = _make_data()
    estimator = TsgamForecastEstimator(config=_forecast_config(horizon=4))

    estimator.fit(X, y)

    assert estimator.freq_ == "1h"
    for horizon, child in estimator.forecast_estimators_.items():
        assert child.freq_ == "1h"
        assert child.time_reference_ == X.index[0] + pd.Timedelta(hours=horizon)
        assert child.time_indices_.shape == (len(X) - horizon,)


def test_independent_forecast_matches_manual_shifted_regressions():
    X, y = _make_data()
    horizon = 3
    X_future = X.iloc[-12:]
    forecast_estimator = TsgamForecastEstimator(config=_forecast_config(horizon=horizon))

    forecast_estimator.fit(X, y)
    forecast_predictions = forecast_estimator.predict(X_future)

    for horizon_ix in range(1, horizon + 1):
        independent = _manual_shifted_fit(X, y, horizon_ix)
        expected = _predict_from_origin(independent, X_future, horizon_ix)
        np.testing.assert_allclose(
            forecast_predictions[f"horizon_{horizon_ix}"].to_numpy(),
            expected,
            rtol=1e-8,
            atol=1e-8,
        )


def test_coupled_zero_roughness_matches_manual_shifted_regressions():
    X, y = _make_data()
    forecast_estimator = TsgamForecastEstimator(
        config=_forecast_config(horizon=2, mode="coupled", roughness_weight=0.0)
    )

    forecast_estimator.fit(X, y)
    forecast_predictions = forecast_estimator.predict(X.iloc[-8:])

    for horizon_ix in (1, 2):
        independent = _manual_shifted_fit(X, y, horizon_ix)
        expected = _predict_from_origin(independent, X.iloc[-8:], horizon_ix)
        np.testing.assert_allclose(
            forecast_predictions[f"horizon_{horizon_ix}"].to_numpy(),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )


def test_coupled_forecast_uses_shared_design_module(monkeypatch):
    X, y = _make_data()
    original_build_design = forecast_module.build_tsgam_design
    calls = []

    def tracking_build_design(config, *args, **kwargs):
        calls.append(config)
        return original_build_design(config, *args, **kwargs)

    def fail_private_helper(*args, **kwargs):
        raise AssertionError("coupled forecast should use shared design functions")

    monkeypatch.setattr(
        forecast_module,
        "build_tsgam_design",
        tracking_build_design,
    )
    monkeypatch.setattr(TsgamEstimator, "_process_exog_config", fail_private_helper)
    monkeypatch.setattr(TsgamEstimator, "_normalize_interaction_pairs", fail_private_helper)
    monkeypatch.setattr(TsgamEstimator, "_make_regularization_matrix", fail_private_helper)

    TsgamForecastEstimator(
        config=_forecast_config(horizon=2, mode="coupled", roughness_weight=0.0)
    ).fit(X, y)

    assert len(calls) == 2


def test_coupled_roughness_smooths_horizon_coefficients():
    X, _ = _make_data(n_samples=120)
    x = X["x"].to_numpy()
    y = np.zeros(len(X))
    true_horizon_coefs = np.array([1.0, 5.0, -1.0, 4.0])
    for h, coef in enumerate(true_horizon_coefs, start=1):
        y[h:] += coef * x[:-h]

    unsmoothed = TsgamForecastEstimator(
        config=_forecast_config(horizon=4, mode="coupled", roughness_weight=0.0)
    ).fit(X, y)
    smoothed = TsgamForecastEstimator(
        config=_forecast_config(horizon=4, mode="coupled", roughness_weight=100.0)
    ).fit(X, y)

    unsmoothed_coefs = np.array(
        [coef.value[0, 0] for coef in unsmoothed.variables_["exog_coef_0"]]
    )
    smoothed_coefs = np.array(
        [coef.value[0, 0] for coef in smoothed.variables_["exog_coef_0"]]
    )

    assert np.sum(np.diff(smoothed_coefs) ** 2) < np.sum(
        np.diff(unsmoothed_coefs) ** 2
    )


def test_public_forecast_imports_are_available():
    assert ShimForecastEstimator is TsgamForecastEstimator


def test_coupled_forecast_rejects_irregular_predict_origins():
    X, y = _make_data()
    estimator = TsgamForecastEstimator(
        config=_forecast_config(horizon=2, mode="coupled", roughness_weight=0.0)
    )
    estimator.fit(X, y)
    irregular_X = X.iloc[[0, 1, 3, 4]]

    with pytest.raises(ValueError, match="regularly spaced"):
        estimator.predict(irregular_X)


def test_forecast_horizon_validation():
    with pytest.raises(ValueError, match="horizon must be positive"):
        TsgamForecastConfig(horizon=0, base_config=_base_config())


def test_coupled_forecast_rejects_outlier_config():
    X, y = _make_data()
    base_config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=[TsgamLinearConfig(lags=[0])],
        outlier_config=TsgamOutlierConfig(reg_weight=0.01),
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )
    estimator = TsgamForecastEstimator(
        config=_forecast_config(horizon=2, mode="coupled", base_config=base_config)
    )

    with pytest.raises(ValueError, match="outlier_config"):
        estimator.fit(X, y)
