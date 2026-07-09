# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cvxpy
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted

from ._design import (
    build_tsgam_design,
    infer_fit_frequency,
    normalize_X,
    resolve_exog_knots,
    sort_fit_inputs,
    sort_predict_X,
    step_timedelta,
    validate_predict_frequency,
)
from ._estimator import (
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
)
from ._problem import (
    evaluate_horizon_prediction,
    horizon_prediction_expression,
    make_horizon_standard_variables,
    solve_problem,
    weighted_squared_loss,
)

@dataclass
class TsgamForecastCouplingConfig:
    """
    Configuration for coupled direct multi-horizon forecasts.

    Parameters
    ----------
    roughness_weight : float, default=1.0
        Weight for coefficient roughness penalties across positive forecast
        horizons. The horizon-zero nowcast remains an uncoupled diagnostic
        baseline.
    roughness_order : {1, 2}, default=1
        Difference order across horizons. Second differences fall back to first
        differences when fewer than three horizons are fitted.
    """

    roughness_weight: float = 1.0
    roughness_order: Literal[1, 2] = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.roughness_weight, (bool, np.bool_))
            or not isinstance(
                self.roughness_weight,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(self.roughness_weight)
            or self.roughness_weight < 0
        ):
            raise ValueError(
                f"roughness_weight must be non-negative and finite, got {self.roughness_weight!r}."
            )
        if (
            isinstance(self.roughness_order, (bool, np.bool_))
            or not isinstance(self.roughness_order, (int, np.integer))
            or self.roughness_order not in (1, 2)
        ):
            raise ValueError(
                f"roughness_order must be 1 or 2, got {self.roughness_order!r}."
            )


@dataclass
class TsgamForecastConfig:
    """
    Configuration for direct multi-horizon forecast mode.

    Forecast mode trains one direct regression per horizon, including the
    horizon-zero nowcast. For horizon ``h``,
    each row uses exogenous data available at the forecast origin and the target
    at ``origin + h``. ``horizon`` is the largest requested horizon, so
    ``predict`` returns ``horizon_0`` through ``horizon_H``.
    """

    horizon: int
    base_config: TsgamEstimatorConfig
    mode: Literal["independent", "coupled"] = "independent"
    coupling_config: TsgamForecastCouplingConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, TsgamEstimatorConfig):
            raise TypeError(
                "base_config must be a TsgamEstimatorConfig, got "
                f"{type(self.base_config).__name__}."
            )
        if not isinstance(self.horizon, (int, np.integer)) or isinstance(self.horizon, bool):
            raise ValueError(f"horizon must be a non-negative integer, got {self.horizon!r}.")
        if self.horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {self.horizon!r}.")
        if self.mode not in ("independent", "coupled"):
            raise ValueError(f"mode must be 'independent' or 'coupled', got {self.mode!r}.")
        if self.mode == "independent":
            if self.coupling_config is not None:
                raise ValueError(
                    "coupling_config is only valid when mode='coupled'."
                )
        elif self.coupling_config is None:
            self.coupling_config = TsgamForecastCouplingConfig()
        elif not isinstance(self.coupling_config, TsgamForecastCouplingConfig):
            raise TypeError(
                "coupling_config must be a TsgamForecastCouplingConfig or None, got "
                f"{type(self.coupling_config).__name__}."
            )


class TsgamForecastEstimator(BaseEstimator, RegressorMixin):
    """Direct multi-horizon forecast estimator built on the TSGAM model API."""

    def __init__(self, config: TsgamForecastConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(config, TsgamForecastConfig):
            raise TypeError(
                f"config must be a TsgamForecastConfig, got {type(config).__name__}."
            )
        self.config = config

    def _target_time_X(self, X: pd.DataFrame, horizon: int) -> pd.DataFrame:
        shifted = X.copy()
        shifted.index = shifted.index + horizon * self.step_
        return shifted

    def _fit_data_for_horizon(
        self,
        X: pd.DataFrame,
        y: ndarray,
        sample_weight: ndarray | None,
        horizon: int,
    ) -> tuple[pd.DataFrame, ndarray, ndarray | None]:
        if horizon == 0:
            return X.copy(), np.asarray(y), sample_weight
        if len(y) <= horizon:
            raise ValueError(
                f"Need more samples than forecast horizon; got {len(y)} samples "
                f"and horizon={horizon}."
            )
        X_horizon = self._target_time_X(X.iloc[:-horizon], horizon)
        y_horizon = np.asarray(y)[horizon:]
        weight_horizon = None
        if sample_weight is not None:
            weight_horizon = np.asarray(sample_weight)[horizon:]
        return X_horizon, y_horizon, weight_horizon

    def _prepare_fit_inputs(
        self,
        X: pd.DataFrame,
        y: ndarray,
        sample_weight: ndarray | None,
    ) -> tuple[pd.DataFrame, ndarray, ndarray | None]:
        X = normalize_X(X)
        if sample_weight is not None:
            weight = np.asarray(sample_weight)
            if weight.ndim != 1 or weight.shape[0] != len(y):
                raise ValueError(
                    f"sample_weight must have shape (n_samples,) = ({len(y)},), got {weight.shape}"
                )
        X, y, sample_weight = sort_fit_inputs(
            X,
            sort_index=self.config.base_config.sort_index,
            y=y,
            sample_weight=sample_weight,
        )
        timestamps = X.index
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise TypeError("Forecast inputs must have a DatetimeIndex after normalization.")
        self.freq_ = infer_fit_frequency(timestamps)
        self.step_ = step_timedelta(self.freq_)
        self.time_reference_ = timestamps[0]
        return X, np.asarray(y), sample_weight

    def fit(
        self,
        X: pd.DataFrame,
        y: ndarray,
        sample_weight: ndarray | None = None,
    ) -> "TsgamForecastEstimator":
        """Fit direct models for horizons 0 through ``config.horizon``."""
        X, y, sample_weight = self._prepare_fit_inputs(X, y, sample_weight)
        self.horizons_ = list(range(int(self.config.horizon) + 1))
        if self.config.mode == "independent":
            return self._fit_independent(X, y, sample_weight)
        return self._fit_coupled(X, y, sample_weight)

    def _fit_independent(
        self,
        X: pd.DataFrame,
        y: ndarray,
        sample_weight: ndarray | None,
    ) -> "TsgamForecastEstimator":
        self.forecast_estimators_ = {}
        for horizon in self.horizons_:
            X_horizon, y_horizon, weight_horizon = self._fit_data_for_horizon(
                X, y, sample_weight, horizon
            )
            estimator = TsgamEstimator(deepcopy(self.config.base_config))
            estimator.fit(X_horizon, y_horizon, sample_weight=weight_horizon)
            self.forecast_estimators_[horizon] = estimator
        return self

    def _validate_coupled_config(self) -> None:
        base_config = self.config.base_config
        if base_config.outlier_config is not None:
            raise ValueError("Coupled forecast mode does not support outlier_config yet.")
        if (
            base_config.trend_config is not None
            and base_config.trend_config.trend_type != TrendType.NONE
        ):
            raise ValueError("Coupled forecast mode does not support trend_config yet.")
        if base_config.ar_config is not None:
            raise ValueError("Coupled forecast mode does not support ar_config yet.")

    def _fit_coupled(
        self,
        X: pd.DataFrame,
        y: ndarray,
        sample_weight: ndarray | None,
    ) -> "TsgamForecastEstimator":
        self._validate_coupled_config()
        base_config = self.config.base_config
        coupling = self.config.coupling_config
        assert coupling is not None
        roughness_order = coupling.roughness_order
        n_horizons = len(self.horizons_)
        n_forecast_horizons = n_horizons - 1
        if n_forecast_horizons <= roughness_order:
            roughness_order = 1

        def add_horizon_roughness(
            term: cvxpy.Expression,
            coef_by_horizon: cvxpy.Expression,
        ) -> cvxpy.Expression:
            if coupling.roughness_weight == 0 or n_forecast_horizons < 2:
                return term
            forecast_coefs = coef_by_horizon[:, 1:]
            return term + coupling.roughness_weight * cvxpy.sum_squares(
                cvxpy.diff(forecast_coefs, k=roughness_order, axis=1)
            )

        self.exog_knots_ = resolve_exog_knots(base_config, X)
        horizon_data = [
            self._fit_data_for_horizon(X, y, sample_weight, horizon)
            for horizon in self.horizons_
        ]
        designs = [
            build_tsgam_design(
                base_config,
                X_horizon,
                y_horizon,
                weight_horizon,
                knots_by_exog=self.exog_knots_,
                reference=self.time_reference_,
                freq=self.freq_,
            )
            for X_horizon, y_horizon, weight_horizon in horizon_data
        ]

        self.variables_, regularization_term = make_horizon_standard_variables(
            base_config,
            designs,
            horizon_regularizer=add_horizon_roughness,
        )
        losses = []

        for horizon_ix, design in enumerate(designs):
            valid_mask = design.valid_mask
            model_term = horizon_prediction_expression(
                base_config,
                design,
                self.variables_,
                horizon_ix,
                valid_mask,
            )
            assert design.y is not None
            assert design.sample_weight is not None
            y_valid = design.y[valid_mask]
            weight_valid = design.sample_weight[valid_mask]
            losses.append(weighted_squared_loss(y_valid, model_term, weight_valid))

        self.problem_ = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.sum(losses) + regularization_term),
        )
        solve_problem(
            self.problem_,
            base_config.solver_config,
            failure_message="Optimization problem did not converge.",
        )
        return self

    def predict(
        self,
        X: pd.DataFrame,
        remove_periodic: bool = False,
        remove_exogenous: bool = False,
        remove_trend: bool = False,
    ) -> pd.DataFrame:
        """Predict all configured forecast horizons for each forecast origin."""
        check_is_fitted(self, ["horizons_", "freq_", "step_"])
        X = normalize_X(X)
        X = sort_predict_X(X, sort_index=self.config.base_config.sort_index)
        origin_index = X.index
        if not isinstance(origin_index, pd.DatetimeIndex):
            raise TypeError("Forecast inputs must have a DatetimeIndex after normalization.")
        validate_predict_frequency(origin_index, self.freq_)

        columns: dict[str, ndarray] = {}
        if self.config.mode == "independent":
            check_is_fitted(self, ["forecast_estimators_"])
            for horizon in self.horizons_:
                X_horizon = self._target_time_X(X, horizon)
                columns[f"horizon_{horizon}"] = self.forecast_estimators_[horizon].predict(
                    X_horizon,
                    remove_periodic=remove_periodic,
                    remove_exogenous=remove_exogenous,
                    remove_trend=remove_trend,
                )
        else:
            check_is_fitted(self, ["problem_", "variables_", "exog_knots_"])
            if remove_periodic or remove_exogenous or remove_trend:
                raise ValueError(
                    "Component removal is not supported for coupled forecast predictions yet."
            )
            for horizon_ix, horizon in enumerate(self.horizons_):
                X_horizon = self._target_time_X(X, horizon)
                design = build_tsgam_design(
                    self.config.base_config,
                    X_horizon,
                    y=None,
                    sample_weight=None,
                    knots_by_exog=self.exog_knots_,
                    reference=self.time_reference_,
                    freq=self.freq_,
                )
                columns[f"horizon_{horizon}"] = evaluate_horizon_prediction(
                    self.config.base_config,
                    design,
                    self.variables_,
                    horizon_ix,
                )

        return pd.DataFrame(columns, index=origin_index)
