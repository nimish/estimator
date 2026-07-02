# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Literal

import cvxpy
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted

from ._estimator import (
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
)

_ForecastVariable = cvxpy.Variable | list[cvxpy.Variable]


@dataclass
class TsgamForecastCouplingConfig:
    """
    Configuration for coupled direct multi-horizon forecasts.

    Parameters
    ----------
    roughness_weight : float, default=1.0
        Weight for coefficient roughness penalties across forecast horizons.
    roughness_order : int or None, default=None
        Difference order across horizons. If None, uses 2 when horizon >= 3 and
        1 when horizon == 2.
    """

    roughness_weight: float = 1.0
    roughness_order: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.roughness_weight) or self.roughness_weight < 0:
            raise ValueError(
                f"roughness_weight must be non-negative and finite, got {self.roughness_weight!r}."
            )
        if self.roughness_order is not None and self.roughness_order not in (1, 2):
            raise ValueError(
                f"roughness_order must be 1, 2, or None, got {self.roughness_order!r}."
            )


@dataclass
class TsgamForecastConfig:
    """
    Configuration for direct multi-horizon forecast mode.

    Forecast mode trains one direct regression per horizon. For horizon ``h``,
    each row uses exogenous data available at the forecast origin and the target
    at ``origin + h``. ``predict`` returns one column per horizon.
    """

    horizon: int
    base_config: TsgamEstimatorConfig
    mode: Literal["independent", "coupled"] = "independent"
    coupling_config: TsgamForecastCouplingConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, (int, np.integer)) or isinstance(self.horizon, bool):
            raise ValueError(f"horizon must be a positive integer, got {self.horizon!r}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be positive, got {self.horizon!r}.")
        if self.mode not in ("independent", "coupled"):
            raise ValueError(f"mode must be 'independent' or 'coupled', got {self.mode!r}.")
        if self.mode == "coupled" and self.coupling_config is None:
            self.coupling_config = TsgamForecastCouplingConfig()


class TsgamForecastEstimator(BaseEstimator, RegressorMixin):
    """Direct multi-horizon forecast estimator built on the TSGAM model API."""

    def __init__(self, config: TsgamForecastConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(config, TsgamForecastConfig):
            raise TypeError(
                f"config must be a TsgamForecastConfig, got {type(config).__name__}."
            )
        self.config = config
        self._model_api = TsgamEstimator._internal_api(config.base_config)

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
        X = self._model_api.normalize_X(X)
        if sample_weight is not None:
            weight = np.asarray(sample_weight)
            if weight.ndim != 1 or weight.shape[0] != len(y):
                raise ValueError(
                    f"sample_weight must have shape (n_samples,) = ({len(y)},), got {weight.shape}"
                )
        X, y, sample_weight = self._model_api.sort_fit_inputs(X, y, sample_weight)
        timestamps = X.index
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise TypeError("Forecast inputs must have a DatetimeIndex after normalization.")
        self.freq_ = self._model_api.infer_fit_frequency(timestamps)
        self.step_ = self._model_api.step_timedelta(self.freq_)
        self.time_reference_ = timestamps[0]
        self.origin_index_ = timestamps
        return X, np.asarray(y), sample_weight

    def fit(
        self,
        X: pd.DataFrame,
        y: ndarray,
        sample_weight: ndarray | None = None,
    ) -> "TsgamForecastEstimator":
        """Fit direct forecast models for horizons 1 through ``config.horizon``."""
        X, y, sample_weight = self._prepare_fit_inputs(X, y, sample_weight)
        self.horizons_ = list(range(1, int(self.config.horizon) + 1))
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
            estimator = self._model_api.new_estimator()
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
        coupling = self.config.coupling_config or TsgamForecastCouplingConfig()
        roughness_order = coupling.roughness_order
        if roughness_order is None:
            roughness_order = 2 if self.config.horizon >= 3 else 1
        if self.config.horizon <= roughness_order:
            roughness_order = 1

        def add_horizon_roughness(
            term: cvxpy.Expression,
            coef_by_horizon: cvxpy.Expression,
        ) -> cvxpy.Expression:
            if coupling.roughness_weight == 0 or self.config.horizon < 2:
                return term
            return term + coupling.roughness_weight * cvxpy.sum_squares(
                cvxpy.diff(coef_by_horizon, k=roughness_order, axis=1)
            )

        self.exog_knots_ = self._model_api.resolve_exog_knots(X)
        horizon_data = [
            self._fit_data_for_horizon(X, y, sample_weight, horizon)
            for horizon in self.horizons_
        ]
        designs = [
            self._model_api.build_design(
                X_horizon,
                y_horizon,
                weight_horizon,
                knots_by_exog=self.exog_knots_,
                reference=self.time_reference_,
                freq=self.freq_,
            )
            for X_horizon, y_horizon, weight_horizon in horizon_data
        ]

        n_horizons = len(self.horizons_)
        self.variables_: dict[str, _ForecastVariable] = {
            "constant": cvxpy.Variable(n_horizons)
        }
        regularization_term = cvxpy.Constant(0.0)
        losses = []

        if base_config.exog_config:
            for ix, exog_cfg in enumerate(base_config.exog_config):
                basis_dim = designs[0].exog_Hs[ix][0].shape[1]
                num_lags = len(exog_cfg.lags)
                horizon_vars = [
                    cvxpy.Variable((basis_dim, num_lags))
                    for _ in self.horizons_
                ]
                self.variables_[f"exog_coef_{ix}"] = horizon_vars
                for exog_coef in horizon_vars:
                    regularization_term += exog_cfg.reg_weight * cvxpy.sum_squares(exog_coef)
                    if num_lags > 1:
                        regularization_term += (
                            exog_cfg.diff_reg_weight
                            * cvxpy.sum_squares(cvxpy.diff(exog_coef, axis=1))
                        )
                flattened = cvxpy.hstack(
                    [
                        cvxpy.reshape(exog_coef, (basis_dim * num_lags, 1), order="F")
                        for exog_coef in horizon_vars
                    ]
                )
                regularization_term = add_horizon_roughness(
                    regularization_term,
                    flattened,
                )

        if base_config.multi_periodic_config:
            assert designs[0].fourier_basis is not None
            fourier_dim = designs[0].fourier_basis.shape[1]
            fourier_coef = cvxpy.Variable((fourier_dim, n_horizons))
            self.variables_["fourier_coef"] = fourier_coef
            fourier_regularizer = self._model_api.make_regularization_matrix()
            for horizon_ix in range(n_horizons):
                regularization_term += (
                    base_config.multi_periodic_config.reg_weight
                    * cvxpy.sum_squares(fourier_regularizer @ fourier_coef[:, horizon_ix])
                )
            regularization_term = add_horizon_roughness(
                regularization_term,
                fourier_coef,
            )

        interaction_pairs = designs[0].interaction_pairs
        exog_config = base_config.exog_config
        for pair_ix, (left_ix, right_ix) in enumerate(interaction_pairs):
            assert exog_config is not None
            interaction_dim = designs[0].interaction_Hs[pair_ix].shape[1]
            interaction_coef = cvxpy.Variable((interaction_dim, n_horizons))
            self.variables_[f"interaction_coef_{pair_ix}"] = interaction_coef
            left_cfg = exog_config[left_ix]
            right_cfg = exog_config[right_ix]
            interaction_weight = float(np.sqrt(left_cfg.reg_weight * right_cfg.reg_weight))
            for horizon_ix in range(n_horizons):
                regularization_term += interaction_weight * cvxpy.sum_squares(
                    interaction_coef[:, horizon_ix]
                )
            regularization_term = add_horizon_roughness(
                regularization_term,
                interaction_coef,
            )

        regularization_term = add_horizon_roughness(
            regularization_term,
            cvxpy.reshape(self.variables_["constant"], (1, n_horizons), order="F"),
        )

        for horizon_ix, design in enumerate(designs):
            valid_mask = design.valid_mask
            model_term = self._model_api.prediction_expression(
                design,
                self.variables_,
                horizon_ix,
                valid_mask,
            )
            assert design.y is not None
            assert design.sample_weight is not None
            y_valid = design.y[valid_mask]
            weight_valid = design.sample_weight[valid_mask]
            residual = y_valid - model_term
            losses.append(
                cvxpy.sum_squares(cvxpy.multiply(np.sqrt(weight_valid), residual))
                / np.sum(weight_valid)
            )

        self.problem_ = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.sum(losses) + regularization_term),
        )
        self.problem_.solve(
            solver=base_config.solver_config.solver,
            verbose=base_config.solver_config.verbose,
            warm_start=base_config.solver_config.warm_start,
            **base_config.solver_config._solve_kwargs(),
        )
        if self.problem_.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(
                f"Optimization problem did not converge. Status: {self.problem_.status}."
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
        X = self._model_api.normalize_X(X)
        X = self._model_api.sort_predict_X(X)
        origin_index = X.index
        if not isinstance(origin_index, pd.DatetimeIndex):
            raise TypeError("Forecast inputs must have a DatetimeIndex after normalization.")
        self._model_api.validate_predict_frequency(origin_index, self.freq_)

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
                design = self._model_api.build_design(
                    X_horizon,
                    y=None,
                    sample_weight=None,
                    knots_by_exog=self.exog_knots_,
                    reference=self.time_reference_,
                    freq=self.freq_,
                )
                columns[f"horizon_{horizon}"] = self._model_api.evaluate_prediction(
                    design,
                    self.variables_,
                    horizon_ix,
                )

        return pd.DataFrame(columns, index=origin_index)
