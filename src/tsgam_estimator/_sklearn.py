# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Small scikit-learn parameter adapter for nested config dataclasses."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import Field
from typing import Callable, ClassVar, Self, cast

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.utils.validation import check_consistent_length


class _PredictionRegressor(RegressorMixin, BaseEstimator):
    """Pass already-computed predictions through sklearn's public scorer API."""

    def __init__(self, predictions: ndarray):
        self.predictions = predictions

    def predict(self, X: object) -> ndarray:
        return self.predictions


def make_supported_scorer(scoring: str) -> Callable[..., float]:
    """Wrap a named sklearn regression scorer to omit NaN prediction rows.

    Predict once on the complete input window before filtering: dropping rows
    before prediction would change lag support. Targets must remain finite.
    For multi-output predictions, use rows supported by every output. When
    comparing different lag configurations, their scored supports may differ.
    """
    scorer = get_scorer(scoring)

    def supported_score(estimator, X, y, sample_weight=None) -> float:
        from ._design import sort_fit_inputs
        from ._estimator import TsgamEstimator
        from ._forecast import TsgamForecastEstimator

        # TSGAM returns chronological predictions, unlike ordinary sklearn
        # regressors. Apply the same permutation to targets and weights first.
        if isinstance(estimator, (TsgamEstimator, TsgamForecastEstimator)):
            config = estimator.config.base_config if isinstance(estimator, TsgamForecastEstimator) else estimator.config
            check_consistent_length(X, y, sample_weight)
            X, y, sample_weight = sort_fit_inputs(
                X, sort_index=config.sort_index, y=np.asarray(y), sample_weight=sample_weight,
            )
        predicted = np.asarray(estimator.predict(X), dtype=float)
        target = np.asarray(y, dtype=float)
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]
        if predicted.ndim == 2 and predicted.shape[1] == 1:
            predicted = predicted[:, 0]
        if target.shape != predicted.shape or target.ndim not in (1, 2):
            raise ValueError("Targets and predictions must have matching 1D or 2D shapes.")
        if not np.all(np.isfinite(target)) or np.any(np.isinf(predicted)):
            raise ValueError("Targets must be finite and predictions must not contain infinity.")
        supported = ~np.isnan(predicted)
        if predicted.ndim == 2:
            supported = supported.all(axis=1)
        if not np.any(supported):
            raise ValueError("No supported prediction rows are available for scoring.")
        kwargs = {}
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float)
            check_consistent_length(target, weight)
            if weight.ndim != 1 or not np.all(np.isfinite(weight)) or np.any(weight < 0):
                raise ValueError("sample_weight must be a finite nonnegative 1D array.")
            if weight[supported].sum() <= 0:
                raise ValueError("Supported rows must have positive total sample weight.")
            kwargs["sample_weight"] = weight[supported]
        return float(scorer(
            _PredictionRegressor(predicted[supported]),
            np.empty((int(supported.sum()), 0)), target[supported], **kwargs,
        ))

    return supported_score


class SklearnConfigMixin:
    """Expose dataclass fields through scikit-learn's parameter protocol."""

    __dataclass_fields__: ClassVar[dict[str, Field[object]]]

    def __sklearn_clone__(self) -> Self:
        """Clone config value objects without rerunning dataclass normalization."""
        return deepcopy(self)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        params = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.init
        }
        if not deep:
            return params

        nested_params: dict[str, object] = {}
        for name, value in params.items():
            self._collect_nested_params(name, value, nested_params)
        params.update(nested_params)
        return params

    @classmethod
    def _collect_nested_params(
        cls,
        prefix: str,
        value: object,
        params: dict[str, object],
    ) -> None:
        if isinstance(value, SklearnConfigMixin):
            for name, nested_value in value.get_params(deep=True).items():
                params[f"{prefix}__{name}"] = nested_value
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, SklearnConfigMixin):
                    item_prefix = f"{prefix}__{index}"
                    params[item_prefix] = item
                    cls._collect_nested_params(item_prefix, item, params)

    def set_params(self, **params: object) -> Self:
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        for name, value in params.items():
            if name not in valid_params:
                valid_names = sorted(valid_params)
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {valid_names!r}."
                )
            path = name.split("__")
            self._set_nested_value(self, path, value, full_name=name)
        return self

    @classmethod
    def _set_nested_value(
        cls,
        target: object,
        path: list[str],
        value: object,
        *,
        full_name: str,
    ) -> None:
        component = path[0]
        if isinstance(target, list):
            mutable_target = cast(list[object], target)
            try:
                index = int(component)
            except ValueError as error:
                raise ValueError(
                    f"Invalid list index {component!r} in parameter {full_name!r}."
                ) from error
            if len(path) == 1:
                mutable_target[index] = value
            else:
                cls._set_nested_value(
                    mutable_target[index],
                    path[1:],
                    value,
                    full_name=full_name,
                )
            return

        if len(path) == 1:
            setattr(target, component, value)
            return
        cls._set_nested_value(
            getattr(target, component),
            path[1:],
            value,
            full_name=full_name,
        )
