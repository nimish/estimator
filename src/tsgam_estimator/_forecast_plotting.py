# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Notebook-friendly plotting helpers for direct multi-horizon forecasts."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING, cast

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

if TYPE_CHECKING:
    from matplotlib.axes import Axes


type ForecastPredictions = pd.DataFrame | Mapping[str, pd.DataFrame]
type FrequencyLike = str | pd.Timedelta | BaseOffset


_HORIZON_COLUMN = re.compile(r"^horizon_(\d+)$")


def _validate_time_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a pandas DatetimeIndex.")
    if not index.is_unique:
        raise ValueError(f"{name} index must contain unique timestamps.")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing time order.")
    if index.empty:
        raise ValueError(f"{name} index must not be empty.")
    return index


def _horizon_columns(predictions: pd.DataFrame) -> list[tuple[int, str]]:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("forecast predictions must be a pandas DataFrame.")
    if not predictions.columns.is_unique:
        raise ValueError("forecast prediction columns must be unique.")
    columns = []
    for column in predictions.columns:
        match = _HORIZON_COLUMN.fullmatch(str(column))
        if match is None:
            raise ValueError(
                f"invalid forecast prediction column {column!r}; expected horizon_0, "
                "horizon_1, ..."
            )
        horizon = int(match.group(1))
        canonical = f"horizon_{horizon}"
        if column != canonical:
            raise ValueError(
                f"invalid forecast prediction column {column!r}; use {canonical!r}."
            )
        columns.append((horizon, canonical))
    if not columns:
        raise ValueError(
            "forecast predictions must contain columns named horizon_0, horizon_1, ..."
        )
    columns.sort()
    horizons = [horizon for horizon, _ in columns]
    expected = list(range(horizons[-1] + 1))
    if horizons != expected:
        raise ValueError(
            "forecast prediction columns must be contiguous from horizon_0; "
            f"got {horizons}."
        )
    return columns


def _infer_offset(index: pd.DatetimeIndex) -> BaseOffset | None:
    if index.freq is not None:
        return to_offset(index.freq)
    if len(index) >= 3:
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return to_offset(inferred)
    return None


def _resolve_offset(
    origins: pd.DatetimeIndex,
    actual_index: pd.DatetimeIndex | None,
    freq: FrequencyLike | None,
) -> BaseOffset:
    if freq is not None:
        return to_offset(freq)
    offset = None
    if actual_index is not None:
        offset = _infer_offset(actual_index)
    if offset is None:
        offset = _infer_offset(origins)
    if offset is None:
        raise ValueError(
            "Could not infer forecast frequency. Pass freq explicitly, for example "
            "freq='1h'."
        )
    return offset


def _validate_actual(actual: pd.Series) -> pd.Series:
    if not isinstance(actual, pd.Series):
        raise TypeError("actual must be a pandas Series indexed by target time.")
    _validate_time_index(actual.index, name="actual")
    return actual


def _normalize_forecasts(
    forecasts: ForecastPredictions,
) -> dict[str, pd.DataFrame]:
    if isinstance(forecasts, pd.DataFrame):
        normalized = {"Forecast": forecasts}
    elif isinstance(forecasts, Mapping) and forecasts:
        normalized = {str(label): frame for label, frame in forecasts.items()}
    else:
        raise TypeError(
            "forecasts must be a prediction DataFrame or a non-empty mapping of "
            "labels to prediction DataFrames."
        )

    reference_index: pd.DatetimeIndex | None = None
    for label, predictions in normalized.items():
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError(f"forecast {label!r} must be a pandas DataFrame.")
        index = _validate_time_index(predictions.index, name=f"forecast {label!r}")
        _horizon_columns(predictions)
        if reference_index is None:
            reference_index = index
        elif not index.equals(reference_index):
            raise ValueError(
                "all forecast DataFrames must use the same forecast-origin index."
            )
    return normalized


def _target_times(
    origins: pd.DatetimeIndex,
    horizon: int,
    offset: BaseOffset,
) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(origins + horizon * offset)


def forecast_to_long_dataframe(
    predictions: pd.DataFrame,
    actual: pd.Series | None = None,
    *,
    freq: FrequencyLike | None = None,
    model: str = "Forecast",
) -> pd.DataFrame:
    """Convert origin-indexed horizon columns to target-time plotting data.

    Parameters
    ----------
    predictions
        Output from :meth:`TsgamForecastEstimator.predict`. Rows are forecast
        origins and columns are named ``horizon_0``, ``horizon_1``, and so on.
    actual
        Optional observed target series indexed by target time.
    freq
        Forecast grid frequency. It is inferred from ``predictions`` or
        ``actual`` when possible.
    model
        Label stored in the returned ``model`` column.

    Returns
    -------
    pandas.DataFrame
        Long data with ``origin_time``, ``target_time``, ``horizon``,
        ``prediction``, and, when supplied, ``actual`` columns.
    """

    origins = _validate_time_index(predictions.index, name="predictions")
    horizons = _horizon_columns(predictions)
    actual_index: pd.DatetimeIndex | None = None
    if actual is not None:
        actual = _validate_actual(actual)
        actual_index = cast(pd.DatetimeIndex, actual.index)
    offset = _resolve_offset(origins, actual_index, freq)

    frames = []
    for horizon, column in horizons:
        target_times = _target_times(origins, horizon, offset)
        frame = pd.DataFrame(
            {
                "model": model,
                "origin_time": origins,
                "target_time": target_times,
                "horizon": horizon,
                "prediction": predictions[column].to_numpy(),
            }
        )
        if actual is not None:
            frame["actual"] = actual.reindex(target_times).to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _resolve_origin(
    origin: str | pd.Timestamp | None,
    origins: pd.DatetimeIndex,
) -> pd.Timestamp:
    resolved = cast(
        pd.Timestamp,
        pd.Timestamp(origins[-1] if origin is None else origin),
    )
    if pd.isna(resolved):
        raise ValueError("forecast origin must not be NaT.")
    if resolved not in origins:
        raise ValueError(f"forecast origin {resolved!s} is not in predictions.index.")
    return resolved


def _get_axes(ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on user environment
        raise ImportError(
            "Forecast plotting requires matplotlib. Install tsgam-estimator[viz] "
            "or add matplotlib to the notebook environment."
        ) from error
    _, created_ax = plt.subplots(figsize=figsize, layout="constrained")
    return created_ax


def _finish_axes(ax: Axes, *, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left")
    ax.set_xlabel("Target time")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=30)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), frameon=False, ncols=2)


def plot_forecast_origin(
    forecasts: ForecastPredictions,
    actual: pd.Series,
    *,
    origin: str | pd.Timestamp | None = None,
    history_steps: int = 48,
    freq: FrequencyLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot history and one multi-horizon forecast from a selected origin.

    Pass a mapping such as ``{"Independent": independent_predictions,
    "Coupled": coupled_predictions}`` to compare models on the same axes.
    The forecast region is shaded and begins at the vertical forecast-origin
    marker, making the known past and predicted future explicit.
    """

    if history_steps < 0:
        raise ValueError("history_steps must be non-negative.")
    normalized = _normalize_forecasts(forecasts)
    actual = _validate_actual(actual)
    reference = next(iter(normalized.values()))
    origins = cast(pd.DatetimeIndex, reference.index)
    actual_index = cast(pd.DatetimeIndex, actual.index)
    offset = _resolve_offset(origins, actual_index, freq)
    selected_origin = _resolve_origin(origin, origins)
    max_horizon = max(
        horizon
        for predictions in normalized.values()
        for horizon, _ in _horizon_columns(predictions)
    )
    forecast_end = selected_origin + max_horizon * offset
    history_start = selected_origin - history_steps * offset

    ax = _get_axes(ax, figsize=(11.0, 4.5))
    history = actual.loc[
        (actual.index >= history_start) & (actual.index <= selected_origin)
    ]
    if not history.empty:
        ax.plot(
            history.index,
            history.to_numpy(),
            color="0.25",
            linewidth=1.8,
            label="Observed history",
        )

    future_times = pd.DatetimeIndex(
        [selected_origin + step * offset for step in range(max_horizon + 1)]
    )
    future_actual = actual.reindex(future_times)
    if future_actual.notna().any():
        ax.plot(
            future_times,
            future_actual.to_numpy(),
            color="0.1",
            linewidth=2.0,
            marker="o",
            label="Realized future",
            zorder=3,
        )

    markers = ("o", "s", "^", "D")
    for model_index, (label, predictions) in enumerate(normalized.items()):
        horizons = _horizon_columns(predictions)
        target_times = pd.DatetimeIndex(
            [selected_origin + horizon * offset for horizon, _ in horizons]
        )
        values = [predictions.loc[selected_origin, column] for _, column in horizons]
        ax.plot(
            target_times,
            values,
            linewidth=2.0,
            marker=markers[model_index % len(markers)],
            label=label,
        )

    from matplotlib import dates as mdates

    origin_number = float(mdates.date2num(selected_origin.to_pydatetime()))
    end_number = float(mdates.date2num(forecast_end.to_pydatetime()))
    ax.axvspan(origin_number, end_number, color="0.5", alpha=0.08, zorder=0)
    ax.axvline(
        origin_number,
        color="0.3",
        linestyle=":",
        linewidth=1.5,
        label="Forecast origin",
    )
    left_limit = (
        cast(pd.Timestamp, history.index.min())
        if not history.empty
        else selected_origin - offset
    )
    left_number = float(mdates.date2num(left_limit.to_pydatetime()))
    ax.set_xlim(left_number, end_number)
    _finish_axes(
        ax,
        title=f"Forecast from {selected_origin}",
        ylabel=str(actual.name or "Value"),
    )
    return ax


def plot_forecast_horizon(
    forecasts: ForecastPredictions,
    actual: pd.Series,
    *,
    horizon: int = 1,
    freq: FrequencyLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a fixed forecast horizon against observations over target time."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    normalized = _normalize_forecasts(forecasts)
    actual = _validate_actual(actual)
    reference = next(iter(normalized.values()))
    origins = cast(pd.DatetimeIndex, reference.index)
    actual_index = cast(pd.DatetimeIndex, actual.index)
    offset = _resolve_offset(origins, actual_index, freq)
    target_times = _target_times(origins, horizon, offset)

    ax = _get_axes(ax, figsize=(11.0, 4.0))
    observed = actual.reindex(target_times)
    if observed.notna().any():
        ax.plot(
            target_times,
            observed.to_numpy(),
            color="0.15",
            linewidth=2.0,
            label="Actual",
        )
    column = f"horizon_{horizon}"
    missing = [label for label, frame in normalized.items() if column not in frame]
    if missing:
        raise ValueError(
            f"{column} is missing from forecast models: {', '.join(missing)}."
        )
    markers = ("o", "s", "^", "D")
    for model_index, (label, predictions) in enumerate(normalized.items()):
        ax.plot(
            target_times,
            predictions[column].to_numpy(),
            linewidth=1.8,
            marker=markers[model_index % len(markers)],
            markersize=3.5,
            label=label,
        )
    _finish_axes(
        ax,
        title=f"Horizon {horizon}: forecast and actual over target time",
        ylabel=str(actual.name or "Value"),
    )
    return ax
