# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for notebook-friendly forecast plotting helpers."""

import matplotlib
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from tsgam_estimator import (  # noqa: E402
    forecast_to_long_dataframe,
    plot_forecast_horizon,
    plot_forecast_origin,
)
from tsgam_estimator.tsgam_estimator import (  # noqa: E402
    forecast_to_long_dataframe as shim_forecast_to_long_dataframe,
)
from tsgam_estimator.tsgam_estimator import (  # noqa: E402
    plot_forecast_horizon as shim_plot_forecast_horizon,
)
from tsgam_estimator.tsgam_estimator import (  # noqa: E402
    plot_forecast_origin as shim_plot_forecast_origin,
)


def _plot_data() -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2025-01-01", periods=12, freq="1h")
    actual = pd.Series(np.arange(12, dtype=float), index=timestamps, name="load")
    origins = timestamps[2:8]
    independent = pd.DataFrame(
        {
            "horizon_1": actual.reindex(origins + pd.Timedelta(hours=1)).to_numpy()
            + 0.1,
            "horizon_2": actual.reindex(origins + pd.Timedelta(hours=2)).to_numpy()
            + 0.2,
        },
        index=origins,
    )
    coupled = independent + 0.05
    nowcast = actual.reindex(origins) - 0.05
    nowcast.name = "nowcast"
    return actual, nowcast, independent, coupled


def test_forecast_to_long_dataframe_aligns_origins_to_target_time():
    actual, _, predictions, _ = _plot_data()

    long = forecast_to_long_dataframe(predictions, actual, model="Independent")

    assert list(long.columns) == [
        "model",
        "origin_time",
        "target_time",
        "horizon",
        "prediction",
        "actual",
    ]
    assert len(long) == 2 * len(predictions)
    horizon_two = long[long["horizon"] == 2]
    expected_targets = predictions.index + pd.Timedelta(hours=2)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(horizon_two["target_time"]),
        expected_targets.rename("target_time"),
    )
    np.testing.assert_allclose(
        horizon_two["actual"],
        actual.reindex(expected_targets),
    )
    assert long["model"].eq("Independent").all()


def test_forecast_to_long_dataframe_accepts_explicit_single_origin_frequency():
    actual, _, predictions, _ = _plot_data()
    one_origin = predictions.iloc[[0]]

    long = forecast_to_long_dataframe(one_origin, actual.iloc[[0]], freq="1h")

    assert long["target_time"].tolist() == [
        one_origin.index[0] + pd.Timedelta(hours=1),
        one_origin.index[0] + pd.Timedelta(hours=2),
    ]


def test_plot_forecast_origin_separates_history_and_future():
    actual, nowcast, independent, coupled = _plot_data()

    ax = plot_forecast_origin(
        {"Independent": independent, "Coupled": coupled},
        actual,
        nowcast=nowcast,
        origin=independent.index[2],
        history_steps=2,
    )

    labels = {line.get_label() for line in ax.lines}
    assert labels == {
        "Observed history",
        "Realized future",
        "Independent",
        "Coupled",
        "Nowcast",
        "Forecast origin",
    }
    assert len(ax.patches) == 1
    assert ax.get_xlabel() == "Target time"
    assert ax.get_ylabel() == "load"
    plt.close(ax.figure)


def test_plot_forecast_horizon_uses_target_time_axis():
    actual, _, independent, coupled = _plot_data()

    ax = plot_forecast_horizon(
        {"Independent": independent, "Coupled": coupled},
        actual,
        horizon=2,
    )

    labels = {line.get_label() for line in ax.lines}
    assert labels == {"Actual", "Independent", "Coupled"}
    assert ax.get_title(loc="left") == (
        "Horizon 2: forecast and actual over target time"
    )
    plt.close(ax.figure)


def test_forecast_to_long_dataframe_includes_aligned_nowcast_rows():
    actual, nowcast, predictions, _ = _plot_data()

    long = forecast_to_long_dataframe(predictions, actual, nowcast=nowcast)

    horizon_zero = long[long["horizon"] == 0]
    assert len(horizon_zero) == len(predictions)
    assert horizon_zero["model"].eq("Nowcast").all()
    pd.testing.assert_series_equal(
        horizon_zero["origin_time"].reset_index(drop=True),
        horizon_zero["target_time"].reset_index(drop=True),
        check_names=False,
    )
    np.testing.assert_allclose(horizon_zero["prediction"], nowcast)


def test_plot_forecast_horizon_zero_uses_nowcast():
    actual, nowcast, independent, coupled = _plot_data()

    ax = plot_forecast_horizon(
        {"Independent": independent, "Coupled": coupled},
        actual,
        horizon=0,
        nowcast=nowcast,
    )

    labels = {line.get_label() for line in ax.lines}
    assert labels == {"Actual", "Nowcast"}
    assert ax.get_title(loc="left") == (
        "Horizon 0: forecast and actual over target time"
    )
    plt.close(ax.figure)


def test_forecast_plotting_rejects_invalid_prediction_shapes():
    actual, _, predictions, _ = _plot_data()
    invalid = predictions.rename(
        columns=lambda column: column.replace("horizon", "step")
    )

    with pytest.raises(ValueError, match="horizon_1"):
        forecast_to_long_dataframe(invalid, actual)
    with pytest.raises(ValueError, match="horizon_3"):
        plot_forecast_horizon(predictions, actual, horizon=3)
    with pytest.raises(ValueError, match="nowcast must be provided"):
        plot_forecast_horizon(predictions, actual, horizon=0)
    with pytest.raises(ValueError, match="same forecast-origin index"):
        plot_forecast_origin(
            {"first": predictions, "second": predictions.iloc[1:]},
            actual,
        )


def test_public_forecast_plotting_imports_are_available():
    assert shim_forecast_to_long_dataframe is forecast_to_long_dataframe
    assert shim_plot_forecast_origin is plot_forecast_origin
    assert shim_plot_forecast_horizon is plot_forecast_horizon
