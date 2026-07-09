#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=6.0.0",
#     "clarabel>=0.11.1",
#     "cvxpy>=1.7.3",
#     "marimo>=0.23.4",
#     "numpy>=2.3.0",
#     "pandas>=2.3.0",
#     "scikit-learn>=1.7.0",
#     "scipy>=1.16.0",
#     "spcqe>=0.3.0",
# ]
# ///

"""Interactive explorer for forecast origins, targets, and causal features."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
# Forecast-Origin Explorer

Scrub an evaluation origin to inspect one direct multi-horizon forecast: what was
observed then, which lagged driver values were available, and which future targets
the independent and coupled models predict.

Run with `uv run --group notebooks marimo edit
examples/example_forecast_origin_explorer_marimo.py`.
""")
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tsgam_estimator import (
        TsgamEstimatorConfig,
        TsgamForecastConfig,
        TsgamForecastCouplingConfig,
        TsgamForecastEstimator,
        TsgamLinearConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
    )

    alt.data_transformers.disable_max_rows()

    def style_chart(chart):
        return (
            chart.configure_view(stroke="#d9dee8")
            .configure_axis(
                domainColor="#c7ccd8",
                gridColor="#e6e8ef",
                labelColor="#344054",
                labelFontSize=11,
                tickColor="#c7ccd8",
                titleColor="#1f2937",
                titleFontSize=12,
            )
            .configure_legend(
                labelColor="#1f2937",
                labelFontSize=12,
                orient="top",
                titleColor="#1f2937",
                titleFontSize=12,
            )
            .configure_title(
                anchor="start",
                color="#111827",
                fontSize=16,
                fontWeight="bold",
            )
        )

    return (
        TsgamEstimatorConfig,
        TsgamForecastConfig,
        TsgamForecastCouplingConfig,
        TsgamForecastEstimator,
        TsgamLinearConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        alt,
        mo,
        np,
        pd,
        style_chart,
    )


@app.cell
def _(
    TsgamEstimatorConfig,
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    np,
    pd,
):
    forecast_horizon = 5
    history_hours = 48
    exog_lags = [-2, -1, 0]

    def _base_config():
        return TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[2],
                periods=[24],
                reg_weight=1.0e-5,
            ),
            exog_config=[
                TsgamLinearConfig(
                    lags=exog_lags,
                    reg_weight=1.0e-5,
                    diff_reg_weight=1.0e-3,
                )
            ],
            solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
        )

    _rng = np.random.default_rng(23)
    _samples = 456
    _timestamps = pd.date_range("2025-01-01", periods=_samples, freq="1h")
    _sample_index = np.arange(_samples, dtype=float)
    _driver = np.empty(_samples)
    _driver[0] = _rng.normal()
    for _index in range(1, _samples):
        _driver[_index] = 0.55 * _driver[_index - 1] + _rng.normal(scale=0.8)
    _driver = (_driver - _driver.mean()) / _driver.std()

    _periodic = 1.65 * np.sin(2.0 * np.pi * _sample_index / 24.0) + 0.45 * np.cos(
        2.0 * np.pi * _sample_index / 12.0
    )
    _observed = _periodic.copy()
    _horizon_weights = np.array([0.95, 0.78, 0.62, 0.46, 0.35])
    _lag_weights = np.array([0.20, -0.15, 0.95])
    for _horizon, _horizon_weight in enumerate(_horizon_weights, start=1):
        for _lag, _lag_weight in zip(exog_lags, _lag_weights, strict=True):
            _source_index = np.arange(_samples) - _horizon + _lag
            _valid = _source_index >= 0
            _observed[_valid] += (
                _horizon_weight * _lag_weight * _driver[_source_index[_valid]]
            )
    _observed += _rng.normal(scale=0.08, size=_samples)

    frame = pd.DataFrame(
        {
            "observed": _observed,
            "periodic": _periodic,
            "driver": _driver,
        },
        index=_timestamps,
    )
    _train_stop = 336
    _x_train = frame[["driver"]].iloc[:_train_stop]
    _y_train = frame["observed"].to_numpy()[:_train_stop]
    _x_eval = frame[["driver"]].iloc[_train_stop:-forecast_horizon]

    _independent = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=forecast_horizon,
            base_config=_base_config(),
            mode="independent",
        )
    ).fit(_x_train, _y_train)
    _coupled = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=forecast_horizon,
            base_config=_base_config(),
            mode="coupled",
            coupling_config=TsgamForecastCouplingConfig(roughness_weight=4.0),
        )
    ).fit(_x_train, _y_train)
    independent_prediction = _independent.predict(_x_eval)
    coupled_prediction = _coupled.predict(_x_eval)
    actual = pd.DataFrame(
        {
            f"horizon_{_horizon}": frame["observed"].shift(-_horizon).loc[_x_eval.index]
            for _horizon in range(1, forecast_horizon + 1)
        },
        index=_x_eval.index,
    )
    origin_times = _x_eval.index
    _all_values = np.concatenate(
        [
            frame["observed"].to_numpy(),
            independent_prediction.to_numpy().ravel(),
            coupled_prediction.to_numpy().ravel(),
        ]
    )
    _padding = max(0.25, 0.08 * (np.nanmax(_all_values) - np.nanmin(_all_values)))
    y_domain = (
        float(np.floor(10.0 * (np.nanmin(_all_values) - _padding)) / 10.0),
        float(np.ceil(10.0 * (np.nanmax(_all_values) + _padding)) / 10.0),
    )
    return (
        actual,
        coupled_prediction,
        exog_lags,
        forecast_horizon,
        frame,
        history_hours,
        independent_prediction,
        origin_times,
        y_domain,
    )


@app.cell
def _(mo, origin_times):
    origin_slider = mo.ui.slider(
        start=0,
        stop=len(origin_times) - 1,
        step=1,
        value=min(48, len(origin_times) - 1),
        label="Evaluation forecast origin",
        full_width=True,
    )
    origin_slider
    return (origin_slider,)


@app.cell
def _(mo, origin_slider, origin_times):
    selected_origin = origin_times[origin_slider.value]
    mo.md(
        f"## Selected origin: `{selected_origin:%a, %b %d %Y %H:%M}`\n\n"
        "The shaded region starts at this origin. All models see the same causal "
        "driver window shown below; each horizon maps it to a separate target time."
    )
    return (selected_origin,)


@app.cell
def _(
    actual,
    alt,
    coupled_prediction,
    forecast_horizon,
    frame,
    history_hours,
    independent_prediction,
    np,
    pd,
    selected_origin,
    style_chart,
    y_domain,
):
    _history_start = selected_origin - pd.Timedelta(hours=history_hours)
    _history = frame.loc[_history_start:selected_origin, ["observed"]].reset_index(
        names="timestamp"
    )
    _history = _history.rename(columns={"observed": "value"})
    _history["relative_hour"] = (
        _history["timestamp"] - selected_origin
    ).dt.total_seconds() / 3600.0
    _history["series"] = "Observed history"

    _future = pd.DataFrame(
        {
            "horizon": np.arange(1, forecast_horizon + 1),
            "timestamp": selected_origin
            + pd.to_timedelta(np.arange(1, forecast_horizon + 1), unit="h"),
            "Actual future targets": actual.loc[selected_origin].to_numpy(),
            "Independent forecast": independent_prediction.loc[
                selected_origin
            ].to_numpy(),
            "Coupled forecast": coupled_prediction.loc[selected_origin].to_numpy(),
        }
    )
    _future_long = _future.melt(
        id_vars=["horizon", "timestamp"],
        var_name="series",
        value_name="value",
    )
    _future_long["relative_hour"] = _future_long["horizon"].astype(float)
    _color = alt.Color(
        "series:N",
        legend=alt.Legend(title=None, direction="horizontal"),
        scale=alt.Scale(
            domain=[
                "Observed history",
                "Actual future targets",
                "Independent forecast",
                "Coupled forecast",
            ],
            range=["#334155", "#0f766e", "#d97706", "#2563eb"],
        ),
    )
    _x = alt.X(
        "relative_hour:Q",
        axis=alt.Axis(values=[-48, -36, -24, -12, 0, 1, 2, 3, 4, 5]),
        scale=alt.Scale(domain=[-history_hours, forecast_horizon]),
        title="hours relative to forecast origin",
    )
    _y = alt.Y(
        "value:Q",
        scale=alt.Scale(domain=y_domain),
        title="synthetic target",
    )
    _history_tooltip = [
        alt.Tooltip("timestamp:T", format="%a, %b %d %H:%M", title="observed time"),
        alt.Tooltip("value:Q", format=".3f", title="observed"),
    ]
    _future_tooltip = [
        alt.Tooltip("timestamp:T", format="%a, %b %d %H:%M", title="target time"),
        alt.Tooltip("horizon:Q", title="horizon"),
        alt.Tooltip("series:N", title="series"),
        alt.Tooltip("value:Q", format=".3f", title="value"),
    ]
    _prediction_region = (
        alt.Chart(
            pd.DataFrame(
                {
                    "start": [0.0],
                    "end": [float(forecast_horizon)],
                    "low": [y_domain[0]],
                    "high": [y_domain[1]],
                }
            )
        )
        .mark_rect(color="#e8f1fb", opacity=0.9)
        .encode(
            x=alt.X(
                "start:Q",
                axis=None,
                scale=alt.Scale(domain=[-history_hours, forecast_horizon]),
            ),
            x2="end:Q",
            y=alt.Y("low:Q", axis=None, scale=alt.Scale(domain=y_domain)),
            y2="high:Q",
        )
    )
    _history_layer = (
        alt.Chart(_history)
        .mark_line(strokeWidth=2.6)
        .encode(x=_x, y=_y, color=_color, tooltip=_history_tooltip)
    )
    _actual_layer = (
        alt.Chart(_future_long.query("series == 'Actual future targets'"))
        .mark_line(
            point=alt.OverlayMarkDef(filled=True, size=62),
            strokeDash=[5, 3],
            strokeWidth=2.4,
        )
        .encode(x=_x, y=_y, color=_color, tooltip=_future_tooltip)
    )
    _model_layers = (
        alt.Chart(_future_long.query("series != 'Actual future targets'"))
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=62), strokeWidth=2.4)
        .encode(x=_x, y=_y, color=_color, tooltip=_future_tooltip)
    )
    _origin_rule = (
        alt.Chart(pd.DataFrame({"relative_hour": [0.0]}))
        .mark_rule(color="#111827", strokeDash=[4, 3], strokeWidth=1.6)
        .encode(
            x=alt.X(
                "relative_hour:Q",
                axis=None,
                scale=alt.Scale(domain=[-history_hours, forecast_horizon]),
            )
        )
    )
    _origin_label = (
        alt.Chart(
            pd.DataFrame(
                {
                    "relative_hour": [0.35],
                    "value": [y_domain[1] - 0.05 * (y_domain[1] - y_domain[0])],
                    "label": ["now / forecast origin"],
                }
            )
        )
        .mark_text(align="left", baseline="top", color="#111827", dx=4, fontSize=11)
        .encode(
            x=alt.X(
                "relative_hour:Q",
                axis=None,
                scale=alt.Scale(domain=[-history_hours, forecast_horizon]),
            ),
            y=alt.Y("value:Q", axis=None, scale=alt.Scale(domain=y_domain)),
            text="label:N",
        )
    )
    _origin_chart = alt.layer(
        _prediction_region,
        _history_layer,
        _actual_layer,
        _model_layers,
        _origin_rule,
        _origin_label,
    ).properties(
        height=360,
        title="Observed history and direct forecast paths from the selected origin",
        width=850,
    )
    style_chart(_origin_chart)
    return


@app.cell
def _(
    actual,
    exog_lags,
    forecast_horizon,
    frame,
    mo,
    pd,
    selected_origin,
):
    _feature_rows = []
    for _lag in exog_lags:
        _source_time = selected_origin + pd.Timedelta(hours=_lag)
        _feature_rows.append(
            {
                "available driver": f"driver[origin {_lag:+d}h]",
                "source time": _source_time.strftime("%a, %b %d %H:%M"),
                "standardized value": round(
                    float(frame.loc[_source_time, "driver"]), 3
                ),
            }
        )
    _target_rows = []
    for _horizon in range(1, forecast_horizon + 1):
        _target_time = selected_origin + pd.Timedelta(hours=_horizon)
        _target_rows.append(
            {
                "horizon": f"+{_horizon}h",
                "target time": _target_time.strftime("%a, %b %d %H:%M"),
                "observed target": round(
                    float(actual.loc[selected_origin, f"horizon_{_horizon}"]), 3
                ),
            }
        )
    _provenance_view = mo.vstack(
        [
            mo.md(
                "### Feature provenance\n\n"
                "The direct models receive only these lagged driver values at the "
                "origin. Each forecast horizon is scored against its own later target."
            ),
            mo.hstack(
                [
                    mo.ui.table(pd.DataFrame(_feature_rows), pagination=False),
                    mo.ui.table(pd.DataFrame(_target_rows), pagination=False),
                ]
            ),
        ]
    )
    _provenance_view
    return


if __name__ == "__main__":
    app.run()
