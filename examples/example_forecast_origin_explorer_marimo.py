#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clarabel>=0.11.1",
#     "cvxpy>=1.7.3",
#     "marimo>=0.23.4",
#     "matplotlib>=3.10.0",
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

    import marimo as mo
    import matplotlib.pyplot as plt
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
        plot_forecast_origin,
    )

    return (
        TsgamEstimatorConfig,
        TsgamForecastConfig,
        TsgamForecastCouplingConfig,
        TsgamForecastEstimator,
        TsgamLinearConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        mo,
        np,
        pd,
        plot_forecast_origin,
        plt,
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
            for _horizon in range(forecast_horizon + 1)
        },
        index=_x_eval.index,
    )
    origin_times = _x_eval.index
    return (
        actual,
        coupled_prediction,
        exog_lags,
        forecast_horizon,
        frame,
        history_hours,
        independent_prediction,
        origin_times,
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
    coupled_prediction,
    frame,
    history_hours,
    independent_prediction,
    plot_forecast_origin,
    plt,
    selected_origin,
):
    _figure, _axis = plt.subplots(figsize=(12, 5), layout="constrained")
    plot_forecast_origin(
        {
            "Independent forecast": independent_prediction,
            "Coupled forecast": coupled_prediction,
        },
        actual=frame["observed"],
        origin=selected_origin,
        history_steps=history_hours,
        ax=_axis,
    )
    _figure
    return


@app.cell
def _(actual, exog_lags, forecast_horizon, frame, mo, pd, selected_origin):
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
    _target_rows = [
        {
            "horizon": "0h (nowcast)",
            "target time": selected_origin.strftime("%a, %b %d %H:%M"),
            "observed target": round(float(frame.loc[selected_origin, "observed"]), 3),
        }
    ]
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
                "origin. The nowcast is scored at the origin; each forecast horizon "
                "is scored against its own later target."
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
