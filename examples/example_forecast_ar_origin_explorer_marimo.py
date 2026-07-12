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
#     "seaborn>=0.13.2",
#     "spcqe>=0.3.0",
# ]
# ///

"""Interactive explorer for direct target-history AR forecasts."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Direct autoregressive multi-horizon forecasting

    This example isolates the forecasting AR feature added to
    `TsgamForecastEstimator`. The synthetic target has three parts:

    The target is $y_t = s_t + 0.60x_t + r_t$, where $s_t$ is the daily and
    weekly baseline, $x_t$ is the standardized AR(1) driver used in the
    forecast-plotting explorer, and
    $r_t = 0.72r_{t-1} + \epsilon_t$ is an AR(1) residual process.

    The base TSGAM features recover the periodic and linear exogenous terms. The
    direct AR forecast additionally uses target values observed at each forecast
    origin to predict horizons 1 through 12. It never recursively feeds one
    prediction into the next.
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
    import seaborn as sns

    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from tsgam_estimator import (
        TsgamEstimatorConfig,
        TsgamForecastArConfig,
        TsgamForecastConfig,
        TsgamForecastCouplingConfig,
        TsgamForecastEstimator,
        TsgamLinearConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        plot_forecast_origin,
    )

    sns.set_theme(style="whitegrid", context="notebook")
    return (
        TsgamEstimatorConfig,
        TsgamForecastArConfig,
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
def _(mo):
    ar_order_control = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=1,
        label="AR order (number of observed target lags)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "## Model order\n\n"
                "Selecting order $p$ supplies target lags $0, 1, \\ldots, p-1$ "
                "to every positive forecast horizon and refits both AR models."
            ),
            ar_order_control,
        ]
    )
    return (ar_order_control,)


@app.cell
def _(
    TsgamEstimatorConfig,
    TsgamForecastArConfig,
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    ar_order_control,
    np,
    pd,
):
    forecast_horizon = 12
    selected_ar_order = int(ar_order_control.value)
    target_lags = list(range(selected_ar_order))
    train_samples = 720
    total_samples = 1008

    timestamps = pd.date_range("2025-01-01", periods=total_samples, freq="1h")
    sample = np.arange(total_samples, dtype=float)
    periodic_truth = (
        2.0
        + 1.35 * np.sin(2.0 * np.pi * sample / 24.0)
        + 0.40 * np.cos(4.0 * np.pi * sample / 24.0)
        + 0.28 * np.sin(2.0 * np.pi * sample / 168.0)
    )

    rng = np.random.default_rng(7)
    innovation = rng.normal(scale=0.40, size=total_samples)
    ar_residual = np.zeros(total_samples)
    for index in range(1, total_samples):
        ar_residual[index] = 0.72 * ar_residual[index - 1] + innovation[index]

    driver_rng = np.random.default_rng(23)
    driver = np.empty(total_samples)
    driver[0] = driver_rng.normal()
    for index in range(1, total_samples):
        driver[index] = 0.55 * driver[index - 1] + driver_rng.normal(scale=0.8)
    driver = (driver - driver.mean()) / driver.std()
    exogenous_truth = 0.60 * driver
    observed = periodic_truth + exogenous_truth + ar_residual

    frame = pd.DataFrame(
        {
            "observed": observed,
            "periodic truth": periodic_truth,
            "driver": driver,
            "linear exogenous contribution": exogenous_truth,
            "AR residual": ar_residual,
            "innovation": innovation,
        },
        index=timestamps,
    )
    X = frame[["driver"]]
    X_train = X.iloc[:train_samples]
    y_train = frame["observed"].iloc[:train_samples].to_numpy()
    X_eval = X.iloc[train_samples:-forecast_horizon]
    observed_history = frame["observed"]

    def base_config():
        return TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[3, 2],
                periods=[24, 168],
                reg_weight=1.0e-5,
            ),
            exog_config=[
                TsgamLinearConfig(
                    lags=[0],
                    reg_weight=1.0e-5,
                    diff_reg_weight=0.0,
                )
            ],
            solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
        )

    periodic_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=forecast_horizon,
            base_config=base_config(),
            mode="independent",
        )
    ).fit(X_train, y_train)
    independent_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=forecast_horizon,
            base_config=base_config(),
            mode="independent",
            forecast_ar_config=TsgamForecastArConfig(
                lags=target_lags,
                reg_weight=1.0e-5,
            ),
        )
    ).fit(X_train, y_train)
    coupled_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=forecast_horizon,
            base_config=base_config(),
            mode="coupled",
            forecast_ar_config=TsgamForecastArConfig(
                lags=target_lags,
                reg_weight=1.0e-5,
            ),
            coupling_config=TsgamForecastCouplingConfig(
                roughness_weight=0.01,
                roughness_order=1,
            ),
        )
    ).fit(X_train, y_train)

    periodic_prediction = periodic_model.predict(X_eval)
    independent_prediction = independent_model.predict(
        X_eval,
        y_history=observed_history,
    )
    coupled_prediction = coupled_model.predict(
        X_eval,
        y_history=observed_history,
    )
    origin_times = X_eval.index
    return (
        X_eval,
        coupled_model,
        coupled_prediction,
        forecast_horizon,
        frame,
        independent_model,
        independent_prediction,
        origin_times,
        periodic_prediction,
        selected_ar_order,
        target_lags,
        train_samples,
    )


@app.cell
def _(frame, mo, selected_ar_order, train_samples):
    mo.md(
        f"""
        ## The generated problem

        The first **{train_samples} hours** are used for fitting. Everything after
        the vertical split is a sequential evaluation period. At each evaluation
        origin, the model may use target observations at or before that origin.

        The selected **AR({selected_ar_order})** model uses target lags
        **0 through {selected_ar_order - 1}** at each positive horizon.
        """
    )
    return


@app.cell
def _(frame, plt, train_samples):
    figure_components, axes_components = plt.subplots(
        4,
        1,
        figsize=(12, 9),
        sharex=True,
        layout="constrained",
    )
    split_time = frame.index[train_samples]
    axes_components[0].plot(frame.index, frame["periodic truth"], color="#6f42c1")
    axes_components[0].set_ylabel("periodic")
    axes_components[0].set_title("Known components of the synthetic target", loc="left")
    axes_components[1].plot(
        frame.index,
        frame["linear exogenous contribution"],
        color="#2a9d8f",
    )
    axes_components[1].set_ylabel("0.60 x driver")
    axes_components[2].plot(frame.index, frame["AR residual"], color="#2878b5")
    axes_components[2].set_ylabel("AR residual")
    axes_components[3].plot(frame.index, frame["observed"], color="#222222")
    axes_components[3].set_ylabel("observed target")
    axes_components[3].set_xlabel("time")
    for component_axis in axes_components:
        component_axis.axvline(split_time, color="#d1495b", linestyle="--")
        component_axis.grid(axis="y", alpha=0.25)
    figure_components
    return


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
        f"""
        ## Forecast issued at `{selected_origin:%a, %b %d %Y %H:%M}`

        The solid history is known when this forecast is issued. The shaded region
        contains the nowcast and twelve direct future predictions. Scrubbing changes
        the origin and recomputes the visible forecast slice; it does not refit models.
        """
    )
    return (selected_origin,)


@app.cell
def _(
    coupled_prediction,
    frame,
    independent_prediction,
    periodic_prediction,
    plot_forecast_origin,
    plt,
    selected_origin,
):
    figure_origin, axis_origin = plt.subplots(
        figsize=(12, 5),
        layout="constrained",
    )
    plot_forecast_origin(
        {
            "TSGAM without target AR": periodic_prediction,
            "Independent AR": independent_prediction,
            "Coupled AR": coupled_prediction,
        },
        actual=frame["observed"],
        origin=selected_origin,
        history_steps=36,
        ax=axis_origin,
    )
    figure_origin
    return


@app.cell
def _(
    coupled_prediction,
    forecast_horizon,
    frame,
    independent_prediction,
    np,
    pd,
    periodic_prediction,
    plt,
    selected_origin,
):
    selected_horizons = np.arange(forecast_horizon + 1)
    selected_target_times = pd.DatetimeIndex(
        [
            selected_origin + pd.Timedelta(hours=int(_horizon))
            for _horizon in selected_horizons
        ]
    )
    selected_actual = frame["observed"].reindex(selected_target_times).to_numpy()
    selected_predictions = {
        "TSGAM without target AR": periodic_prediction,
        "Independent AR": independent_prediction,
        "Coupled AR": coupled_prediction,
    }
    selected_error_styles = {
        "TSGAM without target AR": ("#666666", ":"),
        "Independent AR": ("#d1495b", "--"),
        "Coupled AR": ("#2878b5", "-"),
    }
    figure_selected_error, axis_selected_error = plt.subplots(
        figsize=(10, 3.5),
        layout="constrained",
    )
    for error_model, error_prediction in selected_predictions.items():
        predicted_path = np.array(
            [
                error_prediction.loc[
                    selected_origin,
                    f"horizon_{int(_horizon)}",
                ]
                for _horizon in selected_horizons
            ]
        )
        path_error = predicted_path - selected_actual
        path_mae = float(np.mean(np.abs(path_error)))
        error_color, error_style = selected_error_styles[error_model]
        axis_selected_error.plot(
            selected_horizons,
            path_error,
            color=error_color,
            linestyle=error_style,
            marker="o",
            label=f"{error_model} (path MAE {path_mae:.3f})",
        )
    axis_selected_error.axhline(0, color="#111111", linewidth=1)
    axis_selected_error.set_title(
        "Selected-origin forecast error: prediction minus actual",
        loc="left",
    )
    axis_selected_error.set_xlabel("forecast horizon")
    axis_selected_error.set_ylabel("signed error")
    axis_selected_error.legend(frameon=False, ncols=2)
    figure_selected_error
    return


@app.cell
def _(frame, mo, pd, selected_origin, target_lags):
    known_rows = [
        {
            "forecast feature": "linear driver at origin",
            "source time": selected_origin.strftime("%a, %b %d %H:%M"),
            "feature value": round(float(frame.loc[selected_origin, "driver"]), 4),
            "role": "exogenous",
        }
    ]
    for lag in target_lags:
        source_time = selected_origin - pd.Timedelta(hours=lag)
        known_rows.append(
            {
                "forecast feature": f"target lag {lag}",
                "source time": source_time.strftime("%a, %b %d %H:%M"),
                "feature value": round(float(frame.loc[source_time, "observed"]), 4),
                "role": "target history",
            }
        )
    mo.vstack(
        [
            mo.md(
                "### Origin-known inputs supplied to the direct forecasts\n\n"
                "The linear driver is observed at the origin. Lag 0 is the target "
                "at the origin; higher target lags are preceding observations. "
                "Horizon 0 cannot use target history, avoiding `y[t] = y[t]`."
            ),
            mo.ui.table(pd.DataFrame(known_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(forecast_horizon, np, pd, target_lags):
    phi = 0.72
    true_lag_0 = np.array(
        [0.0] + [phi**horizon for horizon in range(1, forecast_horizon + 1)]
    )
    true_data = {
        f"lag_{_lag}": (
            true_lag_0 if _lag == 0 else np.zeros(forecast_horizon + 1)
        )
        for _lag in target_lags
    }
    true_coefficients = pd.DataFrame(
        true_data,
        index=pd.Index(range(forecast_horizon + 1), name="horizon"),
    )
    return (true_coefficients,)


@app.cell
def _(coupled_model, independent_model, np, plt, target_lags, true_coefficients):
    coefficient_columns = min(3, len(target_lags))
    coefficient_rows = int(np.ceil(len(target_lags) / coefficient_columns))
    figure_coefficients, coefficient_axes_grid = plt.subplots(
        coefficient_rows,
        coefficient_columns,
        figsize=(12, 3.5 * coefficient_rows),
        sharex=True,
        layout="constrained",
    )
    coefficient_axes = np.atleast_1d(coefficient_axes_grid).ravel()
    for _lag, coefficient_axis in zip(
        target_lags,
        coefficient_axes,
        strict=False,
    ):
        lag_column = f"lag_{_lag}"
        coefficient_axis.plot(
            true_coefficients.index[1:],
            true_coefficients.loc[1:, lag_column],
            color="#111111",
            marker="o",
            label="Population coefficient",
        )
        coefficient_axis.plot(
            independent_model.forecast_ar_coefficients_.index[1:],
            independent_model.forecast_ar_coefficients_.loc[1:, lag_column],
            color="#d1495b",
            linestyle="--",
            marker="o",
            label="Independent AR",
        )
        coefficient_axis.plot(
            coupled_model.forecast_ar_coefficients_.index[1:],
            coupled_model.forecast_ar_coefficients_.loc[1:, lag_column],
            color="#2878b5",
            marker="o",
            label="Coupled AR",
        )
        coefficient_axis.set_title(f"Target lag {_lag}")
        coefficient_axis.set_xlabel("forecast horizon")
        coefficient_axis.set_ylabel("coefficient")
        coefficient_axis.axhline(0, color="0.6", linewidth=1)
    for unused_axis in coefficient_axes[len(target_lags):]:
        unused_axis.set_visible(False)
    coefficient_axes[0].legend(frameon=False)
    figure_coefficients.suptitle(
        "Direct target-history coefficients across horizon",
        fontweight="bold",
    )
    figure_coefficients
    return


@app.cell
def _(
    coupled_prediction,
    forecast_horizon,
    frame,
    independent_prediction,
    np,
    pd,
    periodic_prediction,
):
    metric_rows = []
    prediction_sets = {
        "TSGAM without target AR": periodic_prediction,
        "Independent AR": independent_prediction,
        "Coupled AR": coupled_prediction,
    }
    for model_name, prediction in prediction_sets.items():
        for _horizon in range(forecast_horizon + 1):
            target_times = prediction.index + pd.Timedelta(hours=_horizon)
            actual_values = frame["observed"].reindex(target_times).to_numpy()
            errors = prediction[f"horizon_{_horizon}"].to_numpy() - actual_values
            metric_rows.append(
                {
                    "model": model_name,
                    "horizon": _horizon,
                    "RMSE": float(np.sqrt(np.mean(errors**2))),
                }
            )
    horizon_metrics = pd.DataFrame(metric_rows)
    return (horizon_metrics,)


@app.cell
def _(horizon_metrics, plt):
    figure_metrics, axis_metrics = plt.subplots(
        figsize=(10, 4),
        layout="constrained",
    )
    metric_styles = {
        "TSGAM without target AR": ("#666666", ":"),
        "Independent AR": ("#d1495b", "--"),
        "Coupled AR": ("#2878b5", "-"),
    }
    for metric_model, (metric_color, metric_style) in metric_styles.items():
        model_metrics = horizon_metrics[horizon_metrics["model"] == metric_model]
        axis_metrics.plot(
            model_metrics["horizon"],
            model_metrics["RMSE"],
            color=metric_color,
            linestyle=metric_style,
            marker="o",
            label=metric_model,
        )
    axis_metrics.set_title("Out-of-sample error by forecast horizon", loc="left")
    axis_metrics.set_xlabel("forecast horizon")
    axis_metrics.set_ylabel("RMSE")
    axis_metrics.legend(frameon=False)
    figure_metrics
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What to look for

    - **TSGAM without target AR** uses periodic structure and the linear driver at
      the origin, but cannot react to the latest unexplained residual state.
    - **Independent AR** estimates each horizon separately, so its coefficient path
      can be visibly jagged with limited training data.
    - **Coupled AR** solves one joint problem and penalizes first differences between
      adjacent positive-horizon coefficients.
    - The AR contribution naturally decays with horizon because the planted AR(1)
      process gradually forgets its current state.

    `TsgamArConfig` is not involved here. That older configuration models residuals
    for stochastic sample generation; this notebook uses deterministic direct
    target-history forecasting through `TsgamForecastArConfig`.
    """)
    return


if __name__ == "__main__":
    app.run()
