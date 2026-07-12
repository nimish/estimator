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
    `TsgamForecastEstimator`. The synthetic target has two parts:

    The target is $y_t = s_t + r_t$, where $s_t$ is the daily and weekly
    baseline and $r_t = 0.72r_{t-1} + \epsilon_t$ is an AR(1)
    residual process.

    The periodic TSGAM features recover \(s_t\). The direct AR forecast uses target
    values observed at each forecast origin to predict horizons 1 through 12. It
    never recursively feeds one prediction into the next.
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
    TsgamForecastArConfig,
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    np,
    pd,
):
    forecast_horizon = 12
    target_lags = [0]
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
    observed = periodic_truth + ar_residual

    frame = pd.DataFrame(
        {
            "observed": observed,
            "periodic truth": periodic_truth,
            "AR residual": ar_residual,
            "innovation": innovation,
        },
        index=timestamps,
    )
    X = pd.DataFrame(index=timestamps)
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
            exog_config=None,
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
        target_lags,
        train_samples,
    )


@app.cell
def _(frame, mo, train_samples):
    mo.md(
        f"""
        ## The generated problem

        The first **{train_samples} hours** are used for fitting. Everything after
        the vertical split is a sequential evaluation period. At each evaluation
        origin, the model may use target observations at or before that origin.
        """
    )
    return


@app.cell
def _(frame, plt, train_samples):
    figure_components, axes_components = plt.subplots(
        3,
        1,
        figsize=(12, 7),
        sharex=True,
        layout="constrained",
    )
    split_time = frame.index[train_samples]
    axes_components[0].plot(frame.index, frame["periodic truth"], color="#6f42c1")
    axes_components[0].set_ylabel("periodic")
    axes_components[0].set_title("Known components of the synthetic target", loc="left")
    axes_components[1].plot(frame.index, frame["AR residual"], color="#2878b5")
    axes_components[1].set_ylabel("AR residual")
    axes_components[2].plot(frame.index, frame["observed"], color="#222222")
    axes_components[2].set_ylabel("observed target")
    axes_components[2].set_xlabel("time")
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
            "Periodicity only": periodic_prediction,
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
def _(frame, mo, pd, selected_origin, target_lags):
    known_rows = []
    for lag in target_lags:
        source_time = selected_origin - pd.Timedelta(hours=lag)
        known_rows.append(
            {
                "forecast feature": f"target lag {lag}",
                "source time": source_time.strftime("%a, %b %d %H:%M"),
                "observed value": round(float(frame.loc[source_time, "observed"]), 4),
                "available at origin": "yes",
            }
        )
    mo.vstack(
        [
            mo.md(
                "### Target history supplied to every positive horizon\n\n"
                "Lag 0 is the target observed at the selected origin. Horizon 0 is "
                "constrained not to use it, avoiding the tautology `y[t] = y[t]`."
            ),
            mo.ui.table(pd.DataFrame(known_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(coupled_model, forecast_horizon, independent_model, np, pd):
    phi = 0.72
    true_lag_0 = np.array(
        [0.0] + [phi**horizon for horizon in range(1, forecast_horizon + 1)]
    )
    true_coefficients = pd.DataFrame(
        {"lag_0": true_lag_0},
        index=pd.Index(range(forecast_horizon + 1), name="horizon"),
    )
    return (true_coefficients,)


@app.cell
def _(coupled_model, independent_model, plt, true_coefficients):
    figure_coefficients, axis_coefficients = plt.subplots(
        figsize=(10, 4),
        layout="constrained",
    )
    axis_coefficients.plot(
        true_coefficients.index[1:],
        true_coefficients.loc[1:, "lag_0"],
        color="#111111",
        marker="o",
        label="True direct coefficient",
    )
    axis_coefficients.plot(
        independent_model.forecast_ar_coefficients_.index[1:],
        independent_model.forecast_ar_coefficients_.loc[1:, "lag_0"],
        color="#d1495b",
        linestyle="--",
        marker="o",
        label="Independent AR",
    )
    axis_coefficients.plot(
        coupled_model.forecast_ar_coefficients_.index[1:],
        coupled_model.forecast_ar_coefficients_.loc[1:, "lag_0"],
        color="#2878b5",
        marker="o",
        label="Coupled AR",
    )
    axis_coefficients.set_title(
        "Direct coefficient on the target observed at the origin",
        loc="left",
    )
    axis_coefficients.set_xlabel("forecast horizon")
    axis_coefficients.set_ylabel("coefficient")
    axis_coefficients.axhline(0, color="0.6", linewidth=1)
    axis_coefficients.legend(frameon=False)
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
        "Periodicity only": periodic_prediction,
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
        "Periodicity only": ("#666666", ":"),
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

    - **Periodicity only** cannot react to the latest residual state.
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
