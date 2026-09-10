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

"""Interactive direct multi-horizon forecast explorer for bundled PV data."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Solar PV direct multi-horizon forecast explorer

    This notebook applies `TsgamForecastEstimator` to bundled five-minute PV data.
    It uses POA irradiance observed at the forecast origin plus optional measured
    AC-power history. Each horizon is predicted directly; forecasts are never fed
    recursively into later horizons.

    The comparison is:

    - **TSGAM without target AR:** daily periodic structure and origin POA irradiance.
    - **Independent AR:** one separately fitted target-history model per horizon.
    - **Coupled AR:** one joint fit with first-difference smoothing across horizons.
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
    examples_dir = project_root / "examples"
    for local_path in (src_dir, examples_dir):
        if str(local_path) not in sys.path:
            sys.path.insert(0, str(local_path))

    from forecast_real_data_support import load_all_datasets
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
        load_all_datasets,
        mo,
        np,
        pd,
        plot_forecast_origin,
        plt,
    )


@app.cell
def _(mo):
    solar_ar_order = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=3,
        label="Target-history order (5-minute lags)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "## Forecast configuration\n\n"
                "Order $p$ supplies measured-power lags $0, 1, \\ldots, p-1$ "
                "to every positive horizon and refits the two AR models."
            ),
            solar_ar_order,
        ]
    )
    return (solar_ar_order,)


@app.cell
def _(load_all_datasets, pd):
    solar_dataset = load_all_datasets(["pv_solar"])["pv_solar"]
    solar_frame = solar_dataset.frame.copy()
    solar_spec = solar_dataset.spec
    train_end = solar_spec.train_samples
    eval_end = train_end + solar_spec.eval_samples
    solar_origins = pd.DatetimeIndex(solar_frame.index[train_end:eval_end])
    return (
        eval_end,
        solar_dataset,
        solar_frame,
        solar_origins,
        solar_spec,
        train_end,
    )


@app.cell
def _(mo, solar_dataset, solar_frame, solar_spec, train_end):
    mo.md(
        f"""
        ## Real bundled data

        - **Source:** `examples/data/{solar_spec.source}`
        - **Target:** {solar_spec.target_label}
        - **Native frequency:** {solar_dataset.step}
        - **Selected window:** `{solar_frame.index[0]}` through `{solar_frame.index[-1]}`
        - **Training samples:** {train_end}
        - **Evaluation origins:** {solar_spec.eval_samples}
        - **Maximum horizon:** {solar_spec.horizon} samples =
          {solar_spec.horizon * solar_dataset.step}

        The target is never filled. POA irradiance is causally forward-filled for at
        most {solar_spec.feature_fill_limit} source steps by the shared real-data
        loader. Future irradiance paths are not supplied to the model.
        """
    )
    return


@app.cell
def _(plt, solar_frame, train_end):
    solar_overview_figure, solar_overview_axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        layout="constrained",
    )
    solar_split_time = solar_frame.index[train_end]
    solar_overview_axes[0].plot(
        solar_frame.index,
        solar_frame["target"],
        color="#d18f00",
        linewidth=1.2,
    )
    solar_overview_axes[0].set_title("PV power and POA irradiance", loc="left")
    solar_overview_axes[0].set_ylabel("AC power")
    solar_overview_axes[1].plot(
        solar_frame.index,
        solar_frame["poa_irradiance"],
        color="#2878b5",
        linewidth=1.2,
    )
    solar_overview_axes[1].set_ylabel("POA irradiance")
    solar_overview_axes[1].set_xlabel("time")
    for solar_overview_axis in solar_overview_axes:
        solar_overview_axis.axvline(
            solar_split_time,
            color="#d1495b",
            linestyle="--",
            label="first evaluation origin",
        )
        solar_overview_axis.grid(axis="y", alpha=0.25)
    solar_overview_axes[0].legend(frameon=False)
    solar_overview_figure
    return


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
    eval_end,
    np,
    pd,
    solar_ar_order,
    solar_frame,
    solar_spec,
    train_end,
):
    selected_solar_ar_order = int(solar_ar_order.value)
    solar_target_lags = list(range(selected_solar_ar_order))
    solar_horizon = solar_spec.horizon

    raw_solar_X = solar_frame[["poa_irradiance"]]
    raw_solar_X_train = raw_solar_X.iloc[:train_end]
    raw_solar_X_eval = raw_solar_X.iloc[train_end:eval_end]
    irradiance_center = raw_solar_X_train.mean()
    irradiance_scale = raw_solar_X_train.std(ddof=0).replace(0.0, 1.0)
    solar_X_train = (raw_solar_X_train - irradiance_center) / irradiance_scale
    solar_X_eval = (raw_solar_X_eval - irradiance_center) / irradiance_scale

    solar_target_scale = float(solar_frame["target"].iloc[:train_end].max())
    solar_y_train = (
        solar_frame["target"].iloc[:train_end].to_numpy(dtype=float)
        / solar_target_scale
    )
    normalized_solar_history = solar_frame["target"] / solar_target_scale

    def solar_base_config():
        return TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[4],
                periods=[288.0],
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

    solar_baseline_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=solar_horizon,
            base_config=solar_base_config(),
            mode="independent",
        )
    ).fit(solar_X_train, solar_y_train)
    solar_independent_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=solar_horizon,
            base_config=solar_base_config(),
            mode="independent",
            forecast_ar_config=TsgamForecastArConfig(
                lags=solar_target_lags,
                reg_weight=1.0e-5,
            ),
        )
    ).fit(solar_X_train, solar_y_train)
    solar_coupled_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=solar_horizon,
            base_config=solar_base_config(),
            mode="coupled",
            forecast_ar_config=TsgamForecastArConfig(
                lags=solar_target_lags,
                reg_weight=1.0e-5,
            ),
            coupling_config=TsgamForecastCouplingConfig(
                roughness_weight=0.01,
                roughness_order=1,
            ),
        )
    ).fit(solar_X_train, solar_y_train)

    solar_baseline_prediction = (
        solar_baseline_model.predict(solar_X_eval) * solar_target_scale
    )
    solar_independent_prediction = (
        solar_independent_model.predict(
            solar_X_eval,
            y_history=normalized_solar_history,
        )
        * solar_target_scale
    )
    solar_coupled_prediction = (
        solar_coupled_model.predict(
            solar_X_eval,
            y_history=normalized_solar_history,
        )
        * solar_target_scale
    )
    solar_independent_without_history = (
        solar_independent_model.predict(
            solar_X_eval,
            remove_forecast_ar=True,
        )
        * solar_target_scale
    )
    solar_coupled_without_history = (
        solar_coupled_model.predict(
            solar_X_eval,
            remove_forecast_ar=True,
        )
        * solar_target_scale
    )
    return (
        raw_solar_X_eval,
        selected_solar_ar_order,
        solar_baseline_prediction,
        solar_coupled_prediction,
        solar_coupled_without_history,
        solar_horizon,
        solar_independent_prediction,
        solar_independent_without_history,
        solar_target_lags,
        solar_target_scale,
    )


@app.cell
def _(mo, solar_origins):
    solar_origin_slider = mo.ui.slider(
        start=0,
        stop=len(solar_origins) - 1,
        step=1,
        value=min(120, len(solar_origins) - 1),
        label="Evaluation forecast origin",
        full_width=True,
    )
    solar_origin_slider
    return (solar_origin_slider,)


@app.cell
def _(mo, selected_solar_ar_order, solar_origin_slider, solar_origins):
    selected_solar_origin = solar_origins[solar_origin_slider.value]
    mo.md(
        f"""
        ## Forecast issued at `{selected_solar_origin:%a, %b %d %Y %H:%M}`

        The selected AR({selected_solar_ar_order}) models see measured AC power at
        the origin and the preceding {(selected_solar_ar_order - 1) * 5} minutes,
        plus POA irradiance observed at the origin. The shaded region is the next
        two hours of direct forecasts.
        """
    )
    return (selected_solar_origin,)


@app.cell
def _(
    plot_forecast_origin,
    plt,
    selected_solar_origin,
    solar_baseline_prediction,
    solar_coupled_prediction,
    solar_frame,
    solar_independent_prediction,
):
    solar_origin_figure, solar_origin_axis = plt.subplots(
        figsize=(12, 5),
        layout="constrained",
    )
    plot_forecast_origin(
        {
            "TSGAM without target AR": solar_baseline_prediction,
            "Independent AR": solar_independent_prediction,
            "Coupled AR": solar_coupled_prediction,
        },
        actual=solar_frame["target"],
        origin=selected_solar_origin,
        history_steps=72,
        freq="5min",
        ax=solar_origin_axis,
    )
    solar_origin_figure
    return


@app.cell
def _(
    np,
    pd,
    plt,
    selected_solar_origin,
    solar_baseline_prediction,
    solar_coupled_prediction,
    solar_frame,
    solar_horizon,
    solar_independent_prediction,
):
    solar_selected_horizons = np.arange(solar_horizon + 1)
    solar_selected_times = pd.DatetimeIndex(
        [
            selected_solar_origin + pd.Timedelta(minutes=5 * int(_horizon))
            for _horizon in solar_selected_horizons
        ]
    )
    solar_selected_actual = solar_frame["target"].reindex(
        solar_selected_times
    ).to_numpy()
    solar_selected_predictions = {
        "TSGAM without target AR": solar_baseline_prediction,
        "Independent AR": solar_independent_prediction,
        "Coupled AR": solar_coupled_prediction,
    }
    solar_error_styles = {
        "TSGAM without target AR": ("#666666", ":"),
        "Independent AR": ("#d1495b", "--"),
        "Coupled AR": ("#2878b5", "-"),
    }
    solar_error_figure, solar_error_axis = plt.subplots(
        figsize=(10, 3.5),
        layout="constrained",
    )
    for solar_error_model, solar_error_prediction in solar_selected_predictions.items():
        solar_predicted_path = solar_error_prediction.loc[
            selected_solar_origin
        ].to_numpy()
        solar_path_error = solar_predicted_path - solar_selected_actual
        solar_path_mae = float(np.mean(np.abs(solar_path_error)))
        solar_error_color, solar_error_style = solar_error_styles[solar_error_model]
        solar_error_axis.plot(
            solar_selected_horizons * 5,
            solar_path_error,
            color=solar_error_color,
            linestyle=solar_error_style,
            marker="o",
            markersize=3,
            label=f"{solar_error_model} (path MAE {solar_path_mae:.2f})",
        )
    solar_error_axis.axhline(0, color="#111111", linewidth=1)
    solar_error_axis.set_title(
        "Selected-origin forecast error: prediction minus actual",
        loc="left",
    )
    solar_error_axis.set_xlabel("minutes ahead")
    solar_error_axis.set_ylabel("signed AC-power error")
    solar_error_axis.legend(frameon=False, ncols=2)
    solar_error_figure
    return


@app.cell
def _(
    mo,
    pd,
    raw_solar_X_eval,
    selected_solar_origin,
    solar_frame,
    solar_target_lags,
):
    solar_known_rows = [
        {
            "forecast feature": "POA irradiance at origin",
            "source time": selected_solar_origin.strftime("%a, %b %d %H:%M"),
            "feature value": round(
                float(raw_solar_X_eval.loc[selected_solar_origin, "poa_irradiance"]),
                2,
            ),
            "role": "exogenous",
        }
    ]
    for _lag in solar_target_lags:
        solar_source_time = selected_solar_origin - pd.Timedelta(minutes=5 * _lag)
        solar_known_rows.append(
            {
                "forecast feature": f"AC power lag {_lag}",
                "source time": solar_source_time.strftime("%a, %b %d %H:%M"),
                "feature value": round(
                    float(solar_frame.loc[solar_source_time, "target"]),
                    3,
                ),
                "role": "target history",
            }
        )
    mo.vstack(
        [
            mo.md(
                "### Origin-known inputs\n\n"
                "These are the literal values available when the selected forecast "
                "is issued. No future measured power or irradiance is used."
            ),
            mo.ui.table(pd.DataFrame(solar_known_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(
    np,
    plt,
    selected_solar_origin,
    solar_coupled_prediction,
    solar_coupled_without_history,
    solar_horizon,
    solar_independent_prediction,
    solar_independent_without_history,
):
    solar_adjustment_horizons = np.arange(solar_horizon + 1)
    solar_adjustment_models = {
        "Independent AR": (
            solar_independent_prediction,
            solar_independent_without_history,
            "#d1495b",
            "--",
        ),
        "Coupled AR": (
            solar_coupled_prediction,
            solar_coupled_without_history,
            "#2878b5",
            "-",
        ),
    }
    solar_adjustment_figure, solar_adjustment_axis = plt.subplots(
        figsize=(10, 3.5),
        layout="constrained",
    )
    for solar_adjustment_label, (
        solar_full_prediction,
        solar_no_history_prediction,
        solar_adjustment_color,
        solar_adjustment_style,
    ) in solar_adjustment_models.items():
        solar_history_adjustment = (
            solar_full_prediction.loc[selected_solar_origin].to_numpy()
            - solar_no_history_prediction.loc[selected_solar_origin].to_numpy()
        )
        solar_mean_adjustment = float(np.mean(np.abs(solar_history_adjustment)))
        solar_adjustment_axis.plot(
            solar_adjustment_horizons * 5,
            solar_history_adjustment,
            color=solar_adjustment_color,
            linestyle=solar_adjustment_style,
            marker="o",
            markersize=3,
            label=(
                f"{solar_adjustment_label} "
                f"(mean absolute adjustment {solar_mean_adjustment:.2f})"
            ),
        )
    solar_adjustment_axis.axhline(0, color="#111111", linewidth=1)
    solar_adjustment_axis.set_title(
        "How measured-power history changes the selected forecast",
        loc="left",
    )
    solar_adjustment_axis.set_xlabel("minutes ahead")
    solar_adjustment_axis.set_ylabel("AC-power forecast adjustment")
    solar_adjustment_axis.legend(frameon=False)
    solar_adjustment_figure
    return


@app.cell
def _(
    np,
    pd,
    solar_baseline_prediction,
    solar_coupled_prediction,
    solar_frame,
    solar_horizon,
    solar_independent_prediction,
    solar_target_scale,
):
    solar_metric_rows = []
    solar_prediction_sets = {
        "TSGAM without target AR": solar_baseline_prediction,
        "Independent AR": solar_independent_prediction,
        "Coupled AR": solar_coupled_prediction,
    }
    solar_active_threshold = 0.05 * solar_target_scale
    for solar_metric_model, solar_metric_prediction in solar_prediction_sets.items():
        for _horizon in range(solar_horizon + 1):
            solar_metric_times = pd.DatetimeIndex(
                solar_metric_prediction.index
                + pd.Timedelta(minutes=5 * _horizon)
            )
            solar_metric_actual = solar_frame["target"].reindex(
                solar_metric_times
            ).to_numpy()
            solar_metric_error = (
                solar_metric_prediction[f"horizon_{_horizon}"].to_numpy()
                - solar_metric_actual
            )
            solar_active = solar_metric_actual > solar_active_threshold
            solar_metric_rows.append(
                {
                    "model": solar_metric_model,
                    "horizon": _horizon,
                    "minutes ahead": 5 * _horizon,
                    "all-period RMSE": float(
                        np.sqrt(np.mean(solar_metric_error**2))
                    ),
                    "active-generation RMSE": float(
                        np.sqrt(np.mean(solar_metric_error[solar_active] ** 2))
                    ),
                }
            )
    solar_metrics = pd.DataFrame(solar_metric_rows)
    return (solar_metrics,)


@app.cell
def _(plt, solar_metrics):
    solar_metrics_figure, solar_metrics_axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        sharex=True,
        layout="constrained",
    )
    solar_metric_styles = {
        "TSGAM without target AR": ("#666666", ":"),
        "Independent AR": ("#d1495b", "--"),
        "Coupled AR": ("#2878b5", "-"),
    }
    for solar_metric_axis, solar_metric_name in zip(
        solar_metrics_axes,
        ("all-period RMSE", "active-generation RMSE"),
        strict=True,
    ):
        for solar_metric_label, (
            solar_metric_color,
            solar_metric_style,
        ) in solar_metric_styles.items():
            solar_model_metrics = solar_metrics[
                solar_metrics["model"] == solar_metric_label
            ]
            solar_metric_axis.plot(
                solar_model_metrics["minutes ahead"],
                solar_model_metrics[solar_metric_name],
                color=solar_metric_color,
                linestyle=solar_metric_style,
                marker="o",
                markersize=3,
                label=solar_metric_label,
            )
        solar_metric_axis.set_title(solar_metric_name)
        solar_metric_axis.set_xlabel("minutes ahead")
        solar_metric_axis.set_ylabel("AC-power RMSE")
    solar_metrics_axes[0].legend(frameon=False)
    solar_metrics_figure.suptitle(
        "Out-of-sample forecast error by horizon",
        fontweight="bold",
    )
    solar_metrics_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the results

    - Scrub to a daytime origin to inspect how measured-power history changes the
      near-term forecast during ramps, clouds, and clear production periods.
    - The signed-error plot explains one forecast path. The RMSE panels summarize
      every held-out origin.
    - **Active-generation RMSE** excludes target times below 5% of training peak,
      preventing nighttime zeros from dominating the solar comparison.
    - This is a bounded demonstration window, not a production backtest. The model
      receives only irradiance observed at the origin; forecast irradiance scenarios
      remain a separate future extension.
    """)
    return


if __name__ == "__main__":
    app.run()
