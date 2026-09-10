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

"""Basic marimo walkthrough for periodic multi-horizon TSGAM forecasts."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    intro_panel = mo.md(
        "# Basic Multi-Horizon Forecast: Independent vs Coupled\n\n"
        "This notebook uses a deliberately simple signal: a few periodic "
        "components plus noise, with an optional single linear exogenous "
        "driver. The point is to show what direct multi-horizon forecasting "
        "does over time, and how independent and coupled forecast mode differ."
    )
    intro_panel
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
                orient="right",
                titleColor="#1f2937",
                titleFontSize=12,
            )
            .configure_header(
                labelColor="#1f2937",
                labelFontSize=12,
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
def _(mo):
    controls = (
        mo.md(
            """
            ## 1. Configure and fit

            {horizon}

            {samples}

            {train_fraction}

            {noise_scale}

            {roughness_weight}

            {use_driver}

            {seed}
            """
        )
        .batch(
            horizon=mo.ui.slider(
                start=3,
                stop=12,
                step=1,
                value=8,
                label="Forecast horizon",
                full_width=True,
            ),
            samples=mo.ui.slider(
                start=168,
                stop=1344,
                step=24,
                value=504,
                label="Hourly samples",
                full_width=True,
            ),
            train_fraction=mo.ui.slider(
                start=0.50,
                stop=0.85,
                step=0.05,
                value=0.60,
                label="Training fraction",
                full_width=True,
            ),
            noise_scale=mo.ui.slider(
                start=0.00,
                stop=0.80,
                step=0.01,
                value=0.08,
                label="Noise scale",
                full_width=True,
            ),
            roughness_weight=mo.ui.slider(
                start=0.0,
                stop=20.0,
                step=0.5,
                value=5.0,
                label="Coupled roughness weight",
                full_width=True,
            ),
            use_driver=mo.ui.switch(
                value=True,
                label="Include one linear exogenous driver",
            ),
            seed=mo.ui.number(
                start=0,
                stop=999,
                step=1,
                value=12,
                label="Random seed",
            ),
        )
        .form(
            bordered=False,
            submit_button_label="Generate data and fit models",
        )
    )
    controls
    return (controls,)


@app.cell
def _(controls):
    default_settings = {
        "horizon": 8,
        "samples": 504,
        "train_fraction": 0.60,
        "noise_scale": 0.08,
        "roughness_weight": 5.0,
        "use_driver": True,
        "seed": 12,
    }
    submitted = controls.value or default_settings
    settings = {
        "horizon": int(submitted["horizon"]),
        "samples": int(submitted["samples"]),
        "train_fraction": float(submitted["train_fraction"]),
        "noise_scale": float(submitted["noise_scale"]),
        "roughness_weight": float(submitted["roughness_weight"]),
        "use_driver": bool(submitted["use_driver"]),
        "seed": int(submitted["seed"]),
    }
    return (settings,)


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
    def driver_lags() -> list[int]:
        return [-2, -1, 0]

    def driver_generation_weights() -> dict[int, float]:
        return {-2: 0.15, -1: 0.30, 0: 0.55}

    def make_periodic_data(
        samples: int,
        noise_scale: float,
        seed: int,
        horizon: int = 8,
        use_driver: bool = True,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        timestamp_index = pd.date_range("2024-01-01", periods=samples, freq="1h")
        sample_number = np.arange(samples, dtype=float)
        daily = 1.4 * np.sin(2.0 * np.pi * sample_number / 24.0)
        daily_harmonic = 0.45 * np.cos(2.0 * np.pi * sample_number / 12.0)
        slow_wave = 0.55 * np.sin(2.0 * np.pi * sample_number / 168.0 - 0.5)
        periodic_signal = daily + daily_harmonic + slow_wave
        driver_position = np.linspace(0.0, 1.0, samples)
        driver_ramp = 1.1 * (driver_position - 0.5)
        driver_bump = 0.8 * np.exp(
            -0.5 * ((driver_position - 0.42) / 0.12) ** 2
        )
        driver_step = -0.6 / (1.0 + np.exp(-(driver_position - 0.72) / 0.05))
        raw_driver = (
            driver_ramp
            + driver_bump
            + driver_step
            + rng.normal(scale=0.02, size=samples)
        )
        driver = (raw_driver - raw_driver.mean()) / raw_driver.std()
        driver_contribution = np.zeros(samples)
        driver_contribution_parts = {}
        if use_driver:
            for lag, coefficient in driver_generation_weights().items():
                contribution_part = np.zeros(samples)
                if lag < 0:
                    contribution_part[-lag:] = coefficient * driver[:lag]
                elif lag == 0:
                    contribution_part = coefficient * driver
                else:
                    contribution_part[:-lag] = coefficient * driver[lag:]
                driver_contribution_parts[
                    f"driver contribution lag {lag}"
                ] = contribution_part
                driver_contribution += contribution_part
        else:
            for lag in driver_lags():
                driver_contribution_parts[f"driver contribution lag {lag}"] = np.zeros(
                    samples
                )
        signal = periodic_signal + driver_contribution
        noise = rng.normal(scale=noise_scale, size=samples)
        observed = signal + noise
        return pd.DataFrame(
            {
                "observed": observed,
                "signal": signal,
                "periodic signal": periodic_signal,
                "daily": daily,
                "daily harmonic": daily_harmonic,
                "weekly wave": slow_wave,
                "raw driver profile": raw_driver,
                "driver": driver,
                "driver contribution": driver_contribution,
                "noise": noise,
                **driver_contribution_parts,
            },
            index=timestamp_index,
        )

    def base_config(use_driver: bool) -> TsgamEstimatorConfig:
        return TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[4, 2],
                periods=[24, 168],
                reg_weight=1.0e-8,
            ),
            exog_config=[
                TsgamLinearConfig(
                    lags=driver_lags(),
                    reg_weight=1.0e-10,
                    diff_reg_weight=1.0e-10,
                )
            ]
            if use_driver
            else None,
            solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
        )

    def fit_models(
        frame: pd.DataFrame,
        horizon: int,
        train_fraction: float,
        roughness_weight: float,
        use_driver: bool,
    ) -> dict[str, object]:
        train_stop = int(len(frame) * train_fraction)
        train_stop = max(train_stop, horizon + 72)
        train_stop = min(train_stop, len(frame) - horizon - 1)
        x_all = frame[["driver"]] if use_driver else pd.DataFrame(index=frame.index)
        y_all = frame["observed"].to_numpy()
        x_train = x_all.iloc[:train_stop]
        y_train = y_all[:train_stop]
        x_eval = x_all.iloc[train_stop:-horizon]
        independent = TsgamForecastEstimator(
            TsgamForecastConfig(
                horizon=horizon,
                base_config=base_config(use_driver),
                mode="independent",
            )
        ).fit(x_train, y_train)
        coupled = TsgamForecastEstimator(
            TsgamForecastConfig(
                horizon=horizon,
                base_config=base_config(use_driver),
                mode="coupled",
                coupling_config=TsgamForecastCouplingConfig(
                    roughness_weight=roughness_weight,
                ),
            )
        ).fit(x_train, y_train)
        independent_prediction = independent.predict(x_eval)
        coupled_prediction = coupled.predict(x_eval)
        actual = pd.DataFrame(
            {
                f"horizon_{step}": frame["observed"].shift(-step).loc[x_eval.index]
                for step in range(horizon + 1)
            },
            index=x_eval.index,
        )
        return {
            "actual": actual,
            "coupled": coupled,
            "coupled_prediction": coupled_prediction,
            "independent": independent,
            "independent_prediction": independent_prediction,
            "train_stop": train_stop,
            "use_driver": use_driver,
            "x_eval": x_eval,
        }

    def component_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame[
                [
                    "observed",
                    "signal",
                    "periodic signal",
                    "daily",
                    "daily harmonic",
                    "weekly wave",
                    "driver contribution",
                    "noise",
                ]
            ]
            .reset_index(names="timestamp")
            .melt(id_vars=["timestamp"], var_name="component", value_name="value")
        )

    def driver_contribution_frame(frame: pd.DataFrame) -> pd.DataFrame:
        contribution_columns = [
            "driver contribution",
            *[f"driver contribution lag {lag}" for lag in driver_lags()],
        ]
        contribution_labels = {"driver contribution": "total driver contribution"}
        return (
            frame[contribution_columns]
            .rename(columns=contribution_labels)
            .reset_index(names="timestamp")
            .melt(id_vars=["timestamp"], var_name="component", value_name="value")
        )

    def driver_response_frame(results: dict[str, object]) -> pd.DataFrame:
        response_rows = []
        lags = driver_lags()
        independent = results["independent"]
        horizons = independent.horizons_
        true_weights = (
            driver_generation_weights()
            if results["use_driver"]
            else {lag: 0.0 for lag in lags}
        )

        def append_coefficient(
            response_horizon: int,
            lag: int,
            model_name: str,
            coefficient: float,
        ) -> None:
            response_rows.append(
                {
                    "horizon": response_horizon,
                    "lag": lag,
                    "lag label": f"lag {lag}",
                    "model": model_name,
                    "coefficient": coefficient,
                }
            )

        if results["use_driver"]:
            coupled = results["coupled"]
            for response_horizon in horizons:
                coupled_horizon_ix = coupled.horizons_.index(response_horizon)
                independent_child = independent.forecast_estimators_[response_horizon]
                for lag_ix, lag in enumerate(lags):
                    append_coefficient(
                        response_horizon,
                        lag,
                        "independent",
                        float(
                            independent_child.variables_["exog_coef_0"].value[
                                0, lag_ix
                            ]
                        ),
                    )
                    append_coefficient(
                        response_horizon,
                        lag,
                        "coupled",
                        float(
                            coupled.variables_["exog_coef_0"][
                                coupled_horizon_ix
                            ].value[0, lag_ix]
                        ),
                    )
        else:
            for response_horizon in horizons:
                for lag in lags:
                    append_coefficient(response_horizon, lag, "independent", 0.0)
                    append_coefficient(response_horizon, lag, "coupled", 0.0)

        for response_horizon in horizons:
            for lag in lags:
                append_coefficient(response_horizon, lag, "true", true_weights[lag])
        return pd.DataFrame(response_rows)

    def driver_linear_function_frame(
        response: pd.DataFrame,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        del frame
        return response.copy()

    def horizon_path_frame(
        results: dict[str, object],
        origin_offset: int,
    ) -> pd.DataFrame:
        actual = results["actual"]
        independent_prediction = results["independent_prediction"]
        coupled_prediction = results["coupled_prediction"]
        origin_position = min(origin_offset, len(actual.index) - 1)
        origin_time = actual.index[origin_position]
        rows = []
        for path_column in actual.columns:
            path_horizon = int(path_column.split("_")[1])
            target_time = origin_time + pd.Timedelta(hours=path_horizon)
            rows.extend(
                [
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "horizon": path_horizon,
                        "series": "actual",
                        "value": float(actual.iloc[origin_position][path_column]),
                    },
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "horizon": path_horizon,
                        "series": "independent",
                        "value": float(
                            independent_prediction.iloc[origin_position][path_column]
                        ),
                    },
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "horizon": path_horizon,
                        "series": "coupled",
                        "value": float(
                            coupled_prediction.iloc[origin_position][path_column]
                        ),
                    },
                ]
            )
        return pd.DataFrame(rows)

    def prediction_by_horizon_frame(
        results: dict[str, object],
        horizon: int,
    ) -> pd.DataFrame:
        column_name = f"horizon_{horizon}"
        actual = results["actual"][[column_name]].rename(columns={column_name: "value"})
        independent = results["independent_prediction"][[column_name]].rename(
            columns={column_name: "value"}
        )
        coupled = results["coupled_prediction"][[column_name]].rename(
            columns={column_name: "value"}
        )
        return pd.concat(
            [
                actual.assign(series="actual"),
                independent.assign(series="independent"),
                coupled.assign(series="coupled"),
            ]
        ).reset_index(names="origin_time")

    def model_difference_frame(results: dict[str, object]) -> pd.DataFrame:
        independent_prediction = results["independent_prediction"]
        coupled_prediction = results["coupled_prediction"]
        rows = []
        for origin_time in independent_prediction.index:
            for difference_column in independent_prediction.columns:
                difference_horizon = int(difference_column.split("_")[1])
                rows.append(
                    {
                        "origin_time": origin_time,
                        "horizon": f"horizon {difference_horizon}",
                        "difference": float(
                            coupled_prediction.loc[origin_time, difference_column]
                            - independent_prediction.loc[origin_time, difference_column]
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def metric_frame(results: dict[str, object]) -> pd.DataFrame:
        actual = results["actual"]
        rows = []
        for model_name, prediction_key in [
            ("independent", "independent_prediction"),
            ("coupled", "coupled_prediction"),
        ]:
            prediction = results[prediction_key]
            for metric_column in actual.columns:
                metric_error = prediction[metric_column] - actual[metric_column]
                rows.append(
                    {
                        "model": model_name,
                        "horizon": int(metric_column.split("_")[1]),
                        "rmse": float(np.sqrt(np.mean(metric_error**2))),
                        "mae": float(np.mean(np.abs(metric_error))),
                    }
                )
        return pd.DataFrame(rows)

    return (
        component_frame,
        driver_contribution_frame,
        driver_linear_function_frame,
        driver_response_frame,
        fit_models,
        horizon_path_frame,
        make_periodic_data,
        metric_frame,
        model_difference_frame,
        prediction_by_horizon_frame,
    )


@app.cell
def _(make_periodic_data, settings):
    periodic_data = make_periodic_data(
        samples=settings["samples"],
        horizon=settings["horizon"],
        noise_scale=settings["noise_scale"],
        seed=settings["seed"],
        use_driver=settings["use_driver"],
    )
    return (periodic_data,)


@app.cell
def _(fit_models, periodic_data, settings):
    forecast_results = fit_models(
        frame=periodic_data,
        horizon=settings["horizon"],
        train_fraction=settings["train_fraction"],
        roughness_weight=settings["roughness_weight"],
        use_driver=settings["use_driver"],
    )
    return (forecast_results,)


@app.cell
def _(
    component_frame,
    driver_contribution_frame,
    driver_linear_function_frame,
    driver_response_frame,
    forecast_results,
    metric_frame,
    model_difference_frame,
    periodic_data,
):
    train_stop = forecast_results["train_stop"]
    component_plot_data = component_frame(periodic_data)
    driver_contribution_plot_data = driver_contribution_frame(periodic_data)
    metric_plot_data = metric_frame(forecast_results)
    model_difference_plot_data = model_difference_frame(forecast_results)
    driver_response_plot_data = driver_response_frame(forecast_results)
    driver_linear_function_plot_data = driver_linear_function_frame(
        driver_response_plot_data,
        periodic_data,
    )
    return (
        component_plot_data,
        driver_contribution_plot_data,
        driver_linear_function_plot_data,
        metric_plot_data,
        model_difference_plot_data,
        train_stop,
    )


@app.cell
def _(mo, settings, train_stop):
    setup_panel = mo.md(
        "## 2. The forecasting setup\n\n"
        "The generated target is periodic signal plus noise. Forecast mode is "
        "asked to predict several future steps from each forecast origin. "
        "The optional linear driver adds a small lagged linear contribution "
        "to the target. "
        "When the driver is off, the model uses only "
        "target-time seasonality: horizon 1 predicts the seasonal value at "
        "`t+1`, horizon 2 predicts the seasonal value at `t+2`, and so on.\n\n"
        f"Current fit: horizon `{settings['horizon']}`, "
        f"`{settings['samples']}` hourly samples, train split row `{train_stop}`, "
        f"noise scale `{settings['noise_scale']}`, coupled roughness "
        f"`{settings['roughness_weight']:g}`, linear driver "
        f"`{'on' if settings['use_driver'] else 'off'}`."
    )
    setup_panel
    return


@app.cell
def _(alt, component_plot_data, mo, style_chart):
    component_chart = style_chart(
        (
            alt.Chart(component_plot_data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("component:N", title=None),
                row=alt.Row("component:N", title=None),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="time"),
                    alt.Tooltip("component:N", title="component"),
                    alt.Tooltip("value:Q", title="value", format=".3f"),
                ],
            )
            .properties(height=75, title="Components used to generate the signal")
            .resolve_scale(y="independent")
        )
    )
    component_panel = mo.vstack(
        [
            mo.md(
                "The component chart is the synthetic truth. TSGAM does not see "
                "these component columns directly; it sees timestamps, and when "
                "enabled, the one linear driver. The hidden components let us "
                "check whether the fitted behavior is sensible."
            ),
            component_chart,
        ]
    )
    component_panel
    return


@app.cell
def _(alt, driver_contribution_plot_data, mo, settings, style_chart):
    driver_contribution_chart = style_chart(
        (
            alt.Chart(driver_contribution_plot_data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("component:N", title=None),
                row=alt.Row("component:N", title=None),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="time"),
                    alt.Tooltip("component:N", title="component"),
                    alt.Tooltip("value:Q", title="value", format=".3f"),
                ],
            )
            .properties(height=95, title="How the driver enters the target")
            .resolve_scale(y="independent")
        )
    )
    driver_panel = mo.vstack(
        [
            mo.md(
                "## 4. Optional linear driver\n\n"
                "Here the exogenous driver is a smooth, non-periodic profile "
                "with tiny noise. Think of it as a measured external condition, "
                "not another sine or cosine component.\n\n"
                "When the switch is on, the target gets a known lagged linear "
                "contribution:\n\n"
                "`0.55 * driver[t] + 0.30 * driver[t-1] + "
                "0.15 * driver[t-2]`\n\n"
                "The goal is to see whether independent and coupled forecast "
                "mode recover this easy exogenous effect without "
                "over-penalizing it. When the switch is off, the contribution "
                "curves are zero.\n\n"
                f"Linear driver is currently "
                f"`{'on' if settings['use_driver'] else 'off'}`."
            ),
            driver_contribution_chart,
        ]
    )
    driver_panel
    return


@app.cell
def _(forecast_results, mo):
    max_origin_offset = min(96, len(forecast_results["x_eval"]) - 1)
    max_horizon = max(forecast_results["independent"].horizons_)
    origin_slider = mo.ui.slider(
        start=0,
        stop=max_origin_offset,
        step=1,
        value=0,
        label="Evaluation origin to inspect",
        full_width=True,
    )
    horizon_slider = mo.ui.slider(
        start=0,
        stop=max_horizon,
        step=1,
        value=min(1, max_horizon),
        label="Horizon to plot over evaluation time",
        full_width=True,
    )
    forecast_controls = mo.vstack(
        [
            mo.md("## 5. What forecasts look like as time moves forward"),
            origin_slider,
            horizon_slider,
        ]
    )
    forecast_controls
    return horizon_slider, origin_slider


@app.cell
def _(
    forecast_results,
    horizon_path_frame,
    horizon_slider,
    origin_slider,
    prediction_by_horizon_frame,
):
    path_plot_data = horizon_path_frame(
        forecast_results,
        origin_offset=origin_slider.value,
    )
    horizon_series_plot_data = prediction_by_horizon_frame(
        forecast_results,
        horizon=horizon_slider.value,
    )
    return horizon_series_plot_data, path_plot_data


@app.cell
def _(
    alt,
    horizon_series_plot_data,
    horizon_slider,
    mo,
    path_plot_data,
    style_chart,
):
    path_origin = path_plot_data["origin_time"].iloc[0]
    series_domain = [
        "actual",
        "independent",
        "coupled",
    ]
    series_colors = ["#222222", "#e45756", "#4c78a8"]
    path_chart = style_chart(
        (
            alt.Chart(path_plot_data)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("target_time:T", title="target time"),
                y=alt.Y("value:Q", title="value"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(domain=series_domain, range=series_colors),
                ),
                strokeDash=alt.StrokeDash("series:N", title=None),
                tooltip=[
                    alt.Tooltip("origin_time:T", title="origin"),
                    alt.Tooltip("target_time:T", title="target"),
                    alt.Tooltip("horizon:O", title="horizon"),
                    alt.Tooltip("series:N", title="series"),
                    alt.Tooltip("value:Q", title="value", format=".3f"),
                ],
            )
            .properties(
                height=280,
                title="One forecast origin expanded into horizons",
            )
        )
    )
    horizon_series_chart = style_chart(
        (
            alt.Chart(horizon_series_plot_data)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("origin_time:T", title="forecast origin"),
                y=alt.Y("value:Q", title="value"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(domain=series_domain, range=series_colors),
                ),
                strokeDash=alt.StrokeDash("series:N", title=None),
                tooltip=[
                    alt.Tooltip("origin_time:T", title="origin"),
                    alt.Tooltip("series:N", title="series"),
                    alt.Tooltip("value:Q", title="value", format=".3f"),
                ],
            )
            .properties(
                height=280,
                title=f"Horizon {horizon_slider.value}: forecast vs actual",
            )
        )
    )
    movement_panel = mo.vstack(
        [
            mo.md(
                f"For origin `{path_origin}`, the first chart shows the whole "
                "forecast window. The black line is the actual shifted target; "
                "the colored lines are independent and coupled forecasts. The "
                "second chart fixes one horizon and repeats the same comparison "
                "over evaluation origins."
            ),
            path_chart,
            horizon_series_chart,
        ]
    )
    movement_panel
    return


@app.cell
def _(alt, driver_linear_function_plot_data, mo, settings, style_chart):
    coefficient_models = ["true", "coupled", "independent"]
    coefficient_colors = ["#222222", "#4c78a8", "#e45756"]
    lag_coefficient_chart = style_chart(
        (
            alt.Chart(driver_linear_function_plot_data)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("horizon:O", title="horizon"),
                y=alt.Y("coefficient:Q", title="coefficient"),
                color=alt.Color(
                    "model:N",
                    title="model",
                    sort=coefficient_models,
                    scale=alt.Scale(
                        domain=coefficient_models,
                        range=coefficient_colors,
                    ),
                ),
                column=alt.Column("lag label:N", title=None),
                tooltip=[
                    alt.Tooltip("model:N", title="model"),
                    alt.Tooltip("horizon:O", title="horizon"),
                    alt.Tooltip("lag:O", title="lag"),
                    alt.Tooltip("coefficient:Q", title="coefficient", format=".3f"),
                ],
            )
            .properties(height=260, title="Fitted lag coefficients by horizon")
        )
    )
    fit_panel = mo.vstack(
        [
            mo.md(
                "## 6. Independent vs coupled fit behavior\n\n"
                "Independent mode fits each horizon separately. Coupled mode "
                "solves the horizons together and penalizes rough jumps in "
                "matching coefficients across horizon. When the linear driver "
                "is on, this chart shows the fitted linear coefficients "
                "directly. Each panel is one exogenous lag; the red line is the "
                "independent fit and the blue line is the coupled fit."
            ),
            mo.md(
                f"Linear driver is currently "
                f"`{'on' if settings['use_driver'] else 'off'}`."
            ),
            mo.md(
                "The generated target uses `0.55 * driver[t] + "
                "0.30 * driver[t-1] + 0.15 * driver[t-2]`. The forecast model "
                "uses those same three lags; coupled mode should smooth the "
                "estimated lag coefficients across horizon without washing out "
                "the effect."
            ),
            lag_coefficient_chart,
        ]
    )
    fit_panel
    return


@app.cell
def _(alt, mo, model_difference_plot_data, style_chart):
    difference_chart = style_chart(
        (
            alt.Chart(model_difference_plot_data)
            .mark_rect()
            .encode(
                x=alt.X("origin_time:T", title="forecast origin"),
                y=alt.Y("horizon:O", title="horizon"),
                color=alt.Color(
                    "difference:Q",
                    title="coupled - independent",
                    scale=alt.Scale(scheme="redblue", reverse=True),
                ),
                tooltip=[
                    alt.Tooltip("origin_time:T", title="origin"),
                    alt.Tooltip("horizon:N", title="horizon"),
                    alt.Tooltip(
                        "difference:Q",
                        title="coupled - independent",
                        format=".4f",
                    ),
                ],
            )
            .properties(height=190, title="Where coupled differs from independent")
        )
    )
    difference_panel = mo.vstack(
        [
            mo.md(
                "This chart compares the two fitted forecast modes directly. "
                "White or nearly-white cells mean the two modes are making "
                "almost the same prediction, which is common in this simple "
                "periodic-only example."
            ),
            difference_chart,
        ]
    )
    difference_panel
    return


@app.cell
def _(alt, metric_plot_data, mo, style_chart):
    metric_long = metric_plot_data.melt(
        id_vars=["model", "horizon"],
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    metric_chart = style_chart(
        (
            alt.Chart(metric_long)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("horizon:O", title="horizon"),
                y=alt.Y("value:Q", title="error"),
                color=alt.Color("model:N", title=None),
                strokeDash=alt.StrokeDash("metric:N", title=None),
                tooltip=[
                    alt.Tooltip("model:N", title="model"),
                    alt.Tooltip("metric:N", title="metric"),
                    alt.Tooltip("horizon:O", title="horizon"),
                    alt.Tooltip("value:Q", title="value", format=".3f"),
                ],
            )
            .properties(height=260, title="Held-out error by horizon")
        )
    )
    performance_panel = mo.vstack(
        [
            mo.md(
                "## 7. Performance\n\n"
                "The error chart summarizes forecast quality against the actual "
                "held-out shifted target. With this clean synthetic data, both "
                "modes should perform similarly unless the coupled roughness "
                "weight is pushed high enough to visibly smooth the horizon "
                "coefficients."
            ),
            metric_chart,
        ]
    )
    performance_panel
    return


if __name__ == "__main__":
    app.run()
