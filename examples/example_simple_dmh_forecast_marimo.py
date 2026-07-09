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

"""Small marimo walkthrough for direct multi-horizon TSGAM forecasts."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    walkthrough_intro = mo.md(
        "# A Simple Direct Multi-Horizon Forecast\n\n"
        "This notebook builds the smallest useful forecast-mode example: one "
        "known driver, one target series, and three direct forecast horizons. "
        "It keeps the synthetic data small so the core contract is visible: "
        "`fit(X, y)` learns `demand[t + h]` from information at origin `t`, "
        "and `predict(X_future)` returns one column per horizon.\n\n"
        "The target is generated so that the weather value at a forecast "
        "origin has a different effect at each future horizon."
    )
    walkthrough_intro
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
        TsgamForecastEstimator,
        TsgamLinearConfig,
        TsgamSolverConfig,
    )

    alt.data_transformers.disable_max_rows()
    return (
        TsgamEstimatorConfig,
        TsgamForecastConfig,
        TsgamForecastEstimator,
        TsgamLinearConfig,
        TsgamSolverConfig,
        alt,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    sample_count = mo.ui.slider(
        start=90,
        stop=360,
        step=30,
        value=180,
        label="Number of hourly samples",
        full_width=True,
    )
    train_fraction = mo.ui.slider(
        start=0.55,
        stop=0.85,
        step=0.05,
        value=0.70,
        label="Training fraction",
        full_width=True,
    )
    noise_scale = mo.ui.slider(
        start=0.0,
        stop=0.6,
        step=0.05,
        value=0.15,
        label="Noise scale",
        full_width=True,
    )
    random_seed = mo.ui.number(
        start=0,
        stop=999,
        step=1,
        value=7,
        label="Random seed",
    )
    controls_view = mo.vstack(
        [
            mo.md("## 1. Pick the sample problem"),
            sample_count,
            train_fraction,
            noise_scale,
            random_seed,
        ]
    )
    controls_view
    return noise_scale, random_seed, sample_count, train_fraction


@app.cell
def _(np, pd):
    FORECAST_HORIZON = 3
    TRUE_WEATHER_EFFECT = {
        1: 2.0,
        2: -1.0,
        3: 0.6,
    }

    def make_simple_problem(
        samples: int,
        noise: float,
        seed: int,
    ) -> tuple[pd.DataFrame, dict[int, float]]:
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2024-01-01", periods=samples, freq="1h")
        weather_signal = rng.normal(size=samples)
        demand = rng.normal(scale=noise, size=samples)
        component_columns = {}
        for horizon, coefficient in TRUE_WEATHER_EFFECT.items():
            contribution = np.zeros(samples)
            contribution[horizon:] = coefficient * weather_signal[:-horizon]
            component_columns[f"weather_effect_h{horizon}"] = contribution
            demand += contribution
        problem_data = pd.DataFrame(
            {
                "weather_signal": weather_signal,
                "demand": demand,
                **component_columns,
            },
            index=timestamps,
        )
        return problem_data, TRUE_WEATHER_EFFECT

    def build_actual_targets(
        frame: pd.DataFrame,
        origin_index: pd.DatetimeIndex,
        horizon: int,
    ) -> pd.DataFrame:
        target_series = frame["demand"]
        return pd.DataFrame(
            {
                f"horizon_{step}": target_series.shift(-step).loc[origin_index]
                for step in range(horizon + 1)
            },
            index=origin_index,
        )

    return FORECAST_HORIZON, build_actual_targets, make_simple_problem


@app.cell
def _(make_simple_problem, noise_scale, random_seed, sample_count):
    simple_problem_frame, true_weather_effect = make_simple_problem(
        samples=sample_count.value,
        noise=noise_scale.value,
        seed=random_seed.value,
    )
    return simple_problem_frame, true_weather_effect


@app.cell
def _(FORECAST_HORIZON, mo, pd, true_weather_effect):
    truth_table = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "target": f"demand[t + {horizon}]",
                "feature known at origin": "weather_signal[t]",
                "true coefficient": coefficient,
            }
            for horizon, coefficient in true_weather_effect.items()
        ]
    )
    target_formula_view = mo.vstack(
        [
            mo.md(
                "## 2. What problem did we generate?\n\n"
                "The target is generated from lagged weather effects:\n\n"
                "`demand[t] = 2.0 * weather[t-1] - 1.0 * weather[t-2] + "
                "0.6 * weather[t-3] + noise[t]`\n\n"
                "That means a forecast issued at origin `t` should learn three "
                "different direct relationships from the same known feature "
                "`weather_signal[t]`."
            ),
            mo.ui.table(truth_table, pagination=False),
            mo.md(f"The forecast horizon is fixed at `{FORECAST_HORIZON}` steps."),
        ]
    )
    target_formula_view
    return


@app.cell
def _(FORECAST_HORIZON, simple_problem_frame):
    component_value_columns = [
        f"weather_effect_h{step}" for step in range(1, FORECAST_HORIZON + 1)
    ]
    synthetic_component_frame = (
        simple_problem_frame[component_value_columns + ["demand"]]
        .rename(
            columns={
                "weather_effect_h1": "effect from weather[t-1]",
                "weather_effect_h2": "effect from weather[t-2]",
                "weather_effect_h3": "effect from weather[t-3]",
                "demand": "observed demand",
            }
        )
        .reset_index(names="timestamp")
        .melt(
            id_vars=["timestamp"],
            var_name="series",
            value_name="value",
        )
    )
    observed_signal_frame = (
        simple_problem_frame[["weather_signal", "demand"]]
        .reset_index(names="timestamp")
        .melt(
            id_vars=["timestamp"],
            var_name="series",
            value_name="value",
        )
    )
    return observed_signal_frame, synthetic_component_frame


@app.cell
def _(alt, mo, observed_signal_frame, synthetic_component_frame):
    observed_signal_chart = (
        alt.Chart(observed_signal_frame)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("series:N", title=None),
            row=alt.Row("series:N", title=None),
            tooltip=[
                alt.Tooltip("timestamp:T", title="time"),
                alt.Tooltip("series:N", title="series"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(height=120, title="The observed feature and target")
        .resolve_scale(y="independent")
    )
    component_decomposition_chart = (
        alt.Chart(synthetic_component_frame)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("series:N", title=None),
            row=alt.Row("series:N", title=None),
            tooltip=[
                alt.Tooltip("timestamp:T", title="time"),
                alt.Tooltip("series:N", title="series"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(height=90, title="Hidden components that add up to demand")
        .resolve_scale(y="independent")
    )
    generated_data_view = mo.vstack(
        [
            mo.md(
                "The library will only receive `weather_signal` and `demand`. "
                "The component chart is shown because this is a synthetic "
                "example: it lets us check whether the fitted horizon models "
                "recover the known effects."
            ),
            observed_signal_chart,
            component_decomposition_chart,
        ]
    )
    generated_data_view
    return


@app.cell
def _(FORECAST_HORIZON, pd, simple_problem_frame):
    example_origin_position = 8
    example_origin_time = simple_problem_frame.index[example_origin_position]
    alignment_rows = []
    for alignment_step in range(1, FORECAST_HORIZON + 1):
        alignment_target_time = simple_problem_frame.index[
            example_origin_position + alignment_step
        ]
        alignment_rows.append(
            {
                "horizon": alignment_step,
                "training row feature": f"weather_signal at {example_origin_time}",
                "feature value": round(
                    float(simple_problem_frame.loc[example_origin_time, "weather_signal"]),
                    3,
                ),
                "target": f"demand at {alignment_target_time}",
                "target value": round(
                    float(simple_problem_frame.loc[alignment_target_time, "demand"]),
                    3,
                ),
                "array slice": f"X[:-{alignment_step}] -> y[{alignment_step}:]",
            }
        )
    alignment_table = pd.DataFrame(alignment_rows)
    return (alignment_table,)


@app.cell
def _(alignment_table, mo):
    alignment_view = mo.vstack(
        [
            mo.md(
                "## 3. Direct multi-horizon alignment\n\n"
                "For every forecast origin, direct multi-horizon forecasting "
                "creates one target per future step. Horizon 1 predicts the "
                "next row, horizon 2 predicts two rows ahead, and horizon 3 "
                "predicts three rows ahead.\n\n"
                "This is why the training slices are `X[:-1] -> y[1:]`, "
                "`X[:-2] -> y[2:]`, and `X[:-3] -> y[3:]`."
            ),
            mo.ui.table(alignment_table, pagination=False),
        ]
    )
    alignment_view
    return


@app.cell
def _(
    FORECAST_HORIZON,
    TsgamEstimatorConfig,
    TsgamForecastConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamSolverConfig,
    build_actual_targets,
    np,
    pd,
    simple_problem_frame,
    train_fraction,
):
    train_stop = int(len(simple_problem_frame) * train_fraction.value)
    train_stop = max(train_stop, 24)
    train_stop = min(train_stop, len(simple_problem_frame) - FORECAST_HORIZON - 1)
    X_all = simple_problem_frame[["weather_signal"]]
    y_all = simple_problem_frame["demand"].to_numpy()
    X_train = X_all.iloc[:train_stop]
    y_train = y_all[:train_stop]
    X_eval = X_all.iloc[train_stop:-FORECAST_HORIZON]
    base_config = TsgamEstimatorConfig(
        multi_periodic_config=None,
        exog_config=[TsgamLinearConfig(lags=[0], reg_weight=1.0e-7)],
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )
    simple_forecast_model = TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=FORECAST_HORIZON,
            base_config=base_config,
            mode="independent",
        )
    ).fit(X_train, y_train)
    simple_predictions = simple_forecast_model.predict(X_eval)
    simple_actuals = build_actual_targets(
        simple_problem_frame,
        origin_index=X_eval.index,
        horizon=FORECAST_HORIZON,
    )
    metric_rows = []
    coefficient_rows = []
    for metric_step in range(1, FORECAST_HORIZON + 1):
        metric_column = f"horizon_{metric_step}"
        forecast_error = simple_predictions[metric_column] - simple_actuals[metric_column]
        metric_rows.append(
            {
                "horizon": metric_step,
                "rmse": float(np.sqrt(np.mean(forecast_error**2))),
                "mae": float(np.mean(np.abs(forecast_error))),
            }
        )
        child_model = simple_forecast_model.forecast_estimators_[metric_step]
        coefficient_rows.append(
            {
                "horizon": metric_step,
                "learned coefficient": float(
                    child_model.variables_["exog_coef_0"].value[0, 0]
                ),
            }
        )
    simple_metrics = pd.DataFrame(metric_rows)
    learned_coefficients = pd.DataFrame(coefficient_rows)
    return (
        X_eval,
        learned_coefficients,
        simple_actuals,
        simple_metrics,
        simple_predictions,
        train_stop,
    )


@app.cell
def _(mo, train_stop):
    split_view = mo.md(
        "## 4. Fit `TsgamForecastEstimator`\n\n"
        f"The first `{train_stop}` rows are made available to fitting. Horizon "
        "specific direct regressions then drop the final `h` rows from that "
        "training block because their targets would fall outside the fitted "
        "target vector. Evaluation starts at the first held-out origin and "
        "also drops the dataset tail that cannot provide all future targets.\n\n"
        "The model receives a normal single-output `TsgamEstimatorConfig`, then "
        "`TsgamForecastConfig(horizon=3, mode=\"independent\")` turns it into "
        "three direct forecast regressions with output columns `horizon_1`, "
        "`horizon_2`, and `horizon_3`."
    )
    split_view
    return


@app.cell
def _(FORECAST_HORIZON, pd, simple_problem_frame, train_stop):
    alignment_scatter_rows = []
    for scatter_horizon in range(1, FORECAST_HORIZON + 1):
        for scatter_origin_position in range(
            0,
            len(simple_problem_frame) - scatter_horizon,
        ):
            alignment_scatter_rows.append(
                {
                    "horizon": f"horizon {scatter_horizon}",
                    "origin_time": simple_problem_frame.index[
                        scatter_origin_position
                    ],
                    "split": (
                        "fit row"
                        if scatter_origin_position < train_stop - scatter_horizon
                        else (
                            "scored evaluation origin"
                            if train_stop
                            <= scatter_origin_position
                            < len(simple_problem_frame) - FORECAST_HORIZON
                            else "not fit/scored"
                        )
                    ),
                    "weather at origin": float(
                        simple_problem_frame["weather_signal"].iloc[
                            scatter_origin_position
                        ]
                    ),
                    "future demand target": float(
                        simple_problem_frame["demand"].iloc[
                            scatter_origin_position + scatter_horizon
                        ]
                    ),
                }
            )
    supervised_alignment_frame = pd.DataFrame(alignment_scatter_rows)
    return (supervised_alignment_frame,)


@app.cell
def _(alt, mo, supervised_alignment_frame):
    alignment_points = (
        alt.Chart()
        .mark_circle(size=35, opacity=0.45)
        .encode(
            x=alt.X("weather at origin:Q", title="weather_signal[t]"),
            y=alt.Y("future demand target:Q", title="demand[t + h]"),
            color=alt.Color("split:N", title=None),
            tooltip=[
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("horizon:N", title="horizon"),
                alt.Tooltip("split:N", title="split"),
                alt.Tooltip("weather at origin:Q", title="weather", format=".3f"),
                alt.Tooltip("future demand target:Q", title="target", format=".3f"),
            ],
        )
    )
    alignment_regression = (
        alt.Chart()
        .transform_regression(
            "weather at origin",
            "future demand target",
            groupby=["horizon"],
        )
        .mark_line(color="#222222", strokeWidth=2)
        .encode(
            x="weather at origin:Q",
            y="future demand target:Q",
        )
    )
    supervised_alignment_chart = (
        alt.layer(
            alignment_points,
            alignment_regression,
            data=supervised_alignment_frame,
        )
        .properties(width=250, height=260)
        .facet(column=alt.Column("horizon:N", title=None))
        .properties(title="The supervised regression each horizon actually sees")
    )
    supervised_alignment_view = mo.vstack(
        [
            mo.md(
                "This is the most literal chart for direct multi-horizon "
                "forecasting. Each panel is one fitted problem: the feature is "
                "`weather_signal[t]`, while the target moves from "
                "`demand[t+1]` to `demand[t+3]`. The regression slope changes "
                "because the synthetic truth gave each horizon a different "
                "weather effect."
            ),
            supervised_alignment_chart,
        ]
    )
    supervised_alignment_view
    return


@app.cell
def _(FORECAST_HORIZON, learned_coefficients, pd, true_weather_effect):
    true_coefficients = pd.DataFrame(
        [
            {
                "horizon": step,
                "coefficient type": "true",
                "coefficient": value,
            }
            for step, value in true_weather_effect.items()
        ]
    )
    fitted_coefficients = learned_coefficients.rename(
        columns={"learned coefficient": "coefficient"}
    ).assign(**{"coefficient type": "learned"})
    coefficient_comparison = pd.concat(
        [
            true_coefficients,
            fitted_coefficients[
                ["horizon", "coefficient type", "coefficient"]
            ],
        ],
        ignore_index=True,
    )
    coefficient_error_table = (
        true_coefficients.rename(columns={"coefficient": "true coefficient"})
        .drop(columns=["coefficient type"])
        .merge(
            fitted_coefficients.rename(
                columns={"coefficient": "learned coefficient"}
            )[["horizon", "learned coefficient"]],
            on="horizon",
        )
    )
    coefficient_error_table["learned minus true"] = (
        coefficient_error_table["learned coefficient"]
        - coefficient_error_table["true coefficient"]
    )
    prediction_shape_table = pd.DataFrame(
        [
            {
                "object": "predictions",
                "rows": "one per evaluation origin",
                "columns": ", ".join(
                    f"horizon_{step}" for step in range(1, FORECAST_HORIZON + 1)
                ),
            },
            {
                "object": "actual targets",
                "rows": "same evaluation origins",
                "columns": "same horizon columns for scoring",
            },
        ]
    )
    return (
        coefficient_comparison,
        coefficient_error_table,
        prediction_shape_table,
    )


@app.cell
def _(
    alt,
    coefficient_comparison,
    coefficient_error_table,
    mo,
    prediction_shape_table,
):
    coefficient_chart = (
        alt.Chart(coefficient_comparison)
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon:O", title="forecast horizon"),
            y=alt.Y("coefficient:Q", title="weather coefficient"),
            color=alt.Color("coefficient type:N", title=None),
            tooltip=[
                alt.Tooltip("horizon:O", title="horizon"),
                alt.Tooltip("coefficient type:N", title="type"),
                alt.Tooltip("coefficient:Q", title="coefficient", format=".3f"),
            ],
        )
        .properties(height=280, title="Did each horizon recover its own effect?")
    )
    fitted_model_view = mo.vstack(
        [
            mo.ui.table(prediction_shape_table, pagination=False),
            coefficient_chart,
            mo.md("Coefficient error is a direct check that each horizon learned its own target alignment."),
            mo.ui.table(coefficient_error_table.round(4), pagination=False),
        ]
    )
    fitted_model_view
    return


@app.cell
def _(FORECAST_HORIZON, X_eval, mo):
    max_origin_to_show = min(48, len(X_eval) - 1)
    inspected_origin = mo.ui.slider(
        start=0,
        stop=max_origin_to_show,
        step=1,
        value=0,
        label="Evaluation origin to inspect",
        full_width=True,
    )
    horizon_to_plot = mo.ui.slider(
        start=1,
        stop=FORECAST_HORIZON,
        step=1,
        value=1,
        label="Horizon to plot over time",
        full_width=True,
    )
    forecast_controls = mo.vstack(
        [
            mo.md("## 5. Inspect what the forecasts are doing"),
            inspected_origin,
            horizon_to_plot,
        ]
    )
    forecast_controls
    return horizon_to_plot, inspected_origin


@app.cell
def _(inspected_origin, pd, simple_actuals, simple_predictions):
    origin_position_to_show = inspected_origin.value
    selected_origin_time = simple_predictions.index[origin_position_to_show]
    forecast_path_rows = []
    for path_column in simple_predictions.columns:
        path_step = int(path_column.split("_")[1])
        path_target_time = selected_origin_time + pd.Timedelta(hours=path_step)
        forecast_path_rows.append(
            {
                "series": "actual future demand",
                "horizon": path_step,
                "target_time": path_target_time,
                "value": float(simple_actuals.iloc[origin_position_to_show][path_column]),
            }
        )
        forecast_path_rows.append(
            {
                "series": "forecast",
                "horizon": path_step,
                "target_time": path_target_time,
                "value": float(
                    simple_predictions.iloc[origin_position_to_show][path_column]
                ),
            }
        )
    forecast_path_data = pd.DataFrame(forecast_path_rows)
    return forecast_path_data, selected_origin_time


@app.cell
def _(alt, forecast_path_data, mo, selected_origin_time):
    forecast_path_chart = (
        alt.Chart(forecast_path_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("target_time:T", title="target time"),
            y=alt.Y("value:Q", title="demand"),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("target_time:T", title="target"),
                alt.Tooltip("horizon:O", title="horizon"),
                alt.Tooltip("series:N", title="series"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(height=300, title="One forecast origin gives three targets")
    )
    forecast_path_view = mo.vstack(
        [
            mo.md(
                f"At forecast origin `{selected_origin_time}`, `predict` "
                "returns one row with `horizon_0`, `horizon_1`, `horizon_2`, and "
                "`horizon_3`. The chart unwraps that row into target time."
            ),
            forecast_path_chart,
        ]
    )
    forecast_path_view
    return


@app.cell
def _(horizon_to_plot, simple_actuals, simple_predictions):
    selected_horizon_column = f"horizon_{horizon_to_plot.value}"
    horizon_prediction_frame = (
        simple_actuals[[selected_horizon_column]]
        .rename(columns={selected_horizon_column: "actual"})
        .join(
            simple_predictions[[selected_horizon_column]].rename(
                columns={selected_horizon_column: "forecast"}
            )
        )
        .reset_index(names="origin_time")
        .melt(
            id_vars=["origin_time"],
            var_name="series",
            value_name="value",
        )
    )
    return (horizon_prediction_frame,)


@app.cell
def _(alt, horizon_prediction_frame, horizon_to_plot):
    horizon_prediction_chart = (
        alt.Chart(horizon_prediction_frame)
        .mark_line()
        .encode(
            x=alt.X("origin_time:T", title="forecast origin"),
            y=alt.Y("value:Q", title="demand"),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("series:N", title="series"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(
            height=300,
            title=f"Forecast vs actual over time for horizon {horizon_to_plot.value}",
        )
    )
    horizon_prediction_chart
    return


@app.cell
def _(FORECAST_HORIZON, pd, simple_actuals, simple_predictions):
    calibration_rows = []
    for calibration_horizon in range(1, FORECAST_HORIZON + 1):
        calibration_column = f"horizon_{calibration_horizon}"
        for calibration_origin_time in simple_predictions.index:
            calibration_rows.append(
                {
                    "horizon": f"horizon {calibration_horizon}",
                    "origin_time": calibration_origin_time,
                    "actual": float(
                        simple_actuals.loc[
                            calibration_origin_time,
                            calibration_column,
                        ]
                    ),
                    "forecast": float(
                        simple_predictions.loc[
                            calibration_origin_time,
                            calibration_column,
                        ]
                    ),
                }
            )
    forecast_calibration_frame = pd.DataFrame(calibration_rows)
    return (forecast_calibration_frame,)


@app.cell
def _(alt, forecast_calibration_frame, mo):
    calibration_points = (
        alt.Chart()
        .mark_circle(size=35, opacity=0.45)
        .encode(
            x=alt.X("actual:Q", title="actual future demand"),
            y=alt.Y("forecast:Q", title="forecast demand"),
            color=alt.Color("horizon:N", title=None),
            tooltip=[
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("horizon:N", title="horizon"),
                alt.Tooltip("actual:Q", title="actual", format=".3f"),
                alt.Tooltip("forecast:Q", title="forecast", format=".3f"),
            ],
        )
    )
    calibration_fit = (
        alt.Chart()
        .transform_regression("actual", "forecast", groupby=["horizon"])
        .mark_line(color="#404040", strokeWidth=2)
        .encode(x="actual:Q", y="forecast:Q")
    )
    calibration_chart = (
        alt.layer(
            calibration_points,
            calibration_fit,
            data=forecast_calibration_frame,
        )
        .properties(width=250, height=260)
        .facet(column=alt.Column("horizon:N", title=None))
        .properties(title="Calibration: fitted forecast-vs-actual line by horizon")
    )
    calibration_view = mo.vstack(
        [
            mo.md(
                "Calibration is the simplest held-out sanity check: points on "
                "a tight line mean the forecast tracks the actual future "
                "target. Faceting by horizon shows whether one direct model is "
                "biased, noisy, or has the wrong slope."
            ),
            calibration_chart,
        ]
    )
    calibration_view
    return


@app.cell
def _(simple_actuals, simple_predictions):
    forecast_error_frame = (
        (simple_predictions - simple_actuals)
        .reset_index(names="origin_time")
        .melt(
            id_vars=["origin_time"],
            var_name="horizon",
            value_name="error",
        )
    )
    forecast_error_frame["horizon"] = forecast_error_frame["horizon"].str.replace(
        "horizon_",
        "",
        regex=False,
    )
    forecast_error_frame["horizon_label"] = (
        "horizon " + forecast_error_frame["horizon"]
    )
    forecast_error_frame["absolute_error"] = forecast_error_frame["error"].abs()
    rolling_error_frame = (
        forecast_error_frame.sort_values(["horizon", "origin_time"])
        .assign(
            rolling_absolute_error=lambda frame: frame.groupby("horizon")[
                "absolute_error"
            ].transform(lambda values: values.rolling(12, min_periods=3).mean())
        )
        .dropna(subset=["rolling_absolute_error"])
    )
    return forecast_error_frame, rolling_error_frame


@app.cell
def _(alt, forecast_error_frame, mo, rolling_error_frame, simple_metrics):
    metric_frame_long = simple_metrics.melt(
        id_vars=["horizon"],
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    metric_chart = (
        alt.Chart(metric_frame_long)
        .mark_bar()
        .encode(
            x=alt.X("horizon:O", title="horizon"),
            y=alt.Y("value:Q", title="error"),
            color=alt.Color("metric:N", title=None),
            xOffset=alt.XOffset("metric:N"),
            tooltip=[
                alt.Tooltip("horizon:O", title="horizon"),
                alt.Tooltip("metric:N", title="metric"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(height=260, title="Held-out error by horizon")
    )
    error_heatmap = (
        alt.Chart(forecast_error_frame)
        .mark_rect()
        .encode(
            x=alt.X("origin_time:T", title="forecast origin"),
            y=alt.Y("horizon_label:O", title="horizon"),
            color=alt.Color(
                "error:Q",
                title="forecast error",
                scale=alt.Scale(scheme="redblue", reverse=True),
            ),
            tooltip=[
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("horizon_label:N", title="horizon"),
                alt.Tooltip("error:Q", title="error", format=".3f"),
            ],
        )
        .properties(height=180, title="Forecast error over time")
    )
    residual_boxplot = (
        alt.Chart(forecast_error_frame)
        .mark_boxplot(size=46)
        .encode(
            x=alt.X("horizon_label:O", title="horizon"),
            y=alt.Y("error:Q", title="forecast error"),
            color=alt.Color("horizon_label:N", title=None),
            tooltip=[
                alt.Tooltip("horizon_label:N", title="horizon"),
                alt.Tooltip("error:Q", title="error", format=".3f"),
            ],
        )
        .properties(height=260, title="Residual distribution by horizon")
    )
    rolling_error_chart = (
        alt.Chart(rolling_error_frame)
        .mark_line()
        .encode(
            x=alt.X("origin_time:T", title="forecast origin"),
            y=alt.Y(
                "rolling_absolute_error:Q",
                title="rolling mean absolute error",
            ),
            color=alt.Color("horizon_label:N", title=None),
            tooltip=[
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("horizon_label:N", title="horizon"),
                alt.Tooltip(
                    "rolling_absolute_error:Q",
                    title="rolling abs error",
                    format=".3f",
                ),
            ],
        )
        .properties(height=260, title="12-origin rolling absolute error")
    )
    performance_view = mo.vstack(
        [
            mo.md(
                "## 6. Performance\n\n"
                "The error charts compare the forecast output against the "
                "future-target table created with the same horizon columns. "
                "The bar chart gives the summary, the heatmap shows when "
                "misses happen, and the residual charts show spread and bias."
            ),
            metric_chart,
            error_heatmap,
            residual_boxplot,
            rolling_error_chart,
            mo.ui.table(simple_metrics, pagination=False),
        ]
    )
    performance_view
    return


if __name__ == "__main__":
    app.run()
