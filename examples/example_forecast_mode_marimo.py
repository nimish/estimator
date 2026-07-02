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

"""Interactive marimo notebook for direct and coupled TSGAM forecast mode."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    intro = (
        "# Forecast Mode Walkthrough\n\n"
        "This notebook walks through the new `TsgamForecastEstimator` using a "
        "small demand-forecasting problem. We generate hourly demand from a "
        "known weather signal, turn it into horizon-specific supervised "
        "forecast rows, and compare independent versus coupled forecast mode. "
        "Use it as an explanation of the API shape, not as a benchmark."
    )
    mo.md(intro)
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
    )


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    PROFILE_OPTIONS = ["smooth decay", "jagged response", "delayed peak"]
    return (PROFILE_OPTIONS,)


@app.cell
def _(PROFILE_OPTIONS, mo):
    section_intro = mo.md(
        "## 1. Configure a sample multi-horizon problem\n\n"
        "The synthetic demand has a daily pattern plus a weather signal whose "
        "effect changes by forecast horizon. The weather signal is only used "
        "at the forecast origin, which is the default causal forecast setting."
    )
    horizon_count = mo.ui.slider(
        start=2,
        stop=6,
        step=1,
        value=4,
        label="Forecast horizons",
        full_width=True,
    )
    profile_name = mo.ui.dropdown(
        options=PROFILE_OPTIONS,
        value="jagged response",
        label="True horizon response",
    )
    roughness_weight = mo.ui.slider(
        start=0.0,
        stop=80.0,
        step=1.0,
        value=20.0,
        label="Coupled roughness",
        full_width=True,
    )
    n_samples = mo.ui.slider(
        start=120,
        stop=720,
        step=24,
        value=360,
        label="Samples",
        full_width=True,
    )
    train_fraction = mo.ui.slider(
        start=0.50,
        stop=0.85,
        step=0.05,
        value=0.70,
        label="Train fraction",
        full_width=True,
    )
    noise_scale = mo.ui.slider(
        start=0.00,
        stop=0.60,
        step=0.02,
        value=0.12,
        label="Noise scale",
        full_width=True,
    )
    random_seed = mo.ui.number(
        start=0,
        stop=10_000,
        step=1,
        value=11,
        label="Seed",
        full_width=True,
    )
    controls = mo.vstack(
        [
            section_intro,
            mo.hstack([horizon_count, profile_name, roughness_weight]),
            mo.hstack([n_samples, train_fraction, noise_scale, random_seed]),
        ]
    )
    controls
    return (
        horizon_count,
        n_samples,
        noise_scale,
        profile_name,
        random_seed,
        roughness_weight,
        train_fraction,
    )


@app.cell
def _(horizon_count, mo):
    focus_horizon = mo.ui.slider(
        start=1,
        stop=horizon_count.value,
        step=1,
        value=min(2, horizon_count.value),
        label="Displayed horizon",
        full_width=True,
    )
    focus_horizon
    return (focus_horizon,)


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
    def horizon_profile(name: str, horizon: int) -> np.ndarray:
        base_profiles = {
            "smooth decay": np.array([2.8, 2.2, 1.6, 1.1, 0.7, 0.4]),
            "jagged response": np.array([1.0, 4.2, -0.8, 3.3, 0.3, 2.1]),
            "delayed peak": np.array([0.2, 0.8, 2.5, 3.2, 1.7, 0.6]),
        }
        return base_profiles[name][:horizon]

    def make_forecast_problem(
        samples: int,
        horizon: int,
        profile: str,
        noise: float,
        seed: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range("2024-01-01", periods=samples, freq="1h")
        sample_ix = np.arange(samples, dtype=float)
        weather_signal = rng.normal(0.0, 1.0, size=samples)
        daily_pattern = (
            0.8 * np.sin(2.0 * np.pi * sample_ix / 24.0)
            + 0.25 * np.cos(2.0 * np.pi * sample_ix / 12.0)
        )
        demand = daily_pattern.copy()
        coefficients = horizon_profile(profile, horizon)
        component_columns = {}
        for step, coefficient in enumerate(coefficients, start=1):
            contribution = np.zeros(samples)
            contribution[step:] = coefficient * weather_signal[:-step]
            component_columns[f"weather_effect_horizon_{step}"] = contribution
            demand += contribution
        noise_values = rng.normal(0.0, noise, size=samples)
        demand_without_noise = demand.copy()
        demand += noise_values
        frame = pd.DataFrame(
            {
                "weather_signal": weather_signal,
                "demand": demand,
                "daily_pattern": daily_pattern,
                "demand_without_noise": demand_without_noise,
                "noise": noise_values,
                **component_columns,
            },
            index=timestamps,
        )
        return frame, coefficients

    def base_forecast_config() -> TsgamEstimatorConfig:
        return TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[2],
                periods=[24],
                reg_weight=1.0e-5,
            ),
            exog_config=[TsgamLinearConfig(lags=[0], reg_weight=1.0e-7)],
            solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
        )

    def fit_forecast_models(
        frame: pd.DataFrame,
        horizon: int,
        train_fraction_value: float,
        roughness: float,
    ) -> dict[str, object]:
        train_stop = int(len(frame) * train_fraction_value)
        train_stop = max(train_stop, horizon + 24)
        train_stop = min(train_stop, len(frame) - horizon - 1)
        x_train = frame[["weather_signal"]].iloc[:train_stop]
        y_train = frame["demand"].to_numpy()[:train_stop]
        x_eval = frame[["weather_signal"]].iloc[train_stop:-horizon]
        independent = TsgamForecastEstimator(
            TsgamForecastConfig(
                horizon=horizon,
                base_config=base_forecast_config(),
                mode="independent",
            )
        ).fit(x_train, y_train)
        coupled = TsgamForecastEstimator(
            TsgamForecastConfig(
                horizon=horizon,
                base_config=base_forecast_config(),
                mode="coupled",
                coupling_config=TsgamForecastCouplingConfig(
                    roughness_weight=roughness
                ),
            )
        ).fit(x_train, y_train)
        independent_pred = independent.predict(x_eval)
        coupled_pred = coupled.predict(x_eval)
        actual = pd.DataFrame(
            {
                f"horizon_{step}": frame["demand"].shift(-step).loc[x_eval.index]
                for step in range(1, horizon + 1)
            },
            index=x_eval.index,
        )
        return {
            "actual": actual,
            "coupled": coupled,
            "coupled_pred": coupled_pred,
            "independent": independent,
            "independent_pred": independent_pred,
            "train_stop": train_stop,
            "x_eval": x_eval,
        }

    def split_summary_frame(
        frame: pd.DataFrame,
        horizon: int,
        train_fraction_value: float,
    ) -> pd.DataFrame:
        train_stop = int(len(frame) * train_fraction_value)
        train_stop = max(train_stop, horizon + 24)
        train_stop = min(train_stop, len(frame) - horizon - 1)
        eval_start = train_stop
        eval_stop = len(frame) - horizon
        rows = [
            {
                "piece": "training source block",
                "rows": train_stop,
                "first timestamp": frame.index[0],
                "last timestamp": frame.index[train_stop - 1],
                "why it matters": "candidate rows made available before each horizon drops its own unobservable tail",
            },
            {
                "piece": "shortest training table",
                "rows": train_stop - horizon,
                "first timestamp": frame.index[0],
                "last timestamp": frame.index[train_stop - horizon - 1],
                "why it matters": f"horizon {horizon} fits X[:-{horizon}] against y[{horizon}:]",
            },
            {
                "piece": "evaluation origins",
                "rows": eval_stop - eval_start,
                "first timestamp": frame.index[eval_start],
                "last timestamp": frame.index[eval_stop - 1],
                "why it matters": "origins where predictions are scored",
            },
            {
                "piece": "evaluation targets",
                "rows": eval_stop - eval_start,
                "first timestamp": frame.index[eval_start + 1],
                "last timestamp": frame.index[eval_stop - 1 + horizon],
                "why it matters": "future target values used only for scoring",
            },
            {
                "piece": "tail held out",
                "rows": horizon,
                "first timestamp": frame.index[eval_stop],
                "last timestamp": frame.index[-1],
                "why it matters": "not used as origins because full future horizons are unavailable",
            },
        ]
        return pd.DataFrame(rows)

    def horizon_training_rows_frame(
        frame: pd.DataFrame,
        horizon: int,
        train_fraction_value: float,
    ) -> pd.DataFrame:
        train_stop = int(len(frame) * train_fraction_value)
        train_stop = max(train_stop, horizon + 24)
        train_stop = min(train_stop, len(frame) - horizon - 1)
        rows = []
        for step in range(1, horizon + 1):
            rows.append(
                {
                    "horizon": step,
                    "training rows": train_stop - step,
                    "origin range": f"{frame.index[0]} to {frame.index[train_stop - step - 1]}",
                    "target range": f"{frame.index[step]} to {frame.index[train_stop - 1]}",
                    "operation": f"fit X[:-{step}] against y[{step}:]",
                }
            )
        return pd.DataFrame(rows)

    def synthetic_component_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
        component_columns = ["daily_pattern"] + [
            f"weather_effect_horizon_{step}" for step in range(1, horizon + 1)
        ] + ["noise"]
        labels = {
            "daily_pattern": "daily pattern",
            "noise": "noise",
            **{
                f"weather_effect_horizon_{step}": f"weather effect h={step}"
                for step in range(1, horizon + 1)
            },
        }
        return (
            frame[component_columns]
            .rename(columns=labels)
            .reset_index(names="timestamp")
            .melt(
                id_vars=["timestamp"],
                var_name="component",
                value_name="value",
            )
        )

    def synthetic_total_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame[["demand", "demand_without_noise", "daily_pattern"]]
            .rename(
                columns={
                    "demand": "observed demand",
                    "demand_without_noise": "signal before noise",
                    "daily_pattern": "daily pattern only",
                }
            )
            .reset_index(names="timestamp")
            .melt(
                id_vars=["timestamp"],
                var_name="series",
                value_name="value",
            )
        )

    def metric_frame(results: dict[str, object]) -> pd.DataFrame:
        actual = results["actual"]
        rows = []
        for model_name, pred_key in [
            ("independent", "independent_pred"),
            ("coupled", "coupled_pred"),
        ]:
            predictions = results[pred_key]
            for column in actual.columns:
                error = predictions[column] - actual[column]
                rows.append(
                    {
                        "model": model_name,
                        "horizon": int(column.split("_")[1]),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "mae": float(np.mean(np.abs(error))),
                    }
                )
        return pd.DataFrame(rows)

    def coefficient_frame(
        results: dict[str, object],
        true_coefficients: np.ndarray,
    ) -> pd.DataFrame:
        independent = results["independent"]
        coupled = results["coupled"]
        rows = []
        for horizon, true_coefficient in enumerate(true_coefficients, start=1):
            child = independent.forecast_estimators_[horizon]
            rows.append(
                {
                    "model": "true",
                    "horizon": horizon,
                    "coefficient": float(true_coefficient),
                }
            )
            rows.append(
                {
                    "model": "independent",
                    "horizon": horizon,
                    "coefficient": float(child.variables_["exog_coef_0"].value[0, 0]),
                }
            )
            rows.append(
                {
                    "model": "coupled",
                    "horizon": horizon,
                    "coefficient": float(
                        coupled.variables_["exog_coef_0"][horizon - 1].value[0, 0]
                    ),
                }
            )
        return pd.DataFrame(rows)

    def prediction_frame(
        results: dict[str, object],
        displayed_horizon: int,
    ) -> pd.DataFrame:
        column = f"horizon_{displayed_horizon}"
        actual = results["actual"][[column]].rename(columns={column: "value"})
        independent = results["independent_pred"][[column]].rename(
            columns={column: "value"}
        )
        coupled = results["coupled_pred"][[column]].rename(columns={column: "value"})
        actual = actual.assign(series="actual")
        independent = independent.assign(series="independent")
        coupled = coupled.assign(series="coupled")
        return pd.concat([actual, independent, coupled]).reset_index(
            names="timestamp"
        )

    def forecast_path_frame(
        results: dict[str, object],
        origin_offset: int,
    ) -> pd.DataFrame:
        actual = results["actual"]
        independent_pred = results["independent_pred"]
        coupled_pred = results["coupled_pred"]
        origin_position = min(origin_offset, len(actual.index) - 1)
        origin_time = actual.index[origin_position]
        origin_step = (
            actual.index[1] - actual.index[0]
            if len(actual.index) > 1
            else pd.Timedelta(hours=1)
        )
        rows = []
        for horizon in range(1, len(actual.columns) + 1):
            column = f"horizon_{horizon}"
            target_time = origin_time + horizon * origin_step
            rows.extend(
                [
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "horizon": horizon,
                        "series": "actual future demand",
                        "value": float(actual.iloc[origin_position][column]),
                    },
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "horizon": horizon,
                        "series": "independent forecast",
                        "value": float(
                            independent_pred.iloc[origin_position][column]
                        ),
                    },
                    {
                        "origin_time": origin_time,
                        "target_time": target_time,
                        "horizon": horizon,
                        "series": "coupled forecast",
                        "value": float(coupled_pred.iloc[origin_position][column]),
                    },
                ]
            )
        return pd.DataFrame(rows)

    def forecast_error_frame(results: dict[str, object]) -> pd.DataFrame:
        actual = results["actual"]
        origin_step = (
            actual.index[1] - actual.index[0]
            if len(actual.index) > 1
            else pd.Timedelta(hours=1)
        )
        rows = []
        for model_name, pred_key in [
            ("independent", "independent_pred"),
            ("coupled", "coupled_pred"),
        ]:
            predictions = results[pred_key]
            for origin_time in actual.index:
                for column in actual.columns:
                    horizon = int(column.split("_")[1])
                    error = float(
                        predictions.loc[origin_time, column]
                        - actual.loc[origin_time, column]
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "origin_time": origin_time,
                            "target_time": origin_time + horizon * origin_step,
                            "horizon": horizon,
                            "error": error,
                            "absolute_error": abs(error),
                        }
                    )
        return pd.DataFrame(rows)

    def actual_vs_predicted_frame(
        results: dict[str, object],
        displayed_horizon: int,
    ) -> pd.DataFrame:
        column = f"horizon_{displayed_horizon}"
        actual = results["actual"][column]
        rows = []
        for model_name, pred_key in [
            ("independent", "independent_pred"),
            ("coupled", "coupled_pred"),
        ]:
            predictions = results[pred_key][column]
            for origin_time in actual.index:
                rows.append(
                    {
                        "model": model_name,
                        "origin_time": origin_time,
                        "actual": float(actual.loc[origin_time]),
                        "predicted": float(predictions.loc[origin_time]),
                    }
                )
        return pd.DataFrame(rows)

    def estimator_mechanics_frame(horizon: int, roughness: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "stage": "base regression config",
                    "what forecast mode does": "reuse a normal TsgamEstimatorConfig with daily Fourier terms and weather_signal lag=0",
                    "result": "one regression recipe shared by every horizon",
                },
                {
                    "stage": "horizon alignment",
                    "what forecast mode does": "for each h, align origin weather_signal rows with demand h steps later",
                    "result": "direct supervised problem for each horizon",
                },
                {
                    "stage": "independent mode",
                    "what forecast mode does": f"fit {horizon} separate TsgamEstimator models",
                    "result": f"{horizon} child estimators, one per horizon",
                },
                {
                    "stage": "coupled mode",
                    "what forecast mode does": f"solve all {horizon} horizons in one CVXPY problem with roughness weight {roughness:g}",
                    "result": "one model whose horizon coefficients are smoothed together",
                },
                {
                    "stage": "predict",
                    "what forecast mode does": "return one row per forecast origin and one column per horizon",
                    "result": "DataFrame columns horizon_1 ... horizon_T",
                },
            ]
        )

    def forecast_shape_frame(results: dict[str, object]) -> pd.DataFrame:
        actual = results["actual"]
        independent_pred = results["independent_pred"]
        coupled_pred = results["coupled_pred"]
        return pd.DataFrame(
            [
                {
                    "object": "actual future targets",
                    "rows": actual.shape[0],
                    "columns": ", ".join(actual.columns),
                    "index meaning": "forecast origin timestamp",
                },
                {
                    "object": "independent predictions",
                    "rows": independent_pred.shape[0],
                    "columns": ", ".join(independent_pred.columns),
                    "index meaning": "forecast origin timestamp",
                },
                {
                    "object": "coupled predictions",
                    "rows": coupled_pred.shape[0],
                    "columns": ", ".join(coupled_pred.columns),
                    "index meaning": "forecast origin timestamp",
                },
            ]
        )

    return (
        actual_vs_predicted_frame,
        coefficient_frame,
        estimator_mechanics_frame,
        fit_forecast_models,
        forecast_error_frame,
        forecast_path_frame,
        forecast_shape_frame,
        horizon_training_rows_frame,
        make_forecast_problem,
        metric_frame,
        prediction_frame,
        split_summary_frame,
        synthetic_component_frame,
        synthetic_total_frame,
    )


@app.cell
def _(
    horizon_count,
    make_forecast_problem,
    n_samples,
    noise_scale,
    profile_name,
    random_seed,
):
    problem_frame, true_coefficients = make_forecast_problem(
        samples=n_samples.value,
        horizon=horizon_count.value,
        profile=profile_name.value,
        noise=noise_scale.value,
        seed=random_seed.value,
    )
    return problem_frame, true_coefficients


@app.cell
def _(horizon_count, mo, pd, profile_name, true_coefficients):
    truth_rows = pd.DataFrame(
        [
            {
                "forecast horizon": horizon,
                "true weather effect": round(float(coefficient), 3),
                "meaning": f"demand at origin + {horizon} uses weather_signal at origin",
            }
            for horizon, coefficient in enumerate(true_coefficients, start=1)
        ]
    )
    problem_statement = mo.vstack(
        [
            mo.md(
                "## 2. The generated forecasting problem\n\n"
                "The demand series is generated as\n\n"
                "`demand[t] = daily_pattern[t] + sum(beta_h * weather_signal[t-h]) + noise[t]`.\n\n"
                "At forecast origin `o`, horizon `h` predicts `demand[o+h]` "
                "using `weather_signal[o]`. Because `demand[o+h]` contains "
                "`beta_h * weather_signal[o]`, a good horizon-specific forecast "
                "model should recover the corresponding `beta_h`."
            ),
            mo.md(
                f"Selected response profile: `{profile_name.value}` with "
                f"{horizon_count.value} forecast horizons."
            ),
            mo.ui.table(truth_rows, pagination=False),
        ]
    )
    problem_statement
    return (truth_rows,)


@app.cell
def _(horizon_count, problem_frame, synthetic_component_frame, synthetic_total_frame):
    synthetic_components = synthetic_component_frame(
        problem_frame,
        horizon=horizon_count.value,
    )
    synthetic_totals = synthetic_total_frame(problem_frame)
    return synthetic_components, synthetic_totals


@app.cell
def _(alt, mo, synthetic_components, synthetic_totals):
    component_chart = (
        alt.Chart(synthetic_components)
        .mark_line()
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
        .properties(height=70, title="Hidden additive components of demand")
        .resolve_scale(y="independent")
    )
    total_chart = (
        alt.Chart(synthetic_totals)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title="demand"),
            color=alt.Color("series:N", title=None),
            strokeDash=alt.StrokeDash("series:N", title=None),
            tooltip=[
                alt.Tooltip("timestamp:T", title="time"),
                alt.Tooltip("series:N", title="series"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(height=260, title="How those components add up to demand")
    )
    component_view = mo.vstack(
        [
            mo.md(
                "The model only sees `weather_signal` at each forecast origin "
                "and the demand history used for training. These charts expose "
                "the synthetic truth: daily seasonality, horizon-specific "
                "weather effects, and noise add together to create the demand "
                "series being forecast."
            ),
            component_chart,
            total_chart,
        ]
    )
    component_view
    return


@app.cell
def _(
    horizon_count,
    horizon_training_rows_frame,
    mo,
    problem_frame,
    split_summary_frame,
    train_fraction,
):
    split_summary = split_summary_frame(
        problem_frame,
        horizon=horizon_count.value,
        train_fraction_value=train_fraction.value,
    )
    horizon_rows = horizon_training_rows_frame(
        problem_frame,
        horizon=horizon_count.value,
        train_fraction_value=train_fraction.value,
    )
    split_view = mo.vstack(
        [
            mo.md(
                "## 3. What the train/evaluation split looks like\n\n"
                "Training uses only early forecast origins. Each horizon then "
                "drops a different tail from that source block because the "
                "target `y[t + h]` must also be inside the training target "
                "vector. Evaluation uses later origins shared by every horizon, "
                "and the final `horizon` rows are not used as origins because "
                "their full future demand window is unavailable.\n\n"
                "Each horizon has its own supervised training table: horizon 1 "
                "fits `X[:-1] -> y[1:]`, horizon 2 fits `X[:-2] -> y[2:]`, "
                "and so on."
            ),
            mo.ui.table(split_summary, pagination=False),
            mo.md("### Per-horizon supervised training rows"),
            mo.ui.table(horizon_rows, pagination=False),
        ]
    )
    split_view
    return horizon_rows, split_summary


@app.cell
def _(horizon_count, mo, pd, problem_frame):
    origin_position = min(24, len(problem_frame) - horizon_count.value - 1)
    origin_time = problem_frame.index[origin_position]
    alignment_rows = []
    for horizon in range(1, horizon_count.value + 1):
        target_time = problem_frame.index[origin_position + horizon]
        alignment_rows.append(
            {
                "forecast_origin": origin_time,
                "horizon": horizon,
                "uses_weather_signal_at": origin_time,
                "predicts_demand_at": target_time,
                "demand_value": round(float(problem_frame.loc[target_time, "demand"]), 3),
            }
        )
    alignment_frame = pd.DataFrame(alignment_rows)
    alignment_view = mo.vstack(
        [
            mo.md(
                "## 4. One forecast origin expands into several targets\n\n"
                "For each horizon, the model uses information available at the "
                "forecast origin and learns the demand value `h` steps later. "
                "The fitted regression row is indexed at the target time so "
                "periodic terms describe the time being predicted."
            ),
            mo.ui.table(alignment_frame, pagination=False),
        ]
    )
    alignment_view
    return


@app.cell
def _(alt, horizon_count, problem_frame, pd, train_fraction):
    train_stop = int(len(problem_frame) * train_fraction.value)
    train_stop = max(train_stop, horizon_count.value + 24)
    train_stop = min(train_stop, len(problem_frame) - horizon_count.value - 1)
    eval_stop = len(problem_frame) - horizon_count.value
    observed_frame = (
        problem_frame[["weather_signal", "demand"]]
        .reset_index(names="timestamp")
        .assign(
            split=lambda df: pd.Series(
                [
                    "train source"
                    if row < train_stop
                    else (
                        "scored evaluation"
                        if row < eval_stop
                        else "not an origin"
                    )
                    for row in range(len(df))
                ]
            )
        )
        .melt(
            id_vars=["timestamp", "split"],
            value_vars=["weather_signal", "demand"],
            var_name="series",
            value_name="value",
        )
    )
    signal_chart = (
        alt.Chart(observed_frame)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("split:N", title=None),
            row=alt.Row("series:N", title=None),
        )
        .properties(
            height=120,
            title="Observed weather signal and demand, colored by forecast split",
        )
        .resolve_scale(y="independent")
    )
    signal_chart
    return


@app.cell
def _(
    fit_forecast_models,
    horizon_count,
    problem_frame,
    roughness_weight,
    train_fraction,
):
    # Independent mode fits T separate direct regressions; coupled mode solves one
    # joint problem with a roughness penalty across horizon-specific coefficients.
    forecast_results = fit_forecast_models(
        frame=problem_frame,
        horizon=horizon_count.value,
        train_fraction_value=train_fraction.value,
        roughness=roughness_weight.value,
    )
    return (forecast_results,)


@app.cell
def _(forecast_results, forecast_shape_frame, mo):
    shape_summary = forecast_shape_frame(forecast_results)
    output_shape_view = mo.vstack(
        [
            mo.md(
                "## 5. What `predict` returns\n\n"
                "Forecast mode returns one row per forecast origin. Each column "
                "is a forecast horizon. The actual future-target table is built "
                "with the same shape so the two prediction modes can be scored "
                "horizon by horizon."
            ),
            mo.ui.table(shape_summary, pagination=False),
        ]
    )
    output_shape_view
    return (shape_summary,)


@app.cell
def _(forecast_results, mo):
    max_origin_offset = min(72, len(forecast_results["x_eval"]) - 1)
    inspected_origin = mo.ui.slider(
        start=0,
        stop=max_origin_offset,
        step=1,
        value=0,
        label="Evaluation origin to inspect",
        full_width=True,
    )
    inspected_origin
    return (inspected_origin,)


@app.cell
def _(forecast_path_frame, forecast_results, inspected_origin):
    displayed_forecast_path = forecast_path_frame(
        forecast_results,
        origin_offset=inspected_origin.value,
    )
    return (displayed_forecast_path,)


@app.cell
def _(alt, displayed_forecast_path, mo):
    _forecast_origin_time = displayed_forecast_path["origin_time"].iloc[0]
    forecast_path_chart = (
        alt.Chart(displayed_forecast_path)
        .mark_line(point=True)
        .encode(
            x=alt.X("target_time:T", title="target time"),
            y=alt.Y("value:Q", title="demand"),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("target_time:T", title="target"),
                alt.Tooltip("horizon:O", title="horizon"),
                alt.Tooltip("series:N", title="series"),
                alt.Tooltip("value:Q", title="value", format=".3f"),
            ],
        )
        .properties(
            height=300,
            title=f"One forecast origin expands into a {len(displayed_forecast_path) // 3}-step forecast path",
        )
    )
    path_view = mo.vstack(
        [
            mo.md(
                f"Forecast origin `{_forecast_origin_time}` produces one value for each "
                "future target time. This is the shape users consume in "
                "`predict`: the row is the origin, while each horizon column is "
                "a different future target."
            ),
            forecast_path_chart,
        ]
    )
    path_view
    return


@app.cell
def _(coefficient_frame, forecast_results, true_coefficients):
    forecast_coefficients = coefficient_frame(forecast_results, true_coefficients)
    return (forecast_coefficients,)


@app.cell
def _(metric_frame, forecast_results):
    forecast_metrics = metric_frame(forecast_results)
    return (forecast_metrics,)


@app.cell
def _(forecast_error_frame, forecast_results):
    forecast_errors = forecast_error_frame(forecast_results)
    return (forecast_errors,)


@app.cell
def _(estimator_mechanics_frame, horizon_count, mo, roughness_weight):
    model_rows = estimator_mechanics_frame(
        horizon=horizon_count.value,
        roughness=roughness_weight.value,
    )
    model_walkthrough = mo.vstack(
        [
            mo.md(
                "## 6. Fit the new forecast estimator two ways\n\n"
                "Forecast mode is direct, not recursive: it does not predict "
                "horizon 1 and feed that prediction into horizon 2. It builds "
                "a separate supervised target for each horizon.\n\n"
                "Independent mode is equivalent to:\n\n"
                "```python\n"
                "for h in range(1, T + 1):\n"
                "    X_h = X_train.iloc[:-h]\n"
                "    X_h.index = X_h.index + h * freq  # target-time features\n"
                "    y_h = y_train[h:]\n"
                "    child_model[h].fit(X_h, y_h)\n"
                "```\n\n"
                "Prediction performs the same target-time shift internally, "
                "then returns a DataFrame indexed by the original forecast "
                "origins. Coupled mode builds those same `X_h, y_h` designs, "
                "then solves one joint optimization problem with a roughness "
                "penalty across the horizon-specific coefficients. Increase "
                "the coupling weight only when a smooth horizon response is a "
                "reasonable modeling assumption."
            ),
            mo.ui.table(model_rows, pagination=False),
        ]
    )
    model_walkthrough
    return


@app.cell
def _(alt, forecast_coefficients):
    # This chart is the most direct view of the new coupling behavior.
    coefficient_chart = (
        alt.Chart(forecast_coefficients)
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon:O", title="horizon"),
            y=alt.Y("coefficient:Q", title="weather signal coefficient"),
            color=alt.Color("model:N", title=None),
        )
        .properties(
            height=260,
            title="Can each model recover the horizon-specific weather effect?",
        )
    )
    coefficient_chart
    return


@app.cell
def _(forecast_coefficients, mo, np, pd):
    roughness_rows = []
    for model_name, group in forecast_coefficients.groupby("model"):
        ordered = group.sort_values("horizon")
        coefficients = ordered["coefficient"].to_numpy()
        if len(coefficients) >= 3:
            roughness = float(np.sum(np.diff(coefficients, n=2) ** 2))
        else:
            roughness = float(np.sum(np.diff(coefficients) ** 2))
        roughness_rows.append(
            {
                "model": model_name,
                "horizon_roughness": round(roughness, 4),
            }
        )
    roughness_frame = pd.DataFrame(roughness_rows)
    roughness_view = mo.vstack(
        [
            mo.md(
                "The roughness score summarizes how much the learned weather "
                "effect bends across horizons. Coupling adds this idea directly "
                "to the optimization problem. That can reduce variance when the "
                "true response is smooth, but it can bias a deliberately jagged "
                "response; the coefficient and error charts below show that "
                "tradeoff rather than assuming coupling is always better."
            ),
            mo.ui.table(roughness_frame, pagination=False),
        ]
    )
    roughness_view
    return (roughness_frame,)


@app.cell
def _(alt, forecast_metrics, mo):
    metric_long = forecast_metrics.melt(
        id_vars=["model", "horizon"],
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    metric_chart = (
        alt.Chart(metric_long)
        .mark_line(point=True)
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
        .properties(
            height=280,
            title="Held-out forecast error by model and horizon",
        )
    )
    rmse_comparison = (
        forecast_metrics.pivot(index="horizon", columns="model", values="rmse")
        .reset_index()
        .rename_axis(columns=None)
    )
    rmse_comparison["coupled minus independent RMSE"] = (
        rmse_comparison["coupled"] - rmse_comparison["independent"]
    )
    rmse_comparison["lower RMSE model"] = rmse_comparison.apply(
        lambda row: "coupled"
        if row["coupled"] < row["independent"]
        else "independent",
        axis=1,
    )
    metric_view = mo.vstack(
        [
            mo.md(
                "Held-out error is computed against the future target table "
                "with the same origin-by-horizon shape as `predict`. The line "
                "chart shows whether errors grow with horizon. The table makes "
                "the coupling tradeoff explicit: negative delta means coupling "
                "reduced RMSE for that horizon."
            ),
            metric_chart,
            mo.ui.table(rmse_comparison.round(4), pagination=False),
        ]
    )
    metric_view
    return


@app.cell
def _(alt, forecast_errors):
    error_heatmap = (
        alt.Chart(forecast_errors)
        .mark_rect()
        .encode(
            x=alt.X("origin_time:T", title="forecast origin"),
            y=alt.Y("horizon:O", title="horizon"),
            color=alt.Color(
                "error:Q",
                title="forecast error",
                scale=alt.Scale(scheme="redblue", reverse=True),
            ),
            column=alt.Column("model:N", title=None),
            tooltip=[
                alt.Tooltip("model:N", title="model"),
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("target_time:T", title="target"),
                alt.Tooltip("horizon:O", title="horizon"),
                alt.Tooltip("error:Q", title="error", format=".3f"),
                alt.Tooltip("absolute_error:Q", title="absolute error", format=".3f"),
            ],
        )
        .properties(
            width=360,
            height=180,
            title="Where each model misses across evaluation time",
        )
    )
    error_heatmap
    return


@app.cell
def _(focus_horizon, forecast_results, prediction_frame):
    displayed_predictions = prediction_frame(
        forecast_results,
        displayed_horizon=focus_horizon.value,
    )
    return (displayed_predictions,)


@app.cell
def _(alt, displayed_predictions):
    prediction_chart = (
        alt.Chart(displayed_predictions)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value:Q", title="demand"),
            color=alt.Color("series:N", title=None),
        )
        .properties(
            height=300,
            title="Actual vs predicted future demand for the selected horizon",
        )
    )
    prediction_chart
    return


@app.cell
def _(actual_vs_predicted_frame, focus_horizon, forecast_results):
    actual_predicted_points = actual_vs_predicted_frame(
        forecast_results,
        displayed_horizon=focus_horizon.value,
    )
    return (actual_predicted_points,)


@app.cell
def _(actual_predicted_points, alt, focus_horizon, pd):
    diagonal_min = min(
        actual_predicted_points["actual"].min(),
        actual_predicted_points["predicted"].min(),
    )
    diagonal_max = max(
        actual_predicted_points["actual"].max(),
        actual_predicted_points["predicted"].max(),
    )
    diagonal_frame = pd.DataFrame(
        {
            "actual": [diagonal_min, diagonal_max],
            "predicted": [diagonal_min, diagonal_max],
        }
    )
    calibration_points = (
        alt.Chart(actual_predicted_points)
        .mark_circle(size=42, opacity=0.55)
        .encode(
            x=alt.X("actual:Q", title="actual future demand"),
            y=alt.Y("predicted:Q", title="predicted demand"),
            color=alt.Color("model:N", title=None),
            tooltip=[
                alt.Tooltip("model:N", title="model"),
                alt.Tooltip("origin_time:T", title="origin"),
                alt.Tooltip("actual:Q", title="actual", format=".3f"),
                alt.Tooltip("predicted:Q", title="predicted", format=".3f"),
            ],
        )
    )
    calibration_line = (
        alt.Chart(diagonal_frame)
        .mark_line(color="#525252", strokeDash=[4, 4])
        .encode(x="actual:Q", y="predicted:Q")
    )
    calibration_chart = (calibration_points + calibration_line).properties(
        height=320,
        title=f"Actual vs predicted demand for horizon {focus_horizon.value}",
    )
    calibration_chart
    return


@app.cell
def _(forecast_coefficients, forecast_metrics, mo, roughness_frame, shape_summary):
    summary_table = mo.vstack(
        [
            mo.md(
                "## 7. Inspect the fitted forecast objects\n\n"
                "The tables expose the same information as the charts: fitted "
                "weather coefficients by horizon, forecast error by horizon, and "
                "the roughness diagnostic used to understand coupling."
            ),
            mo.ui.table(shape_summary, pagination=False),
            mo.ui.table(forecast_coefficients, page_size=18),
            mo.ui.table(forecast_metrics, page_size=12),
            mo.ui.table(roughness_frame, pagination=False),
        ]
    )
    summary_table
    return


if __name__ == "__main__":
    app.run()
