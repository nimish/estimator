#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive marimo notebook for an idealized synthetic TSGAM problem.
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        "# Synthetic TSGAM Explorer\n\n"
        "Configure an idealized synthetic problem with known periodic "
        "structure, linear and nonlinear regressors, and optional trend. "
        "Then fit TSGAM and compare the recovered model against the ground "
        "truth."
    )
    return


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import sys

    _project_root = Path(__file__).resolve().parent.parent
    _src_dir = _project_root / "src"
    _examples_dir = _project_root / "examples"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    if str(_examples_dir) not in sys.path:
        sys.path.insert(0, str(_examples_dir))

    from synthetic_problem import (
        SyntheticPeriodicComponent,
        SyntheticProblemConfig,
        SyntheticRegressorRelationship,
        SyntheticRegressorSpec,
        SyntheticTrendKind,
        SyntheticTrendSpec,
        build_estimator_config,
        generate_synthetic_problem,
        split_problem_frames,
        synthetic_metrics,
    )
    from tsgam_estimator import TsgamEstimator

    return (
        SyntheticPeriodicComponent,
        SyntheticProblemConfig,
        SyntheticRegressorRelationship,
        SyntheticRegressorSpec,
        SyntheticTrendKind,
        SyntheticTrendSpec,
        TsgamEstimator,
        alt,
        build_estimator_config,
        generate_synthetic_problem,
        mo,
        np,
        pd,
        split_problem_frames,
        synthetic_metrics,
    )


@app.cell
def _():
    FREQ_OPTIONS = ["15min", "30min", "1h", "2h"]
    SOLVER_OPTIONS = ["CLARABEL", "SCS"]
    REG_WEIGHT_OPTIONS = {
        "lighter (1e-6)": 1.0e-6,
        "shared (1e-5)": 1.0e-5,
        "stronger (1e-4)": 1.0e-4,
    }
    return FREQ_OPTIONS, REG_WEIGHT_OPTIONS, SOLVER_OPTIONS


@app.cell
def _(FREQ_OPTIONS, REG_WEIGHT_OPTIONS, SOLVER_OPTIONS, mo):
    start_date = mo.ui.date(value="2024-01-01", label="Start date")
    n_samples = mo.ui.number(
        start=24,
        stop=24 * 120,
        step=24,
        value=24 * 30,
        label="Samples",
        full_width=True,
    )
    freq = mo.ui.dropdown(options=FREQ_OPTIONS, value="1h", label="Frequency")
    train_fraction = mo.ui.number(
        start=0.50,
        stop=0.90,
        step=0.05,
        value=0.75,
        label="Train fraction",
        full_width=True,
    )
    seed = mo.ui.number(
        start=0,
        stop=10_000,
        step=1,
        value=7,
        label="Seed",
        full_width=True,
    )
    noise_scale = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.08,
        label="Noise scale",
        full_width=True,
    )
    solver_name = mo.ui.dropdown(
        options=SOLVER_OPTIONS,
        value="CLARABEL",
        label="Solver",
    )
    fourier_reg = mo.ui.dropdown(
        options=list(REG_WEIGHT_OPTIONS),
        value="shared (1e-5)",
        label="Fourier regularization",
    )
    linear_reg = mo.ui.dropdown(
        options=list(REG_WEIGHT_OPTIONS),
        value="shared (1e-5)",
        label="Linear regularization",
    )
    spline_reg = mo.ui.dropdown(
        options=list(REG_WEIGHT_OPTIONS),
        value="shared (1e-5)",
        label="Spline regularization",
    )
    spline_diff_reg = mo.ui.number(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=0.6,
        label="Spline lag smoothing",
        full_width=True,
    )
    trend_reg = mo.ui.number(
        start=0.1,
        stop=50.0,
        step=0.5,
        value=10.0,
        label="Trend regularization",
        full_width=True,
    )
    run_model = mo.ui.run_button(label="Run model")
    return (
        fourier_reg,
        freq,
        linear_reg,
        n_samples,
        noise_scale,
        run_model,
        seed,
        solver_name,
        spline_diff_reg,
        spline_reg,
        start_date,
        train_fraction,
        trend_reg,
    )


@app.cell
def _(mo):
    daily_on = mo.ui.switch(label="Daily", value=True)
    daily_period = mo.ui.number(
        start=2.0,
        stop=96.0,
        step=1.0,
        value=24.0,
        label="Period (hours)",
        full_width=True,
    )
    daily_harmonics = mo.ui.number(
        start=1,
        stop=8,
        step=1,
        value=2,
        label="Harmonics",
        full_width=True,
    )
    daily_amplitude = mo.ui.number(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=1.2,
        label="Amplitude",
        full_width=True,
    )

    weekly_on = mo.ui.switch(label="Weekly", value=True)
    weekly_period = mo.ui.number(
        start=8.0,
        stop=24.0 * 14.0,
        step=1.0,
        value=24.0 * 7.0,
        label="Period (hours)",
        full_width=True,
    )
    weekly_harmonics = mo.ui.number(
        start=1,
        stop=6,
        step=1,
        value=1,
        label="Harmonics",
        full_width=True,
    )
    weekly_amplitude = mo.ui.number(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=0.5,
        label="Amplitude",
        full_width=True,
    )

    custom_on = mo.ui.switch(label="Custom", value=False)
    custom_period = mo.ui.number(
        start=2.0,
        stop=24.0 * 30.0,
        step=1.0,
        value=12.0,
        label="Period (hours)",
        full_width=True,
    )
    custom_harmonics = mo.ui.number(
        start=1,
        stop=8,
        step=1,
        value=1,
        label="Harmonics",
        full_width=True,
    )
    custom_amplitude = mo.ui.number(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=0.4,
        label="Amplitude",
        full_width=True,
    )
    return (
        custom_amplitude,
        custom_harmonics,
        custom_on,
        custom_period,
        daily_amplitude,
        daily_harmonics,
        daily_on,
        daily_period,
        weekly_amplitude,
        weekly_harmonics,
        weekly_on,
        weekly_period,
    )


@app.cell
def _(SyntheticRegressorRelationship, mo):
    relationship_options = [item.value for item in SyntheticRegressorRelationship]

    temp_on = mo.ui.switch(label="temp", value=True)
    temp_relationship = mo.ui.dropdown(
        options=relationship_options,
        value="linear",
        label="Relationship",
    )
    temp_effect = mo.ui.number(
        start=0.1,
        stop=2.5,
        step=0.1,
        value=0.9,
        label="Effect scale",
        full_width=True,
    )
    temp_driver_period = mo.ui.number(
        start=2.0,
        stop=24.0 * 14.0,
        step=1.0,
        value=24.0,
        label="Driver period (hours)",
        full_width=True,
    )
    temp_driver_noise = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.20,
        label="Driver noise",
        full_width=True,
    )
    temp_knots = mo.ui.number(
        start=4,
        stop=12,
        step=1,
        value=7,
        label="Spline knots",
        full_width=True,
    )

    wind_on = mo.ui.switch(label="wind", value=True)
    wind_relationship = mo.ui.dropdown(
        options=relationship_options,
        value="nonlinear",
        label="Relationship",
    )
    wind_effect = mo.ui.number(
        start=0.1,
        stop=2.5,
        step=0.1,
        value=0.7,
        label="Effect scale",
        full_width=True,
    )
    wind_driver_period = mo.ui.number(
        start=2.0,
        stop=24.0 * 14.0,
        step=1.0,
        value=12.0,
        label="Driver period (hours)",
        full_width=True,
    )
    wind_driver_noise = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.15,
        label="Driver noise",
        full_width=True,
    )
    wind_knots = mo.ui.number(
        start=4,
        stop=12,
        step=1,
        value=7,
        label="Spline knots",
        full_width=True,
    )

    pressure_on = mo.ui.switch(label="pressure", value=False)
    pressure_relationship = mo.ui.dropdown(
        options=relationship_options,
        value="linear",
        label="Relationship",
    )
    pressure_effect = mo.ui.number(
        start=0.1,
        stop=2.5,
        step=0.1,
        value=0.5,
        label="Effect scale",
        full_width=True,
    )
    pressure_driver_period = mo.ui.number(
        start=2.0,
        stop=24.0 * 14.0,
        step=1.0,
        value=8.0,
        label="Driver period (hours)",
        full_width=True,
    )
    pressure_driver_noise = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.10,
        label="Driver noise",
        full_width=True,
    )
    pressure_knots = mo.ui.number(
        start=4,
        stop=12,
        step=1,
        value=6,
        label="Spline knots",
        full_width=True,
    )
    return (
        pressure_driver_noise,
        pressure_driver_period,
        pressure_effect,
        pressure_knots,
        pressure_on,
        pressure_relationship,
        temp_driver_noise,
        temp_driver_period,
        temp_effect,
        temp_knots,
        temp_on,
        temp_relationship,
        wind_driver_noise,
        wind_driver_period,
        wind_effect,
        wind_knots,
        wind_on,
        wind_relationship,
    )


@app.cell
def _(SyntheticTrendKind, mo):
    trend_kind = mo.ui.dropdown(
        options=[item.value for item in SyntheticTrendKind],
        value="linear",
        label="Trend",
    )
    trend_amplitude = mo.ui.number(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=0.35,
        label="Trend amplitude",
        full_width=True,
    )
    trend_grouping_hours = mo.ui.number(
        start=1.0,
        stop=24.0 * 14.0,
        step=1.0,
        value=24.0,
        label="Trend grouping (hours)",
        full_width=True,
    )
    return trend_amplitude, trend_grouping_hours, trend_kind


@app.cell
def _(
    custom_amplitude,
    custom_harmonics,
    custom_on,
    custom_period,
    daily_amplitude,
    daily_harmonics,
    daily_on,
    daily_period,
    fourier_reg,
    freq,
    linear_reg,
    mo,
    n_samples,
    noise_scale,
    pressure_driver_noise,
    pressure_driver_period,
    pressure_effect,
    pressure_knots,
    pressure_on,
    pressure_relationship,
    run_model,
    seed,
    solver_name,
    spline_diff_reg,
    spline_reg,
    start_date,
    temp_driver_noise,
    temp_driver_period,
    temp_effect,
    temp_knots,
    temp_on,
    temp_relationship,
    train_fraction,
    trend_amplitude,
    trend_grouping_hours,
    trend_kind,
    trend_reg,
    weekly_amplitude,
    weekly_harmonics,
    weekly_on,
    weekly_period,
    wind_driver_noise,
    wind_driver_period,
    wind_effect,
    wind_knots,
    wind_on,
    wind_relationship,
):
    periodic_section = mo.vstack(
        [
            mo.md("### Periodic truth"),
            mo.hstack([daily_on, daily_period, daily_harmonics, daily_amplitude], gap=1),
            mo.hstack([weekly_on, weekly_period, weekly_harmonics, weekly_amplitude], gap=1),
            mo.hstack([custom_on, custom_period, custom_harmonics, custom_amplitude], gap=1),
        ]
    )
    regressor_section = mo.vstack(
        [
            mo.md("### Synthetic regressors"),
            mo.hstack(
                [
                    temp_on,
                    temp_relationship,
                    temp_effect,
                    temp_driver_period,
                    temp_driver_noise,
                    temp_knots,
                ],
                gap=1,
            ),
            mo.hstack(
                [
                    wind_on,
                    wind_relationship,
                    wind_effect,
                    wind_driver_period,
                    wind_driver_noise,
                    wind_knots,
                ],
                gap=1,
            ),
            mo.hstack(
                [
                    pressure_on,
                    pressure_relationship,
                    pressure_effect,
                    pressure_driver_period,
                    pressure_driver_noise,
                    pressure_knots,
                ],
                gap=1,
            ),
        ]
    )
    setup_section = mo.vstack(
        [
            mo.md("### Generation and fit settings"),
            mo.hstack([start_date, n_samples, freq, train_fraction], gap=1),
            mo.hstack([seed, noise_scale, solver_name, run_model], gap=1),
            mo.hstack(
                [
                    fourier_reg,
                    linear_reg,
                    spline_reg,
                    spline_diff_reg,
                    trend_reg,
                ],
                gap=1,
            ),
            mo.hstack([trend_kind, trend_amplitude, trend_grouping_hours], gap=1),
        ]
    )
    mo.vstack([setup_section, periodic_section, regressor_section])
    return


@app.cell
def _(
    SyntheticPeriodicComponent,
    SyntheticProblemConfig,
    SyntheticRegressorRelationship,
    SyntheticRegressorSpec,
    SyntheticTrendKind,
    SyntheticTrendSpec,
    custom_amplitude,
    custom_harmonics,
    custom_on,
    custom_period,
    daily_amplitude,
    daily_harmonics,
    daily_on,
    daily_period,
    freq,
    n_samples,
    noise_scale,
    pressure_driver_noise,
    pressure_driver_period,
    pressure_effect,
    pressure_knots,
    pressure_on,
    pressure_relationship,
    seed,
    start_date,
    temp_driver_noise,
    temp_driver_period,
    temp_effect,
    temp_knots,
    temp_on,
    temp_relationship,
    train_fraction,
    trend_amplitude,
    trend_grouping_hours,
    trend_kind,
    weekly_amplitude,
    weekly_harmonics,
    weekly_on,
    weekly_period,
    wind_driver_noise,
    wind_driver_period,
    wind_effect,
    wind_knots,
    wind_on,
    wind_relationship,
):
    periodic_components = []
    if daily_on.value:
        periodic_components.append(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=float(daily_period.value),
                harmonics=int(daily_harmonics.value),
                amplitude=float(daily_amplitude.value),
            )
        )
    if weekly_on.value:
        periodic_components.append(
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=float(weekly_period.value),
                harmonics=int(weekly_harmonics.value),
                amplitude=float(weekly_amplitude.value),
            )
        )
    if custom_on.value:
        periodic_components.append(
            SyntheticPeriodicComponent(
                name="custom",
                period_hours=float(custom_period.value),
                harmonics=int(custom_harmonics.value),
                amplitude=float(custom_amplitude.value),
            )
        )

    regressor_specs = []
    if temp_on.value:
        regressor_specs.append(
            SyntheticRegressorSpec(
                name="temp",
                relationship=SyntheticRegressorRelationship(temp_relationship.value),
                effect_scale=float(temp_effect.value),
                driver_period_hours=float(temp_driver_period.value),
                driver_noise_scale=float(temp_driver_noise.value),
                n_knots=int(temp_knots.value),
            )
        )
    if wind_on.value:
        regressor_specs.append(
            SyntheticRegressorSpec(
                name="wind",
                relationship=SyntheticRegressorRelationship(wind_relationship.value),
                effect_scale=float(wind_effect.value),
                driver_period_hours=float(wind_driver_period.value),
                driver_noise_scale=float(wind_driver_noise.value),
                n_knots=int(wind_knots.value),
            )
        )
    if pressure_on.value:
        regressor_specs.append(
            SyntheticRegressorSpec(
                name="pressure",
                relationship=SyntheticRegressorRelationship(pressure_relationship.value),
                effect_scale=float(pressure_effect.value),
                driver_period_hours=float(pressure_driver_period.value),
                driver_noise_scale=float(pressure_driver_noise.value),
                n_knots=int(pressure_knots.value),
            )
        )

    problem_config = SyntheticProblemConfig(
        start=str(start_date.value),
        n_samples=int(n_samples.value),
        freq=freq.value,
        train_fraction=float(train_fraction.value),
        seed=int(seed.value),
        noise_scale=float(noise_scale.value),
        periodic_components=tuple(periodic_components),
        regressors=tuple(regressor_specs),
        trend=SyntheticTrendSpec(
            kind=SyntheticTrendKind(trend_kind.value),
            amplitude=float(trend_amplitude.value),
            grouping_hours=float(trend_grouping_hours.value),
        ),
    )
    return problem_config,


@app.cell
def _(generate_synthetic_problem, pd, problem_config, split_problem_frames):
    problem = generate_synthetic_problem(problem_config)
    split = split_problem_frames(problem)

    summary_rows = []
    for component in problem_config.periodic_components:
        summary_rows.append(
            {
                "group": "periodic",
                "name": component.name,
                "detail": (
                    f"period={component.period_hours:.1f}h, "
                    f"harmonics={component.harmonics}, amplitude={component.amplitude:.2f}"
                ),
            }
        )
    for spec in problem_config.regressors:
        summary_rows.append(
            {
                "group": "regressor",
                "name": spec.name,
                "detail": (
                    f"{spec.relationship.value}, effect={spec.effect_scale:.2f}, "
                    f"driver={spec.driver_period_hours:.1f}h"
                ),
            }
        )
    summary_rows.append(
        {
            "group": "trend",
            "name": problem_config.trend.kind.value,
            "detail": (
                f"amplitude={problem_config.trend.amplitude:.2f}, "
                f"grouping={problem_config.trend.grouping_hours:.1f}h"
            ),
        }
    )
    summary_df = pd.DataFrame(summary_rows)
    return problem, split, summary_df


@app.cell
def _(alt, mo, pd, problem, summary_df):
    _summary = mo.ui.table(summary_df, pagination=False)

    _overview = pd.DataFrame(
        {
            "datetime": problem.y.index,
            "target": problem.y.values,
            "signal": problem.signal.values,
            "noise": problem.noise.values,
        }
    ).melt("datetime", var_name="series", value_name="value")

    _chart = (
        alt.Chart(_overview)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("value:Q", title="value"),
            color=alt.Color("series:N"),
        )
        .properties(width="container", height=280, title="Generated target, latent signal, and noise")
    )
    generated_overview = mo.vstack(
        [
            mo.md("## Generated problem"),
            _summary,
            mo.ui.altair_chart(_chart),
        ]
    )
    return generated_overview,


@app.cell
def _(
    REG_WEIGHT_OPTIONS,
    TsgamEstimator,
    build_estimator_config,
    fourier_reg,
    linear_reg,
    np,
    problem_config,
    run_model,
    solver_name,
    spline_diff_reg,
    spline_reg,
    split,
    synthetic_metrics,
    trend_reg,
):
    fit_bundle = None
    if run_model.value:
        estimator_config = build_estimator_config(
            problem_config,
            solver_name=solver_name.value,
            fourier_reg_weight=REG_WEIGHT_OPTIONS[fourier_reg.value],
            linear_reg_weight=REG_WEIGHT_OPTIONS[linear_reg.value],
            spline_reg_weight=REG_WEIGHT_OPTIONS[spline_reg.value],
            spline_diff_reg_weight=float(spline_diff_reg.value),
            trend_reg_weight=float(trend_reg.value),
        )
        try:
            estimator = TsgamEstimator(config=estimator_config)
            estimator.fit(split.X_train, split.y_train.to_numpy())
            y_pred_train = estimator.predict(split.X_train)
            y_pred_test = estimator.predict(split.X_test)
            residual_train = split.y_train.to_numpy() - y_pred_train
            residual_test = split.y_test.to_numpy() - y_pred_test
            fit_bundle = {
                "estimator": estimator,
                "estimator_config": estimator_config,
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test,
                "residual_train": residual_train,
                "residual_test": residual_test,
                "metrics_train": synthetic_metrics(split.y_train.to_numpy(), y_pred_train),
                "metrics_test": synthetic_metrics(split.y_test.to_numpy(), y_pred_test),
                "status": estimator.problem_.status,
            }
        except Exception as exc:
            fit_bundle = {"error": f"{type(exc).__name__}: {exc}"}
    return fit_bundle,


@app.cell
def _(alt, fit_bundle, mo, pd, problem, split):
    if fit_bundle is None:
        fit_tab = mo.md("Click `Run model` to fit the estimator on the current synthetic problem.")
    elif "error" in fit_bundle:
        fit_tab = mo.md(f"## Fit error\n\n```text\n{fit_bundle['error']}\n```")
    else:
        _fit_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "datetime": split.y_train.index,
                        "observed": split.y_train.values,
                        "predicted": fit_bundle["y_pred_train"],
                        "segment": "train",
                    }
                ),
                pd.DataFrame(
                    {
                        "datetime": split.y_test.index,
                        "observed": split.y_test.values,
                        "predicted": fit_bundle["y_pred_test"],
                        "segment": "test",
                    }
                ),
            ],
            ignore_index=True,
        ).melt(
            id_vars=["datetime", "segment"],
            var_name="series",
            value_name="value",
        )
        _fit_chart = (
            alt.Chart(_fit_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("datetime:T", title=None),
                y=alt.Y("value:Q", title="value"),
                color=alt.Color("series:N"),
                strokeDash=alt.StrokeDash("segment:N"),
            )
            .properties(width="container", height=320, title="Observed vs predicted")
        )
        _metrics_df = pd.DataFrame(
            [
                {"split": "train", **fit_bundle["metrics_train"]},
                {"split": "test", **fit_bundle["metrics_test"]},
            ]
        )
        fit_tab = mo.vstack(
            [
                mo.md(f"## Fit diagnostics\n\nSolver status: `{fit_bundle['status']}`"),
                mo.ui.table(_metrics_df.round(4), pagination=False),
                mo.ui.altair_chart(_fit_chart),
            ]
        )
    return fit_tab,


@app.cell
def _(alt, mo, pd, problem):
    _components = problem.truth_components.copy()
    _components["datetime"] = _components.index
    _component_long = _components.melt(
        id_vars="datetime",
        var_name="component",
        value_name="value",
    )
    _stack_chart = (
        alt.Chart(_component_long)
        .mark_area(opacity=0.7)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("value:Q", title="truth contribution", stack="zero"),
            color=alt.Color("component:N"),
        )
        .properties(width="container", height=320, title="True component stack")
    )
    components_tab = mo.ui.altair_chart(_stack_chart)
    return components_tab,


@app.cell
def _(alt, fit_bundle, mo, np, pd, split):
    if fit_bundle is None:
        residual_tab = mo.md("Residual diagnostics appear after fitting the model.")
    elif "error" in fit_bundle:
        residual_tab = mo.md("Residual diagnostics unavailable because fitting failed.")
    else:
        _residual_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "datetime": split.y_train.index,
                        "residual": fit_bundle["residual_train"],
                        "segment": "train",
                    }
                ),
                pd.DataFrame(
                    {
                        "datetime": split.y_test.index,
                        "residual": fit_bundle["residual_test"],
                        "segment": "test",
                    }
                ),
            ],
            ignore_index=True,
        )
        _residual_chart = (
            alt.Chart(_residual_df)
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X("datetime:T", title=None),
                y=alt.Y("residual:Q", title="residual"),
                color=alt.Color("segment:N"),
            )
            .properties(width="container", height=220, title="Residual series")
        )
        _hist_df = pd.DataFrame(
            {
                "residual": np.concatenate(
                    [fit_bundle["residual_train"], fit_bundle["residual_test"]]
                )
            }
        )
        _hist_chart = (
            alt.Chart(_hist_df)
            .mark_bar()
            .encode(
                x=alt.X("residual:Q", bin=alt.Bin(maxbins=40), title="residual"),
                y=alt.Y("count():Q", title="count"),
            )
            .properties(width="container", height=220, title="Residual histogram")
        )
        residual_tab = mo.vstack(
            [
                mo.ui.altair_chart(_residual_chart),
                mo.ui.altair_chart(_hist_chart),
            ]
        )
    return residual_tab,


@app.cell
def _(components_tab, fit_tab, generated_overview, mo, residual_tab):
    mo.ui.tabs(
        {
            "Generated problem": generated_overview,
            "Fit": fit_tab,
            "Components": components_tab,
            "Residuals": residual_tab,
        }
    )
    return


if __name__ == "__main__":
    app.run()
