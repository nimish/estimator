#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive marimo notebook for an idealized synthetic TSGAM problem.
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    _intro = (
        "# Synthetic TSGAM Explorer\n\n"
        "Build a controlled time-series problem, fit TSGAM, and compare the "
        "model against the known truth. Start with the defaults, then change "
        "one part of the synthetic data at a time."
    )
    mo.md(_intro)
    return


@app.cell
def _():
    import contextlib
    import io
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
        SyntheticDriverNoiseDistribution,
        SyntheticHarmonicProfile,
        SyntheticNonlinearCurve,
        SyntheticPeriodicComponent,
        SyntheticPeriodicInteractionSpec,
        SyntheticProblemConfig,
        SyntheticRegressorRelationship,
        SyntheticRegressorSpec,
        SyntheticTrendKind,
        SyntheticTrendSpec,
        build_estimator_config,
        component_summary_rows,
        component_fit_quality_rows,
        component_fit_stat_rows,
        cross_basis_coefficient_frame,
        describe_problem_config,
        estimator_config_rows,
        fitted_component_frame,
        fourier_coefficient_frame,
        generate_synthetic_problem,
        problem_dashboard_rows,
        problem_summary_rows,
        regressor_response_frame,
        regressor_inspection_frame,
        residual_summary_rows,
        split_problem_frames,
        synthetic_metrics,
        true_regressor_response_frame,
    )
    from tsgam_estimator import TsgamEstimator

    return (
        SyntheticDriverNoiseDistribution,
        SyntheticHarmonicProfile,
        SyntheticNonlinearCurve,
        SyntheticPeriodicComponent,
        SyntheticPeriodicInteractionSpec,
        SyntheticProblemConfig,
        SyntheticRegressorRelationship,
        SyntheticRegressorSpec,
        SyntheticTrendKind,
        SyntheticTrendSpec,
        TsgamEstimator,
        alt,
        build_estimator_config,
        component_fit_quality_rows,
        component_fit_stat_rows,
        component_summary_rows,
        contextlib,
        cross_basis_coefficient_frame,
        describe_problem_config,
        estimator_config_rows,
        fitted_component_frame,
        fourier_coefficient_frame,
        generate_synthetic_problem,
        io,
        mo,
        np,
        pd,
        problem_dashboard_rows,
        problem_summary_rows,
        regressor_inspection_frame,
        regressor_response_frame,
        residual_summary_rows,
        split_problem_frames,
        synthetic_metrics,
        true_regressor_response_frame,
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
    TREND_KIND_OPTIONS = ["none", "linear", "nonlinear_inc", "nonlinear_dec"]
    return FREQ_OPTIONS, REG_WEIGHT_OPTIONS, SOLVER_OPTIONS, TREND_KIND_OPTIONS


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
    solver_verbose = mo.ui.switch(label="Solver verbose output", value=False)
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
        start=0.0,
        stop=10.0,
        step=0.01,
        value=0.1,
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
        solver_verbose,
        spline_diff_reg,
        spline_reg,
        start_date,
        train_fraction,
        trend_reg,
    )


@app.cell
def _(SyntheticHarmonicProfile, mo):
    _harmonic_profile_options = [item.value for item in SyntheticHarmonicProfile]
    daily_on = mo.ui.switch(label="Daily", value=True)
    daily_period = mo.ui.number(
        start=2.0,
        stop=96.0,
        step=1.0,
        value=24.0,
        label="Period (hours)",
        full_width=True,
    )
    daily_harmonics = mo.ui.slider(
        start=1,
        stop=32,
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
    daily_harmonic_profile = mo.ui.dropdown(
        options=_harmonic_profile_options,
        value="power",
        label="Harmonic profile",
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
    weekly_harmonics = mo.ui.slider(
        start=1,
        stop=32,
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
    weekly_harmonic_profile = mo.ui.dropdown(
        options=_harmonic_profile_options,
        value="power",
        label="Harmonic profile",
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
    custom_harmonics = mo.ui.slider(
        start=1,
        stop=32,
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
    custom_harmonic_profile = mo.ui.dropdown(
        options=_harmonic_profile_options,
        value="power",
        label="Harmonic profile",
    )
    daily_weekly_cross_on = mo.ui.switch(label="Daily x weekly", value=True)
    daily_weekly_cross_effect = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.05,
        value=0.25,
        label="Effect scale",
        full_width=True,
    )
    daily_custom_cross_on = mo.ui.switch(label="Daily x custom", value=False)
    daily_custom_cross_effect = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.05,
        value=0.15,
        label="Effect scale",
        full_width=True,
    )
    weekly_custom_cross_on = mo.ui.switch(label="Weekly x custom", value=False)
    weekly_custom_cross_effect = mo.ui.number(
        start=-2.0,
        stop=2.0,
        step=0.05,
        value=0.15,
        label="Effect scale",
        full_width=True,
    )
    return (
        custom_amplitude,
        custom_harmonic_profile,
        custom_harmonics,
        custom_on,
        custom_period,
        daily_amplitude,
        daily_custom_cross_effect,
        daily_custom_cross_on,
        daily_harmonic_profile,
        daily_harmonics,
        daily_on,
        daily_period,
        daily_weekly_cross_effect,
        daily_weekly_cross_on,
        weekly_amplitude,
        weekly_custom_cross_effect,
        weekly_custom_cross_on,
        weekly_harmonic_profile,
        weekly_harmonics,
        weekly_on,
        weekly_period,
    )


@app.cell
def _(
    SyntheticDriverNoiseDistribution,
    SyntheticHarmonicProfile,
    SyntheticNonlinearCurve,
    SyntheticRegressorRelationship,
    mo,
):
    _relationship_options = [item.value for item in SyntheticRegressorRelationship]
    _driver_noise_options = [item.value for item in SyntheticDriverNoiseDistribution]
    _harmonic_profile_options = [item.value for item in SyntheticHarmonicProfile]
    _nonlinear_curve_options = [item.value for item in SyntheticNonlinearCurve]

    def _regressor_controls(
        name,
        *,
        enabled,
        relationship,
        effect,
        driver_period,
        driver_noise,
        knots,
    ):
        return {
            "on": mo.ui.switch(label=name, value=enabled),
            "relationship": mo.ui.dropdown(
                options=_relationship_options,
                value=relationship,
                label="Effect function",
            ),
            "effect": mo.ui.number(
                start=0.1,
                stop=2.5,
                step=0.1,
                value=effect,
                label="Effect scale",
                full_width=True,
            ),
            "driver_period": mo.ui.number(
                start=2.0,
                stop=24.0 * 14.0,
                step=1.0,
                value=driver_period,
                label="Driver period (hours)",
                full_width=True,
            ),
            "driver_noise": mo.ui.number(
                start=0.0,
                stop=1.0,
                step=0.05,
                value=driver_noise,
                label="Driver noise",
                full_width=True,
            ),
            "driver_distribution": mo.ui.dropdown(
                options=_driver_noise_options,
                value="gaussian",
                label="Driver noise distribution",
            ),
            "knots": mo.ui.number(
                start=4,
                stop=12,
                step=1,
                value=knots,
                label="Spline knots",
                full_width=True,
            ),
            "driver_harmonics": mo.ui.slider(
                start=1,
                stop=32,
                step=1,
                value=2,
                label="Driver temporal complexity",
                full_width=True,
            ),
            "driver_harmonic_profile": mo.ui.dropdown(
                options=_harmonic_profile_options,
                value="power",
                label="Driver harmonic profile",
            ),
            "nonlinear_scale": mo.ui.number(
                start=0.25,
                stop=3.0,
                step=0.25,
                value=1.25,
                label="Nonlinear response sharpness",
                full_width=True,
            ),
            "nonlinear_curve": mo.ui.dropdown(
                options=_nonlinear_curve_options,
                value="tanh",
                label="Nonlinear curve",
            ),
        }

    _temp_controls = _regressor_controls(
        "temp",
        enabled=True,
        relationship="linear",
        effect=0.9,
        driver_period=24.0,
        driver_noise=0.20,
        knots=7,
    )
    _wind_controls = _regressor_controls(
        "wind",
        enabled=True,
        relationship="nonlinear",
        effect=0.7,
        driver_period=12.0,
        driver_noise=0.15,
        knots=7,
    )
    _pressure_controls = _regressor_controls(
        "pressure",
        enabled=False,
        relationship="linear",
        effect=0.5,
        driver_period=8.0,
        driver_noise=0.10,
        knots=6,
    )

    temp_on = _temp_controls["on"]
    temp_relationship = _temp_controls["relationship"]
    temp_effect = _temp_controls["effect"]
    temp_driver_period = _temp_controls["driver_period"]
    temp_driver_noise = _temp_controls["driver_noise"]
    temp_driver_distribution = _temp_controls["driver_distribution"]
    temp_knots = _temp_controls["knots"]
    temp_driver_harmonics = _temp_controls["driver_harmonics"]
    temp_driver_harmonic_profile = _temp_controls["driver_harmonic_profile"]
    temp_nonlinear_scale = _temp_controls["nonlinear_scale"]
    temp_nonlinear_curve = _temp_controls["nonlinear_curve"]

    wind_on = _wind_controls["on"]
    wind_relationship = _wind_controls["relationship"]
    wind_effect = _wind_controls["effect"]
    wind_driver_period = _wind_controls["driver_period"]
    wind_driver_noise = _wind_controls["driver_noise"]
    wind_driver_distribution = _wind_controls["driver_distribution"]
    wind_knots = _wind_controls["knots"]
    wind_driver_harmonics = _wind_controls["driver_harmonics"]
    wind_driver_harmonic_profile = _wind_controls["driver_harmonic_profile"]
    wind_nonlinear_scale = _wind_controls["nonlinear_scale"]
    wind_nonlinear_curve = _wind_controls["nonlinear_curve"]

    pressure_on = _pressure_controls["on"]
    pressure_relationship = _pressure_controls["relationship"]
    pressure_effect = _pressure_controls["effect"]
    pressure_driver_period = _pressure_controls["driver_period"]
    pressure_driver_noise = _pressure_controls["driver_noise"]
    pressure_driver_distribution = _pressure_controls["driver_distribution"]
    pressure_knots = _pressure_controls["knots"]
    pressure_driver_harmonics = _pressure_controls["driver_harmonics"]
    pressure_driver_harmonic_profile = _pressure_controls["driver_harmonic_profile"]
    pressure_nonlinear_scale = _pressure_controls["nonlinear_scale"]
    pressure_nonlinear_curve = _pressure_controls["nonlinear_curve"]

    return (
        pressure_driver_distribution,
        pressure_driver_harmonic_profile,
        pressure_driver_harmonics,
        pressure_driver_noise,
        pressure_driver_period,
        pressure_effect,
        pressure_knots,
        pressure_nonlinear_curve,
        pressure_nonlinear_scale,
        pressure_on,
        pressure_relationship,
        temp_driver_distribution,
        temp_driver_harmonic_profile,
        temp_driver_harmonics,
        temp_driver_noise,
        temp_driver_period,
        temp_effect,
        temp_knots,
        temp_nonlinear_curve,
        temp_nonlinear_scale,
        temp_on,
        temp_relationship,
        wind_driver_distribution,
        wind_driver_harmonic_profile,
        wind_driver_harmonics,
        wind_driver_noise,
        wind_driver_period,
        wind_effect,
        wind_knots,
        wind_nonlinear_curve,
        wind_nonlinear_scale,
        wind_on,
        wind_relationship,
    )


@app.cell
def _(TREND_KIND_OPTIONS, mo):
    trend_kind = mo.ui.dropdown(
        options=TREND_KIND_OPTIONS,
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
    trend_breakpoints = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=4,
        label="Nonlinear jump breakpoints",
        full_width=True,
    )
    return trend_amplitude, trend_breakpoints, trend_grouping_hours, trend_kind


@app.cell
def _(
    custom_amplitude,
    custom_harmonic_profile,
    custom_harmonics,
    custom_on,
    custom_period,
    daily_amplitude,
    daily_custom_cross_effect,
    daily_custom_cross_on,
    daily_harmonic_profile,
    daily_harmonics,
    daily_on,
    daily_period,
    daily_weekly_cross_effect,
    daily_weekly_cross_on,
    freq,
    mo,
    n_samples,
    noise_scale,
    pressure_driver_distribution,
    pressure_driver_harmonic_profile,
    pressure_driver_harmonics,
    pressure_driver_noise,
    pressure_driver_period,
    pressure_effect,
    pressure_knots,
    pressure_nonlinear_curve,
    pressure_nonlinear_scale,
    pressure_on,
    pressure_relationship,
    seed,
    start_date,
    temp_driver_distribution,
    temp_driver_harmonic_profile,
    temp_driver_harmonics,
    temp_driver_noise,
    temp_driver_period,
    temp_effect,
    temp_knots,
    temp_nonlinear_curve,
    temp_nonlinear_scale,
    temp_on,
    temp_relationship,
    train_fraction,
    trend_amplitude,
    trend_breakpoints,
    trend_grouping_hours,
    trend_kind,
    weekly_amplitude,
    weekly_custom_cross_effect,
    weekly_custom_cross_on,
    weekly_harmonic_profile,
    weekly_harmonics,
    weekly_on,
    weekly_period,
    wind_driver_distribution,
    wind_driver_harmonic_profile,
    wind_driver_harmonics,
    wind_driver_noise,
    wind_driver_period,
    wind_effect,
    wind_knots,
    wind_nonlinear_curve,
    wind_nonlinear_scale,
    wind_on,
    wind_relationship,
):
    scenario_section = mo.vstack(
        [
            mo.md(
                "### Data\n\n"
                "Set the time grid, train/test split, seed, and observation noise. "
                "The synthetic data preview updates immediately."
            ),
            mo.hstack([start_date, freq, n_samples, train_fraction], gap=1),
            mo.hstack([seed, noise_scale], gap=1),
        ],
        gap=1,
    )
    _daily_rows = [
        mo.hstack(
            [
                daily_on,
                daily_period,
                daily_harmonics,
                daily_amplitude,
                daily_harmonic_profile,
            ],
            gap=1,
        ),
    ]
    _weekly_rows = [
        mo.hstack(
            [
                weekly_on,
                weekly_period,
                weekly_harmonics,
                weekly_amplitude,
                weekly_harmonic_profile,
            ],
            gap=1,
        ),
    ]
    _custom_rows = [
        mo.hstack(
            [
                custom_on,
                custom_period,
                custom_harmonics,
                custom_amplitude,
                custom_harmonic_profile,
            ],
            gap=1,
        ),
    ]
    _daily_controls = mo.vstack(_daily_rows, gap=1)
    _weekly_controls = mo.vstack(_weekly_rows, gap=1)
    _custom_controls = mo.vstack(_custom_rows, gap=1)
    _cross_controls = mo.vstack(
        [
            mo.md(
                "Use periodic cross terms when the amplitude or shape of one "
                "cycle depends on another, such as daily structure changing "
                "through the week."
            ),
            mo.hstack([daily_weekly_cross_on, daily_weekly_cross_effect], gap=1),
            mo.hstack([daily_custom_cross_on, daily_custom_cross_effect], gap=1),
            mo.hstack([weekly_custom_cross_on, weekly_custom_cross_effect], gap=1),
        ],
        gap=1,
    )
    periodic_section = mo.vstack(
        [
            mo.md("### Periodic\n\nChoose known periodic terms in the synthetic truth."),
            mo.ui.tabs(
                {
                    "Daily": _daily_controls,
                    "Weekly": _weekly_controls,
                    "Custom": _custom_controls,
                    "Cross terms": _cross_controls,
                }
            ),
        ],
        gap=1,
    )
    _regressor_controls = {
        "temp": {
            "on": temp_on,
            "relationship": temp_relationship,
            "effect": temp_effect,
            "driver_period": temp_driver_period,
            "driver_noise": temp_driver_noise,
            "driver_distribution": temp_driver_distribution,
            "knots": temp_knots,
            "driver_harmonics": temp_driver_harmonics,
            "driver_harmonic_profile": temp_driver_harmonic_profile,
            "nonlinear_scale": temp_nonlinear_scale,
            "nonlinear_curve": temp_nonlinear_curve,
        },
        "wind": {
            "on": wind_on,
            "relationship": wind_relationship,
            "effect": wind_effect,
            "driver_period": wind_driver_period,
            "driver_noise": wind_driver_noise,
            "driver_distribution": wind_driver_distribution,
            "knots": wind_knots,
            "driver_harmonics": wind_driver_harmonics,
            "driver_harmonic_profile": wind_driver_harmonic_profile,
            "nonlinear_scale": wind_nonlinear_scale,
            "nonlinear_curve": wind_nonlinear_curve,
        },
        "pressure": {
            "on": pressure_on,
            "relationship": pressure_relationship,
            "effect": pressure_effect,
            "driver_period": pressure_driver_period,
            "driver_noise": pressure_driver_noise,
            "driver_distribution": pressure_driver_distribution,
            "knots": pressure_knots,
            "driver_harmonics": pressure_driver_harmonics,
            "driver_harmonic_profile": pressure_driver_harmonic_profile,
            "nonlinear_scale": pressure_nonlinear_scale,
            "nonlinear_curve": pressure_nonlinear_curve,
        },
    }
    _regressor_help = mo.md(
        "### Regressors\n\n"
        "Enable standardized input drivers, then choose the response function "
        "from each driver into the target. Linear means `scale * x`; nonlinear "
        "uses a selected truth curve (`tanh`, `sigmoid`, `bell`, or `poly`) that "
        "the model fits with a spline. Driver shape controls affect the input "
        "time series itself, not the response function."
    )
    def _render_regressor_controls(controls):
        rows = [
            mo.hstack(
                [controls["on"], controls["relationship"], controls["effect"]],
                gap=1,
            ),
            mo.hstack(
                [
                    controls["driver_period"],
                    controls["driver_noise"],
                    controls["driver_distribution"],
                ],
                gap=1,
            ),
            mo.hstack(
                [
                    controls["driver_harmonics"],
                    controls["driver_harmonic_profile"],
                ],
                gap=1,
            ),
        ]
        if controls["relationship"].value == "nonlinear":
            rows.append(
                mo.hstack(
                    [controls["nonlinear_curve"], controls["knots"]],
                    gap=1,
                )
            )
            rows.append(mo.hstack([controls["nonlinear_scale"]], gap=1))
        return mo.vstack(rows, gap=1)

    regressor_section = mo.vstack(
        [
            _regressor_help,
            mo.ui.tabs(
                {
                    name: _render_regressor_controls(controls)
                    for name, controls in _regressor_controls.items()
                }
            ),
        ],
        gap=1,
    )
    trend_section = mo.vstack(
        [
            mo.md(
                "### Trend\n\n"
                "Choose the low-frequency truth component in the generated signal. "
                "`linear` is a sampled ramp. Nonlinear trend options create a "
                "piecewise-constant step function with a handful of seeded, "
                "irregularly spaced jump breakpoints, flat between jumps."
            ),
            mo.hstack(
                [
                    trend_kind,
                    trend_amplitude,
                    trend_grouping_hours,
                    trend_breakpoints,
                ],
                gap=1,
            ),
        ],
        gap=1,
    )
    configuration_panel = mo.ui.tabs(
        {
            "Data": scenario_section,
            "Periodic": periodic_section,
            "Regressors": regressor_section,
            "Trend": trend_section,
        }
    )
    mo.vstack(
        [
            mo.md(
                "## 1. Configure Synthetic Data Generator\n\n"
                "Start here. These controls define the synthetic time series and "
                "its known truth components. The generated-data plots below update "
                "as soon as these controls change."
            ),
            configuration_panel,
        ],
        gap=1,
    )


    return


@app.cell
def _(
    SyntheticDriverNoiseDistribution,
    SyntheticHarmonicProfile,
    SyntheticNonlinearCurve,
    SyntheticPeriodicComponent,
    SyntheticPeriodicInteractionSpec,
    SyntheticProblemConfig,
    SyntheticRegressorRelationship,
    SyntheticRegressorSpec,
    SyntheticTrendKind,
    SyntheticTrendSpec,
    custom_amplitude,
    custom_harmonic_profile,
    custom_harmonics,
    custom_on,
    custom_period,
    daily_amplitude,
    daily_custom_cross_effect,
    daily_custom_cross_on,
    daily_harmonic_profile,
    daily_harmonics,
    daily_on,
    daily_period,
    daily_weekly_cross_effect,
    daily_weekly_cross_on,
    freq,
    n_samples,
    noise_scale,
    pressure_driver_distribution,
    pressure_driver_harmonic_profile,
    pressure_driver_harmonics,
    pressure_driver_noise,
    pressure_driver_period,
    pressure_effect,
    pressure_knots,
    pressure_nonlinear_curve,
    pressure_nonlinear_scale,
    pressure_on,
    pressure_relationship,
    seed,
    start_date,
    temp_driver_distribution,
    temp_driver_harmonic_profile,
    temp_driver_harmonics,
    temp_driver_noise,
    temp_driver_period,
    temp_effect,
    temp_knots,
    temp_nonlinear_curve,
    temp_nonlinear_scale,
    temp_on,
    temp_relationship,
    train_fraction,
    trend_amplitude,
    trend_breakpoints,
    trend_grouping_hours,
    trend_kind,
    weekly_amplitude,
    weekly_custom_cross_effect,
    weekly_custom_cross_on,
    weekly_harmonic_profile,
    weekly_harmonics,
    weekly_on,
    weekly_period,
    wind_driver_distribution,
    wind_driver_harmonic_profile,
    wind_driver_harmonics,
    wind_driver_noise,
    wind_driver_period,
    wind_effect,
    wind_knots,
    wind_nonlinear_curve,
    wind_nonlinear_scale,
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
                harmonic_profile=SyntheticHarmonicProfile(daily_harmonic_profile.value),
            )
        )
    if weekly_on.value:
        periodic_components.append(
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=float(weekly_period.value),
                harmonics=int(weekly_harmonics.value),
                amplitude=float(weekly_amplitude.value),
                harmonic_profile=SyntheticHarmonicProfile(weekly_harmonic_profile.value),
            )
        )
    if custom_on.value:
        periodic_components.append(
            SyntheticPeriodicComponent(
                name="custom",
                period_hours=float(custom_period.value),
                harmonics=int(custom_harmonics.value),
                amplitude=float(custom_amplitude.value),
                harmonic_profile=SyntheticHarmonicProfile(custom_harmonic_profile.value),
            )
        )

    periodic_names = {component.name for component in periodic_components}
    periodic_interactions = []

    def _add_periodic_cross(left, right, switch, effect):
        if switch.value and left in periodic_names and right in periodic_names:
            periodic_interactions.append(
                SyntheticPeriodicInteractionSpec(
                    left=left,
                    right=right,
                    effect_scale=float(effect.value),
                )
            )

    _add_periodic_cross(
        "daily",
        "weekly",
        daily_weekly_cross_on,
        daily_weekly_cross_effect,
    )
    _add_periodic_cross(
        "daily",
        "custom",
        daily_custom_cross_on,
        daily_custom_cross_effect,
    )
    _add_periodic_cross(
        "weekly",
        "custom",
        weekly_custom_cross_on,
        weekly_custom_cross_effect,
    )

    regressor_specs = []

    def _add_regressor(
        name,
        *,
        on,
        relationship,
        effect,
        driver_period,
        driver_harmonics,
        driver_noise,
        driver_distribution,
        driver_harmonic_profile,
        knots,
        nonlinear_scale,
        nonlinear_curve,
    ):
        if on.value:
            regressor_specs.append(
                SyntheticRegressorSpec(
                    name=name,
                    relationship=SyntheticRegressorRelationship(relationship.value),
                    effect_scale=float(effect.value),
                    driver_period_hours=float(driver_period.value),
                    driver_harmonics=int(driver_harmonics.value),
                    driver_noise_scale=float(driver_noise.value),
                    driver_noise_distribution=SyntheticDriverNoiseDistribution(
                        driver_distribution.value
                    ),
                    driver_harmonic_profile=SyntheticHarmonicProfile(
                        driver_harmonic_profile.value
                    ),
                    n_knots=int(knots.value),
                    nonlinear_scale=float(nonlinear_scale.value),
                    nonlinear_curve=SyntheticNonlinearCurve(nonlinear_curve.value),
                )
            )

    _add_regressor(
        "temp",
        on=temp_on,
        relationship=temp_relationship,
        effect=temp_effect,
        driver_period=temp_driver_period,
        driver_harmonics=temp_driver_harmonics,
        driver_noise=temp_driver_noise,
        driver_distribution=temp_driver_distribution,
        driver_harmonic_profile=temp_driver_harmonic_profile,
        knots=temp_knots,
        nonlinear_scale=temp_nonlinear_scale,
        nonlinear_curve=temp_nonlinear_curve,
    )
    _add_regressor(
        "wind",
        on=wind_on,
        relationship=wind_relationship,
        effect=wind_effect,
        driver_period=wind_driver_period,
        driver_harmonics=wind_driver_harmonics,
        driver_noise=wind_driver_noise,
        driver_distribution=wind_driver_distribution,
        driver_harmonic_profile=wind_driver_harmonic_profile,
        knots=wind_knots,
        nonlinear_scale=wind_nonlinear_scale,
        nonlinear_curve=wind_nonlinear_curve,
    )
    _add_regressor(
        "pressure",
        on=pressure_on,
        relationship=pressure_relationship,
        effect=pressure_effect,
        driver_period=pressure_driver_period,
        driver_harmonics=pressure_driver_harmonics,
        driver_noise=pressure_driver_noise,
        driver_distribution=pressure_driver_distribution,
        driver_harmonic_profile=pressure_driver_harmonic_profile,
        knots=pressure_knots,
        nonlinear_scale=pressure_nonlinear_scale,
        nonlinear_curve=pressure_nonlinear_curve,
    )

    problem_config = SyntheticProblemConfig(
        start=str(start_date.value),
        n_samples=int(n_samples.value),
        freq=freq.value,
        train_fraction=float(train_fraction.value),
        seed=int(seed.value),
        noise_scale=float(noise_scale.value),
        periodic_components=tuple(periodic_components),
        periodic_interactions=tuple(periodic_interactions),
        regressors=tuple(regressor_specs),
        trend=SyntheticTrendSpec(
            kind=SyntheticTrendKind(trend_kind.value),
            amplitude=float(trend_amplitude.value),
            grouping_hours=float(trend_grouping_hours.value),
            breakpoints=int(trend_breakpoints.value),
        ),
    )
    return (problem_config,)


@app.cell
def _(
    component_summary_rows,
    describe_problem_config,
    generate_synthetic_problem,
    pd,
    problem_config,
    problem_dashboard_rows,
    problem_summary_rows,
    regressor_inspection_frame,
    split_problem_frames,
    true_regressor_response_frame,
):
    problem = generate_synthetic_problem(problem_config)
    split = split_problem_frames(problem)
    summary_df = pd.DataFrame(problem_summary_rows(problem_config))
    dashboard_df = pd.DataFrame(problem_dashboard_rows(problem, split))
    component_summary_df = pd.DataFrame(component_summary_rows(problem_config, problem))
    regressor_inspection_df = regressor_inspection_frame(problem)
    true_regressor_response_df = true_regressor_response_frame(problem)
    scenario_description = describe_problem_config(problem_config)
    return (
        component_summary_df,
        dashboard_df,
        problem,
        regressor_inspection_df,
        scenario_description,
        split,
        summary_df,
        true_regressor_response_df,
    )


@app.cell
def _(
    REG_WEIGHT_OPTIONS,
    build_estimator_config,
    estimator_config_rows,
    fourier_reg,
    linear_reg,
    pd,
    problem_config,
    solver_name,
    solver_verbose,
    spline_diff_reg,
    spline_reg,
    trend_reg,
):
    try:
        current_estimator_config = build_estimator_config(
            problem_config,
            solver_name=solver_name.value,
            solver_verbose=bool(solver_verbose.value),
            fourier_reg_weight=REG_WEIGHT_OPTIONS[fourier_reg.value],
            linear_reg_weight=REG_WEIGHT_OPTIONS[linear_reg.value],
            spline_reg_weight=REG_WEIGHT_OPTIONS[spline_reg.value],
            spline_diff_reg_weight=float(spline_diff_reg.value),
            trend_reg_weight=float(trend_reg.value),
        )
        model_config_error = None
        model_config_df = pd.DataFrame(
            estimator_config_rows(problem_config, current_estimator_config)
        )
    except ValueError as exc:
        current_estimator_config = None
        model_config_error = str(exc)
        model_config_df = pd.DataFrame(
            [
                {
                    "section": "Validation",
                    "term": "periodic configuration",
                    "value": model_config_error,
                }
            ]
        )
    fit_signature = repr((problem_config, current_estimator_config))
    return current_estimator_config, fit_signature, model_config_df, model_config_error


@app.cell
def _(components_tab, generated_overview, mo, regressors_tab):
    generated_data_tabs = mo.ui.tabs(
        {
            "Truth Overview": generated_overview,
            "Components": components_tab,
            "Regressors": regressors_tab,
        }
    )
    mo.vstack(
        [
            mo.md(
                "## 2. Inspect Generated Data\n\n"
                "Use these plots before fitting to check that the synthetic "
                "target, components, and regressors match the scenario you meant "
                "to create."
            ),
            generated_data_tabs,
        ],
        gap=1,
    )
    return


@app.cell
def _(
    fourier_reg,
    linear_reg,
    mo,
    model_config_df,
    model_config_error,
    problem_config,
    run_model,
    solver_name,
    solver_verbose,
    spline_diff_reg,
    spline_reg,
    trend_reg,
):
    _active_relationships = [
        spec.relationship.value
        for spec in problem_config.regressors
    ]
    _has_linear_regressor = "linear" in _active_relationships
    _has_spline_regressor = "nonlinear" in _active_relationships
    _reg_regularization_items = [
        item
        for item in [
            linear_reg if _has_linear_regressor else None,
            spline_reg if _has_spline_regressor else None,
            spline_diff_reg if _has_spline_regressor else None,
        ]
        if item is not None
    ]
    _reg_regularization_row = (
        mo.hstack(_reg_regularization_items, gap=1)
        if _reg_regularization_items
        else mo.md("No regressor regularization controls apply.")
    )
    _validation_message = (
        mo.md(f"### Validation\n\n```text\n{model_config_error}\n```")
        if model_config_error
        else mo.md("")
    )
    mo.vstack(
        [
            mo.md(
                "## 3. Configure and Run Model Fit\n\n"
                "After inspecting the generated data, choose the fitted TSGAM "
                "settings here. The fit only runs when you click `Run model`, "
                "using the current synthetic data and model configuration."
            ),
            mo.hstack([solver_name, solver_verbose, fourier_reg, trend_reg], gap=1),
            _reg_regularization_row,
            _validation_message,
            mo.ui.table(model_config_df, pagination=False),
            mo.hstack([run_model], justify="end"),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    alt,
    dashboard_df,
    mo,
    pd,
    problem,
    scenario_description,
    split,
    summary_df,
):
    _summary = mo.ui.table(summary_df, pagination=False)
    _dashboard = mo.ui.table(dashboard_df, pagination=False)

    _overview = pd.DataFrame(
        {
            "datetime": problem.y.index,
            "target": problem.y.values,
            "signal": problem.signal.values,
            "noise": problem.noise.values,
        }
    ).melt("datetime", var_name="series", value_name="value")

    _line_chart = (
        alt.Chart(_overview)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("value:Q", title="value"),
            color=alt.Color("series:N"),
        )
    )
    _split_rule_df = pd.DataFrame({"datetime": [split.y_test.index[0]]})
    _split_rule = (
        alt.Chart(_split_rule_df)
        .mark_rule(strokeDash=[5, 4], color="#666")
        .encode(x=alt.X("datetime:T"))
    )
    _chart = (
        alt.layer(_line_chart, _split_rule)
        .properties(
            width="container",
            height=320,
            title="Generated target, latent signal, and noise",
        )
    )
    generated_overview = mo.vstack(
        [
            mo.md(
                "## Current Scenario\n\n"
                f"{scenario_description}\n\n"
                "The table lists the truth components used to synthesize the "
                "target. The chart shows the observed target alongside the "
                "noise-free signal. The dashed rule marks the train/test split."
            ),
            _dashboard,
            _summary,
            mo.ui.altair_chart(_chart),
        ],
        gap=1,
    )
    return (generated_overview,)


@app.cell
def _(mo):
    get_fit_bundle, set_fit_bundle = mo.state(None)
    return get_fit_bundle, set_fit_bundle


@app.cell
def _(
    TsgamEstimator,
    component_fit_quality_rows,
    component_fit_stat_rows,
    contextlib,
    cross_basis_coefficient_frame,
    current_estimator_config,
    fit_signature,
    fitted_component_frame,
    fourier_coefficient_frame,
    get_fit_bundle,
    io,
    model_config_df,
    model_config_error,
    problem,
    problem_config,
    regressor_response_frame,
    run_model,
    set_fit_bundle,
    split,
    synthetic_metrics,
):
    fit_bundle = get_fit_bundle()
    if run_model.value:
        _solver_log_buffer = io.StringIO()
        if current_estimator_config is None:
            fit_bundle = {
                "error": model_config_error or "Model configuration is invalid.",
                "solver_output": "",
                "fit_signature": fit_signature,
                "problem_config": problem_config,
                "problem": problem,
                "split": split,
                "model_config_df": model_config_df.copy(),
            }
        else:
            try:
                estimator = TsgamEstimator(config=current_estimator_config)
                with contextlib.redirect_stdout(_solver_log_buffer), contextlib.redirect_stderr(
                    _solver_log_buffer
                ):
                    estimator.fit(split.X_train, split.y_train.to_numpy())
                solver_output = _solver_log_buffer.getvalue().strip()
                y_pred_train = estimator.predict(split.X_train)
                y_pred_test = estimator.predict(split.X_test)
                residual_train = split.y_train.to_numpy() - y_pred_train
                residual_test = split.y_test.to_numpy() - y_pred_test
                fitted_components_train = fitted_component_frame(estimator, split.X_train)
                fitted_components_test = fitted_component_frame(estimator, split.X_test)
                component_quality = component_fit_quality_rows(
                    config=problem_config,
                    truth_components=problem.truth_components,
                    fitted_train=fitted_components_train,
                    fitted_test=fitted_components_test,
                )
                component_fit_stats = component_fit_stat_rows(component_quality)
                fourier_coefficients = fourier_coefficient_frame(problem_config, estimator)
                cross_basis_coefficients = cross_basis_coefficient_frame(
                    problem_config,
                    estimator,
                )
                regressor_responses = regressor_response_frame(estimator, problem)
                fit_bundle = {
                    "estimator": estimator,
                    "estimator_config": current_estimator_config,
                    "fit_signature": fit_signature,
                    "problem_config": problem_config,
                    "problem": problem,
                    "split": split,
                    "model_config_df": model_config_df.copy(),
                    "fitted_components_train": fitted_components_train,
                    "fitted_components_test": fitted_components_test,
                    "component_quality": component_quality,
                    "component_fit_stats": component_fit_stats,
                    "fourier_coefficients": fourier_coefficients,
                    "cross_basis_coefficients": cross_basis_coefficients,
                    "regressor_responses": regressor_responses,
                    "solver_output": solver_output,
                    "y_pred_train": y_pred_train,
                    "y_pred_test": y_pred_test,
                    "residual_train": residual_train,
                    "residual_test": residual_test,
                    "metrics_train": synthetic_metrics(split.y_train.to_numpy(), y_pred_train),
                    "metrics_test": synthetic_metrics(split.y_test.to_numpy(), y_pred_test),
                    "status": estimator.problem_.status,
                    "objective_value": estimator.problem_.value,
                }
            except Exception as exc:
                solver_output = _solver_log_buffer.getvalue().strip()
                fit_bundle = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "solver_output": solver_output,
                    "fit_signature": fit_signature,
                    "problem_config": problem_config,
                    "problem": problem,
                    "split": split,
                    "model_config_df": model_config_df.copy(),
                }
        set_fit_bundle(fit_bundle)
    if fit_bundle is not None:
        fit_bundle = {
            **fit_bundle,
            "stale": fit_bundle.get("fit_signature") != fit_signature,
        }
    return (fit_bundle,)


@app.cell
def _(alt, fit_bundle, mo, model_config_df, pd):
    if fit_bundle is None:
        fit_tab = mo.vstack(
            [
                mo.md(
                    "## Fit Performance\n\n"
                    "Click `Run model` in the model-fit section above to fit "
                    "TSGAM on the current synthetic problem. The table below is "
                    "the model form that will be used for the next fit."
                ),
                mo.ui.table(model_config_df, pagination=False),
            ],
            gap=1,
        )
    elif "error" in fit_bundle:
        _solver_output = fit_bundle.get("solver_output") or "No solver output captured."
        _fit_model_config_df = fit_bundle.get("model_config_df", model_config_df)
        _stale_text = (
            "\n\nThis fit was run with an earlier configuration; rerun to refresh."
            if fit_bundle.get("stale")
            else ""
        )
        fit_tab = mo.vstack(
            [
                mo.md(
                    "## Fit Error\n\n"
                    "The current scenario or solver settings produced an error. "
                    "Try fewer harmonics, less noise, or the other solver.\n\n"
                    f"```text\n{fit_bundle['error']}\n```"
                    f"{_stale_text}"
                ),
                mo.ui.table(_fit_model_config_df, pagination=False),
                mo.md(f"### Solver Output\n\n```text\n{_solver_output}\n```"),
            ],
            gap=1,
        )
    else:
        _fit_split = fit_bundle["split"]
        _stale_text = (
            "\n\nThis fit was run with an earlier configuration; rerun to refresh."
            if fit_bundle.get("stale")
            else ""
        )
        _fit_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "datetime": _fit_split.y_train.index,
                        "observed": _fit_split.y_train.values,
                        "predicted": fit_bundle["y_pred_train"],
                        "segment": "train",
                    }
                ),
                pd.DataFrame(
                    {
                        "datetime": _fit_split.y_test.index,
                        "observed": _fit_split.y_test.values,
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
        _test_metrics = fit_bundle["metrics_test"]
        _objective_value = fit_bundle.get("objective_value")
        _objective_text = (
            "not reported"
            if _objective_value is None
            else f"{float(_objective_value):.4g}"
        )
        _metric_cards = mo.Html(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0.75rem;">
              <div style="padding:0.75rem;border:1px solid var(--border-color);border-radius:0.6rem;">
                <div style="font-size:0.8rem;color:var(--muted-foreground);">Test RMSE</div>
                <div style="font-size:1.4rem;font-weight:650;">{_test_metrics['rmse']:.3f}</div>
              </div>
              <div style="padding:0.75rem;border:1px solid var(--border-color);border-radius:0.6rem;">
                <div style="font-size:0.8rem;color:var(--muted-foreground);">Test MAE</div>
                <div style="font-size:1.4rem;font-weight:650;">{_test_metrics['mae']:.3f}</div>
              </div>
              <div style="padding:0.75rem;border:1px solid var(--border-color);border-radius:0.6rem;">
                <div style="font-size:0.8rem;color:var(--muted-foreground);">Test R²</div>
                <div style="font-size:1.4rem;font-weight:650;">{_test_metrics['r2']:.3f}</div>
              </div>
            </div>
            """
        )
        _solver_output = fit_bundle.get("solver_output") or "No solver output captured."
        fit_tab = mo.vstack(
            [
                mo.md(
                    "## Fit Performance\n\n"
                    f"Solver status: `{fit_bundle['status']}`. "
                    f"Objective: `{_objective_text}`."
                    f"{_stale_text}"
                ),
                _metric_cards,
                mo.ui.table(_metrics_df.round(4), pagination=False),
                mo.ui.table(fit_bundle["model_config_df"], pagination=False),
                mo.ui.altair_chart(_fit_chart),
                mo.md(f"### Solver Output\n\n```text\n{_solver_output}\n```"),
            ],
            gap=1,
        )
    return (fit_tab,)


@app.cell
def _(alt, component_summary_df, mo, problem):
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
    _facet_chart = (
        alt.Chart(_component_long)
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("component:N", legend=None),
        )
        .properties(width="container", height=105)
        .facet(row=alt.Row("component:N", title=None))
        .resolve_scale(y="independent")
        .properties(title="True components by term")
    )
    components_tab = mo.vstack(
        [
            mo.md(
                "## Generated Components\n\n"
                "These are the additive components used to generate the "
                "noise-free signal. Use the summary table to compare scale, "
                "then switch between the stacked and faceted views."
            ),
            mo.ui.table(component_summary_df.round(4), pagination=False),
            mo.ui.tabs(
                {
                    "Stacked": mo.ui.altair_chart(_stack_chart),
                    "Faceted": mo.ui.altair_chart(_facet_chart),
                }
            ),
        ],
        gap=1,
    )
    return (components_tab,)


@app.cell
def _(alt, mo, regressor_inspection_df, true_regressor_response_df):
    if regressor_inspection_df.empty:
        regressors_tab = mo.md(
            "## Regressors\n\n"
            "No regressors are enabled in the current synthetic configuration."
        )
    else:
        _plot_df = regressor_inspection_df.copy()
        _plot_df["series_label"] = _plot_df["series"].replace(
            {
                "driver": "input driver x(t)",
                "true contribution": "true target contribution f(x)",
            }
        )
        _plot_df["regressor_label"] = (
            _plot_df["regressor"].astype(str)
            + " ("
            + _plot_df["curve"].astype(str)
            + " truth)"
        )
        _series_order = ["input driver x(t)", "true target contribution f(x)"]
        _regressor_chart = (
            alt.Chart(_plot_df)
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X("datetime:T", title=None),
                y=alt.Y("value:Q", title="value"),
                color=alt.Color(
                    "series_label:N",
                    title="Line",
                    scale=alt.Scale(
                        domain=_series_order,
                        range=["#4C78A8", "#F58518"],
                    ),
                    legend=alt.Legend(orient="top", direction="horizontal"),
                ),
                strokeDash=alt.StrokeDash(
                    "series_label:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=_series_order,
                        range=[[1, 0], [5, 3]],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("datetime:T", title="time"),
                    alt.Tooltip("regressor:N", title="regressor"),
                    alt.Tooltip("relationship:N", title="relationship"),
                    alt.Tooltip("curve:N", title="truth curve"),
                    alt.Tooltip("series_label:N", title="line"),
                    alt.Tooltip("value:Q", title="value", format=".3f"),
                ],
            )
            .properties(width="container", height=140)
            .facet(row=alt.Row("regressor_label:N", title=None))
            .resolve_scale(y="independent")
            .properties(title="Generated input driver vs true target contribution")
        )
        if true_regressor_response_df.empty:
            _response_preview = mo.md("No response curves are available to inspect.")
        else:
            _response_chart = (
                alt.Chart(true_regressor_response_df)
                .mark_line(strokeWidth=2.0)
                .encode(
                    x=alt.X("x:Q", title="standardized driver value"),
                    y=alt.Y("value:Q", title="true response f(x)"),
                    color=alt.Color("curve:N", title="Truth curve"),
                    tooltip=[
                        alt.Tooltip("regressor:N", title="regressor"),
                        alt.Tooltip("relationship:N", title="relationship"),
                        alt.Tooltip("curve:N", title="curve"),
                        alt.Tooltip("x:Q", title="x", format=".3f"),
                        alt.Tooltip("value:Q", title="response", format=".3f"),
                    ],
                )
                .properties(width="container", height=150)
                .facet(row=alt.Row("regressor:N", title=None))
                .resolve_scale(y="independent")
                .properties(title="Generated true response curves before fitting")
            )
            _response_preview = mo.vstack(
                [
                    mo.md(
                        "Inspect the generated response curve before fitting. The "
                        "line is the ground-truth function used to turn each "
                        "standardized driver value into its target contribution."
                    ),
                    mo.ui.altair_chart(_response_chart),
                ],
                gap=1,
            )
        regressors_tab = mo.vstack(
            [
                mo.md(
                    "## Regressors\n\n"
                    "Each panel shows two different things for one generated regressor:\n\n"
                    "- **input driver `x(t)`**: the standardized synthetic input series.\n"
                    "- **true target contribution `f(x)`**: the additive ground-truth "
                    "response placed into the target before noise is added.\n\n"
                    "For a linear effect these lines have the same shape, scaled by the "
                    "effect size. For a nonlinear effect, `f(x)` bends or saturates."
                ),
                mo.ui.tabs(
                    {
                        "Time Series": mo.ui.altair_chart(_regressor_chart),
                        "Response Curves": _response_preview,
                    }
                ),
            ],
            gap=1,
        )
    return (regressors_tab,)


@app.cell
def _(alt, fit_bundle, mo, pd):
    if fit_bundle is None:
        component_diagnostics_tab = mo.md(
            "## Component Diagnostics\n\n"
            "Run the model to compare fitted additive components against the "
            "known synthetic truth."
        )
    elif "error" in fit_bundle:
        component_diagnostics_tab = mo.md(
            "## Component Diagnostics\n\n"
            "Component diagnostics are unavailable because fitting failed."
        )
    else:
        _fit_problem = fit_bundle["problem"]
        _quality_df = pd.DataFrame(fit_bundle["component_quality"])
        _component_stats_df = pd.DataFrame(fit_bundle["component_fit_stats"])
        _fitted_components = pd.concat(
            [
                fit_bundle["fitted_components_train"].assign(segment="train"),
                fit_bundle["fitted_components_test"].assign(segment="test"),
            ]
        )
        _component_names = [
            column
            for column in _fitted_components.columns
            if column not in {"constant", "fitted", "segment"}
        ]
        _truth_compare = pd.DataFrame(index=_fitted_components.index)
        if "periodic" in _component_names:
            _periodic_truth_columns = [
                column
                for column in _fit_problem.truth_components.columns
                if column.startswith("periodic:")
                or column.startswith("periodic_cross:")
            ]
            if _periodic_truth_columns:
                _truth_compare["periodic"] = _fit_problem.truth_components[
                    _periodic_truth_columns
                ].sum(axis=1)
        for _component_name in _component_names:
            if _component_name in _fit_problem.truth_components.columns:
                _truth_compare[_component_name] = _fit_problem.truth_components[
                    _component_name
                ]
        _component_names = [
            component
            for component in _component_names
            if component in _truth_compare.columns
        ]
        _diagnostics_intro = mo.md(
            "## Component Diagnostics\n\n"
            "Each row compares a fitted additive term with the matching "
            "known truth. Periodic terms are scored as one aggregate "
            "Fourier block; regressors are scored individually as linear "
            "or spline terms."
        )
        if _quality_df.empty or not _component_names:
            component_diagnostics_tab = mo.vstack(
                [
                    _diagnostics_intro,
                    mo.md("No component-level diagnostics for this scenario."),
                ],
                gap=1,
            )
        else:
            _fitted_long = (
                _fitted_components[_component_names]
                .assign(datetime=_fitted_components.index, source="fitted model")
                .melt(
                    id_vars=["datetime", "source"],
                    var_name="component",
                    value_name="value",
                )
            )
            _truth_long = (
                _truth_compare[_component_names]
                .assign(datetime=_truth_compare.index, source="synthetic truth")
                .melt(
                    id_vars=["datetime", "source"],
                    var_name="component",
                    value_name="value",
                )
            )
            _compare_df = pd.concat([_truth_long, _fitted_long], ignore_index=True)
            _compare_df["centered_value"] = _compare_df["value"] - _compare_df.groupby(
                ["component", "source"]
            )["value"].transform("mean")
            _compare_chart = (
                alt.Chart(_compare_df)
                .mark_line(strokeWidth=1.2)
                .encode(
                    x=alt.X("datetime:T", title=None),
                    y=alt.Y("centered_value:Q", title="centered value"),
                    color=alt.Color(
                        "source:N",
                        title="Legend",
                        scale=alt.Scale(
                            domain=["synthetic truth", "fitted model"],
                            range=["#4C78A8", "#F58518"],
                        ),
                        legend=alt.Legend(orient="top", direction="horizontal"),
                    ),
                    strokeDash=alt.StrokeDash("source:N", legend=None),
                    tooltip=[
                        alt.Tooltip("datetime:T", title="time"),
                        alt.Tooltip("component:N", title="component"),
                        alt.Tooltip("source:N", title="line"),
                        alt.Tooltip("value:Q", title="raw value", format=".3f"),
                        alt.Tooltip(
                            "centered_value:Q",
                            title="centered value",
                            format=".3f",
                        ),
                    ],
                )
                .properties(width="container", height=115)
                .facet(row=alt.Row("component:N", title=None))
                .resolve_scale(y="independent")
                .properties(title="Centered fitted component shape vs synthetic truth")
            )
            component_diagnostics_tab = mo.vstack(
                [
                    mo.md(
                        "## Component Diagnostics\n\n"
                        "The fitted component baselines are not uniquely identifiable: "
                        "the intercept, trend, and additive terms can trade constant "
                        "offsets while their sum stays the same. The chart below is "
                        "mean-centered per line so it compares recovered **shape**. "
                        "The stats table reports fit quality for every fitted "
                        "component and split: RMSE/MAE are absolute error, relative "
                        "RMSE is scaled by the true component RMS, R² and correlation "
                        "measure recovered shape, and mean offset captures raw "
                        "vertical baseline shift."
                    ),
                    mo.ui.table(_component_stats_df.round(4), pagination=False),
                    mo.ui.table(_quality_df.round(4), pagination=False),
                    mo.ui.altair_chart(_compare_chart),
                ],
                gap=1,
            )
    return (component_diagnostics_tab,)


@app.cell
def _(alt, fit_bundle, mo):
    if fit_bundle is None:
        fourier_coefficients_tab = mo.md(
            "## Fourier Coefficients\n\n"
            "Run the model to compare fitted Fourier coefficients with the "
            "known synthetic periodic coefficients."
        )
    elif "error" in fit_bundle:
        fourier_coefficients_tab = mo.md(
            "## Fourier Coefficients\n\n"
            "Fourier coefficient diagnostics are unavailable because fitting failed."
        )
    else:
        _coef_df = fit_bundle["fourier_coefficients"]
        if _coef_df.empty:
            fourier_coefficients_tab = mo.md(
                "## Fourier Coefficients\n\n"
                "No periodic Fourier terms are enabled in this scenario."
            )
        else:
            _coef_plot_df = _coef_df.copy()
            _coef_plot_df["basis_term"] = (
                _coef_plot_df["harmonic"].astype(str)
                + _coef_plot_df["term"].str[0]
            )
            _coef_plot_df["zero"] = 0.0
            _truth_ticks = (
                alt.Chart(_coef_plot_df)
                .mark_tick(thickness=2, size=24, color="#4C78A8")
                .encode(
                    x=alt.X("basis_term:N", title="basis term"),
                    y=alt.Y("truth_coefficient:Q", title="coefficient"),
                    tooltip=[
                        alt.Tooltip("component:N", title="component"),
                        alt.Tooltip("harmonic:O", title="harmonic"),
                        alt.Tooltip("term:N", title="term"),
                        alt.Tooltip(
                            "truth_coefficient:Q",
                            title="truth",
                            format=".4f",
                        ),
                    ],
                )
            )
            _fitted_points = (
                alt.Chart(_coef_plot_df)
                .mark_point(filled=True, size=55, color="#F58518")
                .encode(
                    x=alt.X("basis_term:N", title="basis term"),
                    y=alt.Y("fitted_coefficient:Q", title="coefficient"),
                    tooltip=[
                        alt.Tooltip("component:N", title="component"),
                        alt.Tooltip("harmonic:O", title="harmonic"),
                        alt.Tooltip("term:N", title="term"),
                        alt.Tooltip(
                            "fitted_coefficient:Q",
                            title="fitted",
                            format=".4f",
                        ),
                    ],
                )
            )
            _paired_chart = (
                (_truth_ticks + _fitted_points)
                .facet(row=alt.Row("component:N", title=None))
                .properties(
                    width=260,
                    height=120,
                    title="Truth ticks vs fitted points",
                )
                .resolve_scale(y="independent")
            )
            _zero_rule = alt.Chart(_coef_plot_df).mark_rule(
                color="black",
                opacity=0.45,
            ).encode(y="zero:Q")
            _difference_chart = (
                (
                    alt.Chart(_coef_plot_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("basis_term:N", title="basis term"),
                        y=alt.Y("difference:Q", title="fitted minus truth"),
                        color=alt.condition(
                            alt.datum.difference >= 0,
                            alt.value("#54A24B"),
                            alt.value("#E45756"),
                        ),
                        tooltip=[
                            alt.Tooltip("component:N", title="component"),
                            alt.Tooltip("harmonic:O", title="harmonic"),
                            alt.Tooltip("term:N", title="term"),
                            alt.Tooltip(
                                "truth_coefficient:Q",
                                title="truth",
                                format=".4f",
                            ),
                            alt.Tooltip(
                                "fitted_coefficient:Q",
                                title="fitted",
                                format=".4f",
                            ),
                            alt.Tooltip(
                                "difference:Q",
                                title="fitted - truth",
                                format=".4f",
                            ),
                        ],
                    )
                    + _zero_rule
                )
                .facet(row=alt.Row("component:N", title=None))
                .properties(
                    width=260,
                    height=120,
                    title="Coefficient differences",
                )
                .resolve_scale(y="independent")
            )
            _legend = mo.md(
                "**Reading this:** blue ticks are synthetic truth, orange dots "
                "are fitted coefficients, and the difference bars show "
                "`fitted - truth`. Nothing is stacked."
            )
            fourier_coefficients_tab = mo.vstack(
                [
                    mo.md(
                        "## Fourier Coefficients\n\n"
                        "This view compares the known truncated Fourier series "
                        "coefficients with the coefficients learned by the model. "
                        "The difference chart is the main diagnostic: zero means "
                        "the fitted coefficient matches the synthetic truth."
                    ),
                    _legend,
                    mo.ui.table(_coef_df.round(5), pagination=False),
                    mo.ui.tabs(
                        {
                            "Differences": mo.ui.altair_chart(_difference_chart),
                            "Truth vs Fitted": mo.ui.altair_chart(_paired_chart),
                        }
                    ),
                ],
                gap=1,
            )
    return (fourier_coefficients_tab,)


@app.cell
def _(alt, fit_bundle, mo, pd):
    if fit_bundle is None:
        cross_basis_coefficients_tab = mo.md(
            "## Cross-Basis Coefficients\n\n"
            "Run the model to inspect periodic cross-basis coefficients."
        )
    elif "error" in fit_bundle:
        cross_basis_coefficients_tab = mo.md(
            "## Cross-Basis Coefficients\n\n"
            "Cross-basis diagnostics are unavailable because fitting failed."
        )
    else:
        _cross_df = fit_bundle["cross_basis_coefficients"]
        if _cross_df.empty:
            cross_basis_coefficients_tab = mo.md(
                "## Cross-Basis Coefficients\n\n"
                "No periodic cross basis is present; enable at least two periodic terms."
            )
        else:
            _cross_plot_df = _cross_df.copy()
            _cross_plot_df["pair"] = (
                _cross_plot_df["left_component"]
                + " x "
                + _cross_plot_df["right_component"]
            )
            _cross_plot_df["basis_term"] = (
                _cross_plot_df["left_harmonic"].astype(str)
                + _cross_plot_df["left_term"].str[0]
                + " × "
                + _cross_plot_df["right_harmonic"].astype(str)
                + _cross_plot_df["right_term"].str[0]
            )
            _cross_long = _cross_plot_df.melt(
                id_vars=["pair", "basis_term", "truth_interaction"],
                value_vars=["truth_coefficient", "fitted_coefficient"],
                var_name="source",
                value_name="coefficient",
            )
            _cross_long["source"] = _cross_long["source"].replace(
                {
                    "truth_coefficient": "synthetic truth",
                    "fitted_coefficient": "fitted model",
                }
            )
            _summary_source = _cross_df.assign(
                abs_difference=_cross_df["difference"].abs(),
                squared_difference=_cross_df["difference"] ** 2,
            )
            _cross_summary_df = (
                _summary_source.groupby(
                    ["left_component", "right_component", "truth_interaction"],
                    as_index=False,
                )
                .agg(
                    n_terms=("difference", "size"),
                    coefficient_mae=("abs_difference", "mean"),
                    coefficient_rmse=("squared_difference", "mean"),
                    max_abs_difference=("abs_difference", "max"),
                )
            )
            _cross_summary_df["coefficient_rmse"] = (
                _cross_summary_df["coefficient_rmse"] ** 0.5
            )
            _cross_chart = (
                alt.Chart(_cross_long)
                .mark_bar()
                .encode(
                    x=alt.X("basis_term:N", title="basis term", sort=None),
                    y=alt.Y("coefficient:Q", title="coefficient"),
                    color=alt.Color(
                        "source:N",
                        title="Legend",
                        scale=alt.Scale(
                            domain=["synthetic truth", "fitted model"],
                            range=["#4C78A8", "#F58518"],
                        ),
                        legend=alt.Legend(orient="top", direction="horizontal"),
                    ),
                    row=alt.Row("pair:N", title=None),
                    tooltip=[
                        alt.Tooltip("pair:N", title="pair"),
                        alt.Tooltip("basis_term:N", title="basis"),
                        alt.Tooltip("truth_interaction:N", title="truth interaction"),
                        alt.Tooltip("source:N", title="line"),
                        alt.Tooltip("coefficient:Q", title="coefficient", format=".5f"),
                    ],
                )
                .properties(width="container", height=135, title="Cross-basis coefficients")
                .resolve_scale(y="independent")
            )
            cross_basis_coefficients_tab = mo.vstack(
                [
                    mo.md(
                        "## Cross-Basis Coefficients\n\n"
                        "This compares learned Fourier pair-product coefficients "
                        "with the synthetic cross-term coefficients implied by the "
                        "enabled truth interactions. Rows with `truth_interaction = none` "
                        "represent model capacity that was not present in the synthetic truth."
                    ),
                    mo.ui.table(_cross_summary_df.round(5), pagination=False),
                    mo.ui.table(_cross_df.round(5), pagination=True),
                    mo.ui.altair_chart(_cross_chart),
                ],
                gap=1,
            )
    return (cross_basis_coefficients_tab,)


@app.cell
def _(alt, fit_bundle, mo):
    if fit_bundle is None:
        regressor_response_tab = mo.md(
            "## Regressor Responses\n\n"
            "Run the model to compare fitted response curves with the synthetic "
            "linear or saturating response functions."
        )
    elif "error" in fit_bundle:
        regressor_response_tab = mo.md(
            "## Regressor Responses\n\n"
            "Regressor response diagnostics are unavailable because fitting failed."
        )
    else:
        _response_df = fit_bundle["regressor_responses"]
        if _response_df.empty:
            regressor_response_tab = mo.md(
                "## Regressor Responses\n\n"
                "No lag-0 regressors are available for response-curve diagnostics."
            )
        else:
            _response_chart = (
                alt.Chart(_response_df)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("x:Q", title="standardized driver value"),
                    y=alt.Y("value:Q", title="response"),
                    color=alt.Color(
                        "source:N",
                        title="Legend",
                        scale=alt.Scale(
                            domain=["synthetic truth", "fitted model"],
                            range=["#4C78A8", "#F58518"],
                        ),
                        legend=alt.Legend(orient="top", direction="horizontal"),
                    ),
                    strokeDash=alt.StrokeDash("source:N", legend=None),
                    tooltip=[
                        alt.Tooltip("regressor:N", title="regressor"),
                        alt.Tooltip("relationship:N", title="truth"),
                        alt.Tooltip("curve:N", title="curve"),
                        alt.Tooltip("source:N", title="line"),
                        alt.Tooltip("x:Q", title="x", format=".3f"),
                        alt.Tooltip("value:Q", title="response", format=".3f"),
                    ],
                )
                .properties(width="container", height=160)
                .facet(row=alt.Row("regressor:N", title=None))
                .resolve_scale(y="independent")
                .properties(title="True vs fitted regressor response functions")
            )
            regressor_response_tab = mo.vstack(
                [
                    mo.md(
                        "## Regressor Responses\n\n"
                        "Each row evaluates the known synthetic response and the "
                        "fitted model response across the observed standardized "
                        "driver range. This isolates the learned function shape "
                        "from the driver's time-series pattern."
                    ),
                    mo.ui.altair_chart(_response_chart),
                ],
                gap=1,
            )
    return (regressor_response_tab,)


@app.cell
def _(alt, fit_bundle, mo, np, pd, residual_summary_rows):
    if fit_bundle is None:
        residual_tab = mo.md(
            "## Residuals\n\n"
            "Residual diagnostics appear after fitting the model."
        )
    elif "error" in fit_bundle:
        residual_tab = mo.md(
            "## Residuals\n\n"
            "Residual diagnostics are unavailable because fitting failed."
        )
    else:
        _fit_split = fit_bundle["split"]
        _residual_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "datetime": _fit_split.y_train.index,
                        "residual": fit_bundle["residual_train"],
                        "segment": "train",
                    }
                ),
                pd.DataFrame(
                    {
                        "datetime": _fit_split.y_test.index,
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
        _residual_summary_df = pd.DataFrame(
            residual_summary_rows(
                fit_bundle["residual_train"],
                fit_bundle["residual_test"],
            )
        )
        residual_tab = mo.vstack(
            [
                mo.md(
                    "## Residuals\n\n"
                    "Check whether errors are centered and whether the test "
                    "period has a visibly different residual pattern."
                ),
                mo.ui.table(_residual_summary_df.round(4), pagination=False),
                mo.ui.altair_chart(_residual_chart),
                mo.ui.altair_chart(_hist_chart),
            ],
            gap=1,
        )
    return (residual_tab,)


@app.cell
def _(
    component_diagnostics_tab,
    cross_basis_coefficients_tab,
    fit_tab,
    fourier_coefficients_tab,
    mo,
    regressor_response_tab,
    residual_tab,
):
    fit_inspection_tabs = mo.ui.tabs(
        {
            "Fit Performance": fit_tab,
            "Fit Components": component_diagnostics_tab,
            "Fourier Coefficients": fourier_coefficients_tab,
            "Cross-Basis Coefficients": cross_basis_coefficients_tab,
            "Regressor Responses": regressor_response_tab,
            "Residuals": residual_tab,
        }
    )
    mo.vstack(
        [
            mo.md(
                "## 4. Inspect Model Fit\n\n"
                "After running the model, use these diagnostics to inspect "
                "prediction accuracy, fitted components, and residual structure."
            ),
            fit_inspection_tabs,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
