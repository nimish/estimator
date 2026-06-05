from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tsgam_estimator import TsgamEstimator, TsgamLinearConfig, TsgamSplineConfig, TrendType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.synthetic_problem import (
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


def _make_problem_config() -> SyntheticProblemConfig:
    return SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=24 * 14,
        freq="1h",
        train_fraction=0.75,
        seed=7,
        noise_scale=0.05,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.2,
            ),
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=24.0 * 7.0,
                harmonics=1,
                amplitude=0.5,
            ),
        ),
        regressors=(
            SyntheticRegressorSpec(
                name="temp",
                relationship=SyntheticRegressorRelationship.LINEAR,
                effect_scale=0.8,
                driver_period_hours=24.0,
                driver_noise_scale=0.2,
                lags=(0,),
            ),
            SyntheticRegressorSpec(
                name="wind",
                relationship=SyntheticRegressorRelationship.NONLINEAR,
                effect_scale=0.6,
                driver_period_hours=12.0,
                driver_noise_scale=0.1,
                lags=(0,),
                n_knots=7,
            ),
        ),
        trend=SyntheticTrendSpec(
            kind=SyntheticTrendKind.LINEAR,
            amplitude=0.4,
            grouping_hours=24.0,
        ),
    )


def test_generate_synthetic_problem_is_deterministic():
    config = _make_problem_config()

    first = generate_synthetic_problem(config)
    second = generate_synthetic_problem(config)

    assert_frame_equal(first.X, second.X)
    assert_series_equal(first.y, second.y)
    assert_frame_equal(first.truth_components, second.truth_components)
    assert_series_equal(first.signal, second.signal)
    assert_series_equal(first.noise, second.noise)


def test_generate_synthetic_problem_tracks_component_breakdown():
    config = _make_problem_config()

    problem = generate_synthetic_problem(config)

    expected_columns = {
        "periodic:daily",
        "periodic:weekly",
        "regressor:temp",
        "regressor:wind",
        "trend",
    }
    assert expected_columns.issubset(problem.truth_components.columns)
    assert list(problem.X.columns) == ["temp", "wind"]
    assert math.isclose(problem.X["temp"].mean(), 0.0, abs_tol=1e-10)
    assert math.isclose(problem.X["wind"].mean(), 0.0, abs_tol=1e-10)
    assert math.isclose(problem.X["temp"].std(ddof=0), 1.0, rel_tol=1e-6)
    assert math.isclose(problem.X["wind"].std(ddof=0), 1.0, rel_tol=1e-6)

    reconstructed = problem.truth_components.sum(axis=1) + problem.noise
    np.testing.assert_allclose(problem.y.values, reconstructed.values)


def test_generate_synthetic_problem_tracks_periodic_cross_term_breakdown():
    config = SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=24 * 7,
        freq="1h",
        seed=11,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=1,
                amplitude=1.0,
            ),
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=24.0 * 7.0,
                harmonics=1,
                amplitude=0.5,
            ),
        ),
        periodic_interactions=(
            SyntheticPeriodicInteractionSpec(
                left="daily",
                right="weekly",
                effect_scale=0.35,
            ),
        ),
    )

    problem = generate_synthetic_problem(config)

    expected = (
        0.35
        * problem.truth_components["periodic:daily"].to_numpy()
        * problem.truth_components["periodic:weekly"].to_numpy()
    )
    assert "periodic_cross:daily x weekly" in problem.truth_components.columns
    np.testing.assert_allclose(
        problem.truth_components["periodic_cross:daily x weekly"].to_numpy(),
        expected,
    )
    reconstructed = problem.truth_components.sum(axis=1) + problem.noise
    np.testing.assert_allclose(problem.y.values, reconstructed.values)


def test_generate_synthetic_problem_supports_nonlinear_increasing_and_decreasing_trends():
    base_kwargs = {
        "start": "2024-01-01",
        "n_samples": 24,
        "freq": "1h",
        "seed": 11,
        "noise_scale": 0.0,
        "periodic_components": (),
        "regressors": (),
    }

    increasing = generate_synthetic_problem(
        SyntheticProblemConfig(
            **base_kwargs,
            trend=SyntheticTrendSpec(
                kind=SyntheticTrendKind.NONLINEAR_INC,
                amplitude=1.0,
                grouping_hours=1.0,
            ),
        )
    ).truth_components["trend"]
    decreasing = generate_synthetic_problem(
        SyntheticProblemConfig(
            **base_kwargs,
            trend=SyntheticTrendSpec(
                kind=SyntheticTrendKind.NONLINEAR_DEC,
                amplitude=1.0,
                grouping_hours=1.0,
            ),
        )
    ).truth_components["trend"]
    legacy = generate_synthetic_problem(
        SyntheticProblemConfig(
            **base_kwargs,
            trend=SyntheticTrendSpec(
                kind=SyntheticTrendKind.NONLINEAR,
                amplitude=1.0,
                grouping_hours=1.0,
            ),
        )
    ).truth_components["trend"]

    assert increasing.iloc[0] < increasing.iloc[-1]
    assert decreasing.iloc[0] > decreasing.iloc[-1]
    assert_series_equal(legacy, decreasing)


def test_nonlinear_trend_uses_sparse_jump_breakpoints_not_smooth_zoh_curve():
    config = SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=24 * 30,
        freq="1h",
        seed=11,
        noise_scale=0.0,
        periodic_components=(),
        regressors=(),
        trend=SyntheticTrendSpec(
            kind=SyntheticTrendKind.NONLINEAR_INC,
            amplitude=2.0,
            grouping_hours=1.0,
            breakpoints=4,
        ),
    )

    trend = generate_synthetic_problem(config).truth_components["trend"]
    jumps = np.flatnonzero(np.diff(trend.to_numpy()) != 0.0)
    segment_lengths = np.diff(np.r_[0, jumps + 1, len(trend)])

    assert len(jumps) == 4
    assert len(set(segment_lengths)) > 1
    assert trend.nunique() == 5
    assert np.all(np.diff(trend.to_numpy()[jumps]) > 0.0)
    assert math.isclose(trend.iloc[0], -1.0)
    assert math.isclose(trend.iloc[-1], 1.0)


def test_periodic_harmonic_profiles_change_component_shape():
    base_config = SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=24 * 7,
        freq="1h",
        seed=11,
        noise_scale=0.0,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=4,
                amplitude=1.0,
                harmonic_profile=SyntheticHarmonicProfile.POWER,
            ),
        ),
    )
    flat_config = SyntheticProblemConfig(
        start=base_config.start,
        n_samples=base_config.n_samples,
        freq=base_config.freq,
        seed=base_config.seed,
        noise_scale=base_config.noise_scale,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=4,
                amplitude=1.0,
                harmonic_profile=SyntheticHarmonicProfile.FLAT,
            ),
        ),
    )
    alternating_config = SyntheticProblemConfig(
        start=base_config.start,
        n_samples=base_config.n_samples,
        freq=base_config.freq,
        seed=base_config.seed,
        noise_scale=base_config.noise_scale,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=4,
                amplitude=1.0,
                harmonic_profile=SyntheticHarmonicProfile.ALTERNATING,
            ),
        ),
    )

    power_problem = generate_synthetic_problem(base_config)
    flat_problem = generate_synthetic_problem(flat_config)
    alternating_problem = generate_synthetic_problem(alternating_config)

    power_component = power_problem.truth_components["periodic:daily"].to_numpy()
    flat_component = flat_problem.truth_components["periodic:daily"].to_numpy()
    alternating_component = alternating_problem.truth_components["periodic:daily"].to_numpy()
    assert not np.allclose(power_component, flat_component)
    assert not np.allclose(power_component, alternating_component)


def test_named_harmonic_profiles_match_common_truncated_fourier_series():
    def _component(profile: SyntheticHarmonicProfile) -> np.ndarray:
        config = SyntheticProblemConfig(
            start="2024-01-01",
            n_samples=64,
            freq="1h",
            seed=11,
            noise_scale=0.0,
            periodic_components=(
                SyntheticPeriodicComponent(
                    name="wave",
                    period_hours=64.0,
                    harmonics=6,
                    amplitude=1.0,
                    phase=0.0,
                    phase_step=0.0,
                    harmonic_profile=profile,
                ),
            ),
            regressors=(),
        )
        problem = generate_synthetic_problem(config)
        return problem.truth_components["periodic:wave"].to_numpy()

    theta = 2.0 * np.pi * np.arange(64) / 64.0
    square_expected = sum(
        (4.0 / np.pi) * np.sin(harmonic * theta) / harmonic
        for harmonic in (1, 3, 5)
    )
    sawtooth_expected = sum(
        (2.0 / np.pi)
        * ((-1.0) ** (harmonic + 1))
        * np.sin(harmonic * theta)
        / harmonic
        for harmonic in range(1, 7)
    )
    triangle_expected = sum(
        (8.0 / np.pi**2)
        * ((-1.0) ** ((harmonic - 1) // 2))
        * np.sin(harmonic * theta)
        / harmonic**2
        for harmonic in (1, 3, 5)
    )

    np.testing.assert_allclose(
        _component(SyntheticHarmonicProfile.SQUARE_WAVE),
        square_expected,
    )
    np.testing.assert_allclose(
        _component(SyntheticHarmonicProfile.SAWTOOTH),
        sawtooth_expected,
    )
    np.testing.assert_allclose(
        _component(SyntheticHarmonicProfile.TRIANGLE_WAVE),
        triangle_expected,
    )


def test_generator_supports_32_harmonics_but_estimator_validates_nyquist_limit():
    config = SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=96,
        freq="1h",
        seed=11,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=32,
                amplitude=1.0,
            ),
        ),
        regressors=(
            SyntheticRegressorSpec(
                name="driver",
                driver_harmonics=32,
            ),
        ),
    )

    problem = generate_synthetic_problem(config)
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()

    with pytest.raises(ValueError, match="Maximum supported harmonics: 12"):
        build_estimator_config(config)
    assert problem.X["driver"].std(ddof=0) > 0.0
    assert notebook_source.count("stop=32,") >= 4


def test_notebook_uses_sliders_for_harmonic_order_controls():
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()

    for control_name in ("daily_harmonics", "weekly_harmonics", "custom_harmonics"):
        assert f"{control_name} = mo.ui.slider(" in notebook_source
    assert '"driver_harmonics": mo.ui.slider(' in notebook_source


def test_notebook_trend_regularization_defaults_low_and_can_go_lower():
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()
    compact_source = " ".join(notebook_source.split())

    assert "trend_reg = mo.ui.number(" in compact_source
    assert "start=0.0, stop=10.0, step=0.01, value=0.1" in compact_source


def test_notebook_exposes_default_on_solver_verbose_output():
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()
    compact_source = " ".join(notebook_source.split())

    assert 'solver_verbose = mo.ui.switch(label="Solver verbose output", value=False)' in compact_source
    assert "solver_verbose=bool(solver_verbose.value)" in compact_source
    assert '"solver_output": solver_output' in compact_source
    assert "### Solver Output" in notebook_source


def test_notebook_trend_dropdown_splits_nonlinear_increasing_and_decreasing():
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()

    assert 'TREND_KIND_OPTIONS = ["none", "linear", "nonlinear_inc", "nonlinear_dec"]' in notebook_source
    assert "options=TREND_KIND_OPTIONS" in notebook_source


def test_notebook_shows_periodic_harmonic_profiles_without_advanced_toggle():
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()
    compact_source = " ".join(notebook_source.split())

    assert (
        "daily_on, daily_period, daily_harmonics, "
        "daily_amplitude, daily_harmonic_profile"
    ) in compact_source
    assert (
        "weekly_on, weekly_period, weekly_harmonics, "
        "weekly_amplitude, weekly_harmonic_profile"
    ) in compact_source
    assert (
        "custom_on, custom_period, custom_harmonics, "
        "custom_amplitude, custom_harmonic_profile"
    ) in compact_source
    assert "Advanced controls" not in notebook_source
    assert 'controls["driver_harmonics"], controls["driver_harmonic_profile"]' in compact_source


def test_regressor_driver_noise_distributions_are_deterministic_and_standardized():
    def _driver(distribution: SyntheticDriverNoiseDistribution) -> pd.Series:
        config = SyntheticProblemConfig(
            start="2024-01-01",
            n_samples=24 * 14,
            freq="1h",
            seed=23,
            periodic_components=(),
            regressors=(
                SyntheticRegressorSpec(
                    name="driver",
                    driver_noise_scale=0.8,
                    driver_noise_distribution=distribution,
                ),
            ),
        )
        return generate_synthetic_problem(config).X["driver"]

    gaussian_first = _driver(SyntheticDriverNoiseDistribution.GAUSSIAN)
    gaussian_second = _driver(SyntheticDriverNoiseDistribution.GAUSSIAN)
    uniform_driver = _driver(SyntheticDriverNoiseDistribution.UNIFORM)
    student_t_driver = _driver(SyntheticDriverNoiseDistribution.STUDENT_T)

    assert_series_equal(gaussian_first, gaussian_second)
    for driver in (gaussian_first, uniform_driver, student_t_driver):
        assert math.isclose(driver.mean(), 0.0, abs_tol=1e-10)
        assert math.isclose(driver.std(ddof=0), 1.0, rel_tol=1e-6)
    assert not np.allclose(gaussian_first.to_numpy(), uniform_driver.to_numpy())
    assert not np.allclose(gaussian_first.to_numpy(), student_t_driver.to_numpy())


def test_build_estimator_config_mixes_linear_and_spline_terms():
    config = _make_problem_config()

    estimator_config = build_estimator_config(
        config,
        solver_name="CLARABEL",
        fourier_reg_weight=1.0e-5,
        linear_reg_weight=2.0e-4,
        spline_reg_weight=3.0e-4,
        spline_diff_reg_weight=0.7,
        trend_reg_weight=5.0,
    )

    assert estimator_config.multi_periodic_config is not None
    assert estimator_config.multi_periodic_config.periods == [24.0, 24.0 * 7.0]
    assert estimator_config.multi_periodic_config.num_harmonics == [2, 1]
    assert estimator_config.multi_periodic_config.reg_weight == 1.0e-5
    assert estimator_config.exog_config is not None
    assert isinstance(estimator_config.exog_config[0], TsgamLinearConfig)
    assert estimator_config.exog_config[0].lags == [0]
    assert estimator_config.exog_config[0].reg_weight == 2.0e-4
    assert isinstance(estimator_config.exog_config[1], TsgamSplineConfig)
    assert estimator_config.exog_config[1].lags == [0]
    assert estimator_config.exog_config[1].n_knots == 7
    assert estimator_config.exog_config[1].reg_weight == 3.0e-4
    assert estimator_config.exog_config[1].diff_reg_weight == 0.7
    assert estimator_config.trend_config is not None
    assert estimator_config.trend_config.trend_type == TrendType.LINEAR
    assert estimator_config.trend_config.grouping == 24.0
    assert estimator_config.trend_config.reg_weight == 5.0


def test_build_estimator_config_converts_hour_periods_to_sample_periods():
    config = SyntheticProblemConfig(
        n_samples=96,
        freq="15min",
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.0,
            ),
        ),
    )

    estimator_config = build_estimator_config(config)

    assert estimator_config.multi_periodic_config is not None
    assert estimator_config.multi_periodic_config.periods == [96.0]


def test_build_estimator_config_defaults_to_light_trend_regularization():
    config = SyntheticProblemConfig(
        trend=SyntheticTrendSpec(
            kind=SyntheticTrendKind.LINEAR,
            amplitude=0.4,
            grouping_hours=24.0,
        ),
    )

    estimator_config = build_estimator_config(config)

    assert estimator_config.trend_config is not None
    assert estimator_config.trend_config.reg_weight == 0.1


def test_build_estimator_config_maps_nonlinear_trend_direction():
    increasing_config = SyntheticProblemConfig(
        periodic_components=(),
        trend=SyntheticTrendSpec(kind=SyntheticTrendKind.NONLINEAR_INC),
    )
    decreasing_config = SyntheticProblemConfig(
        periodic_components=(),
        trend=SyntheticTrendSpec(kind=SyntheticTrendKind.NONLINEAR_DEC),
    )
    legacy_config = SyntheticProblemConfig(
        periodic_components=(),
        trend=SyntheticTrendSpec(kind=SyntheticTrendKind.NONLINEAR),
    )

    increasing = build_estimator_config(increasing_config)
    decreasing = build_estimator_config(decreasing_config)
    legacy = build_estimator_config(legacy_config)

    assert increasing.trend_config is not None
    assert decreasing.trend_config is not None
    assert legacy.trend_config is not None
    assert increasing.trend_config.trend_type == TrendType.NONLINEAR_INC
    assert decreasing.trend_config.trend_type == TrendType.NONLINEAR_DEC
    assert legacy.trend_config.trend_type == TrendType.NONLINEAR


def test_build_estimator_config_can_enable_solver_verbose_output():
    estimator_config = build_estimator_config(
        SyntheticProblemConfig(),
        solver_verbose=True,
    )

    assert estimator_config.solver_config.verbose is True


def test_generate_synthetic_problem_accepts_ui_train_fraction_upper_bound():
    config = SyntheticProblemConfig(
        n_samples=100,
        train_fraction=0.90,
        periodic_components=(),
    )

    problem = generate_synthetic_problem(config)
    split = split_problem_frames(problem)

    assert len(split.y_train) == 90
    assert len(split.y_test) == 10


def test_split_problem_frames_and_metrics_smoke():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)

    split = split_problem_frames(problem)

    assert len(split.X_train) == int(config.n_samples * config.train_fraction)
    assert len(split.X_test) == config.n_samples - len(split.X_train)
    assert split.X_train.index[-1] < split.X_test.index[0]
    assert split.y_train.index.equals(split.X_train.index)
    assert split.y_test.index.equals(split.X_test.index)

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.5, 2.5, 4.5])
    metrics = synthetic_metrics(y_true, y_pred)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert math.isclose(metrics["mae"], 0.5)
    assert math.isclose(metrics["rmse"], 0.5)
    assert math.isclose(metrics["r2"], 0.8)


def test_problem_summary_rows_are_human_readable():
    config = _make_problem_config()

    rows = problem_summary_rows(config)

    assert rows[0] == {
        "section": "Data",
        "setting": "Window",
        "value": "336 samples at 1h; 75% train / 25% test",
    }
    assert {
        "section": "Periodic truth",
        "setting": "daily",
        "value": "24.0h period, 2 harmonics, amplitude 1.20",
    } in rows
    assert {
        "section": "Regressors",
        "setting": "temp",
        "value": "linear effect 0.80, 24.0h driver",
    } in rows
    assert {
        "section": "Regressors",
        "setting": "wind",
        "value": "nonlinear effect 0.60, 12.0h driver, tanh truth, 7 spline knots",
    } in rows
    cross_rows = problem_summary_rows(
        SyntheticProblemConfig(
            periodic_components=(
                SyntheticPeriodicComponent(name="daily", period_hours=24.0),
                SyntheticPeriodicComponent(name="weekly", period_hours=168.0),
            ),
            periodic_interactions=(
                SyntheticPeriodicInteractionSpec(
                    left="daily",
                    right="weekly",
                    effect_scale=0.25,
                ),
            ),
        )
    )
    assert {
        "section": "Periodic cross terms",
        "setting": "daily x weekly",
        "value": "cross-term effect 0.25",
    } in cross_rows
    assert rows[-1] == {
        "section": "Trend",
        "setting": "linear",
        "value": "amplitude 0.40, grouped every 24.0h",
    }


def test_describe_problem_config_gives_compact_overview():
    config = _make_problem_config()

    description = describe_problem_config(config)

    assert "336 samples at 1h" in description
    assert "2 periodic terms" in description
    assert "2 regressors" in description
    assert "linear trend" in description
    assert "noise 0.05" in description


def test_fitted_component_frame_reconstructs_predictions():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)
    estimator_config = build_estimator_config(config, solver_name="CLARABEL")
    estimator = TsgamEstimator(config=estimator_config)
    estimator.fit(problem.X, problem.y.to_numpy())

    component_frame = fitted_component_frame(estimator, problem.X)
    predictions = estimator.predict(problem.X)

    expected_columns = {
        "constant",
        "periodic",
        "regressor:temp",
        "regressor:wind",
        "trend",
        "fitted",
    }
    assert expected_columns.issubset(component_frame.columns)
    assert component_frame.index.equals(problem.X.index)
    np.testing.assert_allclose(component_frame["fitted"].to_numpy(), predictions)
    np.testing.assert_allclose(
        component_frame[["constant", "periodic", "regressor:temp", "regressor:wind", "trend"]]
        .sum(axis=1)
        .to_numpy(),
        predictions,
    )


def test_component_fit_quality_rows_scores_known_truth_terms():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)
    split = split_problem_frames(problem)
    estimator_config = build_estimator_config(config, solver_name="CLARABEL")
    estimator = TsgamEstimator(config=estimator_config)
    estimator.fit(split.X_train, split.y_train.to_numpy())

    train_components = fitted_component_frame(estimator, split.X_train)
    test_components = fitted_component_frame(estimator, split.X_test)
    rows = component_fit_quality_rows(
        config=config,
        truth_components=problem.truth_components,
        fitted_train=train_components,
        fitted_test=test_components,
    )

    labels = {row["component"] for row in rows}
    assert {"periodic", "regressor:temp", "regressor:wind", "trend"}.issubset(labels)
    temp_row = next(row for row in rows if row["component"] == "regressor:temp")
    wind_row = next(row for row in rows if row["component"] == "regressor:wind")
    assert temp_row["model_term"] == "linear"
    assert wind_row["model_term"] == "spline"
    assert temp_row["truth_term"] == "linear"
    assert wind_row["truth_term"] == "nonlinear"
    assert math.isfinite(float(temp_row["train_mean_offset"]))
    assert math.isfinite(float(wind_row["test_mean_offset"]))
    assert math.isfinite(float(temp_row["train_rmse"]))
    assert math.isfinite(float(wind_row["test_rmse"]))
    assert math.isfinite(float(temp_row["train_correlation"]))
    assert math.isfinite(float(wind_row["test_relative_rmse"]))


def test_component_fit_stat_rows_make_split_metrics_readable():
    quality_rows = [
        {
            "component": "regressor:temp",
            "truth_term": "linear",
            "model_term": "linear",
            "train_mean_offset": 0.1,
            "test_mean_offset": -0.2,
            "train_rmse": 0.3,
            "test_rmse": 0.4,
            "train_mae": 0.2,
            "test_mae": 0.25,
            "train_r2": 0.9,
            "test_r2": 0.8,
            "train_correlation": 0.95,
            "test_correlation": 0.85,
            "train_relative_rmse": 0.15,
            "test_relative_rmse": 0.2,
        }
    ]

    rows = component_fit_stat_rows(quality_rows)

    assert rows == [
        {
            "component": "regressor:temp",
            "split": "train",
            "truth": "linear",
            "model": "linear",
            "rmse": 0.3,
            "mae": 0.2,
            "r2": 0.9,
            "correlation": 0.95,
            "relative_rmse": 0.15,
            "mean_offset": 0.1,
        },
        {
            "component": "regressor:temp",
            "split": "test",
            "truth": "linear",
            "model": "linear",
            "rmse": 0.4,
            "mae": 0.25,
            "r2": 0.8,
            "correlation": 0.85,
            "relative_rmse": 0.2,
            "mean_offset": -0.2,
        },
    ]


def test_fourier_coefficient_frame_reports_truth_and_fitted_coefficients():
    config = SyntheticProblemConfig(
        start="2024-01-01",
        n_samples=24 * 14,
        freq="1h",
        train_fraction=0.75,
        seed=5,
        noise_scale=0.0,
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.0,
                phase=0.0,
                phase_step=0.0,
                harmonic_decay=1.0,
            ),
        ),
        regressors=(),
        trend=SyntheticTrendSpec(kind=SyntheticTrendKind.NONE),
    )
    problem = generate_synthetic_problem(config)
    estimator_config = build_estimator_config(config, solver_name="CLARABEL")
    estimator = TsgamEstimator(config=estimator_config)
    estimator.fit(problem.X, problem.y.to_numpy())

    frame = fourier_coefficient_frame(config, estimator)

    assert set(frame.columns) == {
        "component",
        "period_hours",
        "harmonic",
        "term",
        "truth_coefficient",
        "fitted_coefficient",
        "difference",
    }
    assert len(frame) == 4
    truth_by_term = {
        (row["harmonic"], row["term"]): row["truth_coefficient"]
        for row in frame.to_dict("records")
    }
    assert math.isclose(truth_by_term[(1, "sin")], 1.0)
    assert math.isclose(truth_by_term[(1, "cos")], 0.0, abs_tol=1e-12)
    assert math.isclose(truth_by_term[(2, "sin")], 0.5)
    assert math.isclose(truth_by_term[(2, "cos")], 0.0, abs_tol=1e-12)
    assert np.isfinite(frame["fitted_coefficient"]).all()
    assert np.isfinite(frame["difference"]).all()


def test_notebook_fourier_coefficients_show_explicit_differences_not_stacked_bars():
    notebook_source = (
        Path(__file__).resolve().parents[1] / "examples" / "example_synthetic_marimo.py"
    ).read_text()
    fourier_cell = notebook_source.split("def _(alt, fit_bundle, mo):", maxsplit=1)[1].split(
        "@app.cell",
        maxsplit=1,
    )[0]

    assert "_coef_long = _coef_df.melt(" not in fourier_cell
    assert 'y=alt.Y("difference:Q"' in fourier_cell
    assert ".mark_tick(" in fourier_cell
    assert ".mark_bar(" in fourier_cell


def test_fourier_coefficient_frame_ignores_extra_cross_basis_coefficients():
    config = SyntheticProblemConfig(
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.0,
            ),
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=168.0,
                harmonics=1,
                amplitude=0.5,
            ),
        ),
    )
    estimator_config = build_estimator_config(config)
    estimator = SimpleNamespace(
        config=estimator_config,
        variables_={"fourier_coef": SimpleNamespace(value=np.arange(14.0))},
    )

    frame = fourier_coefficient_frame(config, estimator)

    assert len(frame) == 6
    assert frame["component"].tolist() == [
        "weekly",
        "weekly",
        "daily",
        "daily",
        "daily",
        "daily",
    ]
    assert frame["fitted_coefficient"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_cross_basis_coefficient_frame_reports_truth_and_fitted_coefficients():
    config = SyntheticProblemConfig(
        freq="1h",
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=1,
                amplitude=1.0,
                phase=0.0,
                phase_step=0.0,
            ),
            SyntheticPeriodicComponent(
                name="weekly",
                period_hours=168.0,
                harmonics=1,
                amplitude=2.0,
                phase=0.0,
                phase_step=0.0,
            ),
        ),
        periodic_interactions=(
            SyntheticPeriodicInteractionSpec(
                left="daily",
                right="weekly",
                effect_scale=0.5,
            ),
        ),
    )
    estimator_config = build_estimator_config(config)
    estimator = SimpleNamespace(
        config=estimator_config,
        variables_={"fourier_coef": SimpleNamespace(value=np.arange(8.0))},
    )

    frame = cross_basis_coefficient_frame(config, estimator)

    assert set(frame.columns) == {
        "left_component",
        "right_component",
        "left_period_hours",
        "right_period_hours",
        "left_harmonic",
        "left_term",
        "right_harmonic",
        "right_term",
        "truth_interaction",
        "truth_coefficient",
        "fitted_coefficient",
        "difference",
    }
    assert len(frame) == 4
    assert frame["left_component"].unique().tolist() == ["weekly"]
    assert frame["right_component"].unique().tolist() == ["daily"]
    assert frame["fitted_coefficient"].tolist() == [4.0, 5.0, 6.0, 7.0]
    truth_by_terms = {
        (row["left_term"], row["right_term"]): row["truth_coefficient"]
        for row in frame.to_dict("records")
    }
    assert math.isclose(truth_by_terms[("sin", "sin")], 1.0)
    assert math.isclose(truth_by_terms[("cos", "sin")], 0.0, abs_tol=1e-12)


def test_fourier_coefficient_frame_does_not_fail_when_fitted_coefficients_are_short():
    config = SyntheticProblemConfig(
        periodic_components=(
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.0,
            ),
        ),
    )
    estimator_config = build_estimator_config(config)
    estimator = SimpleNamespace(
        config=estimator_config,
        variables_={"fourier_coef": SimpleNamespace(value=np.array([1.0, 2.0]))},
    )

    frame = fourier_coefficient_frame(config, estimator)

    assert len(frame) == 4
    assert frame["fitted_coefficient"].iloc[:2].tolist() == [1.0, 2.0]
    assert frame["fitted_coefficient"].iloc[2:].isna().all()
    assert frame["difference"].iloc[2:].isna().all()


def test_regressor_response_frame_reports_true_and_fitted_response_curves():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)
    split = split_problem_frames(problem)
    estimator_config = build_estimator_config(config, solver_name="CLARABEL")
    estimator = TsgamEstimator(config=estimator_config)
    estimator.fit(split.X_train, split.y_train.to_numpy())

    frame = regressor_response_frame(estimator, problem, grid_size=15)

    assert set(frame.columns) == {
        "regressor",
        "relationship",
        "curve",
        "x",
        "source",
        "value",
    }
    assert set(frame["regressor"]) == {"temp", "wind"}
    assert set(frame["source"]) == {"synthetic truth", "fitted model"}
    assert len(frame) == len(config.regressors) * 15 * 2
    temp_truth = frame[
        (frame["regressor"] == "temp") & (frame["source"] == "synthetic truth")
    ]
    np.testing.assert_allclose(
        temp_truth["value"].to_numpy(),
        0.8 * temp_truth["x"].to_numpy(),
    )
    assert np.isfinite(frame["value"]).all()


def test_true_regressor_response_frame_reports_prefit_curve_shapes():
    config = SyntheticProblemConfig(
        periodic_components=(),
        noise_scale=0.0,
        regressors=(
            SyntheticRegressorSpec(
                name="linear",
                relationship=SyntheticRegressorRelationship.LINEAR,
                effect_scale=0.5,
            ),
            SyntheticRegressorSpec(
                name="sigmoid",
                relationship=SyntheticRegressorRelationship.NONLINEAR,
                effect_scale=1.0,
                nonlinear_curve=SyntheticNonlinearCurve.SIGMOID,
                nonlinear_scale=2.0,
            ),
            SyntheticRegressorSpec(
                name="bell",
                relationship=SyntheticRegressorRelationship.NONLINEAR,
                effect_scale=1.0,
                nonlinear_curve=SyntheticNonlinearCurve.BELL,
                nonlinear_scale=1.5,
            ),
            SyntheticRegressorSpec(
                name="poly",
                relationship=SyntheticRegressorRelationship.NONLINEAR,
                effect_scale=0.75,
                nonlinear_curve=SyntheticNonlinearCurve.POLY,
                nonlinear_scale=1.0,
            ),
        ),
    )
    problem = generate_synthetic_problem(config)

    frame = true_regressor_response_frame(problem, grid_size=31)

    assert set(frame.columns) == {
        "regressor",
        "relationship",
        "curve",
        "x",
        "source",
        "value",
    }
    assert set(frame["source"]) == {"synthetic truth"}
    assert set(frame["curve"]) == {"linear", "sigmoid", "bell", "poly"}

    sigmoid = frame[frame["regressor"] == "sigmoid"].sort_values("x")
    bell = frame[frame["regressor"] == "bell"].sort_values("x")
    poly = frame[frame["regressor"] == "poly"].sort_values("x")

    assert sigmoid["value"].is_monotonic_increasing
    assert bell.loc[bell["x"].abs().idxmin(), "value"] == pytest.approx(
        bell["value"].max()
    )
    np.testing.assert_allclose(
        poly["value"].to_numpy(),
        poly.sort_values("x", ascending=False)["value"].to_numpy(),
    )
    assert poly.loc[poly["x"].abs().idxmin(), "value"] < poly["value"].iloc[0]


def test_nonlinear_regressor_curve_type_changes_truth_contribution():
    base = {
        "name": "driver",
        "relationship": SyntheticRegressorRelationship.NONLINEAR,
        "effect_scale": 1.0,
        "driver_noise_scale": 0.0,
        "lags": (0,),
        "nonlinear_scale": 1.5,
    }
    configs = {
        curve: SyntheticProblemConfig(
            periodic_components=(),
            noise_scale=0.0,
            regressors=(SyntheticRegressorSpec(**base, nonlinear_curve=curve),),
        )
        for curve in SyntheticNonlinearCurve
    }

    contributions = {
        curve: generate_synthetic_problem(config).truth_components[
            "regressor:driver"
        ].to_numpy()
        for curve, config in configs.items()
    }

    for left_curve, left_values in contributions.items():
        for right_curve, right_values in contributions.items():
            if left_curve == right_curve:
                continue
            assert not np.allclose(left_values, right_values)


def test_problem_dashboard_rows_summarize_current_synthetic_data():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)
    split = split_problem_frames(problem)

    rows = problem_dashboard_rows(problem, split)

    metrics = {row["metric"]: row["value"] for row in rows}
    assert metrics["Samples"] == "336"
    assert metrics["Train / test"] == "252 / 84"
    assert metrics["Frequency"] == "1h"
    assert metrics["Regressors"] == "temp, wind"
    assert float(metrics["Signal RMS"]) > 0.0
    assert float(metrics["Noise RMS"]) > 0.0


def test_component_summary_rows_include_truth_scale_and_metadata():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)

    rows = component_summary_rows(config, problem)

    daily_row = next(row for row in rows if row["component"] == "periodic:daily")
    wind_row = next(row for row in rows if row["component"] == "regressor:wind")
    assert daily_row["kind"] == "periodic"
    assert daily_row["detail"] == "24.0h, 2 harmonics"
    assert wind_row["kind"] == "regressor"
    assert wind_row["detail"] == "nonlinear, 12.0h driver"
    assert math.isfinite(float(daily_row["rms"]))
    assert math.isfinite(float(wind_row["std"]))

    cross_config = SyntheticProblemConfig(
        periodic_components=(
            SyntheticPeriodicComponent(name="daily", period_hours=24.0),
            SyntheticPeriodicComponent(name="weekly", period_hours=168.0),
        ),
        periodic_interactions=(
            SyntheticPeriodicInteractionSpec(
                left="daily",
                right="weekly",
                effect_scale=0.25,
            ),
        ),
    )
    cross_problem = generate_synthetic_problem(cross_config)
    cross_rows = component_summary_rows(cross_config, cross_problem)
    cross_row = next(
        row
        for row in cross_rows
        if row["component"] == "periodic_cross:daily x weekly"
    )
    assert cross_row["kind"] == "periodic cross"
    assert cross_row["detail"] == "daily x weekly, effect 0.25"


def test_regressor_inspection_frame_pairs_drivers_with_true_contributions():
    config = _make_problem_config()
    problem = generate_synthetic_problem(config)

    frame = regressor_inspection_frame(problem)

    assert set(frame.columns) == {
        "datetime",
        "regressor",
        "relationship",
        "curve",
        "series",
        "value",
    }
    assert set(frame["regressor"]) == {"temp", "wind"}
    assert set(frame["series"]) == {"driver", "true contribution"}
    assert set(frame["curve"]) == {"linear", "tanh"}
    assert len(frame) == len(problem.X) * len(problem.X.columns) * 2


def test_regressor_inspection_frame_reflects_selected_nonlinear_curve():
    base = {
        "name": "wind",
        "relationship": SyntheticRegressorRelationship.NONLINEAR,
        "effect_scale": 1.0,
        "driver_noise_scale": 0.0,
        "lags": (0,),
        "nonlinear_scale": 1.25,
    }
    tanh_config = SyntheticProblemConfig(
        periodic_components=(),
        noise_scale=0.0,
        regressors=(
            SyntheticRegressorSpec(
                **base,
                nonlinear_curve=SyntheticNonlinearCurve.TANH,
            ),
        ),
    )
    bell_config = SyntheticProblemConfig(
        periodic_components=(),
        noise_scale=0.0,
        regressors=(
            SyntheticRegressorSpec(
                **base,
                nonlinear_curve=SyntheticNonlinearCurve.BELL,
            ),
        ),
    )

    tanh_frame = regressor_inspection_frame(generate_synthetic_problem(tanh_config))
    bell_frame = regressor_inspection_frame(generate_synthetic_problem(bell_config))
    tanh_contribution = tanh_frame[tanh_frame["series"] == "true contribution"]
    bell_contribution = bell_frame[bell_frame["series"] == "true contribution"]

    assert set(tanh_frame["curve"]) == {"tanh"}
    assert set(bell_frame["curve"]) == {"bell"}
    assert tanh_contribution["value"].min() < 0.0
    assert bell_contribution["value"].min() >= 0.0
    assert not np.allclose(
        tanh_contribution["value"].to_numpy(),
        bell_contribution["value"].to_numpy(),
    )


def test_estimator_config_rows_describe_fitted_model_form():
    config = _make_problem_config()
    estimator_config = build_estimator_config(
        config,
        solver_name="SCS",
        fourier_reg_weight=1.0e-5,
        linear_reg_weight=2.0e-4,
        spline_reg_weight=3.0e-4,
        spline_diff_reg_weight=0.7,
        trend_reg_weight=5.0,
    )

    rows = estimator_config_rows(config, estimator_config)

    model_terms = {(row["section"], row["term"]): row["value"] for row in rows}
    assert model_terms[("Solver", "solver")] == "SCS"
    assert model_terms[("Periodic", "periods")] == "24.0h, 168.0h"
    assert model_terms[("Regressors", "temp")] == "linear, lags [0], reg 0.0002"
    assert model_terms[("Regressors", "wind")] == "spline, 7 knots, lags [0], reg 0.0003"
    assert model_terms[("Trend", "trend")] == "linear, grouping 24 samples, reg 5"


def test_residual_summary_rows_report_train_and_test_residual_stats():
    train = np.array([1.0, -1.0, 0.0])
    test = np.array([0.5, -0.5])

    rows = residual_summary_rows(train, test)

    by_split = {row["split"]: row for row in rows}
    assert by_split["train"]["n"] == 3
    assert by_split["test"]["n"] == 2
    assert math.isclose(float(by_split["train"]["mean"]), 0.0)
    assert math.isclose(float(by_split["test"]["rmse"]), 0.5)
