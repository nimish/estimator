#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Compact marimo explorer for constituent-aware tidal modeling with TSGAM.
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    from pathlib import Path
    import sys
    from example_tidal import (
        DEFAULT_STATION,
        STATION_CATALOG,
        TIDAL_COMPONENT_LABELS,
        TIDAL_CONSTITUENT_HARMONICS,
        TIDAL_CONSTITUENT_PERIODS_HOURS,
        TIDE_TO_WEATHER,
    )
    from tidal_analysis_helpers import build_day_hour_matrix
    return (
        DEFAULT_STATION,
        Path,
        STATION_CATALOG,
        TIDAL_COMPONENT_LABELS,
        TIDAL_CONSTITUENT_HARMONICS,
        TIDAL_CONSTITUENT_PERIODS_HOURS,
        TIDE_TO_WEATHER,
        alt,
        build_day_hour_matrix,
        mo,
        pd,
        plt,
        sys,
    )


@app.cell
def _():
    DEFAULT_TRAIN_START = "2022-01-01"
    DEFAULT_TRAIN_END = "2023-12-31"
    DEFAULT_TEST_START = "2024-01-01"
    DEFAULT_TEST_END = "2024-03-31"
    WEATHER_MODE_OPTIONS = ["recommended", "override", "tide-only"]
    return (
        DEFAULT_TEST_END,
        DEFAULT_TEST_START,
        DEFAULT_TRAIN_END,
        DEFAULT_TRAIN_START,
        WEATHER_MODE_OPTIONS,
    )


@app.cell
def _(TIDE_TO_WEATHER):
    TIDE_ONLY_OPTION = "__tide_only__"
    WEATHER_OVERRIDE_OPTIONS = [TIDE_ONLY_OPTION] + [wid for wid, _ in TIDE_TO_WEATHER.values()]
    FOURIER_PRESETS = {
        "lighter (1e-6)": 1.0e-6,
        "shared (1e-5)": 1.0e-5,
        "stronger (1e-4)": 1.0e-4,
    }
    LAG_PRESET_OPTIONS = ["run_tidal defaults"]
    return (
        FOURIER_PRESETS,
        LAG_PRESET_OPTIONS,
        TIDE_ONLY_OPTION,
        WEATHER_OVERRIDE_OPTIONS,
    )


@app.cell
def _(
    Path,
    STATION_CATALOG,
    TIDAL_COMPONENT_LABELS,
    TIDAL_CONSTITUENT_HARMONICS,
    TIDAL_CONSTITUENT_PERIODS_HOURS,
    TIDE_ONLY_OPTION,
    TIDE_TO_WEATHER,
    pd,
    sys,
):
    import json as _json
    from collections import OrderedDict as _OrderedDict

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from example_tidal import (
        download_lcd_weather,
        download_tidal_data,
        load_lcd_weather,
        load_tidal_data,
        make_constituent_multi_periodic,
        merge_tidal_weather,
        resolve_tidal_cache_path,
        _is_date_only_text,
        _window_boundary,
    )
    from tidal_model_shared import (
        make_tidal_spline_configs,
        prepare_split_regressors,
        tidal_metrics,
        usable_lcd_columns,
    )
    from tidal_analysis_helpers import (
        compute_lagged_correlation,
        compute_periodogram,
        extract_fourier_components,
        infer_samples_per_hour,
    )
    from tsgam_estimator import (
        TrendType,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        TsgamTrendConfig,
    )

    _DATA_DIR = Path(__file__).resolve().parent / "data" / "tidal"

    def station_label(station_id):
        meta = STATION_CATALOG.get(station_id)
        return f"{meta['name']} ({station_id})" if meta else station_id

    def weather_label(weather_id):
        for tid, (wid, wname) in TIDE_TO_WEATHER.items():
            if wid == weather_id:
                return f"{wname} ({wid})"
        return weather_id

    def resolve_weather_choice(station_id, weather_mode, manual_weather_id):
        recommended = TIDE_TO_WEATHER.get(station_id, (None, None))
        recommended_id, recommended_name = recommended
        recommended_label = (
            f"{recommended_name} ({recommended_id})"
            if recommended_id
            else "none"
        )
        if weather_mode == "tide-only":
            return {
                "weather_id": None,
                "recommended_label": recommended_label,
                "requested_label": "tide-only",
            }
        if weather_mode == "override":
            override_id = (
                manual_weather_id
                if manual_weather_id != TIDE_ONLY_OPTION
                else None
            )
            return {
                "weather_id": override_id,
                "recommended_label": recommended_label,
                "requested_label": weather_label(override_id) if override_id else "tide-only (manual)",
            }
        return {
            "weather_id": recommended_id,
            "recommended_label": recommended_label,
            "requested_label": recommended_label,
        }

    def load_station_bundle(station_id, weather_mode, manual_weather_id):
        weather_choice = resolve_weather_choice(
            station_id, weather_mode, manual_weather_id
        )
        weather_id = weather_choice["weather_id"]
        station_meta = STATION_CATALOG[station_id]

        data_file = download_tidal_data(_DATA_DIR, station=station_id)
        tidal_df = load_tidal_data(data_file, interpolate_missing=False)

        fallback_reason = None
        effective_source_label = "CO-OPS only"

        if weather_id is not None:
            try:
                download_lcd_weather(_DATA_DIR, station_id=weather_id)
                lcd_df = load_lcd_weather(
                    _DATA_DIR, station_id=weather_id, interpolate_missing=False
                )
                candidate_merged = merge_tidal_weather(
                    tidal_df, lcd_df, interpolate_missing=False
                )
                lcd_candidates = [
                    "air_temp", "dewpoint", "wind_speed", "wind_u", "wind_v", "lcd_slp",
                ]
                lcd_usable = usable_lcd_columns(
                    lcd_df, candidate_merged, lcd_candidates
                )
                if lcd_usable:
                    drop_from_tidal = [
                        c for c in lcd_usable if c in tidal_df.columns
                    ]
                    merged_df = tidal_df.drop(columns=drop_from_tidal).join(
                        candidate_merged[lcd_usable], how="left"
                    )
                    effective_source_label = f"CO-OPS + LCD ({weather_id})"
                else:
                    merged_df = tidal_df
                    fallback_reason = "LCD columns had insufficient coverage"
            except FileNotFoundError:
                merged_df = tidal_df
                fallback_reason = f"LCD files not found for {weather_id}"
        else:
            merged_df = tidal_df

        return {
            "station_id": station_id,
            "station_meta": station_meta,
            "weather_choice": weather_choice,
            "merged_df": merged_df,
            "fallback_reason": fallback_reason,
            "effective_source_label": effective_source_label,
        }

    def _start_in_range(ts, df_index):
        return ts >= df_index.min()

    def _end_in_range(ts, df_index):
        return ts <= df_index.max() + pd.Timedelta("1D")

    def parse_window_inputs(train_start_text, train_end_text, test_start_text, test_end_text):
        return {
            "train_start": _window_boundary(train_start_text, is_end=False),
            "train_end": _window_boundary(train_end_text, is_end=True),
            "test_start": _window_boundary(test_start_text, is_end=False),
            "test_end": _window_boundary(test_end_text, is_end=True),
            "train_start_text": train_start_text,
            "train_end_text": train_end_text,
            "test_start_text": test_start_text,
            "test_end_text": test_end_text,
        }

    def build_regressor_inventory(merged_df, train_start, train_end):
        regressor_candidates = [
            "pressure", "water_temp", "wind_u", "wind_v", "air_temp",
            "dewpoint", "wind_speed", "lcd_slp",
        ]
        rows = []
        train_slice = merged_df.loc[str(train_start):str(train_end)]
        for col in regressor_candidates:
            if col in merged_df.columns:
                total_cov = merged_df[col].notna().mean()
                train_cov = train_slice[col].notna().mean() if len(train_slice) else 0.0
                rows.append({"variable": col, "total_coverage": f"{total_cov:.1%}", "train_coverage": f"{train_cov:.1%}"})
        if not rows:
            return pd.DataFrame(columns=["variable", "total_coverage", "train_coverage"])
        return pd.DataFrame(rows)

    def build_analysis_context(bundle, train_start_text, train_end_text, test_start_text, test_end_text):
        merged_df = bundle["merged_df"]
        windows = parse_window_inputs(train_start_text, train_end_text, test_start_text, test_end_text)

        if not _start_in_range(windows["train_start"], merged_df.index):
            raise ValueError(f"Train start {train_start_text} is before loaded data")
        if not _end_in_range(windows["test_end"], merged_df.index):
            raise ValueError(f"Test end {test_end_text} is after loaded data")

        analysis_df = merged_df.loc[str(train_start_text):str(test_end_text)]
        sph = infer_samples_per_hour(analysis_df.index)

        inventory = build_regressor_inventory(merged_df, windows["train_start"], windows["train_end"])
        regressor_candidates = [
            "pressure", "water_temp", "wind_u", "wind_v", "air_temp",
        ]
        available = [c for c in regressor_candidates if c in merged_df.columns and merged_df[c].notna().mean() > 0.05]
        default = available[:]

        return {
            "analysis_df": analysis_df,
            "samples_per_hour": sph,
            "windows": windows,
            "coverage_table": inventory,
            "regressor_inventory": inventory,
            "available_regressors": available,
            "default_regressors": default,
        }

    def parse_solver_options(text):
        text = text.strip()
        if not text or text == "{}":
            return {}
        return _json.loads(text)

    def _lags_for_preset(preset_name, sph):
        return None

    def build_spline_config_list(selected_regressors, sph, lag_preset_name, knot_count_val):
        configs, default = make_tidal_spline_configs(sph)
        out = []
        for var in selected_regressors:
            cfg = configs.get(var, TsgamSplineConfig(n_knots=knot_count_val, lags=[0], reg_weight=1e-5, diff_reg_weight=0.3))
            out.append(cfg)
        return out if out else None

    def build_multi_periodic_config(sph, harmonic_map, fourier_reg_weight):
        periods = []
        harmonics = []
        for label in TIDAL_COMPONENT_LABELS:
            p_hours = TIDAL_CONSTITUENT_PERIODS_HOURS[label]
            periods.append(p_hours * sph)
            harmonics.append(harmonic_map.get(label, int(TIDAL_CONSTITUENT_HARMONICS[label])))
        return TsgamMultiPeriodicConfig(
            num_harmonics=harmonics,
            periods=periods,
            reg_weight=fourier_reg_weight,
        )

    def prepare_model_frames(merged_df, train_start_text, train_end_text, test_start_text, test_end_text, selected_regressors):
        train_df = merged_df.loc[str(train_start_text):str(train_end_text)].copy()
        test_df = merged_df.loc[str(test_start_text):str(test_end_text)].copy()
        for sub in (train_df, test_df):
            if sub.index.freq is None:
                sub.index.freq = pd.infer_freq(sub.index)
        y_train = train_df["water_level"].values
        y_test = test_df["water_level"].values
        X_train, X_test, active, dropped = prepare_split_regressors(
            train_df, test_df, selected_regressors
        )
        return {
            "train_df": train_df,
            "test_df": test_df,
            "y_train": y_train,
            "y_test": y_test,
            "X_train": X_train,
            "X_test": X_test,
            "active_regressors": active,
            "dropped_regressors": dropped,
        }

    def build_harmonic_candidates(spectrum):
        rows = []
        for label, p_hours in TIDAL_CONSTITUENT_PERIODS_HOURS.items():
            for n in range(1, 9):
                candidate_period = p_hours / n
                rows.append({"candidate": f"{label}/h{n}", "period_hours": candidate_period})
        return pd.DataFrame(rows)

    def summarize_periodogram_peaks(spectrum, top_n=8):
        if spectrum.empty:
            return pd.DataFrame()
        return spectrum.nlargest(top_n, "power")[["period_hours", "power"]].reset_index(drop=True)

    def run_fit_workflow(
        merged_df,
        train_start_text, train_end_text,
        test_start_text, test_end_text,
        selected_regressors,
        trend_enabled, solver_name,
        fourier_reg_weight, harmonic_map,
        lag_preset_name, knot_count,
        trend_grouping_days, solver_options_text,
    ):
        prepared = prepare_model_frames(
            merged_df, train_start_text, train_end_text,
            test_start_text, test_end_text, selected_regressors,
        )
        sph = infer_samples_per_hour(prepared["train_df"].index)
        multi_periodic = build_multi_periodic_config(sph, harmonic_map, fourier_reg_weight)
        exog_config = build_spline_config_list(
            prepared["active_regressors"], sph, lag_preset_name, knot_count,
        )
        trend_config = None
        if trend_enabled:
            trend_config = TsgamTrendConfig(
                trend_type=TrendType.LINEAR,
                grouping=trend_grouping_days * 24.0 * sph,
            )
        solver_opts = parse_solver_options(solver_options_text) or None
        solver_config = TsgamSolverConfig(
            solver=solver_name, verbose=False, solver_opts=solver_opts,
        )
        est = TsgamEstimator(config=TsgamEstimatorConfig(
            multi_periodic_config=multi_periodic,
            exog_config=exog_config,
            trend_config=trend_config,
            solver_config=solver_config,
            random_state=42,
        ))
        est.fit(prepared["X_train"], prepared["y_train"])
        y_pred_train = est.predict(prepared["X_train"])
        y_pred_test = est.predict(prepared["X_test"])

        metrics = tidal_metrics(prepared["y_test"], y_pred_test)

        fourier_est = TsgamEstimator(config=TsgamEstimatorConfig(
            multi_periodic_config=multi_periodic,
            exog_config=None,
            solver_config=TsgamSolverConfig(solver=solver_name, verbose=False),
            random_state=42,
        ))
        fourier_est.fit(
            pd.DataFrame(index=prepared["X_train"].index),
            prepared["y_train"],
        )
        fourier_resid_train = prepared["y_train"] - fourier_est.predict(
            pd.DataFrame(index=prepared["X_train"].index)
        )
        spectrum = compute_periodogram(
            prepared["train_df"].index, fourier_resid_train,
        )
        peaks = summarize_periodogram_peaks(spectrum)
        harmonic_cands = build_harmonic_candidates(spectrum)

        component_dict = {}
        try:
            component_dict = extract_fourier_components(
                est, labels=TIDAL_COMPONENT_LABELS,
            )
        except Exception:
            pass

        preview_hours = min(7 * 24 * sph, len(prepared["train_df"]))
        comp_index = prepared["train_df"].index[:preview_hours]
        comp_observed = prepared["y_train"][:preview_hours]

        lag_correlations = {}
        dt_hours = prepared["train_df"].index.to_series().diff().median() / pd.Timedelta("1h")
        max_lag = int(48.0 / max(dt_hours, 0.1))
        for col in prepared["active_regressors"]:
            if col in prepared["train_df"].columns:
                try:
                    lc = compute_lagged_correlation(
                        fourier_resid_train,
                        prepared["X_train"][col].values,
                        max_lag,
                    )
                    lc["lag_hours"] = lc["lag"] * dt_hours
                    lag_correlations[col] = lc
                except Exception:
                    pass

        weather_summary_lines = []
        if prepared["active_regressors"]:
            weather_summary_lines.append(
                "Active weather regressors: " + ", ".join(prepared["active_regressors"])
            )
        if prepared["dropped_regressors"]:
            weather_summary_lines.append(
                "Dropped: " + ", ".join(prepared["dropped_regressors"])
            )
        weather_summary = "\n".join(weather_summary_lines) if weather_summary_lines else "No weather regressors."

        return {
            "prepared": prepared,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "metrics": metrics,
            "residual_spectrum": spectrum,
            "periodogram_peaks": peaks,
            "harmonic_candidates": harmonic_cands,
            "component_dict": component_dict,
            "component_window_index": comp_index,
            "component_window_observed": comp_observed,
            "lag_correlations": lag_correlations,
            "weather_summary": weather_summary,
            "active_regressors": prepared["active_regressors"],
            "dropped_regressors": prepared["dropped_regressors"],
        }
    return (
        build_analysis_context,
        load_station_bundle,
        run_fit_workflow,
        station_label,
        weather_label,
    )


@app.cell
def _(mo):
    mo.md("""
    # Compact Tidal Explorer

    A compact, PACT-style `marimo` app for NOAA tide gauges, curated weather
    recommendations, and constituent-aware TSGAM fits.
    """)
    return


@app.cell
def _(
    DEFAULT_STATION,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    STATION_CATALOG,
    WEATHER_MODE_OPTIONS,
    mo,
):
    station_select = mo.ui.dropdown(
        options=list(STATION_CATALOG),
        value=DEFAULT_STATION,
        label="Tide station",
    )
    train_start = mo.ui.date(value=DEFAULT_TRAIN_START, label="Train start")
    train_end = mo.ui.date(value=DEFAULT_TRAIN_END, label="Train end")
    test_start = mo.ui.date(value=DEFAULT_TEST_START, label="Test start")
    test_end = mo.ui.date(value=DEFAULT_TEST_END, label="Test end")
    weather_mode = mo.ui.dropdown(
        options=WEATHER_MODE_OPTIONS,
        value="recommended",
        label="Weather source",
    )
    trend_on = mo.ui.switch(label="Trend", value=True)
    solver_select = mo.ui.dropdown(
        options=["SCS", "CLARABEL"],
        value="SCS",
        label="Solver",
    )
    run_model = mo.ui.run_button(label="Run")
    show_advanced = mo.ui.switch(label="Advanced", value=False)
    return (
        run_model,
        show_advanced,
        solver_select,
        station_select,
        test_end,
        test_start,
        train_end,
        train_start,
        trend_on,
        weather_mode,
    )


@app.cell
def _(
    fourier_preset,
    harmonic_inputs,
    knot_count,
    lag_preset,
    mo,
    show_advanced,
    solver_options,
    trend_grouping_days,
):
    if not show_advanced.value:
        mo.stop(True)

    mo.vstack(
        [
            mo.md("### Advanced Controls"),
            mo.hstack([fourier_preset, lag_preset, knot_count, trend_grouping_days]),
            mo.hstack(
                [
                    harmonic_inputs["M2"],
                    harmonic_inputs["S2"],
                    harmonic_inputs["N2"],
                    harmonic_inputs["K1"],
                ]
            ),
            mo.hstack(
                [
                    harmonic_inputs["O1"],
                    harmonic_inputs["Mf"],
                    harmonic_inputs["Mm"],
                    harmonic_inputs["annual"],
                ]
            ),
            solver_options,
        ]
    )
    return


@app.cell
def _(
    TIDE_ONLY_OPTION,
    TIDE_TO_WEATHER,
    WEATHER_OVERRIDE_OPTIONS,
    mo,
    station_select,
):
    recommended_id = TIDE_TO_WEATHER.get(station_select.value, (None, None))[0]
    weather_override = mo.ui.dropdown(
        options=WEATHER_OVERRIDE_OPTIONS,
        value=recommended_id or TIDE_ONLY_OPTION,
        label="Manual weather override",
    )
    return (weather_override,)


@app.cell
def _(
    FOURIER_PRESETS,
    LAG_PRESET_OPTIONS,
    TIDAL_COMPONENT_LABELS,
    TIDAL_CONSTITUENT_HARMONICS,
    mo,
):
    fourier_preset = mo.ui.dropdown(
        options=list(FOURIER_PRESETS),
        value="shared (1e-5)",
        label="Fourier regularization",
    )
    lag_preset = mo.ui.dropdown(
        options=LAG_PRESET_OPTIONS,
        value="run_tidal defaults",
        label="Spline lag preset",
    )
    knot_count = mo.ui.number(start=4, stop=20, value=8, label="Spline knots", full_width=True)
    trend_grouping_days = mo.ui.number(
        start=0.25,
        stop=14.0,
        step=0.25,
        value=1.0,
        label="Trend grouping (days)",
        full_width=True,
    )
    solver_options = mo.ui.text(value="{}", label="Solver options JSON")
    harmonic_inputs = {
        label: mo.ui.number(
            start=1,
            stop=16,
            value=int(TIDAL_CONSTITUENT_HARMONICS[label]),
            label=label,
            full_width=True,
        )
        for label in TIDAL_COMPONENT_LABELS
    }
    return (
        fourier_preset,
        harmonic_inputs,
        knot_count,
        lag_preset,
        solver_options,
        trend_grouping_days,
    )


@app.cell
def _(
    mo,
    run_model,
    show_advanced,
    solver_select,
    station_select,
    test_end,
    test_start,
    train_end,
    train_start,
    trend_on,
    weather_mode,
    weather_override,
):
    mo.vstack(
        [
            mo.hstack([station_select, weather_mode, weather_override, show_advanced]),
            mo.hstack([train_start, train_end, test_start, test_end]),
            mo.hstack([trend_on, solver_select, run_model]),
        ]
    )
    return


@app.cell
def _(load_station_bundle, mo):
    @mo.cache
    def cached_load_station_bundle(
        station_id: str,
        weather_mode: str,
        manual_weather_id: str,
    ) -> dict[str, object]:
        return load_station_bundle(
            station_id=station_id,
            weather_mode=weather_mode,
            manual_weather_id=manual_weather_id,
        )
    return (cached_load_station_bundle,)


@app.cell
def _(
    cached_load_station_bundle,
    station_select,
    weather_mode,
    weather_override,
):
    bundle_error = None
    try:
        station_bundle = cached_load_station_bundle(
            station_id=station_select.value,
            weather_mode=weather_mode.value,
            manual_weather_id=weather_override.value,
        )
    except Exception as exc:
        station_bundle = None
        bundle_error = f"{type(exc).__name__}: {exc}"
    return bundle_error, station_bundle


@app.cell
def _(
    build_analysis_context,
    bundle_error,
    station_bundle,
    test_end,
    test_start,
    train_end,
    train_start,
):
    analysis_error = bundle_error
    analysis_context = None
    if station_bundle is not None and bundle_error is None:
        try:
            analysis_context = build_analysis_context(
                bundle=station_bundle,
                train_start_text=train_start.value,
                train_end_text=train_end.value,
                test_start_text=test_start.value,
                test_end_text=test_end.value,
            )
        except Exception as exc:
            analysis_error = f"{type(exc).__name__}: {exc}"
    return analysis_context, analysis_error


@app.cell
def _(analysis_context, mo):
    options = analysis_context["available_regressors"] if analysis_context is not None else []
    default_value = analysis_context["default_regressors"] if analysis_context is not None else []
    regressor_select = mo.ui.multiselect(
        options=options,
        value=default_value,
        label="Meteorological regressors",
    )
    return (regressor_select,)


@app.cell
def _(
    analysis_context,
    analysis_error,
    mo,
    station_bundle,
    station_label,
    weather_label,
    weather_override,
):
    if analysis_error is not None:
        overview_header = mo.md(f"**Overview error:** `{analysis_error}`")
    else:
        station_meta = station_bundle["station_meta"]
        weather_choice = station_bundle["weather_choice"]
        window = analysis_context["windows"]
        requested_override = weather_label(weather_override.value)
        fallback_line = (
            f"- Fallback: {station_bundle['fallback_reason']}"
            if station_bundle["fallback_reason"] is not None
            else ""
        )
        overview_header = mo.md(
            "\n".join(
                line
                for line in [
                    f"## {station_label(station_bundle['station_id'])}",
                    f"- Region: {station_meta['region']}",
                    f"- Tidal regime: {station_meta['tidal_regime']}",
                    f"- Notes: {station_meta['notes']}",
                    f"- Sampling rate: {analysis_context['samples_per_hour']} samples/hour",
                    f"- Recommended weather: {weather_choice['recommended_label']}",
                    f"- Requested weather source: {weather_choice['requested_label']}",
                    f"- Manual override target: {requested_override}",
                    f"- Effective weather source: {station_bundle['effective_source_label']}",
                    f"- Analysis window: {window['train_start'].date()} to {window['test_end'].date()}",
                    fallback_line,
                ]
                if line
            )
        )
    return (overview_header,)


@app.cell
def _(analysis_context, analysis_error, build_day_hour_matrix, mo, plt):
    if analysis_error is not None:
        overview_plot = mo.md("No preview available until the selected station and dates load cleanly.")
    else:
        _analysis_df = analysis_context["analysis_df"]
        _plot_series = _analysis_df["water_level"]
        if len(_plot_series) > 4000:
            _plot_series = _plot_series.resample("1D").mean()

        _heat_source = _analysis_df["water_level"]
        _heat_limit = max(24, min(len(_heat_source), 90 * analysis_context["samples_per_hour"] * 24))
        _heat_source = _heat_source.iloc[-_heat_limit:]
        _heat_matrix = build_day_hour_matrix(_heat_source.index, _heat_source.to_numpy())

        _fig, _axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
        _axes[0].plot(_plot_series.index, _plot_series.to_numpy(), linewidth=1.0, color="C0")
        _axes[0].set_title("Observed water level")
        _axes[0].set_ylabel("m")
        _axes[0].tick_params(axis="x", rotation=45)

        _image = _axes[1].imshow(
            _heat_matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        _axes[1].set_title("Recent day-hour matrix")
        _axes[1].set_xlabel("hour")
        _axes[1].set_ylabel("day")
        _axes[1].set_xticks(range(0, 24, 3))
        _fig.colorbar(_image, ax=_axes[1], label="m")

        overview_plot = mo.mpl.interactive(_fig)
    return (overview_plot,)


@app.cell
def _(alt, analysis_context, analysis_error, mo):
    if analysis_error is not None:
        data_overview_plot = mo.md("")
    else:
        _adf = analysis_context["analysis_df"]
        _met_cols = [
            c for c in ["pressure", "water_temp", "wind_u", "wind_v", "air_temp",
                        "wind_speed", "lcd_slp", "dewpoint"]
            if c in _adf.columns and _adf[c].notna().any()
        ]
        _all_cols = ["water_level"] + _met_cols
        _plot_df = _adf[_all_cols].copy()
        _plot_df.index.name = "datetime"
        if len(_plot_df) > 5000:
            _plot_df = _plot_df.resample("1h").mean()
        _plot_df = _plot_df.reset_index()
        _long = _plot_df.melt(id_vars="datetime", var_name="variable", value_name="value")
        _long = _long.dropna(subset=["value"])

        _brush = alt.selection_interval(encodings=["x"])
        _chart = (
            alt.Chart(_long)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("datetime:T", title=None),
                y=alt.Y("value:Q", title=""),
                color=alt.value("steelblue"),
            )
            .properties(width="container", height=120)
            .add_params(_brush)
            .facet(
                facet=alt.Facet("variable:N", title=None, sort=_all_cols),
                columns=1,
            )
            .resolve_scale(y="independent")
        )
        data_overview_plot = mo.ui.altair_chart(_chart)
    return (data_overview_plot,)


@app.cell
def _(
    FOURIER_PRESETS,
    TIDAL_COMPONENT_LABELS,
    analysis_error,
    fourier_preset,
    harmonic_inputs,
    knot_count,
    lag_preset,
    mo,
    regressor_select,
    run_fit_workflow,
    run_model,
    solver_options,
    solver_select,
    station_bundle,
    test_end,
    test_start,
    train_end,
    train_start,
    trend_grouping_days,
    trend_on,
):
    fit_bundle = None
    if run_model.value and analysis_error is None:
        harmonic_map = {
            label: int(harmonic_inputs[label].value)
            for label in TIDAL_COMPONENT_LABELS
        }
        try:
            for step in mo.status.progress_bar(
                ["prepare data", "fit model", "fit Fourier baseline", "assemble diagnostics"]
            ):
                if step == "prepare data":
                    continue
                if step == "fit model":
                    fit_bundle = run_fit_workflow(
                        merged_df=station_bundle["merged_df"],
                        train_start_text=train_start.value,
                        train_end_text=train_end.value,
                        test_start_text=test_start.value,
                        test_end_text=test_end.value,
                        selected_regressors=list(regressor_select.value),
                        trend_enabled=bool(trend_on.value),
                        solver_name=solver_select.value,
                        fourier_reg_weight=FOURIER_PRESETS[fourier_preset.value],
                        harmonic_map=harmonic_map,
                        lag_preset_name=lag_preset.value,
                        knot_count=int(knot_count.value),
                        trend_grouping_days=float(trend_grouping_days.value),
                        solver_options_text=solver_options.value,
                    )
        except Exception as exc:
            fit_bundle = {"error": f"{type(exc).__name__}: {exc}"}
    return (fit_bundle,)


@app.cell
def _(fit_bundle, mo, pd, plt):
    if fit_bundle is None:
        fit_tab = mo.md("Click **Run** to fit the current TSGAM configuration.")
    elif "error" in fit_bundle:
        fit_tab = mo.md(f"**Fit error:** `{fit_bundle['error']}`")
    else:
        _prepared = fit_bundle["prepared"]
        _metrics = fit_bundle["metrics"]
        _metric_table = pd.DataFrame(
            [
                {"metric": "RMSE", "value": _metrics["rmse"]},
                {"metric": "MAE", "value": _metrics["mae"]},
                {"metric": "MAPE", "value": _metrics["mape"]},
                {"metric": "R2", "value": _metrics["r2"]},
            ]
        )

        _train_daily = _prepared["train_df"][["water_level"]].assign(
            predicted=fit_bundle["y_pred_train"]
        ).resample("1D").mean()
        _test_daily = _prepared["test_df"][["water_level"]].assign(
            predicted=fit_bundle["y_pred_test"]
        ).resample("1D").mean()
        _test_residual = _prepared["test_df"][["water_level"]].copy()
        _test_residual["residual"] = _prepared["y_test"] - fit_bundle["y_pred_test"]
        if len(_test_residual) > 2500:
            _test_residual = _test_residual.resample("6H").mean()

        _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
        _axes[0].plot(_train_daily.index, _train_daily["water_level"], label="train actual", linewidth=1.2)
        _axes[0].plot(_train_daily.index, _train_daily["predicted"], label="train predicted", linewidth=1.0)
        _axes[0].plot(_test_daily.index, _test_daily["water_level"], label="test actual", linewidth=1.4)
        _axes[0].plot(_test_daily.index, _test_daily["predicted"], label="test predicted", linewidth=1.1)
        _axes[0].set_title("Daily-mean fit overview")
        _axes[0].legend(loc="upper right")
        _axes[0].tick_params(axis="x", rotation=45)

        _axes[1].plot(_test_residual.index, _test_residual["residual"], color="C3", linewidth=1.0)
        _axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        _axes[1].set_title("Test residuals")
        _axes[1].set_ylabel("m")
        _axes[1].tick_params(axis="x", rotation=45)

        _notes = []
        if fit_bundle["active_regressors"]:
            _notes.append(
                "Active regressors: " + ", ".join(fit_bundle["active_regressors"])
            )
        else:
            _notes.append("Active regressors: none (Fourier + optional trend only)")
        if fit_bundle["dropped_regressors"]:
            _notes.append(
                "Dropped after imputation check: " + ", ".join(fit_bundle["dropped_regressors"])
            )
        fit_tab = mo.vstack(
            [
                mo.md("\n".join(f"- {note}" for note in _notes)),
                _metric_table,
                mo.mpl.interactive(_fig),
            ]
        )
    return (fit_tab,)


@app.cell
def _(TIDAL_CONSTITUENT_PERIODS_HOURS, fit_bundle, mo, plt):
    if fit_bundle is None:
        residual_tab = mo.md("Run the model to compute the Fourier-only residual spectrum.")
    elif "error" in fit_bundle:
        residual_tab = mo.md(f"**Residual-spectrum error:** `{fit_bundle['error']}`")
    else:
        _spectrum = fit_bundle["residual_spectrum"]
        _peaks = fit_bundle["periodogram_peaks"]
        _harmonic_candidates = fit_bundle["harmonic_candidates"]

        _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
        _axes[0].plot(_spectrum["period_hours"], _spectrum["power"], linewidth=1.2, color="C0")
        _axes[0].set_title("Fourier-only residual spectrum")
        _axes[0].set_xlabel("period (hours)")
        _axes[0].set_ylabel("power")
        for _constituent_label, _period_hours in TIDAL_CONSTITUENT_PERIODS_HOURS.items():
            _axes[0].axvline(_period_hours, color="0.5", linestyle=":", linewidth=0.8)
            _axes[0].text(_period_hours, _axes[0].get_ylim()[1] * 0.88, _constituent_label, rotation=90, va="top", ha="right")

        _low_freq = _spectrum[_spectrum["period_hours"] >= 120.0]
        _axes[1].plot(_low_freq["period_hours"], _low_freq["power"], linewidth=1.2, color="C1")
        _axes[1].set_title("Low-frequency harmonic diagnostics")
        _axes[1].set_xlabel("period (hours)")
        _axes[1].set_ylabel("power")
        _selected_candidates = _harmonic_candidates[
            _harmonic_candidates["period_hours"].between(120.0, _low_freq["period_hours"].max() if not _low_freq.empty else 120.0)
        ]
        _selected_candidates = _selected_candidates.drop_duplicates(subset=["candidate"]).head(10)
        for _candidate_row in _selected_candidates.itertuples(index=False):
            _axes[1].axvline(_candidate_row.period_hours, color="0.6", linestyle=":", linewidth=0.8)
            _axes[1].text(_candidate_row.period_hours, _axes[1].get_ylim()[1] * 0.88, _candidate_row.candidate, rotation=90, va="top", ha="right")

        residual_tab = mo.vstack(
            [
                mo.md(
                    "Residual peaks are computed on the **Fourier-only** training residual so you can inspect "
                    "what periodic structure still appears to be missing."
                ),
                mo.mpl.interactive(_fig),
                _peaks,
            ]
        )
    return (residual_tab,)


@app.cell
def _(TIDAL_COMPONENT_LABELS, fit_bundle, mo, plt):
    if fit_bundle is None:
        components_tab = mo.md("Run the model to render the constituent decomposition preview.")
    elif "error" in fit_bundle:
        components_tab = mo.md(f"**Component error:** `{fit_bundle['error']}`")
    else:
        _component_index = fit_bundle["component_window_index"]
        _component_dict = fit_bundle["component_dict"]
        _observed = fit_bundle["component_window_observed"]
        _component_labels = [
            _component_label for _component_label in TIDAL_COMPONENT_LABELS if _component_label in _component_dict
        ]

        _n_rows = len(_component_labels) + 1
        _fig, _axes = plt.subplots(
            _n_rows,
            1,
            figsize=(14, max(6, 2.2 * _n_rows)),
            sharex=True,
            constrained_layout=True,
        )

        _preview_len = len(_component_index)
        _axes[0].plot(_component_index, _observed[:_preview_len], label="observed", linewidth=1.0)
        _axes[0].plot(
            _component_index,
            _component_dict["combined"][:_preview_len],
            label="combined Fourier basis",
            linewidth=1.0,
            color="C1",
        )
        _axes[0].set_title("Observed signal vs constituent-aware Fourier basis")
        _axes[0].legend(loc="upper right")

        for _axis, _component_label in zip(_axes[1:], _component_labels, strict=True):
            _axis.plot(_component_index, _component_dict[_component_label][:_preview_len], linewidth=1.0)
            _axis.set_title(_component_label)

        _axes[-1].tick_params(axis="x", rotation=45)
        components_tab = mo.mpl.interactive(_fig)
    return (components_tab,)


@app.cell
def _(
    alt,
    analysis_context,
    analysis_error,
    fit_bundle,
    mo,
    pd,
    regressor_select,
):
    if analysis_error is not None:
        weather_tab = mo.md(f"**Weather diagnostics unavailable:** `{analysis_error}`")
    else:
        _inventory = analysis_context["regressor_inventory"]
        _selected = list(regressor_select.value) if regressor_select.value else []
        _selected_line = (
            "Selected regressors: " + ", ".join(_selected)
            if _selected
            else "Selected regressors: none"
        )

        _adf = analysis_context["analysis_df"]
        _all_wx = [
            c for c in analysis_context["available_regressors"]
            if c in _adf.columns and _adf[c].notna().any()
        ]
        _wx_elements = []

        if _all_wx:
            _wx_df = _adf[_all_wx].copy()
            _wx_df.index.name = "datetime"
            if len(_wx_df) > 5000:
                _wx_df = _wx_df.resample("1h").mean()
            _wx_df = _wx_df.reset_index()
            _wx_long = _wx_df.melt(id_vars="datetime", var_name="variable", value_name="value")
            _wx_long = _wx_long.dropna(subset=["value"])
            _wx_long["selected"] = _wx_long["variable"].isin(_selected)

            _wx_chart = (
                alt.Chart(_wx_long)
                .mark_line(strokeWidth=1)
                .encode(
                    x=alt.X("datetime:T", title=None),
                    y=alt.Y("value:Q", title=""),
                    opacity=alt.condition(
                        alt.datum.selected,
                        alt.value(1.0),
                        alt.value(0.3),
                    ),
                    color=alt.Color("variable:N", legend=None),
                )
                .properties(width="container", height=100)
                .facet(
                    facet=alt.Facet("variable:N", title=None, sort=_all_wx),
                    columns=1,
                )
                .resolve_scale(y="independent")
            )
            _wx_elements.append(mo.ui.altair_chart(_wx_chart))

        if fit_bundle is None or "error" in fit_bundle or not fit_bundle.get("lag_correlations"):
            weather_tab = mo.vstack(
                [
                    mo.md(_selected_line),
                    _inventory,
                    *_wx_elements,
                    mo.md("Run the model to add lag-correlation diagnostics."),
                ]
            )
        else:
            _lag_rows = []
            for _column_name, _lag_corr in fit_bundle["lag_correlations"].items():
                for _, _row in _lag_corr.iterrows():
                    _lag_rows.append({
                        "lag_hours": _row["lag_hours"],
                        "correlation": _row["correlation"],
                        "variable": _column_name,
                    })
            _lag_df = pd.DataFrame(_lag_rows)
            _lag_chart = (
                alt.Chart(_lag_df)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("lag_hours:Q", title="lag (hours)"),
                    y=alt.Y("correlation:Q", title="correlation"),
                    color=alt.Color("variable:N"),
                )
                .properties(width="container", height=300, title="Lagged correlation with Fourier-only residual")
            )
            _zero_rule = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(color="black", strokeDash=[4, 4])
                .encode(y="y:Q")
            )

            weather_tab = mo.vstack(
                [
                    mo.md(_selected_line),
                    _inventory,
                    *_wx_elements,
                    fit_bundle["weather_summary"],
                    mo.ui.altair_chart(_lag_chart + _zero_rule),
                ]
            )
    return (weather_tab,)


@app.cell
def _(
    analysis_context,
    analysis_error,
    data_overview_plot,
    mo,
    overview_header,
    overview_plot,
):
    if analysis_error is not None:
        overview_tab = mo.md(f"**Overview unavailable:** `{analysis_error}`")
    else:
        overview_tab = mo.vstack(
            [
                overview_header,
                analysis_context["coverage_table"],
                overview_plot,
                data_overview_plot,
            ]
        )
    return (overview_tab,)


@app.cell
def _(components_tab, fit_tab, mo, overview_tab, residual_tab, weather_tab):
    mo.ui.tabs(
        {
            "Overview": overview_tab,
            "Fit": fit_tab,
            "Residual Spectrum": residual_tab,
            "Components": components_tab,
            "Weather": weather_tab,
        },
        lazy=False,
    )
    return


if __name__ == "__main__":
    app.run()
