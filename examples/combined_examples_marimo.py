# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Combined Marimo Notebook: TSGAM Examples

This notebook combines three example use cases:
1. Air Quality Forecasting (Beijing PM2.5)
2. LA Energy Demand Forecasting
3. PV/Solar Power Analysis

Select an example from the dropdown to load data and configure the model.
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Combined TSGAM Examples

    This notebook demonstrates TSGAM modeling with three different use cases:
    - **Air Quality**: Forecasting PM2.5 using meteorological variables
    - **LA Energy**: Forecasting energy demand using weather variables
    - **PV/Solar**: Analyzing solar power generation with temperature and irradiance

    Select an example below to get started.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from pathlib import Path
    import sys
    import urllib.request
    import zipfile
    from datetime import timedelta

    # Add src directory to path to import tsgam_estimator
    _project_root = Path(__file__).parent.parent
    _src_dir = _project_root / 'src'
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from tsgam_estimator import (
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiHarmonicConfig,
        TsgamSplineConfig,
        TsgamArConfig,
        TsgamSolverConfig,
        TsgamTrendConfig,
        TrendType,
        PERIOD_HOURLY_DAILY,
        PERIOD_HOURLY_WEEKLY,
        PERIOD_HOURLY_YEARLY,
    )

    # Import DataHandler for PV example
    try:
        from solardatatools import DataHandler
    except ImportError:
        DataHandler = None
        print("Warning: solardatatools not available. PV example will be limited.")
    return (
        DataHandler,
        PERIOD_HOURLY_DAILY,
        PERIOD_HOURLY_WEEKLY,
        PERIOD_HOURLY_YEARLY,
        Path,
        TrendType,
        TsgamArConfig,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiHarmonicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        TsgamTrendConfig,
        mo,
        np,
        pd,
        plt,
        sns,
        stats,
        timedelta,
        urllib,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Select Example & Configure Model

    Select an example below and click "Fit Model" to train. Data loads automatically.
    """)
    return


@app.cell
def _(mo):
    example_select = mo.ui.dropdown(
        options=['Air Quality (Beijing PM2.5)', 'LA Energy Demand', 'PV/Solar Power'],
        value='Air Quality (Beijing PM2.5)',
        label='Select Example'
    )
    fit_model_button = mo.ui.run_button(label='Fit Model')
    mo.hstack([example_select, fit_model_button], justify='start', gap=2)
    return example_select, fit_model_button


@app.cell
def _(
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
    TsgamArConfig,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    df_air_quality,
    example_select,
    fit_model_button,
    np,
    pd,
):
    # Air Quality model fitting
    if example_select.value == 'Air Quality (Beijing PM2.5)' and fit_model_button.value:
        # Pre-canned parameters
        _train_start_aq = '2012-01-01'
        _train_end_aq = '2013-12-31'
        _test_start_aq = '2014-01-01'
        _test_end_aq = '2014-03-31'

        _df_train_aq = df_air_quality[_train_start_aq:_train_end_aq].copy()
        _df_test_aq = df_air_quality[_test_start_aq:_test_end_aq].copy()

        _y_train_aq = np.log(_df_train_aq['pm25'].values + 1.0)
        _y_test_aq = np.log(_df_test_aq['pm25'].values + 1.0)

        X_train_aq = pd.DataFrame({
            'temperature': _df_train_aq['temperature'].values,
            'dewpoint': _df_train_aq['dewpoint'].values,
            'wind_speed': _df_train_aq['wind_speed'].values,
            'pressure': _df_train_aq['pressure'].values,
            'rain_hours': _df_train_aq['rain_hours'].values,
        }, index=_df_train_aq.index)

        X_test_aq = pd.DataFrame({
            'temperature': _df_test_aq['temperature'].values,
            'dewpoint': _df_test_aq['dewpoint'].values,
            'wind_speed': _df_test_aq['wind_speed'].values,
            'pressure': _df_test_aq['pressure'].values,
            'rain_hours': _df_test_aq['rain_hours'].values,
        }, index=_df_test_aq.index)

        # Multi-harmonic config
        # Using 30-day period for better capture of monthly patterns (matching standalone example exactly)
        _periods_list_aq = [PERIOD_HOURLY_YEARLY, float(30*24.0), float(PERIOD_HOURLY_WEEKLY), float(PERIOD_HOURLY_DAILY)]
        _num_harmonics_list_aq = [4, 4, 8, 4]
        _multi_harmonic_config_aq = TsgamMultiHarmonicConfig(
            num_harmonics=_num_harmonics_list_aq,
            periods=_periods_list_aq,
            reg_weight=6e-5
        )

        # Exogenous configs (pre-canned)
        # Note: Only non-negative lags (0, 1, 2, ...) are used for forecasting
        # Negative lags would use future data, which is not available for forecasting
        _exog_config_aq = [
            TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),  # temperature (n_knots=8 to avoid solver issues)
            TsgamSplineConfig(n_knots=10, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),  # dewpoint
            TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),  # wind_speed
            TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5),  # pressure
            TsgamSplineConfig(n_knots=6, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5),  # rain_hours
        ]

        # AR config
        _ar_config_aq = TsgamArConfig(lags=[1, 2, 3, 4], l1_constraint=0.97)

        # Solver config
        _solver_config_aq = TsgamSolverConfig(solver='CLARABEL', verbose=False)

        # Create config (matching standalone example exactly)
        _config_aq = TsgamEstimatorConfig(
            multi_harmonic_config=_multi_harmonic_config_aq,
            exog_config=_exog_config_aq,
            ar_config=_ar_config_aq,
            outlier_config=None,  # use_outlier defaults to False in standalone
            solver_config=_solver_config_aq,
            random_state=42,
            debug=False  # debug defaults to False in standalone
        )

        # Fit model
        print("Fitting Air Quality model...")
        estimator_aq = TsgamEstimator(config=_config_aq)
        estimator_aq.fit(X_train_aq, _y_train_aq)

        print(f"Model fitted! Status: {estimator_aq.problem_.status}")

        # Make predictions
        _predictions_aq_log = estimator_aq.predict(X_test_aq)
        predictions_aq = np.exp(_predictions_aq_log) - 1.0
        y_test_aq_orig = np.exp(_y_test_aq) - 1.0
        df_test_aq = _df_test_aq
        df_train_aq = _df_train_aq
        y_train_aq = _y_train_aq
    else:
        estimator_aq = None
        predictions_aq = None
        y_test_aq_orig = None
        X_train_aq = None
        X_test_aq = None
        df_test_aq = None
        df_train_aq = None
        y_train_aq = None
    return (
        X_train_aq,
        df_test_aq,
        df_train_aq,
        estimator_aq,
        predictions_aq,
        y_test_aq_orig,
        y_train_aq,
    )


@app.cell
def _(
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    create_train_test_split_with_gaps,
    df_la,
    example_select,
    fit_model_button,
    np,
):
    # LA Energy model fitting
    if example_select.value == 'LA Energy Demand' and fit_model_button.value and df_la is not None:
        # Pre-canned parameters
        _target_var_la = 'elec_total_MW'
        _train_mask_la, _test_mask_la = create_train_test_split_with_gaps(df_la, _target_var_la, holdout_days=7)

        _df_train_la = df_la[_train_mask_la].copy()
        _df_test_la = df_la[_test_mask_la].copy()

        _y_train_la = _df_train_la[_target_var_la].values
        _y_test_la = _df_test_la[_target_var_la].values
        _y_train_la_log = np.log(_y_train_la + 1.0)

        X_train_la = _df_train_la[['temperature_degF', 'humidity_pc']].copy()
        X_test_la = _df_test_la[['temperature_degF', 'humidity_pc']].copy()

        # Filter valid samples
        _train_valid_la = ~(_df_train_la[_target_var_la].isna() | X_train_la.isna().any(axis=1))
        _test_valid_la = ~(_df_test_la[_target_var_la].isna() | X_test_la.isna().any(axis=1))

        X_train_la = X_train_la[_train_valid_la]
        X_test_la = X_test_la[_test_valid_la]
        _y_train_la_log = _y_train_la_log[_train_valid_la]
        y_test_la_aligned = _df_test_la[_target_var_la].loc[X_test_la.index].values

        # Multi-harmonic config
        _multi_harmonic_config_la = TsgamMultiHarmonicConfig(
            num_harmonics=[4, 4, 6],
            periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
            reg_weight=6e-5
        )

        # Exogenous configs (pre-canned with requested lags)
        _exog_config_la = [
            TsgamSplineConfig(n_knots=10, lags=[-2, -1, 0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),  # temperature
            TsgamSplineConfig(n_knots=8, lags=[-2, -1, 0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),  # humidity
        ]

        # Solver config
        _solver_config_la = TsgamSolverConfig(solver='CLARABEL', verbose=False)

        # Create config
        _config_la = TsgamEstimatorConfig(
            multi_harmonic_config=_multi_harmonic_config_la,
            exog_config=_exog_config_la,
            ar_config=None,
            solver_config=_solver_config_la,
            random_state=42
        )

        # Fit model
        print("Fitting LA Energy model...")
        estimator_la = TsgamEstimator(config=_config_la)
        estimator_la.fit(X_train_la, _y_train_la_log)

        print(f"Model fitted! Status: {estimator_la.problem_.status}")

        # Make predictions
        _predictions_la_log = estimator_la.predict(X_test_la)
        predictions_la = np.exp(_predictions_la_log) - 1.0
    else:
        estimator_la = None
        predictions_la = None
        X_train_la = None
        X_test_la = None
        y_test_la_aligned = None
    return (
        X_test_la,
        X_train_la,
        estimator_la,
        predictions_la,
        y_test_la_aligned,
    )


@app.cell
def _(
    DataHandler,
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
    df_pv_raw,
    example_select,
    fit_model_button,
    mo,
    np,
    pd,
):
    # PV/Solar model fitting (matching standalone example exactly)
    if example_select.value == 'PV/Solar Power' and fit_model_button.value and df_pv_raw is not None and DataHandler is not None:
        # Pre-canned parameters - use all columns (matching standalone col_select.value pattern)
        _df_pv_cols_pv = list(df_pv_raw.columns)

        # Default to power, POA irradiance and temperature columns (matching script defaults)
        _default_primary_col_pv = 'inv_03_ac_power_inv_149593'
        _default_irrad_col_pv = 'poa_irradiance_o_149574'
        _default_temp_col_pv = 'ambient_temperature_o_149575'

        # Use default primary column if available, otherwise use first column
        if _default_primary_col_pv in _df_pv_cols_pv:
            _primary_col_pv = _default_primary_col_pv
        else:
            _primary_col_pv = _df_pv_cols_pv[0] if len(_df_pv_cols_pv) > 0 else None

        # Use defaults if available, otherwise try to identify columns
        if _default_irrad_col_pv in _df_pv_cols_pv:
            _irrad_col_pv = _default_irrad_col_pv
        else:
            _irrad_cols_pv = [c for c in _df_pv_cols_pv if 'irrad' in c.lower() or 'poa' in c.lower()]
            _irrad_col_pv = _irrad_cols_pv[0] if _irrad_cols_pv else _df_pv_cols_pv[2] if len(_df_pv_cols_pv) > 2 else None

        if _default_temp_col_pv in _df_pv_cols_pv:
            _module_temp_col_pv = _default_temp_col_pv
        else:
            _temp_cols_pv = [c for c in _df_pv_cols_pv if 'temp' in c.lower()]
            _module_temp_col_pv = _temp_cols_pv[0] if _temp_cols_pv else _df_pv_cols_pv[1] if len(_df_pv_cols_pv) > 1 else None

        # Create fresh DataHandler and run pipeline (matching standalone example exactly)
        if _primary_col_pv and df_pv_raw is not None and len(df_pv_raw) > 0:
            # Match standalone: if primary_select.value is None, use col_select.value[0]
            _pc = _primary_col_pv

            _dh_pv = DataHandler(df_pv_raw)
            _dh_pv.fix_dst()  # Always fix DST (matching standalone fix_dst_slct.value pattern)
            _lt = 0.1  # linearity_threshold (matching standalone lin_thresh_select.value)

            # Match standalone/script: extra_cols should only include temp and irrad columns
            _extra_cols_pv = [c for c in [_module_temp_col_pv, _irrad_col_pv] if c is not None and c != _pc]

            with mo.capture_stdout():
                if len(_extra_cols_pv) == 0:
                    _dh_pv.run_pipeline(power_col=_pc, max_val=2000, linearity_threshold=_lt)
                else:
                    _dh_pv.run_pipeline(power_col=_pc, max_val=2000, extra_cols=_extra_cols_pv, linearity_threshold=_lt)

            # Process temperature and irradiance matrices (matching standalone exactly)
            if _module_temp_col_pv and _module_temp_col_pv in _dh_pv.extra_matrices:
                _temp_mat_pv = _dh_pv.extra_matrices[_module_temp_col_pv]
                _temp_mat_pv[_temp_mat_pv > 140] = np.nan
            if _irrad_col_pv and _irrad_col_pv in _dh_pv.extra_matrices:
                _irrad_mat_pv = _dh_pv.extra_matrices[_irrad_col_pv]
                _irrad_mat_pv[_irrad_mat_pv < 0] = 0
        else:
            _dh_pv = None
            _module_temp_col_pv = None
            _irrad_col_pv = None

        # Prepare data (matching standalone example exactly)
        if _dh_pv is not None and _module_temp_col_pv and _irrad_col_pv:
            # Match standalone: data_start.value and data_end.value (default to full range)
            _data_start_pv = 0
            _data_end_pv = _dh_pv.raw_data_matrix.shape[1] - 1
            _data_select_pv = np.s_[_data_start_pv:_data_end_pv+1]

            # Prepare target (matching standalone exactly)
            _y_pv = np.copy(_dh_pv.raw_data_matrix)
            _y_max_pv = np.nanmax(_y_pv)  # Calculate max BEFORE filtering
            _y_pv[:, ~_dh_pv.daily_flags.no_errors] = np.nan
            _y_pv[~_dh_pv.boolean_masks.daytime] = np.nan
            _y_pv[_y_pv < 0.01 * np.nanmax(_y_pv)] = np.nan
            _y_pv = _y_pv[:, _data_select_pv].ravel(order='F')
            _y_pv /= _y_max_pv
            _y_pv[_y_pv < 0.0] = np.nan  # target_filter.value defaults to 0
            _y_pv = np.log(_y_pv)  # take_log defaults to True

            # Prepare exogenous variables (matching standalone exactly)
            _x1_pv = np.copy(_dh_pv.extra_matrices[_module_temp_col_pv][:, _data_select_pv].ravel(order='F'))
            _x1_avail_pv = ~np.isnan(_x1_pv)
            _x1_pv[~_x1_avail_pv] = 0
            _x1_max_pv = np.max(_x1_pv)
            _x1_pv /= _x1_max_pv

            _x2_pv = np.copy(_dh_pv.extra_matrices[_irrad_col_pv][:, _data_select_pv].ravel(order='F'))
            _x2_pv[_x2_pv < 0] = 0
            _x2_avail_pv = np.logical_and(~np.isnan(_x2_pv), _x2_pv > 0.02 * np.nanquantile(_x2_pv, 0.98))
            _x2_pv[~_x2_avail_pv] = 0
            _x2_max_pv = np.max(_x2_pv)
            _x2_pv /= _x2_max_pv

            # Ensure all arrays have the same length (matching standalone)
            _min_len_pv = min(len(_y_pv), len(_x1_pv), len(_x2_pv))
            _y_pv = _y_pv[:_min_len_pv]
            _x1_pv = _x1_pv[:_min_len_pv]
            _x2_pv = _x2_pv[:_min_len_pv]

            # Filter out NaN in y (matching standalone)
            _valid_mask_pv = ~np.isnan(_y_pv)
            _y_pv = _y_pv[_valid_mask_pv]
            _x1_pv = _x1_pv[_valid_mask_pv]
            _x2_pv = _x2_pv[_valid_mask_pv]

            # Create timestamps (matching standalone exactly)
            _m_pv, _n_pv = _dh_pv.raw_data_matrix.shape
            if hasattr(df_pv_raw, 'index') and len(df_pv_raw.index) > 0:
                _base_time_pv = df_pv_raw.index[0]
                _start_time_pv = _base_time_pv + pd.Timedelta(days=_data_start_pv)
            else:
                _start_time_pv = pd.Timestamp('2020-01-01 00:00:00')

            if not isinstance(_start_time_pv, pd.Timestamp):
                _start_time_pv = pd.Timestamp(_start_time_pv)

            _valid_len_pv = len(_y_pv)
            _timestamps_pv = pd.date_range(
                start=_start_time_pv,
                periods=_valid_len_pv,
                freq='15min'
            )

            _X_pv = pd.DataFrame({'temp': _x1_pv, 'irrad': _x2_pv}, index=_timestamps_pv)
        else:
            _y_pv = None
            _x1_pv = None
            _x2_pv = None
            _y_max_pv = None
            _valid_mask_pv = None
            _x1_max_pv = None
            _x2_max_pv = None
            _X_pv = None

        # Fit model (only if data is ready)
        if _dh_pv is not None and _y_pv is not None and len(_y_pv) > 0 and _X_pv is not None:
            # Multi-harmonic config
            _nvals_pv = _dh_pv.raw_data_matrix.shape[0]
            _period_daily_hours_pv = 24.0
            _period_yearly_hours_pv = 365.2425 * 24.0

            _multi_harmonic_config_pv = TsgamMultiHarmonicConfig(
                num_harmonics=[6, 10],
                periods=[_period_yearly_hours_pv, _period_daily_hours_pv],
                reg_weight=1e-2
            )

            # Exogenous configs (pre-canned)
            _exog_config_pv = [
                TsgamSplineConfig(n_knots=10, lags=[0], reg_weight=1e-4, diff_reg_weight=0.5),  # temp
                TsgamSplineConfig(n_knots=10, lags=[0], reg_weight=1e-4, diff_reg_weight=0.5),  # irrad
            ]

            # Trend config
            _trend_config_pv = TsgamTrendConfig(
                trend_type=TrendType.NONLINEAR,
                grouping=24.0,
                reg_weight=10.0
            )

            # Solver config
            _solver_config_pv = TsgamSolverConfig(solver='CLARABEL', verbose=False)

            # Create config
            _config_pv = TsgamEstimatorConfig(
                multi_harmonic_config=_multi_harmonic_config_pv,
                exog_config=_exog_config_pv,
                trend_config=_trend_config_pv,
                solver_config=_solver_config_pv
            )

            # Fit model
            print("Fitting PV/Solar model...")
            estimator_pv = TsgamEstimator(config=_config_pv)
            estimator_pv.fit(_X_pv, _y_pv)

            print(f"Model fitted! Status: {estimator_pv.problem_.status}")

            # Make predictions
            predictions_pv = estimator_pv.predict(_X_pv)
            valid_mask_pv = _valid_mask_pv
            x1_max_pv = _x1_max_pv
            x1_pv = _x1_pv
            x2_max_pv = _x2_max_pv
            x2_pv = _x2_pv
            y_max_pv = _y_max_pv
            y_pv = _y_pv
        else:
            estimator_pv = None
            predictions_pv = None
            valid_mask_pv = None
            x1_max_pv = None
            x1_pv = None
            x2_max_pv = None
            x2_pv = None
            y_max_pv = None
            y_pv = None
    return (
        estimator_pv,
        predictions_pv,
        valid_mask_pv,
        x1_max_pv,
        x1_pv,
        x2_max_pv,
        x2_pv,
        y_max_pv,
        y_pv,
    )


@app.cell
def _(
    calculate_metrics,
    estimator_aq,
    estimator_la,
    estimator_pv,
    example_select,
    format_metrics_report,
    mo,
    np,
    predictions_aq,
    predictions_la,
    predictions_pv,
    y_max_pv,
    y_pv,
    y_test_aq_orig,
    y_test_la_aligned,
):
    # Consolidated metrics report for all examples
    _report = None

    if example_select.value == 'Air Quality (Beijing PM2.5)' and estimator_aq is not None and predictions_aq is not None:
        _metrics = calculate_metrics(predictions_aq, y_test_aq_orig)
        _report = mo.md(format_metrics_report("Air Quality (Beijing PM2.5)", _metrics, "μg/m³"))

    elif example_select.value == 'LA Energy Demand' and estimator_la is not None and predictions_la is not None:
        _metrics = calculate_metrics(predictions_la, y_test_la_aligned)
        _report = mo.md(format_metrics_report("LA Energy Demand", _metrics, "MW"))

    elif example_select.value == 'PV/Solar Power' and estimator_pv is not None and predictions_pv is not None:
        # Convert from log space to original space for metrics
        _y_orig_pv = y_max_pv * np.exp(y_pv)
        _pred_orig_pv = y_max_pv * np.exp(predictions_pv)
        _metrics = calculate_metrics(_pred_orig_pv, _y_orig_pv)
        _report = mo.md(format_metrics_report("PV/Solar Power", _metrics))

    _report
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualizations

    Charts will be displayed based on the selected example.
    """)
    return


@app.cell
def _(
    X_train_aq,
    df_test_aq,
    df_train_aq,
    estimator_aq,
    example_select,
    mo,
    np,
    plt,
    predictions_aq,
    y_test_aq_orig,
    y_train_aq,
):
    if example_select.value == 'Air Quality (Beijing PM2.5)' and estimator_aq is not None:
        # Air Quality visualizations
        _fig_aq, _axes_aq = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # Time series
        _ax1_aq = _axes_aq[0, 0]
        _test_idx_aq = df_test_aq.index[:len(y_test_aq_orig)]
        _ax1_aq.plot(df_train_aq.index, np.exp(y_train_aq) - 1.0, 'b-', alpha=0.6, label='Training', linewidth=0.5)
        _ax1_aq.plot(_test_idx_aq, y_test_aq_orig, 'g-', alpha=0.7, label='Actual (test)', linewidth=1)
        _ax1_aq.plot(_test_idx_aq, predictions_aq[:len(y_test_aq_orig)], 'r-', label='Forecast', linewidth=1.5)
        _ax1_aq.set_ylabel('PM2.5 (μg/m³)', fontsize=11)
        _ax1_aq.set_title('Air Quality Forecast', fontsize=12, fontweight='bold')
        _ax1_aq.legend()
        _ax1_aq.grid(True, alpha=0.3)

        # Scatter plot
        _ax2_aq = _axes_aq[0, 1]
        _valid_aq = np.isfinite(predictions_aq) & np.isfinite(y_test_aq_orig) & (y_test_aq_orig > 0)
        if np.any(_valid_aq):
            _ax2_aq.scatter(y_test_aq_orig[_valid_aq], predictions_aq[_valid_aq], alpha=0.5, s=1)
            _min_val_aq = min(y_test_aq_orig[_valid_aq].min(), predictions_aq[_valid_aq].min())
            _max_val_aq = max(y_test_aq_orig[_valid_aq].max(), predictions_aq[_valid_aq].max())
            _ax2_aq.plot([_min_val_aq, _max_val_aq], [_min_val_aq, _max_val_aq], 'r--', linewidth=2)
        _ax2_aq.set_xlabel('Actual (μg/m³)', fontsize=11)
        _ax2_aq.set_ylabel('Predicted (μg/m³)', fontsize=11)
        _ax2_aq.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
        _ax2_aq.grid(True, alpha=0.3)

        # Residuals
        _ax3_aq = _axes_aq[1, 0]
        if np.any(_valid_aq):
            _residuals_aq = y_test_aq_orig[_valid_aq] - predictions_aq[_valid_aq]
            _ax3_aq.plot(_test_idx_aq[_valid_aq], _residuals_aq, 'k-', alpha=0.6, linewidth=0.5)
        _ax3_aq.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        _ax3_aq.set_xlabel('Date', fontsize=11)
        _ax3_aq.set_ylabel('Residual (μg/m³)', fontsize=11)
        _ax3_aq.set_title('Residuals', fontsize=12, fontweight='bold')
        _ax3_aq.grid(True, alpha=0.3)

        # Response functions
        _ax4_aq = _axes_aq[1, 1]
        if hasattr(estimator_aq, 'variables_') and 'exog_coef_0' in estimator_aq.variables_:
            _exog_coef_aq = estimator_aq.variables_['exog_coef_0'].value
            if _exog_coef_aq is not None and estimator_aq.exog_knots_ and len(estimator_aq.exog_knots_) > 0:
                _knots_aq = estimator_aq.exog_knots_[0]
                _x_vals_aq = X_train_aq['temperature'].values
                _H_aq = estimator_aq._make_H(_x_vals_aq, _knots_aq, include_offset=False)
                _log_response_aq = _H_aq @ _exog_coef_aq[:, 0]
                _ax4_aq.scatter(_x_vals_aq, _log_response_aq, s=1, alpha=0.3)
                _ax4_aq.axhline(y=0, color='r', linestyle='--', linewidth=1)
                _ax4_aq.set_xlabel('Temperature (°C)', fontsize=11)
                _ax4_aq.set_ylabel('Log Response', fontsize=11)
                _ax4_aq.set_title('Temperature Response Function', fontsize=12, fontweight='bold')
                _ax4_aq.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        _fig_aq = plt.figure(figsize=(10, 6))
        _ax_placeholder_aq = _fig_aq.add_subplot(111)
        _ax_placeholder_aq.text(0.5, 0.5, 'Select "Air Quality (Beijing PM2.5)" and fit the model to see visualizations.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder_aq.transAxes)
        _ax_placeholder_aq.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(
    X_test_la,
    X_train_la,
    estimator_la,
    example_select,
    mo,
    np,
    plt,
    predictions_la,
    y_test_la_aligned,
):
    if example_select.value == 'LA Energy Demand' and estimator_la is not None:
        # LA Energy visualizations
        _fig_la, _axes_la = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # Time series
        _ax1_la = _axes_la[0, 0]
        _test_idx_la = X_test_la.index[:len(y_test_la_aligned)]
        _ax1_la.plot(_test_idx_la, y_test_la_aligned, 'b-', alpha=0.7, label='Actual', linewidth=1)
        _ax1_la.plot(_test_idx_la, predictions_la[:len(y_test_la_aligned)], 'r-', label='Predicted', linewidth=1.5)
        _ax1_la.set_ylabel('Energy (MW)', fontsize=11)
        _ax1_la.set_title('LA Energy Forecast', fontsize=12, fontweight='bold')
        _ax1_la.legend()
        _ax1_la.grid(True, alpha=0.3)

        # Scatter plot
        _ax2_la = _axes_la[0, 1]
        _valid_la = np.isfinite(predictions_la) & np.isfinite(y_test_la_aligned) & (y_test_la_aligned > 0)
        if np.any(_valid_la):
            _ax2_la.scatter(y_test_la_aligned[_valid_la], predictions_la[_valid_la], alpha=0.5, s=10)
            _min_val_la = min(y_test_la_aligned[_valid_la].min(), predictions_la[_valid_la].min())
            _max_val_la = max(y_test_la_aligned[_valid_la].max(), predictions_la[_valid_la].max())
            _ax2_la.plot([_min_val_la, _max_val_la], [_min_val_la, _max_val_la], 'r--', linewidth=2)
        _ax2_la.set_xlabel('Actual (MW)', fontsize=11)
        _ax2_la.set_ylabel('Predicted (MW)', fontsize=11)
        _ax2_la.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
        _ax2_la.grid(True, alpha=0.3)

        # Temperature response
        _ax3_la = _axes_la[1, 0]
        if hasattr(estimator_la, 'variables_') and 'exog_coef_0' in estimator_la.variables_:
            _exog_coef_la = estimator_la.variables_['exog_coef_0'].value
            if _exog_coef_la is not None and estimator_la.exog_knots_ and len(estimator_la.exog_knots_) > 0:
                _knots_la = estimator_la.exog_knots_[0]
                _x_vals_la = X_train_la['temperature_degF'].values
                _H_la = estimator_la._make_H(_x_vals_la, _knots_la, include_offset=False)
                _log_response_la = _H_la @ _exog_coef_la[:, 0]  # Use lag 0
                _ax3_la.scatter(_x_vals_la, _log_response_la, s=1, alpha=0.3)
                _ax3_la.axhline(y=0, color='r', linestyle='--', linewidth=1)
                _ax3_la.set_xlabel('Temperature (°F)', fontsize=11)
                _ax3_la.set_ylabel('Log Response', fontsize=11)
                _ax3_la.set_title('Temperature Response Function', fontsize=12, fontweight='bold')
                _ax3_la.grid(True, alpha=0.3)

        # Humidity response
        _ax4_la = _axes_la[1, 1]
        if hasattr(estimator_la, 'variables_') and 'exog_coef_1' in estimator_la.variables_:
            _exog_coef_la_hum = estimator_la.variables_['exog_coef_1'].value
            if _exog_coef_la_hum is not None and estimator_la.exog_knots_ and len(estimator_la.exog_knots_) > 1:
                _knots_la_hum = estimator_la.exog_knots_[1]
                _x_vals_la_hum = X_train_la['humidity_pc'].values
                _H_la_hum = estimator_la._make_H(_x_vals_la_hum, _knots_la_hum, include_offset=False)
                _log_response_la_hum = _H_la_hum @ _exog_coef_la_hum[:, 0]  # Use lag 0
                _ax4_la.scatter(_x_vals_la_hum, _log_response_la_hum, s=1, alpha=0.3, color='green')
                _ax4_la.axhline(y=0, color='r', linestyle='--', linewidth=1)
                _ax4_la.set_xlabel('Humidity (%)', fontsize=11)
                _ax4_la.set_ylabel('Log Response', fontsize=11)
                _ax4_la.set_title('Humidity Response Function', fontsize=12, fontweight='bold')
                _ax4_la.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        _fig_la = plt.figure(figsize=(10, 6))
        _ax_placeholder_la = _fig_la.add_subplot(111)
        _ax_placeholder_la.text(0.5, 0.5, 'Select "LA Energy Demand" and fit the model to see visualizations.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder_la.transAxes)
        _ax_placeholder_la.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(
    estimator_pv,
    example_select,
    mo,
    np,
    plt,
    predictions_pv,
    valid_mask_pv,
    x1_max_pv,
    x1_pv,
    x2_max_pv,
    x2_pv,
    y_max_pv,
    y_pv,
):
    if example_select.value == 'PV/Solar Power' and estimator_pv is not None:
        # PV/Solar visualizations
        _fig_pv, _axes_pv = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # Model fit scatter
        _ax1_pv = _axes_pv[0, 0]
        _ax1_pv.scatter(predictions_pv, y_pv, marker='.', s=1, alpha=0.5)
        _xlim_pv = _ax1_pv.get_xlim()
        _ylim_pv = _ax1_pv.get_ylim()
        _ax1_pv.plot([-1e4, 1e4], [-1e4, 1e4], color='red', ls='--', linewidth=1)
        _ax1_pv.set_xlim(_xlim_pv)
        _ax1_pv.set_ylim(_ylim_pv)
        _ax1_pv.set_xlabel('Predicted (log space)', fontsize=11)
        _ax1_pv.set_ylabel('Actual (log space)', fontsize=11)
        _ax1_pv.set_title('Model Fit', fontsize=12, fontweight='bold')
        _ax1_pv.grid(True, alpha=0.3)

        # Original space
        _ax2_pv = _axes_pv[0, 1]
        _y_orig_pv = y_max_pv * np.exp(y_pv)
        _pred_orig_pv = y_max_pv * np.exp(predictions_pv)
        _ax2_pv.scatter(_pred_orig_pv, _y_orig_pv, marker='.', s=1, alpha=0.5)
        _xlim_pv2 = _ax2_pv.get_xlim()
        _ylim_pv2 = _ax2_pv.get_ylim()
        _ax2_pv.plot([-1e4, 1e4], [-1e4, 1e4], color='red', ls='--', linewidth=1)
        _ax2_pv.set_xlim(_xlim_pv2)
        _ax2_pv.set_ylim(_ylim_pv2)
        _ax2_pv.set_xlabel('Predicted', fontsize=11)
        _ax2_pv.set_ylabel('Actual', fontsize=11)
        _ax2_pv.set_title('Model Fit (Original Space)', fontsize=12, fontweight='bold')
        _ax2_pv.grid(True, alpha=0.3)

        # Temperature response
        _ax3_pv = _axes_pv[1, 0]
        if hasattr(estimator_pv, 'variables_') and 'exog_coef_0' in estimator_pv.variables_:
            _exog_coef_pv = estimator_pv.variables_['exog_coef_0'].value
            if _exog_coef_pv is not None and estimator_pv.exog_knots_ and len(estimator_pv.exog_knots_) > 0:
                _knots_pv = estimator_pv.exog_knots_[0]
                _x_vals_pv = x1_pv[valid_mask_pv] if len(x1_pv) == len(valid_mask_pv) else x1_pv
                _H_pv = estimator_pv._make_H(_x_vals_pv, _knots_pv, include_offset=False)
                _log_response_pv = _H_pv @ _exog_coef_pv[:, 0]
                _ax3_pv.scatter(_x_vals_pv * x1_max_pv, np.exp(_log_response_pv), s=1, alpha=0.3)
                _ax3_pv.set_xlabel('Temperature (normalized)', fontsize=11)
                _ax3_pv.set_ylabel('Correction Factor', fontsize=11)
                _ax3_pv.set_title('Temperature Response', fontsize=12, fontweight='bold')
                _ax3_pv.grid(True, alpha=0.3)

        # Irradiance response
        _ax4_pv = _axes_pv[1, 1]
        if hasattr(estimator_pv, 'variables_') and 'exog_coef_1' in estimator_pv.variables_:
            _exog_coef_pv_irr = estimator_pv.variables_['exog_coef_1'].value
            if _exog_coef_pv_irr is not None and estimator_pv.exog_knots_ and len(estimator_pv.exog_knots_) > 1:
                _knots_pv_irr = estimator_pv.exog_knots_[1]
                _x_vals_pv_irr = x2_pv[valid_mask_pv] if len(x2_pv) == len(valid_mask_pv) else x2_pv
                _H_pv_irr = estimator_pv._make_H(_x_vals_pv_irr, _knots_pv_irr, include_offset=False)
                _log_response_pv_irr = _H_pv_irr @ _exog_coef_pv_irr[:, 0]
                _ax4_pv.scatter(_x_vals_pv_irr * x2_max_pv, np.exp(_log_response_pv_irr), s=1, alpha=0.3, color='orange')
                _ax4_pv.set_xlabel('Irradiance (normalized)', fontsize=11)
                _ax4_pv.set_ylabel('Correction Factor', fontsize=11)
                _ax4_pv.set_title('Irradiance Response', fontsize=12, fontweight='bold')
                _ax4_pv.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        _fig_pv = plt.figure(figsize=(10, 6))
        _ax_placeholder_pv = _fig_pv.add_subplot(111)
        _ax_placeholder_pv.text(0.5, 0.5, 'Select "PV/Solar Power" and fit the model to see visualizations.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder_pv.transAxes)
        _ax_placeholder_pv.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(
    X_train_aq,
    estimator_aq,
    example_select,
    mo,
    np,
    plt,
    predictions_aq,
    stats,
    y_test_aq_orig,
):
    # Air Quality: Expanded visualizations - Response Functions & Residual Analysis
    if example_select.value == 'Air Quality (Beijing PM2.5)' and estimator_aq is not None:
        _fig_aq2, _axes_aq2 = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

        # Response functions for all variables
        _var_names_aq = ['temperature', 'dewpoint', 'wind_speed', 'pressure', 'rain_hours']
        _var_labels_aq = ['Temperature (°C)', 'Dewpoint (°C)', 'Wind Speed (m/s)', 'Pressure (hPa)', 'Rain Hours']
        _colors_aq = ['coral', 'dodgerblue', 'green', 'purple', 'brown']

        for _idx, (_var_name, _var_label, _color) in enumerate(zip(_var_names_aq, _var_labels_aq, _colors_aq)):
            _ax = _axes_aq2.flatten()[_idx]
            _var_key = f'exog_coef_{_idx}'
            if hasattr(estimator_aq, 'variables_') and _var_key in estimator_aq.variables_:
                _exog_coef = estimator_aq.variables_[_var_key].value
                if _exog_coef is not None and estimator_aq.exog_knots_ and len(estimator_aq.exog_knots_) > _idx:
                    _knots = estimator_aq.exog_knots_[_idx]
                    _x_vals = X_train_aq[_var_name].values
                    _H = estimator_aq._make_H(_x_vals, _knots, include_offset=False)
                    _log_response = _H @ _exog_coef[:, 0]
                    _ax.scatter(_x_vals, _log_response, s=1, alpha=0.3, color=_color)
                    _ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
                    _ax.set_xlabel(_var_label, fontsize=10)
                    _ax.set_ylabel('Log Response', fontsize=10)
                    _ax.set_title(f'{_var_name.replace("_", " ").title()} Response', fontsize=11, fontweight='bold')
                    _ax.grid(True, alpha=0.3)

        # Residual distribution (6th subplot)
        _ax_resid = _axes_aq2.flatten()[5]
        _valid_aq2 = np.isfinite(predictions_aq) & np.isfinite(y_test_aq_orig) & (y_test_aq_orig > 0)
        if np.any(_valid_aq2):
            _residuals = y_test_aq_orig[_valid_aq2] - predictions_aq[_valid_aq2]
            _ax_resid.hist(_residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            # Fit and plot distributions
            _xs = np.linspace(np.min(_residuals), np.max(_residuals), 200)
            _lap_loc, _lap_scale = stats.laplace.fit(_residuals)
            _nor_loc, _nor_scale = stats.norm.fit(_residuals)
            _ax_resid.plot(_xs, stats.laplace.pdf(_xs, _lap_loc, _lap_scale), 'dodgerblue', linewidth=2, label='Laplace fit')
            _ax_resid.plot(_xs, stats.norm.pdf(_xs, _nor_loc, _nor_scale), 'lime', linewidth=2, label='Normal fit')
            _ax_resid.axvline(x=0, color='r', linestyle='--', linewidth=2)
            _ax_resid.set_xlabel('Residual (μg/m³)', fontsize=10)
            _ax_resid.set_ylabel('Density', fontsize=10)
            _ax_resid.set_title('Residual Distribution', fontsize=11, fontweight='bold')
            _ax_resid.legend(fontsize=9)
            _ax_resid.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        _fig_aq2 = plt.figure(figsize=(10, 6))
        _ax_placeholder = _fig_aq2.add_subplot(111)
        _ax_placeholder.text(0.5, 0.5, 'Fit Air Quality model to see response functions.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder.transAxes)
        _ax_placeholder.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(df_air_quality, example_select, mo, plt, sns):
    # Air Quality: Data Heatmaps
    if example_select.value == 'Air Quality (Beijing PM2.5)' and df_air_quality is not None:
        # Create heatmap data
        _df_reshaped = df_air_quality.copy()
        _df_reshaped['day'] = (_df_reshaped.index - _df_reshaped.index.min()).days
        _df_reshaped['hour'] = _df_reshaped.index.hour

        _fig_hm, _axes_hm = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # PM2.5 heatmap
        _heatmap_pm25 = _df_reshaped.pivot_table(index='day', columns='hour', values='pm25', aggfunc='mean')
        sns.heatmap(_heatmap_pm25, cmap='Reds', ax=_axes_hm[0, 0], cbar_kws={'label': 'PM2.5 (μg/m³)'})
        _axes_hm[0, 0].set_title('PM2.5: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_hm[0, 0].set_xlabel('Hour of Day')
        _axes_hm[0, 0].set_ylabel('Day')
        _axes_hm[0, 0].tick_params(axis='y', labelsize=6)

        # Temperature heatmap
        _heatmap_temp = _df_reshaped.pivot_table(index='day', columns='hour', values='temperature', aggfunc='mean')
        sns.heatmap(_heatmap_temp, cmap='coolwarm', ax=_axes_hm[0, 1], cbar_kws={'label': 'Temperature (°C)'})
        _axes_hm[0, 1].set_title('Temperature: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_hm[0, 1].set_xlabel('Hour of Day')
        _axes_hm[0, 1].set_ylabel('Day')
        _axes_hm[0, 1].tick_params(axis='y', labelsize=6)

        # Dewpoint heatmap
        _heatmap_dewp = _df_reshaped.pivot_table(index='day', columns='hour', values='dewpoint', aggfunc='mean')
        sns.heatmap(_heatmap_dewp, cmap='Blues', ax=_axes_hm[1, 0], cbar_kws={'label': 'Dewpoint (°C)'})
        _axes_hm[1, 0].set_title('Dewpoint: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_hm[1, 0].set_xlabel('Hour of Day')
        _axes_hm[1, 0].set_ylabel('Day')
        _axes_hm[1, 0].tick_params(axis='y', labelsize=6)

        # Wind speed heatmap
        _heatmap_wind = _df_reshaped.pivot_table(index='day', columns='hour', values='wind_speed', aggfunc='mean')
        sns.heatmap(_heatmap_wind, cmap='Greens', ax=_axes_hm[1, 1], cbar_kws={'label': 'Wind Speed (m/s)'})
        _axes_hm[1, 1].set_title('Wind Speed: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_hm[1, 1].set_xlabel('Hour of Day')
        _axes_hm[1, 1].set_ylabel('Day')
        _axes_hm[1, 1].tick_params(axis='y', labelsize=6)

        plt.tight_layout()
    else:
        _fig_hm = plt.figure(figsize=(10, 6))
        _ax_placeholder = _fig_hm.add_subplot(111)
        _ax_placeholder.text(0.5, 0.5, 'Select "Air Quality" to see data heatmaps.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder.transAxes)
        _ax_placeholder.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(
    X_test_la,
    X_train_la,
    estimator_la,
    example_select,
    mo,
    np,
    plt,
    predictions_la,
    stats,
    y_test_la_aligned,
):
    # LA Energy: Expanded visualizations - Residual Analysis & Fourier Contributions
    if example_select.value == 'LA Energy Demand' and estimator_la is not None:
        _fig_la2, _axes_la2 = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # Residuals over time
        _ax1 = _axes_la2[0, 0]
        _valid_la2 = np.isfinite(predictions_la) & np.isfinite(y_test_la_aligned) & (y_test_la_aligned > 0)
        if np.any(_valid_la2):
            _residuals_la = y_test_la_aligned[_valid_la2] - predictions_la[_valid_la2]
            _test_idx_la2 = X_test_la.index[:len(y_test_la_aligned)][_valid_la2]
            _ax1.plot(_test_idx_la2, _residuals_la, 'b-', linewidth=0.5, alpha=0.7)
            _ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
            _ax1.set_ylabel('Residual (MW)', fontsize=10)
            _ax1.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
            _ax1.grid(True, alpha=0.3)

        # Residual distribution
        _ax2 = _axes_la2[0, 1]
        if np.any(_valid_la2):
            _ax2.hist(_residuals_la, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            # Fit and plot distributions
            _xs_la = np.linspace(np.min(_residuals_la), np.max(_residuals_la), 200)
            _lap_loc_la, _lap_scale_la = stats.laplace.fit(_residuals_la)
            _nor_loc_la, _nor_scale_la = stats.norm.fit(_residuals_la)
            _ax2.plot(_xs_la, stats.laplace.pdf(_xs_la, _lap_loc_la, _lap_scale_la), 'dodgerblue', linewidth=2, label='Laplace fit')
            _ax2.plot(_xs_la, stats.norm.pdf(_xs_la, _nor_loc_la, _nor_scale_la), 'lime', linewidth=2, label='Normal fit')
            _ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
            _ax2.set_xlabel('Residual (MW)', fontsize=10)
            _ax2.set_ylabel('Density', fontsize=10)
            _ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
            _ax2.legend(fontsize=9)
            _ax2.grid(True, alpha=0.3)

        # Fourier/harmonic contribution
        _ax3 = _axes_la2[1, 0]
        if hasattr(estimator_la, 'variables_') and 'fourier_coef' in estimator_la.variables_:
            _fourier_coef = estimator_la.variables_['fourier_coef'].value
            if _fourier_coef is not None and hasattr(estimator_la, 'F_') and estimator_la.F_ is not None:
                _fourier_contrib = estimator_la.F_ @ _fourier_coef
                _n_plot = min(len(_fourier_contrib), 2000)
                _ax3.plot(X_train_la.index[:_n_plot], _fourier_contrib[:_n_plot], 'b-', linewidth=0.5, alpha=0.7)
                _ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                _ax3.set_ylabel('Fourier Contribution (log)', fontsize=10)
                _ax3.set_title('Seasonal/Harmonic Pattern', fontsize=12, fontweight='bold')
                _ax3.grid(True, alpha=0.3)

        # Scatter: Residuals vs Temperature
        _ax4 = _axes_la2[1, 1]
        if np.any(_valid_la2):
            _temp_test = X_test_la['temperature_degF'].values[:len(y_test_la_aligned)][_valid_la2]
            _ax4.scatter(_temp_test, _residuals_la, s=5, alpha=0.5, color='coral')
            _ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
            _ax4.set_xlabel('Temperature (°F)', fontsize=10)
            _ax4.set_ylabel('Residual (MW)', fontsize=10)
            _ax4.set_title('Residuals vs Temperature', fontsize=12, fontweight='bold')
            _ax4.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        _fig_la2 = plt.figure(figsize=(10, 6))
        _ax_placeholder = _fig_la2.add_subplot(111)
        _ax_placeholder.text(0.5, 0.5, 'Fit LA Energy model to see residual analysis.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder.transAxes)
        _ax_placeholder.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(df_la, example_select, mo, plt, sns):
    # LA Energy: Data Heatmaps
    if example_select.value == 'LA Energy Demand' and df_la is not None:
        _df_la_copy = df_la.copy()
        _df_la_copy['day'] = (_df_la_copy.index - _df_la_copy.index.min()).days
        _df_la_copy['hour'] = _df_la_copy.index.hour

        _fig_la_hm, _axes_la_hm = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Energy demand heatmap
        _heatmap_energy = _df_la_copy.pivot_table(index='day', columns='hour', values='elec_total_MW', aggfunc='mean')
        sns.heatmap(_heatmap_energy, cmap='viridis', ax=_axes_la_hm[0], cbar_kws={'label': 'Energy (MW)'})
        _axes_la_hm[0].set_title('Energy Demand: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_la_hm[0].set_xlabel('Hour of Day')
        _axes_la_hm[0].set_ylabel('Day')
        _axes_la_hm[0].tick_params(axis='y', labelsize=6)

        # Temperature heatmap
        _heatmap_temp_la = _df_la_copy.pivot_table(index='day', columns='hour', values='temperature_degF', aggfunc='mean')
        sns.heatmap(_heatmap_temp_la, cmap='plasma', ax=_axes_la_hm[1], cbar_kws={'label': 'Temperature (°F)'})
        _axes_la_hm[1].set_title('Temperature: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_la_hm[1].set_xlabel('Hour of Day')
        _axes_la_hm[1].set_ylabel('Day')
        _axes_la_hm[1].tick_params(axis='y', labelsize=6)

        # Humidity heatmap
        _heatmap_humid = _df_la_copy.pivot_table(index='day', columns='hour', values='humidity_pc', aggfunc='mean')
        sns.heatmap(_heatmap_humid, cmap='Blues', ax=_axes_la_hm[2], cbar_kws={'label': 'Humidity (%)'})
        _axes_la_hm[2].set_title('Humidity: Days vs Hours', fontsize=12, fontweight='bold')
        _axes_la_hm[2].set_xlabel('Hour of Day')
        _axes_la_hm[2].set_ylabel('Day')
        _axes_la_hm[2].tick_params(axis='y', labelsize=6)

        plt.tight_layout()
    else:
        _fig_la_hm = plt.figure(figsize=(10, 6))
        _ax_placeholder = _fig_la_hm.add_subplot(111)
        _ax_placeholder.text(0.5, 0.5, 'Select "LA Energy Demand" to see data heatmaps.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder.transAxes)
        _ax_placeholder.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(estimator_pv, example_select, mo, np, plt, predictions_pv, stats, y_pv):
    # PV/Solar: Expanded visualizations - Residuals & Trend
    if example_select.value == 'PV/Solar Power' and estimator_pv is not None:
        _fig_pv2, _axes_pv2 = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        # Residuals in log space
        _ax1_pv2 = _axes_pv2[0, 0]
        _residuals_pv_log = y_pv - predictions_pv
        _ax1_pv2.plot(_residuals_pv_log, 'b-', linewidth=0.3, alpha=0.5)
        _ax1_pv2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        _ax1_pv2.set_xlabel('Sample Index', fontsize=10)
        _ax1_pv2.set_ylabel('Residual (log space)', fontsize=10)
        _ax1_pv2.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        _ax1_pv2.grid(True, alpha=0.3)

        # Residual distribution
        _ax2_pv2 = _axes_pv2[0, 1]
        _valid_resid = np.isfinite(_residuals_pv_log)
        _resid_valid_pv = _residuals_pv_log[_valid_resid]
        _ax2_pv2.hist(_resid_valid_pv, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        # Fit and plot distributions
        _xs_pv = np.linspace(np.min(_resid_valid_pv), np.max(_resid_valid_pv), 200)
        _lap_loc_pv, _lap_scale_pv = stats.laplace.fit(_resid_valid_pv)
        _nor_loc_pv, _nor_scale_pv = stats.norm.fit(_resid_valid_pv)
        _ax2_pv2.plot(_xs_pv, stats.laplace.pdf(_xs_pv, _lap_loc_pv, _lap_scale_pv), 'dodgerblue', linewidth=2, label='Laplace fit')
        _ax2_pv2.plot(_xs_pv, stats.norm.pdf(_xs_pv, _nor_loc_pv, _nor_scale_pv), 'lime', linewidth=2, label='Normal fit')
        _ax2_pv2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        _ax2_pv2.set_xlabel('Residual (log space)', fontsize=10)
        _ax2_pv2.set_ylabel('Density', fontsize=10)
        _ax2_pv2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        _ax2_pv2.legend(fontsize=9)
        _ax2_pv2.grid(True, alpha=0.3)

        # Trend/degradation over time
        _ax3_pv2 = _axes_pv2[1, 0]
        if hasattr(estimator_pv, 'variables_') and 'trend' in estimator_pv.variables_:
            _trend = estimator_pv.variables_['trend'].value
            if _trend is not None:
                _years = np.arange(len(_trend)) / 365.0
                _ax3_pv2.plot(_years, np.exp(_trend), 'b-', linewidth=2)
                _ax3_pv2.set_xlabel('Time (years)', fontsize=10)
                _ax3_pv2.set_ylabel('Trend Factor', fontsize=10)
                _ax3_pv2.set_title('Degradation/Trend Over Time', fontsize=12, fontweight='bold')
                _ax3_pv2.grid(True, alpha=0.3)
        else:
            _ax3_pv2.text(0.5, 0.5, 'No trend component', ha='center', va='center', fontsize=12)
            _ax3_pv2.axis('off')

        # Fourier/seasonal contribution
        _ax4_pv2 = _axes_pv2[1, 1]
        if hasattr(estimator_pv, 'variables_') and 'fourier_coef' in estimator_pv.variables_:
            _fourier_coef_pv = estimator_pv.variables_['fourier_coef'].value
            if _fourier_coef_pv is not None and hasattr(estimator_pv, 'F_') and estimator_pv.F_ is not None:
                _fourier_contrib_pv = estimator_pv.F_ @ _fourier_coef_pv
                _n_plot_pv = min(len(_fourier_contrib_pv), 5000)
                _ax4_pv2.plot(_fourier_contrib_pv[:_n_plot_pv], 'b-', linewidth=0.3, alpha=0.7)
                _ax4_pv2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                _ax4_pv2.set_xlabel('Sample Index', fontsize=10)
                _ax4_pv2.set_ylabel('Fourier Contribution (log)', fontsize=10)
                _ax4_pv2.set_title('Seasonal/Harmonic Pattern', fontsize=12, fontweight='bold')
                _ax4_pv2.grid(True, alpha=0.3)
        else:
            _ax4_pv2.text(0.5, 0.5, 'No Fourier component', ha='center', va='center', fontsize=12)
            _ax4_pv2.axis('off')

        plt.tight_layout()
    else:
        _fig_pv2 = plt.figure(figsize=(10, 6))
        _ax_placeholder = _fig_pv2.add_subplot(111)
        _ax_placeholder.text(0.5, 0.5, 'Fit PV/Solar model to see residual analysis.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder.transAxes)
        _ax_placeholder.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(df_pv_raw, example_select, mo, plt, sns):
    # PV/Solar: Data Heatmaps
    if example_select.value == 'PV/Solar Power' and df_pv_raw is not None:
        _fig_pv_hm, _axes_pv_hm = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Get column names
        _cols_pv = list(df_pv_raw.columns)
        _power_col = 'inv_03_ac_power_inv_149593' if 'inv_03_ac_power_inv_149593' in _cols_pv else _cols_pv[0]
        _temp_col = 'ambient_temperature_o_149575' if 'ambient_temperature_o_149575' in _cols_pv else None
        _irrad_col = 'poa_irradiance_o_149574' if 'poa_irradiance_o_149574' in _cols_pv else None

        # Reshape for heatmap (time of day vs day)
        _df_pv_copy = df_pv_raw.copy()
        _df_pv_copy['day'] = (_df_pv_copy.index - _df_pv_copy.index.min()).days
        _df_pv_copy['time_slot'] = _df_pv_copy.index.hour + _df_pv_copy.index.minute / 60.0

        # Power heatmap
        _heatmap_power = _df_pv_copy.pivot_table(index='day', columns='time_slot', values=_power_col, aggfunc='mean')
        sns.heatmap(_heatmap_power, cmap='plasma', ax=_axes_pv_hm[0], cbar_kws={'label': 'Power'})
        _axes_pv_hm[0].set_title('Power Output: Days vs Time', fontsize=12, fontweight='bold')
        _axes_pv_hm[0].set_xlabel('Hour of Day')
        _axes_pv_hm[0].set_ylabel('Day')
        _axes_pv_hm[0].tick_params(axis='y', labelsize=6)

        # Temperature heatmap
        if _temp_col and _temp_col in _df_pv_copy.columns:
            _heatmap_temp_pv = _df_pv_copy.pivot_table(index='day', columns='time_slot', values=_temp_col, aggfunc='mean')
            sns.heatmap(_heatmap_temp_pv, cmap='coolwarm', ax=_axes_pv_hm[1], cbar_kws={'label': 'Temperature'})
            _axes_pv_hm[1].set_title('Temperature: Days vs Time', fontsize=12, fontweight='bold')
        else:
            _axes_pv_hm[1].text(0.5, 0.5, 'No temperature data', ha='center', va='center', fontsize=12)
            _axes_pv_hm[1].axis('off')
        _axes_pv_hm[1].set_xlabel('Hour of Day')
        _axes_pv_hm[1].set_ylabel('Day')
        _axes_pv_hm[1].tick_params(axis='y', labelsize=6)

        # Irradiance heatmap
        if _irrad_col and _irrad_col in _df_pv_copy.columns:
            _heatmap_irrad = _df_pv_copy.pivot_table(index='day', columns='time_slot', values=_irrad_col, aggfunc='mean')
            sns.heatmap(_heatmap_irrad, cmap='YlOrRd', ax=_axes_pv_hm[2], cbar_kws={'label': 'Irradiance'})
            _axes_pv_hm[2].set_title('Irradiance: Days vs Time', fontsize=12, fontweight='bold')
        else:
            _axes_pv_hm[2].text(0.5, 0.5, 'No irradiance data', ha='center', va='center', fontsize=12)
            _axes_pv_hm[2].axis('off')
        _axes_pv_hm[2].set_xlabel('Hour of Day')
        _axes_pv_hm[2].set_ylabel('Day')
        _axes_pv_hm[2].tick_params(axis='y', labelsize=6)

        plt.tight_layout()
    else:
        _fig_pv_hm = plt.figure(figsize=(10, 6))
        _ax_placeholder = _fig_pv_hm.add_subplot(111)
        _ax_placeholder.text(0.5, 0.5, 'Select "PV/Solar Power" to see data heatmaps.',
                ha='center', va='center', fontsize=14, transform=_ax_placeholder.transAxes)
        _ax_placeholder.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(Path, np, pd, urllib, zipfile):
    def download_beijing_air_quality_data(data_dir: Path):
        """Download Beijing air quality dataset from UCI ML Repository."""
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "PRSA_data_2010.1.1-2014.12.31.csv"

        if data_file.exists():
            print(f"Data file already exists: {data_file}")
            return data_file

        print("Downloading Beijing air quality dataset...")
        url = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"

        try:
            zip_path = data_dir / "beijing_pm25.zip"
            urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    zip_ref.extract(csv_files[0], data_dir)
                    extracted_file = data_dir / csv_files[0]
                    if extracted_file != data_file:
                        extracted_file.rename(data_file)

            zip_path.unlink()
            print(f"Downloaded and extracted data to: {data_file}")
            return data_file

        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Please download manually from:")
            print("https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data")
            print(f"and place it in: {data_file}")
            raise

    def load_beijing_air_quality(data_file: Path):
        """Load and preprocess Beijing air quality data."""
        print(f"Loading data from: {data_file}")
        df = pd.read_csv(data_file)

        # Create datetime index
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.set_index('datetime')

        # Select relevant columns
        df = df[['pm2.5', 'TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd']].copy()
        df.columns = ['pm25', 'temperature', 'dewpoint', 'pressure', 'wind_speed', 'snow_hours', 'rain_hours', 'wind_dir']

        # Filter to reasonable ranges
        df = df[(df['pm25'] > 0) | df['pm25'].isna()]
        df = df[(df['pm25'] < 1000) | df['pm25'].isna()]
        df = df[(df['temperature'] > -50) | df['temperature'].isna()]
        df = df[(df['temperature'] < 50) | df['temperature'].isna()]
        df = df[(df['dewpoint'] > -50) | df['dewpoint'].isna()]
        df = df[(df['dewpoint'] < 50) | df['dewpoint'].isna()]
        df = df[(df['pressure'] > 900) | df['pressure'].isna()]
        df = df[(df['pressure'] < 1100) | df['pressure'].isna()]
        df = df[(df['wind_speed'] >= 0) | df['wind_speed'].isna()]
        df = df[(df['wind_speed'] < 100) | df['wind_speed'].isna()]

        df = df.sort_index()

        # Create regular hourly index
        date_range = pd.date_range(
            start=df.index.min().floor('h'),
            end=df.index.max().floor('h'),
            freq='h'
        )

        df = df.reindex(date_range)

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df.select_dtypes(exclude=[np.number]).columns

        # Only interpolate numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Drop rows with NaN in numeric columns (wind_dir may have NaN, which is fine)
        df = df.dropna(subset=numeric_cols)

        print(f"Loaded {len(df)} samples")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"PM2.5 range: {df['pm25'].min():.1f} to {df['pm25'].max():.1f} μg/m³")
        print(f"Temperature range: {df['temperature'].min():.1f} to {df['temperature'].max():.1f} °C")

        return df
    return download_beijing_air_quality_data, load_beijing_air_quality


@app.cell
def _(pd, timedelta):
    def create_train_test_split_with_gaps(df, target_col, holdout_days=7):
        """
        Create train/test split holding out the last N days of each month.
        This creates gaps in the training data to test the estimator's ability
        to handle missing time periods.
        """
        train_mask = pd.Series(True, index=df.index)
        test_mask = pd.Series(False, index=df.index)

        df_with_month = df.copy()
        df_with_month['year_month'] = df_with_month.index.to_period('M')

        for year_month in df_with_month['year_month'].unique():
            month_data = df_with_month[df_with_month['year_month'] == year_month]

            if len(month_data) == 0:
                continue

            month_end = month_data.index.max()
            holdout_start = month_end - timedelta(days=holdout_days-1)
            holdout_start = holdout_start.replace(hour=0, minute=0, second=0, microsecond=0)

            month_mask = (df.index >= holdout_start) & (df.index <= month_end) & (df_with_month['year_month'] == year_month)
            test_mask[month_mask] = True
            train_mask[month_mask] = False

        return train_mask, test_mask
    return (create_train_test_split_with_gaps,)


@app.cell
def _(np):
    def calculate_metrics(predictions, actuals):
        """
        Calculate MAE, RMSE, MAPE, R² for given predictions and actuals.
        Returns a dict with metric values, or NaN values if no valid data.
        """
        valid = np.isfinite(predictions) & np.isfinite(actuals) & (actuals > 0)
        if not np.any(valid):
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}

        pred_valid = predictions[valid]
        actual_valid = actuals[valid]

        mae = np.mean(np.abs(pred_valid - actual_valid))
        rmse = np.sqrt(np.mean((pred_valid - actual_valid) ** 2))
        mape = np.mean(np.abs((pred_valid - actual_valid) / actual_valid)) * 100
        ss_res = np.sum((actual_valid - pred_valid) ** 2)
        ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

    def format_metrics_report(title, metrics, unit="-"):
        """
        Format metrics as a markdown report string.
        """
        return f"""
    ## Model Performance Report: {title}

    **Forecast Performance Metrics:**

    | Metric | Value | Unit |
    |--------|-------|------|
    | **MAE** | {metrics['mae']:.2f} | {unit} |
    | **RMSE** | {metrics['rmse']:.2f} | {unit} |
    | **MAPE** | {metrics['mape']:.2f} | % |
    | **R²** | {metrics['r2']:.4f} | - |

    **Interpretation:**
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
    - **RMSE (Root Mean Squared Error)**: Penalizes larger errors more than MAE
    - **MAPE (Mean Absolute Percentage Error)**: Percentage error relative to actual values
    - **R² (Coefficient of Determination)**: Proportion of variance explained by the model (closer to 1.0 is better)
    """
    return calculate_metrics, format_metrics_report


@app.cell
def _(Path, download_beijing_air_quality_data, load_beijing_air_quality):
    # Load Air Quality data (always loaded)
    _examples_dir_aq = Path(__file__).parent
    _data_dir_aq = _examples_dir_aq / "data" / "air_quality"
    _data_file_aq = download_beijing_air_quality_data(_data_dir_aq)
    df_air_quality = load_beijing_air_quality(_data_file_aq)
    return (df_air_quality,)


@app.cell
def _(Path, pd):
    # Load LA Energy data (always loaded)
    _examples_dir_la = Path(__file__).parent
    _energy_dir_la = _examples_dir_la / "data" / "energy"
    _default_weather_la = "weather_CA_Los Angeles.csv"
    _default_energy_la = "CA_Los Angeles_R.csv"

    _weather_file_la = _energy_dir_la / _default_weather_la
    _energy_file_la = _energy_dir_la / _default_energy_la

    if _weather_file_la.exists() and _energy_file_la.exists():
        _df_weather_la = pd.read_csv(_weather_file_la)
        _df_weather_la['timestamp'] = pd.to_datetime(_df_weather_la['timestamp'])
        _df_weather_la = _df_weather_la.set_index('timestamp')

        _df_energy_la = pd.read_csv(_energy_file_la)
        _df_energy_la['timestamp'] = pd.to_datetime(_df_energy_la['timestamp'])
        _df_energy_la = _df_energy_la.set_index('timestamp')

        df_la = pd.merge(_df_weather_la, _df_energy_la, left_index=True, right_index=True, how='inner')
        df_la = df_la.sort_index()
        print(f"LA Energy data loaded: {len(df_la)} samples")
    else:
        df_la = None
        print("LA Energy data files not found")
    return (df_la,)


@app.cell
def _(DataHandler, Path, pd):
    # Load PV data (always loaded)
    _examples_dir_pv = Path(__file__).parent
    _pv_data_dir_pv = _examples_dir_pv / "data" / "pv"
    _default_pv_file_pv = _pv_data_dir_pv / "2107_data_combined.csv"

    if _default_pv_file_pv.exists() and DataHandler is not None:
        _df_pv_raw_pv = pd.read_csv(_default_pv_file_pv, parse_dates=[0], index_col=0)
        _df_pv_raw_pv = _df_pv_raw_pv.resample('15min').mean()

        dh_pv = DataHandler(_df_pv_raw_pv)
        dh_pv.fix_dst()

        df_pv_raw = _df_pv_raw_pv
        print(f"PV data loaded: {df_pv_raw.shape}")
    else:
        df_pv_raw = None
        dh_pv = None
        if not _default_pv_file_pv.exists():
            print(f"PV data file not found at {_default_pv_file_pv}")
        if DataHandler is None:
            print("DataHandler not available")
    return (df_pv_raw,)


if __name__ == "__main__":
    app.run()
