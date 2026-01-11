import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(mo):
    file_slct = mo.ui.file_browser()
    file_slct
    return (file_slct,)


@app.cell
def _(file_slct, load_file):
    df = load_file(file_slct)
    df = df.resample('15min').mean()
    return (df,)


@app.cell
def _(df, mo):
    col_select = mo.ui.table(list(df.columns), selection='multi')
    return (col_select,)


@app.cell
def _(col_select, mo):
    primary_select = mo.ui.dropdown(col_select.value, label='select primary column')
    module_temp_select = mo.ui.dropdown(col_select.value, label='select module temperature column')
    irrad_select = mo.ui.dropdown(col_select.value, label='select irradiance column')
    lin_thresh_select = mo.ui.slider(start=0, stop=1, step=0.05, label='select linearity threshold', value=0.1)
    fix_dst_slct = mo.ui.switch(label='fix DST')
    return (
        fix_dst_slct,
        irrad_select,
        lin_thresh_select,
        module_temp_select,
        primary_select,
    )


@app.cell
def _(mo):
    run_sdt = mo.ui.run_button(label='run SDT')
    return (run_sdt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Configure and run SDT
    """)
    return


@app.cell
def _(
    col_select,
    fix_dst_slct,
    irrad_select,
    lin_thresh_select,
    mo,
    module_temp_select,
    primary_select,
    run_sdt,
):
    mo.vstack([mo.hstack([col_select, col_select.value]), primary_select, module_temp_select, irrad_select, mo.hstack([lin_thresh_select, fix_dst_slct, run_sdt])])
    return


@app.cell
def _(
    DataHandler,
    col_select,
    df,
    fix_dst_slct,
    irrad_select,
    lin_thresh_select,
    mo,
    module_temp_select,
    np,
    primary_select,
    run_sdt,
):
    mo.stop(not run_sdt.value)

    if primary_select.value is None:
        _pc = col_select.value[0]
    else:
        _pc = primary_select.value

    dh = DataHandler(df)
    if fix_dst_slct.value:
        dh.fix_dst()
    _lt = lin_thresh_select.value
    with mo.capture_stdout() as buffer:
        if len(col_select.value) == 1:
            dh.run_pipeline(power_col=_pc, max_val=2000, linearity_threshold=_lt)
        else:
            dh.run_pipeline(power_col=_pc, max_val=2000, extra_cols=[_c for _c in col_select.value if _pc != _c],
                            linearity_threshold=_lt)
    temp_mat = dh.extra_matrices[module_temp_select.value]
    temp_mat[temp_mat > 140] = np.nan
    irrad_mat = dh.extra_matrices[irrad_select.value]
    irrad_mat[irrad_mat < 0] = 0
    return (dh,)


@app.cell
def _(dh):
    print(dh.raw_data_matrix.shape)
    return


@app.cell
def _(dh, mo):
    with mo.redirect_stdout():
        dh.report()
    return


@app.cell
def _(mo):
    mo.md("""
    ### View data
    """)
    return


@app.cell
def _(dh):
    dh.plot_data_quality_scatter()
    return


@app.cell
def _(dh, mo):
    start_day = mo.ui.slider(start=0, stop=dh.num_days, step=1, value=0, label='start', full_width=True)
    num_days = mo.ui.slider(start=1, stop=30, step=1, value=3, label='num days')
    mo.vstack([start_day, num_days])
    return num_days, start_day


@app.cell
def _(dh, num_days, plt, start_day):
    dh.plot_daily_signals(start_day=start_day.value, num_days=num_days.value, filled=False)
    plt.gcf()
    return


@app.cell
def _(dh, plt):
    dh.plot_heatmap('raw')
    plt.gcf()
    return


@app.cell
def _(dh, irrad_select, plot_2d, plt):
    plot_2d(dh.extra_matrices[irrad_select.value], units='W/m^2')
    plt.title("irradiance")
    return


@app.cell
def _(dh, module_temp_select, plot_2d, plt):

    plot_2d(dh.extra_matrices[module_temp_select.value], units='deg C')
    plt.title("module temp")
    # plt.plot(_temp_mat.ravel(order='F'))
    return


@app.cell
def _(dh, irrad_select, module_temp_select, plt):
    _fig, _ax = plt.subplots(ncols=2, sharey=True, figsize=(12,6))
    _ax[0].plot(dh.extra_matrices[module_temp_select.value][:, dh.daily_flags.no_errors].ravel(),
                dh.raw_data_matrix[:, dh.daily_flags.no_errors].ravel(),
                marker='.', ls='none', markersize=0.5)
    _ax[1].plot(dh.extra_matrices[irrad_select.value][:, dh.daily_flags.no_errors].ravel(),
                dh.raw_data_matrix[:, dh.daily_flags.no_errors].ravel(),
                marker='.', ls='none', markersize=0.5)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Construct time-series GAM using TsgamEstimator
    """)
    return


@app.cell
def _(dh, mo):
    data_start = mo.ui.number(start=0, stop=dh.raw_data_matrix.shape[1]-1, value=0, label='data start')
    data_end = mo.ui.number(start=0, stop=dh.raw_data_matrix.shape[1]-1, value=dh.raw_data_matrix.shape[1]-1, label='data end')
    take_log = mo.ui.switch(label='take log of target data')
    target_filter = mo.ui.slider(start=0, stop=1, step=0.01, value=0, label='target min value filter')
    fit_model = mo.ui.run_button(label='fit model')
    solver_slct = mo.ui.switch(label='solver: CLARABEL <-> MOSEK')
    trend_slct = mo.ui.dropdown(['linear', 'nonlinear', 'none'], value='nonlinear', label='trend term')
    return (
        data_end,
        data_start,
        fit_model,
        solver_slct,
        take_log,
        target_filter,
        trend_slct,
    )


@app.cell
def _(data_end, data_start, dh, irrad_select, module_temp_select, np, plt):
    # Prepare data for visualization
    _data_select = np.s_[data_start.value:data_end.value]
    _y = np.copy(dh.raw_data_matrix)
    _y[:, ~dh.daily_flags.no_errors] = np.nan
    _y[~dh.boolean_masks.daytime] = np.nan
    _y = _y[:, _data_select].ravel(order='F')

    _x1 = np.copy(dh.extra_matrices[module_temp_select.value][:, _data_select].ravel(order='F'))
    _x2 = np.copy(dh.extra_matrices[irrad_select.value][:, _data_select].ravel(order='F'))

    _fig, _ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
    _ax[0].plot(_x1, _y, marker='.', ls='none', markersize=0.5)
    _ax[0].set_title('temperature response')
    _ax[1].plot(_x2, _y, marker='.', ls='none', markersize=0.5)
    _ax[1].set_title('irradiance response')
    scatter_plots = _fig
    return (scatter_plots,)


@app.cell
def _(data_end, data_start, mo, scatter_plots, take_log, target_filter):
    _form = mo.md(f"""
    **Select data to use in tsGAM.**

    {data_start}

    {data_end}

    {target_filter}

    {take_log}
    """)
    mo.hstack([_form, scatter_plots])
    return


@app.cell
def _(
    data_end,
    data_start,
    dh,
    irrad_select,
    module_temp_select,
    np,
    plt,
    sns,
):
    _fig, _ax = plt.subplots(nrows=3, figsize=(12,7))
    _m1 = np.copy(dh.raw_data_matrix)
    _m2 = np.copy(dh.extra_matrices[module_temp_select.value])
    _m3 = np.copy(dh.extra_matrices[irrad_select.value])
    _m1[~dh.boolean_masks.daytime] = np.nan
    _m1[:, ~dh.daily_flags.no_errors] = np.nan
    _m2[~dh.boolean_masks.daytime] = np.nan
    _m2[:, ~dh.daily_flags.no_errors] = np.nan
    _m3[~dh.boolean_masks.daytime] = np.nan
    _m3[:, ~dh.daily_flags.no_errors] = np.nan
    sns.heatmap(_m1, cmap='plasma', ax=_ax[0])
    sns.heatmap(_m2, cmap='plasma', ax=_ax[1])
    sns.heatmap(_m3, cmap='plasma', ax=_ax[2])
    for _ix in range(3):
        _ax[_ix].axvline(data_start.value, color='dodgerblue', ls='--')
        _ax[_ix].axvline(data_end.value, color='lime', ls='--')
        _ax[_ix].axis("off")
    plt.gcf()
    return


@app.cell
def _(fit_model, mo, solver_slct, trend_slct):
    _form = mo.md(f"""
    **Configure and fit model.**

    {solver_slct}

    {trend_slct}

    {fit_model}
    """)
    _form
    return


@app.cell
def _(
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
    data_end,
    data_start,
    df,
    dh,
    fit_model,
    irrad_select,
    mo,
    module_temp_select,
    np,
    pd,
    solver_slct,
    take_log,
    target_filter,
    trend_slct,
):
    mo.stop(not fit_model.value)

    # Prepare data
    _data_select = np.s_[data_start.value:data_end.value]
    y = np.copy(dh.raw_data_matrix)
    y_max = np.nanmax(y)
    y[:, ~dh.daily_flags.no_errors] = np.nan
    y[~dh.boolean_masks.daytime] = np.nan
    y[y < 0.01 * np.nanmax(y)] = np.nan
    y = y[:, _data_select].ravel(order='F')
    y /= y_max
    y[y < target_filter.value] = np.nan
    if take_log.value:
        y = np.log(y)

    # Prepare exogenous variables
    x1 = np.copy(dh.extra_matrices[module_temp_select.value][:, _data_select].ravel(order='F'))
    x1_avail = ~np.isnan(x1)
    x1[~x1_avail] = 0
    x1_max = np.max(x1)
    x1 /= x1_max

    x2 = np.copy(dh.extra_matrices[irrad_select.value][:, _data_select].ravel(order='F'))
    x2[x2 < 0] = 0
    x2_avail = np.logical_and(~np.isnan(x2), x2 > 0.02 * np.nanquantile(x2, 0.98))
    x2[~x2_avail] = 0
    x2_max = np.max(x2)
    x2 /= x2_max

    # Create timestamps for the data
    # DataHandler uses a matrix format, so we need to reconstruct timestamps
    # Use the actual length of the data arrays to ensure consistency
    _data_len = len(y)  # This is the actual length after all processing
    _m, _n = dh.raw_data_matrix.shape
    _n_selected = data_end.value - data_start.value + 1

    # Get start time from original dataframe
    # Account for data_start offset - we need the timestamp for the first selected day
    if hasattr(df, 'index') and len(df.index) > 0:
        # Find the timestamp corresponding to the start of the selected data range
        # DataHandler matrix columns correspond to days, so we need to find the first day's timestamp
        _base_time = df.index[0]
        # Add the data_start offset (in days)
        _start_time = _base_time + pd.Timedelta(days=data_start.value)
    else:
        _start_time = pd.Timestamp('2020-01-01 00:00:00')

    # Ensure all arrays have the same length first
    _min_len = min(len(y), len(x1), len(x2))
    y = y[:_min_len]
    x1 = x1[:_min_len]
    x2 = x2[:_min_len]

    # Store full arrays BEFORE filtering (for visualization)
    y_full = y.copy()
    x1_full = x1.copy()
    x2_full = x2.copy()

    # Filter out NaN in y first (before creating timestamps)
    # This ensures timestamps remain regularly spaced after filtering
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    x1 = x1[valid_mask]
    x2 = x2[valid_mask]
    _valid_len = len(y)

    # Create timestamps for filtered data with regular 15-minute spacing
    # Use date_range to ensure regular frequency that pandas can infer
    # Start time should be a valid timestamp
    if not isinstance(_start_time, pd.Timestamp):
        _start_time = pd.Timestamp(_start_time)

    # Create regularly spaced timestamps with explicit frequency
    timestamps = pd.date_range(
        start=_start_time,
        periods=_valid_len,
        freq='15min'
    )

    # Verify frequency can be inferred (should be '15T' or '15min')
    _inferred_freq = pd.infer_freq(timestamps)
    if _inferred_freq is None:
        # Force frequency if inference fails
        timestamps = timestamps.asfreq('15min')

    # Create DataFrame with exogenous variables (now all have same length)
    X = pd.DataFrame({
        'temp': x1,
        'irrad': x2
    }, index=timestamps)

    # Configure model
    # Multi-harmonic Fourier: yearly and daily patterns
    # periods in hours: yearly = 365.2425 * 24, daily = 24 (for 15-min data, adjust)
    _nvals = dh.raw_data_matrix.shape[0]  # samples per day
    _period_daily_hours = 24.0  # daily period in hours
    _period_yearly_hours = 365.2425 * 24.0  # yearly period in hours
    # For 15-min data, daily period is _nvals samples = 24 hours
    # Yearly period is 365.2425 * _nvals samples = 365.2425 * 24 hours

    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[6, 10],
        periods=[_period_yearly_hours, _period_daily_hours],
        reg_weight=1e-2
    )

    # Exogenous variables: temperature and irradiance splines
    exog_config = [
        TsgamSplineConfig(
            n_knots=10,
            lags=[0],  # No lead/lag for now
            reg_weight=1e-4
        ),
        TsgamSplineConfig(
            n_knots=10,
            lags=[0],
            reg_weight=1e-4
        )
    ]

    # Trend configuration
    trend_type_map = {
        'linear': TrendType.LINEAR,
        'nonlinear': TrendType.NONLINEAR,
        'none': TrendType.NONE
    }
    trend_config = TsgamTrendConfig(
        trend_type=trend_type_map[trend_slct.value],
        grouping=24.0,  # Daily trend (period in hours)
        reg_weight=10.0
    )

    # Solver configuration
    solver_name = "MOSEK" if solver_slct.value else "CLARABEL"
    solver_config = TsgamSolverConfig(
        solver=solver_name,
        verbose=True
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=exog_config,
        trend_config=trend_config,
        solver_config=solver_config
    )

    # Create and fit estimator
    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)

    # Get predictions
    model = estimator.predict(X)

    # Store original arrays (before filtering) and filtered versions for visualization
    # x1, x2 are the normalized versions before filtering
    x1_full = x1.copy()
    x2_full = x2.copy()
    y_full = y.copy()  # This is already the processed y (normalized, log if needed)

    # Store for visualization
    return (
        estimator,
        model,
        valid_mask,
        x1_full,
        x1_max,
        x2_full,
        x2_max,
        y,
        y_full,
        y_max,
    )


@app.cell
def _(mo):
    mo.md("""
    ### View model fit
    """)
    return


@app.cell
def _(estimator):
    print(f"Problem status: {estimator.problem_.status}")
    if hasattr(estimator.problem_, 'value'):
        print(f"Optimal value: {estimator.problem_.value:.6e}")
    return


@app.cell
def _(model, np, plt, take_log, y, y_max):
    _fig, _ax = plt.subplots(ncols=2, figsize=(10, 5))
    _ax[0].plot(model, y, marker='.', linewidth=.5, markersize=1, alpha=0.5)
    if take_log.value:
        _ax[1].plot(y_max * np.exp(model), y_max * np.exp(y), marker='.', linewidth=.5, markersize=1, alpha=0.5)
    else:
        _ax[1].plot(y_max * model, y_max * y, marker='.', linewidth=.5, markersize=1, alpha=0.5)
    for _ix in range(2):
        _xlim = _ax[_ix].get_xlim()
        _ylim = _ax[_ix].get_ylim()
        _ax[_ix].plot([-1e4, 1e4], [-1e4, 1e4], color='red', ls='--', linewidth=1)
        _ax[_ix].set_xlim(_xlim)
        _ax[_ix].set_ylim(_ylim)
        _ax[_ix].set_ylabel('actual')
        _ax[_ix].set_xlabel('predicted')
    _ax[0].set_title('transformed data')
    _ax[1].set_title('original data')
    plt.gcf()
    return


@app.cell
def _(data_end, data_start, mo):
    day_slct = mo.ui.slider(start=0, stop=data_end.value - data_start.value + 1, step=1, value=206, full_width=True)
    day_slct
    return (day_slct,)


@app.cell
def _(
    day_slct,
    dh,
    model,
    np,
    plt,
    take_log,
    valid_mask,
    x1_full,
    x1_max,
    x2_full,
    x2_max,
    y_full,
    y_max,
):
    _fig, _ax = plt.subplots(nrows=4, sharex=True, figsize=(12, 6))
    _n = dh.filled_data_matrix.shape[0]  # samples per day
    _day_idx = day_slct.value
    _day_window = 5

    # Get the valid indices for the selected day range
    # valid_mask is boolean array of length of truncated data (y_full length)
    # We need to find which valid samples correspond to the selected days
    valid_indices = np.where(valid_mask)[0]

    # Calculate start and end indices for the selected day range
    # These are relative to the truncated array (after data_start selection)
    # _day_idx is relative to data_start, so we don't need to add data_start.value
    _start_idx = _n * _day_idx
    _end_idx = _n * (_day_idx + _day_window)

    # Ensure indices are within bounds of y_full
    _start_idx = max(0, min(_start_idx, len(y_full)))
    _end_idx = max(0, min(_end_idx, len(y_full)))

    # Find valid samples in this range
    _range_mask = (valid_indices >= _start_idx) & (valid_indices < _end_idx)
    _range_valid_indices = valid_indices[_range_mask]
    _range_relative_indices = _range_valid_indices - _start_idx

    # Plot target and model (only valid samples)
    if len(_range_valid_indices) > 0:
        # _range_valid_indices are indices into y_full (which has length len(valid_mask))
        # Ensure all indices are within bounds
        _range_valid_indices = _range_valid_indices[_range_valid_indices < len(y_full)]
        if len(_range_valid_indices) > 0:
            _y_range = y_full[_range_valid_indices]
            # Update _range_mask to match the filtered indices
            _range_mask = np.isin(valid_indices, _range_valid_indices)
            _model_range = model[_range_mask]
            _range_relative_indices = _range_valid_indices - _start_idx

            _ax[0].plot(_range_relative_indices, y_max * (np.exp(_y_range) if take_log.value else _y_range),
                        linewidth=1, color='black', label='target', marker='.', markersize=2)
            _ax[0].plot(_range_relative_indices, y_max * (np.exp(_model_range) if take_log.value else _model_range),
                        label='model', marker='.', markersize=2)

            residuals = y_max * ((np.exp(_y_range) - np.exp(_model_range)) if take_log.value else (_y_range - _model_range))
            _ax[1].plot(_range_relative_indices, residuals, label='residuals', marker='.', markersize=2)
        else:
            residuals = np.array([])
    else:
        residuals = np.array([])

    # Plot exogenous variables (all samples in range, not just valid)
    _x1_range = x1_full[_start_idx:min(_end_idx, len(x1_full))] * x1_max
    _x2_range = x2_full[_start_idx:min(_end_idx, len(x2_full))] * x2_max
    _ax[2].plot(_x1_range, label='module temp')
    _ax[3].plot(_x2_range, label='POA')

    _ax[0].legend()
    _ax[1].legend()
    _ax[2].legend()
    _ax[3].legend()
    _ax[0].set_ylabel('target')
    _ax[1].set_ylabel('residual')
    _ax[2].set_ylabel('deg C')
    _ax[3].set_ylabel('W/m^2')
    _fig
    return (residuals,)


@app.cell
def _(estimator, np, plt, sns):
    # Plot trend if available
    if hasattr(estimator, 'variables_') and 'trend' in estimator.variables_:
        trend = estimator.variables_['trend'].value
        if trend is not None:
            with sns.axes_style('whitegrid'):
                plt.plot(np.arange(len(trend)) / 365, np.exp(trend))
                plt.xlabel('years')
                plt.title('Degradation term over time')
            plt.gcf()
    return


@app.cell
def _(
    data_end,
    data_start,
    dh,
    model,
    np,
    plt,
    sns,
    take_log,
    valid_mask,
    y_full,
    y_max,
):
    plt.figure(figsize=(10, 6))
    # Calculate residuals for valid samples
    y_valid = y_full[valid_mask]
    if take_log.value:
        residuals = y_max * (np.exp(y_valid) - np.exp(model))
    else:
        residuals = y_max * (y_valid - model)
    _ = plt.figure(figsize=(10, 5))
    # Reshape residuals to match original matrix format
    # Note: residuals are for valid samples only, so we need to map them back
    _n_per_day = dh.raw_data_matrix.shape[0]
    _n_selected = data_end.value - data_start.value + 1
    _reshaped = np.full((_n_per_day, _n_selected), np.nan)

    # Map residuals back to their positions in the original matrix
    valid_indices = np.where(valid_mask)[0]
    for i, orig_idx in enumerate(valid_indices):
        _day = orig_idx // _n_per_day
        _time = orig_idx % _n_per_day
        if _day < _n_selected and _time < _n_per_day:
            _reshaped[_time, _day] = residuals[i]

    sns.heatmap(_reshaped, cmap='seismic', center=0, ax=plt.gca())
    plt.title("residuals heatmap")
    plt.gcf()
    return (residuals,)


@app.cell
def _(np, plt, primary_select, residuals, stats):
    plt.figure(figsize=(12, 6))
    _r = residuals
    _s = ~np.isnan(_r)
    plt.hist(_r, bins=200, density=True)
    _xs = np.linspace(np.min(_r[_s]), np.max(_r[_s]), 1001)
    lap_loc, lap_scale = stats.laplace.fit(_r[_s])
    nor_loc, nor_scale = stats.norm.fit(_r[_s])
    plt.plot(_xs, stats.laplace.pdf(_xs, lap_loc, lap_scale), label='laplace fit', linewidth=1, color='dodgerblue')
    plt.plot(_xs, stats.norm.pdf(_xs, nor_loc, nor_scale), label='normal fit', linewidth=1, color='lime')

    plt.axvline(np.nanquantile(_r, .2), color='yellow', ls='--', label='60% confidence bounds', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .8), color='yellow', ls='--', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .025), color='orange', ls='--', label='95% confidence bounds', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .975), color='orange', ls='--', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .005), color='red', ls='--', label='99% confidence bounds', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .995), color='red', ls='--', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .0005), color='black', ls='--', label='99.9% confidence bounds', linewidth=0.5)
    plt.axvline(np.nanquantile(_r, .9995), color='black', ls='--', linewidth=0.5)
    plt.xlabel('residual: '+primary_select.value)
    plt.legend()
    plt.title('distribution of residuals')
    return


@app.cell
def _(plt, residuals):
    plt.figure(figsize=(12, 6))
    _r = residuals
    plt.plot(_r)
    plt.title('residuals over time')
    return


@app.cell
def _(mo):
    res_dist = mo.ui.switch(label='residual distribution: Normal <> Laplace')
    res_dist
    return (res_dist,)


@app.cell
def _(model, np, plt, res_dist, sm, stats, take_log, valid_mask, y_full):
    if take_log.value:
        _r = np.exp(y_full[valid_mask]) - np.exp(model)
    else:
        _r = y_full[valid_mask] - model
    _s = ~np.isnan(_r)
    if not res_dist.value:
        _d = stats.norm
        _t = "Normal"
    else:
        _d = stats.laplace
        _t = "Laplace"

    _fig, _ax = plt.subplots(ncols=2, figsize=(10,4))

    _pplot = sm.ProbPlot(_r[_s], _d, fit=True)
    _fig1 = _pplot.ppplot(line="45", ax=_ax[0])
    _h1 = plt.title("P-P plot for "+_t)

    _pplot2 = sm.ProbPlot(_r[_s], _d, fit=True)
    _fig2 = _pplot.qqplot(line="45", ax=_ax[1])
    _h2 = plt.title("Q-Q plot for "+_t)
    _fig
    return


@app.cell
def _(estimator, np, plt, valid_mask, x1_full, x1_max):
    # Plot temperature response
    if hasattr(estimator, 'variables_') and 'exog_coef_0' in estimator.variables_:
        exog_coef = estimator.variables_['exog_coef_0'].value
        if exog_coef is not None:
            # Get knots
            knots = estimator.exog_knots_[0] if estimator.exog_knots_ and len(estimator.exog_knots_) > 0 else None
            if knots is not None:
                # Use only valid samples for visualization
                x1_valid = x1_full[valid_mask]
                # Use the estimator's _make_H method
                H1 = estimator._make_H(x1_valid, knots, include_offset=False)
                plt.plot(x1_valid * x1_max, np.exp(H1 @ exog_coef[:, 0]), ls='none', marker='.', markersize=1)
                plt.title('Inferred temperature response')
                plt.xlabel('module temp [deg C]')
                plt.ylabel('correction factor [1]')
                plt.gcf()
    return


@app.cell
def _(estimator, np, plt, valid_mask, x2_full, x2_max):
    # Plot irradiance response
    if hasattr(estimator, 'variables_') and 'exog_coef_1' in estimator.variables_:
        exog_coef = estimator.variables_['exog_coef_1'].value
        if exog_coef is not None:
            knots = estimator.exog_knots_[1] if estimator.exog_knots_ and len(estimator.exog_knots_) > 1 else None
            if knots is not None:
                # Use only valid samples for visualization
                x2_valid = x2_full[valid_mask]
                H2 = estimator._make_H(x2_valid, knots, include_offset=False)
                plt.plot(x2_valid * x2_max, np.exp(H2 @ exog_coef[:, 0]), ls='none', marker='.', markersize=1)
                plt.title('Inferred irradiance response')
                plt.xlabel('POA irradiance [W/m^2]')
                plt.ylabel('correction factor [1]')
                plt.gcf()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Imports and function definitions
    """)
    return


@app.cell
def _():
    import marimo as mo
    from solardatatools import DataHandler, plot_2d
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import statsmodels.api as sm
    import sys

    # Add parent directory to path to import tsgam_estimator
    # This notebook is in Archive/, so parent directory is project root
    _current = Path.cwd()
    if _current.name == 'Archive':
        _project_root = _current.parent
    else:
        # Try to find project root by looking for tsgam_estimator.py
        _project_root = _current
        for _ in range(5):  # Max 5 levels up
            if (_project_root / 'tsgam_estimator.py').exists():
                break
            _project_root = _project_root.parent

    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from tsgam_estimator import (
        TrendType,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiHarmonicConfig,
        TsgamSplineConfig,
        TsgamSolverConfig,
        TsgamTrendConfig,
    )
    return (
        DataHandler,
        TrendType,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiHarmonicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        TsgamTrendConfig,
        mo,
        np,
        pd,
        plot_2d,
        plt,
        sm,
        sns,
        stats,
    )


@app.cell
def _(pd):
    def load_file(file_ui):
        if len(file_ui.value) == 1:
            df = pd.read_csv(file_ui.value[0].id, parse_dates=[0], index_col=0)
        else:
            dfs = [pd.read_csv(_v.id, parse_dates=[0], index_col=0) for _v in file_ui.value]
            df = pd.concat(dfs, axis=1)
        return df
    return (load_file,)


if __name__ == "__main__":
    app.run()
