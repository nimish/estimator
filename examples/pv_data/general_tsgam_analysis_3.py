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
def _(file_slct, load_file, mo):
    mo.stop(not file_slct.value)

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
    ## Construct time-series GAM
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
    trend_slct = mo.ui.dropdown(['linear', 'non-linear', 'none'], value='non-linear', label='trend term')
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
def _(plt, x1, x2, y):
    _fig, _ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
    _ax[0].plot(x1, y, marker='.', ls='none', markersize=0.5)
    _ax[0].set_title('temperature response')
    _ax[1].plot(x2, y, marker='.', ls='none', markersize=0.5)
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
    data_end,
    data_start,
    dh,
    irrad_select,
    make_H,
    make_basis_matrix,
    make_regularization_matrix,
    module_temp_select,
    np,
    take_log,
    target_filter,
):
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
    # x1 is module temperature
    x1 = np.copy(dh.extra_matrices[module_temp_select.value][:, _data_select].ravel(order='F'))
    # x1[x1 > 200] = np.nan
    x1_avail = ~np.isnan(x1)
    x1[~x1_avail] = 0
    x1_max = np.max(x1)
    x1 /= x1_max
    # x2 is POA irradiance
    x2 = np.copy(dh.extra_matrices[irrad_select.value][:, _data_select].ravel(order='F'))
    x2[x2 < 0] = 0
    x2_avail = np.logical_and(~np.isnan(x2), x2 > 0.02 * np.nanquantile(x2, 0.98))
    x2[~x2_avail] = 0
    x2_max = np.max(x2)
    x2 /= x2_max
    # mult-Fourier basis matrix with cross terms
    nharmon = [6, 10]
    _nvals = dh.raw_data_matrix.shape[0]
    periods = [365.2425 * _nvals, _nvals]
    F = make_basis_matrix(
        num_harmonics=nharmon,
        length=len(y),
        periods=periods
    )
    # weight matrix for regularized Fourier parameters
    Wf = make_regularization_matrix(
        num_harmonics=nharmon,
        weight=1,
        periods=periods
    )
    # Temperature and irradiance basis matrices
    # Natural cubic splice basis expansion
    nK = 10 # 10
    knots1 = np.linspace(np.nanmin(x1), np.max(x1), nK)
    knots2 = np.linspace(np.nanmin(x2), np.max(x2), nK)
    H1 = make_H(x1, knots1, include_offset=False)
    H1[~x1_avail, :] = np.nan
    H2 = make_H(x2, knots2, include_offset=False)
    H2[~x2_avail, :] = np.nan
    Hs = [H1, H2]
    regressor_use_set = np.logical_and(x1_avail, x2_avail)
    target_use_set = ~np.isnan(y)
    use_set = np.logical_and(regressor_use_set, target_use_set)
    x1[~x1_avail] = np.nan
    x2[~x2_avail] = np.nan
    return F, H1, H2, Wf, use_set, x1, x1_max, x2, x2_max, y, y_max


@app.cell
def _(
    F,
    H1,
    H2,
    Wf,
    cvx,
    data_end,
    data_start,
    dh,
    fit_model,
    mo,
    np,
    solver_slct,
    trend_slct,
    use_set,
    y,
):
    mo.stop(not fit_model.value)

    if not solver_slct.value:
        _solver = "CLARABEL"
    else:
        _solver = "MOSEK"

    a = cvx.Variable(F.shape[1]) # coefficients for time features
    b = cvx.Variable(H1.shape[1]) # coefficients for temperature features
    c = cvx.Variable(H2.shape[1]) # coefficients for irradiance features
    print(dh.raw_data_matrix)
    _m, _ = dh.raw_data_matrix.shape
    print(dh.raw_data_matrix.shape)
    _n = data_end.value - data_start.value + 1
    T = np.zeros((len(y), _n))
    for _ix in range(_n - 1):
        T[_ix*_m:(_ix+1)*_m, _ix] = np.ones(_m)
    trend = cvx.Variable(_n)
    _s = use_set
    error = cvx.sum_squares(y[_s] - F[_s] @ a - H1[_s] @ b - H2[_s] @ c - (T @ trend)[_s]) / np.sum(_s)
    regularization = 1e-2 * cvx.norm(Wf @ a) + 1e-4 * cvx.norm(b) + 1e-4 * cvx.norm(c) + 1e1 * cvx.sum_squares(cvx.diff(trend))
    # error = cvx.sum_squares(y[_s] - F[_s] @ a - H1[_s] @ b) / np.sum(_s)
    # regularization = 1e-4 * cvx.norm(Wf @ a) + 1e-4 * cvx.norm(b)
    constraints = [trend[0] == 0]
    if trend_slct.value == 'linear':
        slope = cvx.Variable()
        constraints.append(cvx.diff(trend) == slope)
    elif trend_slct.value == 'non-linear':
        constraints.append(cvx.diff(trend) <= 0)
    else:
        constraints.append(trend == 0)
    problem = cvx.Problem(cvx.Minimize(error + regularization), constraints)
    problem.solve(solver=_solver, verbose=True)
    model = (F @ a + H1 @ b + H2 @ c + T @ trend).value
    # model = (F @ a + H1 @ b).value
    return b, c, error, model, trend


@app.cell
def _(mo):
    mo.md("""
    ### View model fit
    """)
    return


@app.cell
def _(error):
    error.value
    return


@app.cell
def _(model, np, plt, take_log, y, y_max):
    # _model = (F[_s] @ a + H1[_s] @ b + H2[_s] @ c).value
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
    data_end,
    data_start,
    day_slct,
    dh,
    model,
    np,
    plt,
    residuals,
    take_log,
    x1,
    x1_max,
    x2,
    x2_max,
    y,
    y_max,
):
    _fig, _ax = plt.subplots(nrows=4, sharex=True, figsize=(12, 6))
    _n = dh.filled_data_matrix.shape[0]
    _ix = _n * day_slct.value
    _day_window = 5
    _m = np.copy(model)
    _m[~dh.boolean_masks.daytime[:, data_start.value:data_end.value].ravel(order='F')] = np.nan
    _ax[0].plot(dh.raw_data_matrix.ravel(order='F')[_ix:_ix+_n*_day_window], linewidth=1, color='black', label='original data')
    if take_log.value:
        _ax[0].plot(y_max * np.exp(y[_ix:_ix+_n*_day_window]), label='target')
        _ax[0].plot(y_max * np.exp(_m[_ix:_ix+_n*_day_window]), label='model')
    else:
        _ax[0].plot(y_max * y[_ix:_ix+_n*_day_window], label='target')
        _ax[0].plot(y_max * _m[_ix:_ix+_n*_day_window], label='model')
    _ax[1].plot(residuals.ravel(order='F')[_ix:_ix+_n*_day_window], label='residuals')
    _ax[2].plot(x1[_ix:_ix+_n*_day_window] * x1_max, label='module temp')
    _ax[3].plot(x2[_ix:_ix+_n*_day_window] * x2_max, label='POA')
    _ax[0].legend()
    _ax[1].legend()
    _ax[2].legend()
    _ax[3].legend()
    _ax[0].set_ylabel('target')
    _ax[1].set_ylabel('residual')
    _ax[2].set_ylabel('deg C')
    _ax[3].set_ylabel('W/m^2')
    _fig
    return


@app.cell
def _(np, plt, sns, trend):
    with sns.axes_style('whitegrid'):
        plt.plot(np.arange(len(trend.value)) / (365), np.exp(trend.value))
        # plt.axvline(365, color='orange', ls=':', label='year marker')
        # plt.axvline(365*2, color='orange', ls=':')
        # plt.axvline(365*3, color='orange', ls=':')
        # plt.axvline(365*4, color='orange', ls=':')
        plt.xlabel('years')
        # plt.legend()
        plt.title('Degradation term over time')
    plt.gcf()
    return


@app.cell
def _(dh, model, np, plt, sns, take_log, y, y_max):
    plt.figure(figsize=(10, 6))
    if take_log.value:
        residuals = y_max * (np.exp(y) - np.exp(model))
    else:
        residuals = y_max * (y - model)
    _ = plt.figure(figsize=(10, 5))
    sns.heatmap(residuals.reshape((dh.raw_data_matrix.shape[0], -1), order='F'), cmap='seismic', center=0, ax=plt.gca())
    plt.title("residuals heatmap")
    # mo.mpl.interactive(plt.gcf())
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
def _(model, np, plt, res_dist, sm, stats, y):
    _r  = np.exp(y) - np.exp(model)
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
def _(H1, b, np, plt, x1, x1_max):
    # plt.plot(x1, y, marker='.', ls='none', alpha=0.4)
    plt.plot(x1 * x1_max, np.exp((H1 @ b).value), ls='none', marker='.', markersize=1)
    plt.title('Inferred temperature response')
    plt.xlabel('module temp [deg C]')
    plt.ylabel('correction factor [1]')
    plt.gcf()
    return


@app.cell
def _(H2, c, np, plt, x2, x2_max):
    plt.plot(x2 * x2_max, np.exp((H2 @ c).value), ls='none', marker='.', markersize=1)
    plt.title('Inferred irradiance response')
    plt.xlabel('POA irradiance [W/m^s]')
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
    import cvxpy as cvx
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import statsmodels.api as sm
    from sklearn.metrics import r2_score
    from sklearn.mixture import GaussianMixture
    import statsmodels.api as sm
    from spcqe import make_basis_matrix, make_regularization_matrix
    from io import BytesIO
    return (
        DataHandler,
        cvx,
        make_basis_matrix,
        make_regularization_matrix,
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
def _(np):
    def d_func(x, k, k_max):
        n1 = np.clip(np.power(x - k, 3), 0, np.inf)
        n2 = np.clip(np.power(x - k_max, 3), 0, np.inf)
        d1 = k_max - k
        out = (n1 - n2) / d1
        return out


    def make_H(x, knots, include_offset=False):
        nK = len(knots)
        H = np.ones((len(x), nK), dtype=float)
        H[:, 1] = x
        for _i in range(nK - 2):
            _j = _i + 2
            H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(
                x, knots[-2], knots[-1]
            )
        if include_offset:
            return H
        else:
            return H[:, 1:]
    return (make_H,)


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
