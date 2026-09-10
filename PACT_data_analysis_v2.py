import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(load_data_button, mo, slct_sample):
    mo.hstack([slct_sample, load_data_button])
    return


@app.cell
def _(mo):
    fit_model = mo.ui.run_button(label='fit model')
    target = mo.ui.dropdown(['pmp', 'imp', 'vmp'], value='pmp', label='target data')
    use_periodic = mo.ui.switch(label='periodic terms', value=True)
    use_exog = mo.ui.switch(label='exogenous terms', value=True)
    trend_type = mo.ui.dropdown(options=['none', 'linear', 'nonlinear'], label='trend type', value='linear')
    take_log = mo.ui.switch(label='take log', value=True)
    num_knots = mo.ui.number(start=2, stop=50, value=16, label='number of spline knots', full_width=True)
    holdout_slct = mo.ui.dropdown(options=['last 20%', 'every 5th day', 'every 5th week'], label='holdout', value='last 20%')
    harmonics_1 = mo.ui.number(start=0, stop=10, value=1, label='harmonics for period 1', full_width=True)
    harmonics_2 = mo.ui.number(start=0, stop=100, value=16, label='harmonics for period 2', full_width=True)
    mo.vstack([
        fit_model,
        mo.hstack([target, take_log, use_periodic, harmonics_1, harmonics_2]),
        mo.hstack([use_exog, num_knots, trend_type, holdout_slct])
    ])
    return (
        fit_model,
        harmonics_1,
        harmonics_2,
        holdout_slct,
        num_knots,
        take_log,
        target,
        trend_type,
        use_exog,
        use_periodic,
    )


@app.cell
def _(
    error_results,
    holdout_scatter,
    irrad_response,
    mo,
    pi_plot,
    temp_response,
    train_scatter,
):
    mo.ui.tabs({
        'Error summary': error_results,
        'Test scatterplot': holdout_scatter,
        'Train scatterplot': train_scatter,
        'PI and degradation': pi_plot,
        'Irrad response': irrad_response,
        'Temp response': temp_response
    }, lazy=False)
    return


@app.cell
def _(test_ts):
    test_ts
    return


@app.cell
def _(train_ts):
    train_ts
    return


@app.cell
def _(np, pd, result):
    error_results = pd.DataFrame(np.array([["RMSE", f"{result['rmse']:.2e}"],
                                           ["RMSE %", f"{result['rmse%']:.3}%"],
                                           ["out of sample R2", f"{result['oos-R2']:.3}"]]),
                                 columns=['metric', 'value'])
    return (error_results,)


@app.cell
def _(mo, np, pd, plt, result, slct_sample, take_log, target):
    # TODO: fix this analysis when not using log(y) model. The additive trend offset doesn't directly translate into a PI trend, in the way
    # that the multiplicative trend does with a log model.
    if take_log.value:
        _func = lambda _x: np.exp(_x)
        _offset = 0
    else:
        _func = lambda _x: _x
        _offset = 1
    temp_df = pd.DataFrame(_func(result['train'][target.value]))
    temp_df.index = pd.DatetimeIndex(temp_df.index)
    try:
        temp_df['est'] = _func(result['yhat_train_notrend'])
    except NameError:
        mo.stop(True)
    temp_grouped = temp_df.groupby(pd.Grouper(freq='1D')).aggregate(np.nansum)
    # temp_grouped.loc[temp_grouped['est'] < np.nanquantile(temp_grouped[target.value], 0.95) * 0.3, ['est']] = np.nan
    temp_grouped['PI'] = temp_grouped[target.value] / temp_grouped['est']
    try:
        trend =result['tsgam'].variables_['trend'].value
    except KeyError:
        trend = np.zeros_like(temp_grouped['PI'].index, dtype=float) if take_log.value else np.ones_like(temp_grouped['PI'].index, dtype=float)
    try:
        plt.plot(temp_grouped['PI'].index[:-1], temp_grouped['PI'].values[:-1], color='C0', label='tsgam PI', marker='.', ls='none')
        plt.plot(temp_grouped['PI'].index[:-1], _func(trend)+_offset, color='C1', label='trend')
    except ValueError:
        plt.plot(temp_grouped['PI'].index, temp_grouped['PI'].values, color='C0', label='tsgam PI', marker='.', ls='none')
        plt.plot(temp_grouped['PI'].index, _func(trend)+_offset, color='C1', label='trend')
    # _ax[1].plot(temp_grouped[target.value])
    # _ax[1].plot(temp_grouped['est'])
    # plt.ylim((.88 * np.min(_func(trend)), 1.15 * np.max(_func(trend))))
    plt.xticks(rotation=45)
    plt.title('sample: '+slct_sample.value + ', '+ target.value + ', daily trend analysis')
    if np.min(_func(trend)) <= 0.8:
        plt.axhline(0.8, ls='--', color='red', label='80% loss')
    plt.legend()
    plt.tight_layout()
    # plt.ylim(0.8, 1.15)
    pi_plot = plt.gcf()
    return (pi_plot,)


@app.cell
def _(np, plt, result, take_log, target):
    if take_log.value:
        _func = lambda _x: np.exp(_x)
    else:
        _func = lambda _x: _x
    plt.scatter(_func(result['yhat_test']), _func(result['test'][target.value]), s=1)
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-100, 100], [-100, 100], color='red', ls='--')
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('Holdout prediction')
    # plt.axhline(np.median(np.exp(test_data[target.value])), color='gray', ls='--', linewidth=1)
    holdout_scatter = plt.gcf()
    return (holdout_scatter,)


@app.cell
def _(np, plt, result, take_log, target):
    if take_log.value:
        _func = lambda _x: np.exp(_x)
    else:
        _func = lambda _x: _x
    plt.scatter(_func(result['yhat_train']), _func(result['train'][target.value]), s=1, c=result['train'].index.day_of_year / 365 + result['train'].index.year)
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-100, 100], [-100, 100], color='red', ls='--')
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('Training data prediction')
    _cbar = plt.colorbar()
    _cbar.set_label('time [year]', rotation=90, labelpad=10) 
    train_scatter = plt.gcf()
    return (train_scatter,)


@app.cell
def _(holdout_slct, mo, np, plt, result, take_log, target):
    if take_log.value:
        _func = lambda _x: np.exp(_x)
    else:
        _func = lambda _x: _x
    plt.plot(result['test'].index, _func(result['test'][target.value]), linewidth=.2, marker='.', markersize=1, label='actual')
    plt.plot(result['test'].index, _func(result['yhat_test']), linewidth=.2, marker='.', markersize=1, label='predicted')
    plt.xticks(rotation=45)
    plt.title('Actual and predicted on test data set, '+holdout_slct.value)
    plt.legend()
    plt.tight_layout()
    test_ts = mo.mpl.interactive(plt.gcf())
    return (test_ts,)


@app.cell
def _(mo, np, plt, result, take_log, target):
    if take_log.value:
        _func = lambda _x: np.exp(_x)
    else:
        _func = lambda _x: _x
    plt.plot(result['train'].index, _func(result['train'][target.value]), linewidth=.2, marker='.', markersize=1, label='actual')
    plt.plot(result['train'].index, _func(result['yhat_train']), linewidth=.2, marker='.', markersize=1, label='predicted')
    plt.xticks(rotation=45)
    plt.title('Actual and predicted on train data set')
    plt.legend()
    plt.tight_layout()
    train_ts = mo.mpl.interactive(plt.gcf())
    return (train_ts,)


@app.cell
def _(np, plt, result, take_log, target):
    if take_log.value:
        _func = lambda _x: np.exp(_x)
    else:
        _func = lambda _x: _x
    irrad_vars = result['tsgam'].variables_['exog_coef_0'].value
    # irrad_knots = tsgam.exog_knots_[0]
    _x_vals = result['train']['poa_global'].values
    _H = result['tsgam']._make_H(_x_vals, result['knot_points'], include_offset=False)
    plt.scatter(result['train']['poa_global'].values, _func(result['train'][target.value].values), s=1, 
                c=result['train'].index.day_of_year / 365 + result['train'].index.year, label='measured')
    plt.plot(_x_vals, _func(_H @ irrad_vars - np.mean(_H @ irrad_vars) + np.mean(result['train'][target.value].values)), marker='.', ls='none', color='black', label='average model response')
    plt.legend()
    _ax = plt.gca()
    _cbar = plt.colorbar()
    _cbar.set_label('time [year]', rotation=90, labelpad=10) 
    _ax.set_xlabel('irradiance')
    _ax.set_ylabel(target.value)
    irrad_response = plt.gcf()
    return (irrad_response,)


@app.cell
def _(np, plt, result, take_log, target):
    if take_log.value:
        _func = lambda _x: np.exp(_x)
    else:
        _func = lambda _x: _x
    temp_vars = result['tsgam'].variables_['exog_coef_1'].value
    temp_knots = result['tsgam'].exog_knots_[1]
    _x_vals = result['train']['temperature_module'].values
    _H = result['tsgam']._make_H(_x_vals, temp_knots, include_offset=False)
    plt.scatter(result['train']['temperature_module'].values, _func(result['train'][target.value].values), s=1, 
                c=result['train'].index.day_of_year / 365 + result['train'].index.year, label='measured')
    plt.plot(_x_vals, _func(_H @ temp_vars - np.quantile(_H @ temp_vars, .9) + np.quantile(result['train'][target.value].values, .9)), marker='.', ls='none', color='black', label='average model response')
    plt.legend()
    _ax = plt.gca()
    _cbar = plt.colorbar()
    _cbar.set_label('time [year]', rotation=90, labelpad=10) 
    _ax.set_xlabel('module temp')
    _ax.set_ylabel(target.value)
    temp_response = plt.gcf()
    return (temp_response,)


@app.cell
def _(fit_model, mo, run_config, run_experiment):
    mo.stop(not fit_model.value)

    result = run_experiment(run_config)
    return (result,)


@app.cell
def _(
    harmonics_1,
    harmonics_2,
    holdout_slct,
    num_knots,
    slct_sample,
    take_log,
    target,
    trend_type,
    use_exog,
    use_periodic,
):
    run_config = [
        slct_sample.value,
        target.value,
        take_log.value,
        use_periodic.value,
        use_exog.value,
        trend_type.value,
        num_knots.value,
        holdout_slct.value,
        harmonics_1.value,
        harmonics_2.value
    ]
    return (run_config,)


@app.cell
def _():
    import marimo as mo
    import glob as glob
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from solardatatools import standardize_time_axis, make_2d, plot_2d
    from tsgam_estimator import TsgamEstimator, TsgamEstimatorConfig, TsgamMultiPeriodicConfig, TsgamSplineConfig, TsgamTrendConfig, TsgamArConfig, TsgamSolverConfig
    from typing import Tuple, Literal
    import itertools

    return (
        Literal,
        Path,
        TsgamArConfig,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        TsgamTrendConfig,
        Tuple,
        glob,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    _options = [
        'P-0038-01',
        'P-0038-02',
        'P-0038-03',
        'P-0050-01',
        'P-0050-04',
        'P-0050-05',
        'P-0061-02',
        'P-0061-12',
        'P-0061-15'
    ]
    slct_sample = mo.ui.dropdown(_options, label='select sample', value=_options[3])
    load_data_button = mo.ui.run_button(label='load data')
    return load_data_button, slct_sample


@app.cell
def _(
    SolverError,
    TsgamEstimator,
    load_data,
    make_config,
    make_irrad_knots,
    make_train_test,
    np,
):
    def run_experiment(run_config):
        df = load_data(run_config[0])
        train, test = make_train_test(df, run_config[0], run_config[1], run_config[2], run_config[7])
        knot_points = make_irrad_knots(train, run_config[6])
        config = make_config(knot_points, run_config[3], run_config[4], run_config[5], 
                             first_harmonic=run_config[8], second_harmonic=run_config[9])
        try:
            tsgam = TsgamEstimator(config)
            tsgam.fit(X=train[['poa_global', 'temperature_module']], y=train[run_config[1]])
        except SolverError:
            print(f'mosek failed on {run_config}, trying clarabel')
            try:
                tsgam.config.solver_config.solver = 'clarabel'
                tsgam.fit(X=train[['poa_global', 'temperature_module']], y=train[run_config[1]])
            except SolverError:
                print(f'clarabel failed on {run_config}, trying scs')
                try:
                    tsgam.config.solver_config.solver = 'scs'
                    tsgam.fit(X=train[['poa_global', 'temperature_module']], y=train[run_config[1]])
                except SolverError:
                    print('all solvers failed :(')
                    return
                else:
                    print('yay scs worked :)')
            else:
                print('yay clarabel worked :)')
        y_hat = tsgam.predict(X=test[['poa_global', 'temperature_module']])
        y_hat_train = tsgam.predict(X=train[['poa_global', 'temperature_module']])
        y_hat_train_no_trend = tsgam.predict(X=train[['poa_global', 'temperature_module']], remove_trend=True)
        ssquare = lambda x: np.nansum(np.square(x))
        if run_config[2]:
            rmse = np.sqrt(np.mean(np.power(np.exp(y_hat) - np.exp(test[run_config[1]]), 2)))
            rmse_percent = rmse / np.median(np.exp(test[run_config[1]])) * 100
            rsquare = 1 - ssquare(np.exp(y_hat) - np.exp(test[run_config[1]])) / ssquare(np.exp(np.nanmean(train[run_config[1]])) - np.exp(test[run_config[1]]))
        else:
            rmse = np.sqrt(np.mean(np.power((y_hat) - (test[run_config[1]]), 2)))
            rmse_percent = rmse / np.median((test[run_config[1]])) * 100
            rsquare = 1 - ssquare(y_hat - test[run_config[1]]) / ssquare(np.nanmean(train[run_config[1]]) - test[run_config[1]])
        out_dict = {
            'tsgam': tsgam,
            'config': config,
            'train': train,
            'test': test,
            'yhat_test': y_hat,
            'yhat_train': y_hat_train,
            'yhat_train_notrend': y_hat_train_no_trend,
            'rmse': rmse,
            'rmse%': rmse_percent,
            'oos-R2': rsquare,
            'knot_points': knot_points
            }
        return out_dict

    return (run_experiment,)


@app.cell
def _(Path, glob, mo, np, pd):
    @mo.cache
    def load_data(sample):
        path_to_data = Path('.') / 'data' / (sample + '_Sample_MPPT')
        data_files = glob.glob(str(path_to_data / '*.csv'))
        data_files.sort()
        dfs = [pd.read_csv(_f, index_col=0, parse_dates=[0]) for _f in data_files]
        df = pd.concat(dfs)
        if True: # process time stamps
            try:
                df = df.tz_convert('Etc/GMT+6').tz_localize(None)
            except TypeError:
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.tz_convert('Etc/GMT+6').tz_localize(None)
            try:
                df.sort_index(inplace=True)
            except TypeError:
                pass
        df[df.values > 1e10] = np.nan # very large values are not usually correct
        df['pmp'] = df['imp'] * df['vmp']
        grouped = df.groupby(pd.Grouper(freq='5min'))
        df_5min = grouped.aggregate(np.nanmean)
        return df_5min

    return (load_data,)


@app.cell
def _(np):
    def replace_outliers_with_nan(array, remove='both', size=1.5):
        Q1 = np.nanquantile(array, q=0.25)
        Q3 = np.nanquantile(array, q=0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - size * IQR
        upper_bound = Q3 + size * IQR

        # Replace outliers with NaN
        data_cleaned = array.astype(float)
        if remove == 'both':
            data_cleaned[(data_cleaned < lower_bound) | (data_cleaned > upper_bound)] = np.nan
        elif remove == 'upper':
            data_cleaned[(data_cleaned > upper_bound)] = np.nan
        elif remove == 'lower':
            data_cleaned[(data_cleaned < lower_bound)] = np.nan
        return data_cleaned

    return (replace_outliers_with_nan,)


@app.cell
def _(mo, np, replace_outliers_with_nan, train_test_split_temporal):
    @mo.cache
    def make_train_test(df_5min, sample, target, take_log, holdout_method):
        training_data = df_5min.dropna()[['poa_global', 'temperature_module', target]].copy()
        if 'P-0038' in sample:
            training_data['temperature_module'] = replace_outliers_with_nan(training_data['temperature_module'].values, 
                                                                            remove='both', size=1.5)
        if target == 'vmp':
            training_data.loc[training_data[target] < 1.5, target] = np.nan
        training_data.loc[training_data[target] < 0.01 * np.quantile(training_data[target], 0.98), target] = np.nan
        training_data.loc[training_data['poa_global'] < 0, 'poa_global'] = np.nan
        training_data = training_data.dropna()
        if take_log:
            training_data[target] = np.log(training_data[target])
        # and now split the data
        if holdout_method == 'last 20%':
            data_size = len(training_data)
            test_data = training_data.iloc[int(data_size * .8):]
            training_data = training_data.iloc[:int(data_size * .8)]
        elif holdout_method == 'every 5th day':
            train_ix, test_ix = train_test_split_temporal(training_data.index, N=5, how='daily')
            test_data = training_data.loc[test_ix]
            training_data = training_data.loc[train_ix]
        elif holdout_method == 'every 5th week':
            train_ix, test_ix = train_test_split_temporal(training_data.index, N=5, how='weekly')
            test_data = training_data.loc[test_ix]
            training_data = training_data.loc[train_ix]
        return training_data, test_data

    return (make_train_test,)


@app.cell
def _(np):
    def make_irrad_knots(train_data, num_knots):
        knot_points = np.logspace(np.log10(np.min(train_data['poa_global'])), 
                                  np.log10(np.max(train_data['poa_global'])), num_knots)
        return knot_points

    return (make_irrad_knots,)


@app.cell
def _(
    TsgamArConfig,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
):
    def make_config(knot_points, use_periodic, use_exog, trend_type, first_harmonic=2, second_harmonic=24):
        multiperiodic_config = TsgamMultiPeriodicConfig(
                num_harmonics=[first_harmonic, second_harmonic],
                periods=[365.2425*24*60/5, 24*60/5],
                reg_weight=1e-2
            )
        if trend_type == 'linear':
            trend_config = TsgamTrendConfig(
                trend_type='linear',
                grouping=24*60/5
            )
        elif trend_type == 'nonlinear':
            trend_config = TsgamTrendConfig(
                trend_type='nonlinear',
                grouping=24*60/5,
                reg_weight=1e-12
            )
        elif trend_type == 'none':
            trend_config = None
        exog_config_irrad = TsgamSplineConfig(
            knots=knot_points,
            n_knots=None,
            # lags=[-2, -1, 0, 1, 2],
            lags=[0],
            reg_weight=1e0,  # Regularization weight for coefficients
            diff_reg_weight=1e-2  # Regularization weight for differences between lags
        )
        exog_config_temp = TsgamSplineConfig(
            knots=[],  # Empty list means knots will be auto-generated from data
            n_knots=10,  # Number of knots to generate
            lags=[0],
            reg_weight=1e0,  # Regularization weight for coefficients
            diff_reg_weight=1e-2  # Regularization weight for differences between lags
        )

        ar_config = TsgamArConfig(
            lags = list(range(1,int(24*2*60/5)))
        )
        solver_config = TsgamSolverConfig(
            solver='mosek',
            verbose=False
        )
        if not use_periodic:
            multiperiodic_config = None
        if not use_exog:
            exog_config = None
        else:
            exog_config = [exog_config_irrad, exog_config_temp]
        tsgam_config = TsgamEstimatorConfig(
            multi_periodic_config=multiperiodic_config,
            # multi_periodic_config=None,
            exog_config=exog_config,
            # exog_config=None,
            trend_config=trend_config,
            ar_config = None,
            solver_config = solver_config,
            debug=True
        )
        return tsgam_config

    return (make_config,)


@app.cell
def _(Literal, Tuple, pd):
    def train_test_split_temporal(
        dt_index: pd.DatetimeIndex,
        N: int,
        how: Literal["daily", "weekly"] = "daily"
    ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Partition a datetime index into training and test sets based on temporal holdout.

        Parameters
        ----------
        dt_index : pd.DatetimeIndex
            The datetime index to partition (may be tz-aware)
        N : int
            The denominator of the holdout fraction. Every Nth day/week goes to test set.
            E.g., N=5 means 1/5 (20%) of days/weeks are held out for testing.
        how : {"daily", "weekly"}, default "daily"
            Whether to holdout by day or by week.
            - "daily": Every Nth day is held out
            - "weekly": Every Nth week is held out

        Returns
        -------
        train_index : pd.DatetimeIndex
            The training set datetime index
        test_index : pd.DatetimeIndex
            The test/holdout set datetime index

        Examples
        --------
        >>> dt_index = pd.date_range('2023-01-01', '2023-01-10', freq='H')
        >>> train, test = train_test_split_temporal(dt_index, N=5, how='daily')
        >>> # Days 5 and 10 (every 5th day) will be in test set
        """
        if not isinstance(dt_index, pd.DatetimeIndex):
            raise TypeError("dt_index must be a pandas DatetimeIndex")

        if N < 2:
            raise ValueError("N must be at least 2")

        if how not in ["daily", "weekly"]:
            raise ValueError("how must be either 'daily' or 'weekly'")

        # Normalize to date level (preserve timezone if present)
        if dt_index.tz is not None:
            dates = dt_index.tz_convert('UTC').normalize().tz_convert(dt_index.tz)
        else:
            dates = dt_index.normalize()

        # Get unique dates
        unique_dates = dates.unique()

        if how == "daily":
            # Get the day number since the first date
            first_date = unique_dates[0]
            day_numbers = (unique_dates - first_date).days

            # Every Nth day (starting from day N-1, using 0-indexing) goes to test
            # This means days at positions N-1, 2N-1, 3N-1, etc.
            test_mask = (day_numbers + 1) % N == 0
            test_dates = unique_dates[test_mask]

        else:  # weekly
            # Get ISO week number and year
            # We need to create a consistent week counter from the start
            unique_dates_series = pd.Series(unique_dates)

            # Calculate week number from the first date
            first_date = unique_dates[0]
            week_numbers = ((unique_dates - first_date).days // 7)

            # Every Nth week goes to test
            test_mask = (week_numbers + 1) % N == 0
            test_dates = unique_dates[test_mask]

        # Create boolean mask for original index
        test_mask_full = dates.isin(test_dates)
        train_mask_full = ~test_mask_full

        # Split the index
        train_index = dt_index[train_mask_full]
        test_index = dt_index[test_mask_full]

        return train_index, test_index

    return (train_test_split_temporal,)


@app.cell
def _():
    return


@app.cell
def _(load_data, load_data_button, mo, slct_sample):
    mo.stop(not load_data_button.value)

    df = load_data(slct_sample.value)
    return


if __name__ == "__main__":
    app.run()
