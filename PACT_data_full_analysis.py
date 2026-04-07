import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up runs
    """)
    return


@app.cell
def _(glob, mo, reload_config_files):
    reload_config_files.value
    configs = glob.glob('*.yaml')
    configs.sort()
    slct_config = mo.ui.dropdown(configs, label='select experiment config file', value=configs[0])
    return (slct_config,)


@app.cell
def _(mo):
    reload_config_files = mo.ui.button(label='reload config files')
    return (reload_config_files,)


@app.cell
def _(mo, reload_config_files, slct_config):
    mo.hstack([slct_config, reload_config_files])
    return


@app.cell
def _(mo, slct_config):
    key = slct_config.value.split('.')[0]
    mo.md('### '+key)
    return (key,)


@app.cell
def _(load_config, slct_config):
    config, runs = load_config(slct_config.value)
    num_runs = len(runs)
    return config, num_runs, runs


@app.cell
def _(config):
    config
    return


@app.cell
def _(mo, num_runs):
    mo.md(f"""
    Given configuration will result in **{num_runs}** runs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run experiments
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label='run/load experiment')
    run_button
    return (run_button,)


@app.cell
def _(Path, key, mo, np, pd, run_button, run_experiment, runs):
    mo.stop(not run_button.value)
    if Path(key+'.csv').exists():
        results_p = pd.read_csv(key+'.csv', index_col=0)
    else:
        results = pd.DataFrame(columns=['sample', 'target', 'takeLog', 'inclPeriodic', 'inclExog', 'trend', 'knots', 'holdout', 'firstHarmonic', 'secondHarmonic', 'rmse', 'rmse%', 'R2'])
        for _ix, _r in enumerate(mo.status.progress_bar(runs)):
            rmse, rmse_percent, rsquare = run_experiment(_r)
            results.loc[_ix] = np.r_[_r, [rmse, rmse_percent, rsquare]]
        results_p = results.copy()
        # fix data types
        results_p['rmse'] = np.array(results_p['rmse'].values, dtype=float)
        results_p['rmse%'] = np.array(results_p['rmse%'].values, dtype=float)
        results_p['R2'] = np.array(results_p['R2'].values, dtype=float)
        results_p['knots'] = np.array(results_p['knots'].values, dtype=int)
        results_p['firstHarmonic'] = np.array(results_p['firstHarmonic'].values, dtype=int)
        results_p['secondHarmonic'] = np.array(results_p['secondHarmonic'].values, dtype=int)
        results_p['takeLog'] = results_p['takeLog'] == 'True'
        results_p['inclPeriodic'] = results_p['inclPeriodic'] == 'True'
        results_p['inclExog'] = results_p['inclExog'] == 'True'
        results_p.to_csv(key+'.csv')
    return (results_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## View Results
    """)
    return


@app.cell
def _(config, mo):
    slct_sample = mo.ui.dropdown(options=config['sample_options'], value=config['sample_options'][0], label='select sample')
    slct_target = mo.ui.dropdown(options=config['targets'], value=config['targets'][0], label='select target')
    mo.hstack([slct_sample, slct_target])
    return slct_sample, slct_target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Holdout every 5th week
    """)
    return


@app.cell
def _(results_p, slct_sample, slct_target):
    results_p[(results_p['sample'] == slct_sample.value) & (results_p['target'] == slct_target.value) & (results_p['holdout'] == 'every 5th week')]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Holdout last 20%
    """)
    return


@app.cell
def _(results_p, slct_sample, slct_target):
    results_p[(results_p['sample'] == slct_sample.value) & (results_p['target'] == slct_target.value) & (results_p['holdout'] == 'last 20%')]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Holdout every 5th day
    """)
    return


@app.cell
def _(results_p, slct_sample, slct_target):
    results_p[(results_p['sample'] == slct_sample.value) & (results_p['target'] == slct_target.value) & (results_p['holdout'] == 'every 5th day')]
    return


@app.cell
def _(config, mo):
    slct_holdout = mo.ui.dropdown(config['holdout_method'], value=config['holdout_method'][1], label='select holdout method')
    slct_holdout
    return (slct_holdout,)


@app.cell
def _(attributions, config, np, plt, results_p, slct_holdout, waterfall_plot):
    baseline = model_wrapper(results_p[results_p['holdout'] == slct_holdout.value], [0, 0, 0])['rmse%'].values[0]
    data = {
        "amount": np.r_[[baseline], list(attributions.values())]
    }


    index = np.r_[['offset'], list(attributions.keys())]
    waterfall_plot(data, index)
    plt.title(f'component impact on RMSE reduction, {config['sample_options'][0]}, {config['targets'][0]}, holdout {slct_holdout.value}')
    return


@app.cell
def _(attribut_r2, config, np, plt, results_p, slct_holdout, waterfall_plot):
    _baseline = model_wrapper(results_p[results_p['holdout'] == slct_holdout.value], [0, 0, 0])['R2'].values[0]
    _data = {
        "amount": np.r_[[_baseline], list(attribut_r2.values())]
    }


    _index = np.r_[['offset'], list(attribut_r2.keys())]
    waterfall_plot(_data, _index, n_digits=3)
    plt.title(f'component impact on out-of-sample $R^2$, {config['sample_options'][0]}, {config['targets'][0]}, holdout {slct_holdout.value}')
    plt.ylabel('$R^2$')
    plt.ylim(-0.1, 1.3)
    plt.gcf()
    return


@app.cell
def _(results_p, slct_holdout):
    model_wrapper(results_p[results_p['holdout'] == slct_holdout.value], [0, 0, 0])['R2'].values[0]
    return


@app.cell
def _(key, mo, results_p, shapely_analysis, slct_holdout):
    if 'shapley' not in key:
        mo.stop(True)
    attributions = shapely_analysis(results_p[results_p['holdout'] == slct_holdout.value])
    attribut_r2 = shapely_analysis(results_p[results_p['holdout'] == slct_holdout.value], metric='R2')
    # plt.stem(np.abs(list(attributions.values())))
    # plt.xticks(range(3), list(attributions.keys()))
    # plt.title(f'component strength, {config['sample_options'][0]}, {config['targets'][0]}, holdout {slct_holdout.value}')
    # plt.gcf()
    return attribut_r2, attributions


@app.cell
def _(np, plt, results_p, slct_holdout, slct_sample, slct_target):
    er1 = np.array(results_p[(results_p['sample'] == slct_sample.value) 
        & (results_p['target'] == slct_target.value) 
        & (results_p['holdout'] == slct_holdout.value)
        & (results_p['trend'] == 'none')]['rmse%'].values, dtype=float)
    er2 = np.array(results_p[(results_p['sample'] == slct_sample.value) 
        & (results_p['target'] == slct_target.value) 
        & (results_p['holdout'] == slct_holdout.value)
        & (results_p['trend'] == 'linear')]['rmse%'].values, dtype=float)
    er3 = np.array(results_p[(results_p['sample'] == slct_sample.value) 
        & (results_p['target'] == slct_target.value) 
        & (results_p['holdout'] == slct_holdout.value)
        & (results_p['trend'] == 'nonlinear')]['rmse%'].values, dtype=float)
    plt.hist(er1, bins=20, label='none', alpha=0.5)
    plt.hist(er2, bins=20, label='linear', alpha=0.5)
    plt.hist(er3, bins=20, label='nonlinear', alpha=0.5)
    plt.legend()
    plt.title('RMSE%, '+slct_sample.value+', '+slct_target.value+', hold out '+slct_holdout.value)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports and functions
    """)
    return


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
    from cvxpy import SolverError
    import yaml
    import itertools

    return (
        Literal,
        Path,
        SolverError,
        TsgamArConfig,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        TsgamTrendConfig,
        Tuple,
        glob,
        itertools,
        mo,
        np,
        pd,
        plt,
        yaml,
    )


@app.cell
def _(itertools, yaml):
    def load_config(yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        # Generate all combinations
        runs = list(itertools.product(
            config['sample_options'],
            config['targets'],
            config['take_log'],
            config['use_periodic'],
            config['use_exog'],
            config['trends'],
            config['num_knots'],
            config['holdout_method'],
            config['first_harmonics'],
            config['second_harmonics']
        ))

        return config, runs

    return (load_config,)


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
                    return np.nan, np.nan, np.nan
                else:
                    print('yay scs worked :)')
            else:
                print('yay clarabel worked :)')
        y_hat = tsgam.predict(X=test[['poa_global', 'temperature_module']])
        ssquare = lambda x: np.nansum(np.square(x))
        if run_config[2]:
            rmse = np.sqrt(np.mean(np.power(np.exp(y_hat) - np.exp(test[run_config[1]]), 2)))
            rmse_percent = rmse / np.median(np.exp(test[run_config[1]])) * 100
            # rsquare = (ssquare(np.exp(test[run_config[1]])) - ssquare(np.exp(y_hat) - np.exp(test[run_config[1]]))) / ssquare(np.exp(test[run_config[1]]))
            rsquare = 1 - ssquare(np.exp(y_hat) - np.exp(test[run_config[1]])) / ssquare(np.exp(np.nanmean(train[run_config[1]])) - np.exp(test[run_config[1]]))
        else:
            rmse = np.sqrt(np.mean(np.power((y_hat) - (test[run_config[1]]), 2)))
            rmse_percent = rmse / np.median((test[run_config[1]])) * 100
            # rsquare = (ssquare(test[run_config[1]]) - ssquare(y_hat - test[run_config[1]])) / ssquare(test[run_config[1]])
            rsquare = 1 - ssquare(y_hat - test[run_config[1]]) / ssquare(np.nanmean(train[run_config[1]]) - test[run_config[1]])
        return rmse, rmse_percent, rsquare

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
                reg_weight=1e-12
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
            verbose=False,
            # solver_opts={
            #     "mosek_params": {"MSK_IPAR_INTPNT_MAX_ITERATIONS": 1600}
            # }
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
def _(np):
    def enumerate_paths_full(origin, destination, path=None):
        """
        recursive algorithm for generating all possible monotonically increasing paths between
        two points on a n-dimensional hypercube
        """
        origin = list(origin)
        destination = list(destination)
        correct_ordering = np.all(
            np.asarray(destination, dtype=int) - np.asarray(origin, dtype=int) >= 0
        )
        if not correct_ordering:
            raise ValueError("destination must be larger than origin in all dimensions")
        if path is None:
            path = []
        paths = []
        if origin == destination:
            # a path has been completed
            paths.append(path + [origin])
        else:
            # find the next index that can be incremented
            for i in range(len(origin)):
                if origin[i] != destination[i]:
                    # create the next point in this path
                    next_position = list(origin)
                    next_position[i] = destination[0]
                    # recurse to finish all paths that begin on this path
                    paths.extend(
                        enumerate_paths_full(next_position, destination, path + [origin])
                    )

        return paths


    def enumerate_paths(n, dtype=int):
        """
        enumerates all possible paths from the origin to the ones vector in R^n
        """
        origin = np.zeros(n, dtype=dtype)
        destination = np.ones(n, dtype=dtype)
        return np.asarray(enumerate_paths_full(origin, destination))

    return (enumerate_paths,)


@app.function
def model_wrapper(df, encoding):
    a = bool(encoding[0])
    b = bool(encoding[1])
    s = set(df['trend'])
    c = 'none' if encoding[2] == 0 else s.difference(set(['none'])).pop()
    out = df[(df['inclPeriodic'] == a)
        & (df['inclExog'] == b)
        & (df['trend'] == c)]
    return out


@app.cell
def _(enumerate_paths, np):
    def shapely_analysis(df, metric='rmse%'):
        labels=['Periodic', 'Exog', 'Trend']
        paths = enumerate_paths(3)
        rmses = np.zeros((paths.shape[0], paths.shape[1]))
        for _ix, _path in enumerate(paths):
                for _jx, _point in enumerate(_path):
                    rmses[_ix, _jx] = model_wrapper(df, _point)[metric].values[0]
        lifts = np.diff(rmses, axis=1)
        path_diffs = np.diff(paths, axis=1)
        ordering = np.argmax(path_diffs, axis=-1)
        ordered_lifts = np.take_along_axis(lifts, np.argsort(ordering, axis=1), axis=1)
        attributions = np.average(ordered_lifts, axis=0)
        return {_k: _v for _k, _v in zip(labels, attributions)}

    return (shapely_analysis,)


@app.cell
def _(np, pd):
    def waterfall_plot(data, index, n_digits=1, figsize=(10, 4)):
        """
        Create a waterfall plot to visualize the breakdown of energy loss factors.

        This function generates a waterfall plot to display the cumulative impact of sequential
        loss factors. Each bar in the plot represents a specific loss factor..

        :param data: Data to be plotted. This should be a `pandas.Series` or `pandas.DataFrame`
                     where the index represents categories and the values represent the amounts.
        :type data: pd.Series or pd.DataFrame
        :param index: Index to use for the plot. Should match the length of the `data`.
        :type index: pd.Index
        :param figsize: Size of the figure to create, given as (width, height) in inches.
                        Defaults to (10, 4).
        :type figsize: tuple of int, optional

        :return: The figure object containing the waterfall plot.
        :rtype: matplotlib.figure.Figure
        """
        # Store data and create a blank series to use for the waterfall
        trans = pd.DataFrame(data=data, index=index)
        blank = trans.amount.cumsum().shift(1).fillna(0)
        bl = data['amount'][0]

        # Get the net total number for the final element in the waterfall
        total = trans.sum().amount
        trans.loc["full model"] = total
        blank.loc["full model"] = total

        # The steps graphically show the levels as well as used for label placement
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan

        # When plotting the last element, we want to show the full bar,
        # Set the blank to 0
        blank.loc["full model"] = 0

        # Plot and label
        my_plot = trans.plot(
            kind="bar",
            stacked=True,
            bottom=blank,
            legend=None,
            figsize=figsize,
            title="Component predictive power waterfall",
        )
        my_plot.plot(step.index, step.values, "k")
        my_plot.set_xlabel("Model components")
        my_plot.set_ylabel("RMSE (%)")

        # Get the y-axis position for the labels
        y_height = trans.amount.cumsum().shift(1).fillna(0)

        # Get an offset so labels don't sit right on top of the bar
        max = trans.max()
        max = max.iloc[0]
        neg_offset = max / 25
        pos_offset = max / 50
        plot_offset = int(max / 15)

        # Start label loop
        loop = 0
        str_code = f":,.{n_digits}f"
        for index, row in trans.iterrows():
            # For the last item in the list, we don't want to double count
            if row["amount"] == total:
                y = y_height.iloc[loop]
            else:
                y = y_height.iloc[loop] + row["amount"]
            # Determine if we want a neg or pos offset
            if row["amount"] > 0:
                y += pos_offset
            else:
                y -= neg_offset
            my_plot.annotate(("{"+str_code+"}").format(row["amount"]), (loop, y), ha="center")
            loop += 1

        # Scale up the y axis so there is room for the labels
        # my_plot.set_ylim(0, 1.1*(blank.max() + int(plot_offset))) # 1.1 for a lil bit extra
        # print(bl, trans.min(), max, total)
        ymin = np.min([np.min([bl, total]) - 0.1 * np.abs(bl - total), 0])
        ymax = np.max([bl, total]) + 0.2 * np.abs(bl - total)
        my_plot.set_ylim(ymin, ymax)
        # Rotate the labels
        my_plot.set_xticklabels(trans.index, rotation=0)
        fig = my_plot.get_figure()
        fig.set_layout_engine(layout="tight")
        return fig

    return (waterfall_plot,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
