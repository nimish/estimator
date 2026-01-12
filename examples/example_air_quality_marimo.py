# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Marimo Notebook: Air Quality Forecasting with TSGAM

This notebook demonstrates forecasting PM2.5 air quality using:
- Multi-harmonic Fourier basis for seasonal patterns (daily, weekly, yearly)
- Temperature, dewpoint, wind speed, and pressure as exogenous variables with spline basis
- Autoregressive (AR) modeling of residuals

The notebook uses real air quality data from Beijing, China, which is publicly
available and includes both PM2.5 and meteorological measurements.

Adapted from general_tsgam_analysis_3.py plotting and analysis patterns.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Air Quality Forecasting with TSGAM

    This notebook demonstrates forecasting PM2.5 air quality using TSGAM.

    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import statsmodels.api as sm
    from pathlib import Path
    import sys
    import urllib.request
    import zipfile

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
        TsgamOutlierConfig,
        TsgamSolverConfig,
        PERIOD_HOURLY_DAILY,
        PERIOD_HOURLY_WEEKLY,
        PERIOD_HOURLY_YEARLY
    )
    return (
        PERIOD_HOURLY_DAILY,
        PERIOD_HOURLY_WEEKLY,
        PERIOD_HOURLY_YEARLY,
        Path,
        TsgamArConfig,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiHarmonicConfig,
        TsgamOutlierConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        mo,
        np,
        pd,
        plt,
        sm,
        sns,
        stats,
        urllib,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load and Prepare Data
    """)
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
def _(Path, download_beijing_air_quality_data, load_beijing_air_quality):
    # Setup paths
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data"

    # Download and load data
    data_file = download_beijing_air_quality_data(data_dir)
    df = load_beijing_air_quality(data_file)
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### View Data

    * data check:
      - reshape to day and time of day and do a heatmap
      - check for datetime consistency
    * UX: add in some checking for reasonableness
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Data Quality Check
    """)
    return


@app.cell
def _(df, pd):
    # Check datetime consistency and frequency
    _freq = pd.infer_freq(df.index)
    _expected_freq = 'H'  # Hourly

    # Check for missing timestamps
    _date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=_expected_freq)
    _missing_timestamps = _date_range.difference(df.index)

    # Check for duplicates
    _duplicate_timestamps = df.index.duplicated().sum()

    # Summary statistics
    print("Data Quality Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Inferred frequency: {_freq}")
    print(f"  Expected frequency: {_expected_freq}")
    print(f"  Missing timestamps: {len(_missing_timestamps)}")
    print(f"  Duplicate timestamps: {_duplicate_timestamps}")
    print("  Missing values per column:")
    for _col in df.columns:
        _missing = df[_col].isna().sum()
        _pct = (_missing / len(df)) * 100
        print(f"    {_col}: {_missing} ({_pct:.1f}%)")
    return


@app.cell
def _(df, plt, sns):
    # Reshape data for heatmap: days vs hours of day
    # Create day and hour columns
    _df_reshaped = df.copy()
    _df_reshaped['day'] = _df_reshaped.index.date
    _df_reshaped['hour'] = _df_reshaped.index.hour

    # Pivot to create day x hour matrix for PM2.5
    _heatmap_data = _df_reshaped.pivot_table(
        values='pm25',
        index='day',
        columns='hour',
        aggfunc='mean'
    )

    # Create figure with multiple heatmaps (3x2 grid)
    _fig_heatmap, _axes_heatmap = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))

    # PM2.5 heatmap
    sns.heatmap(_heatmap_data, cmap='viridis', ax=_axes_heatmap[0, 0], cbar_kws={'label': 'PM2.5 (μg/m³)'})
    _axes_heatmap[0, 0].set_title('PM2.5: Days vs Hours of Day', fontsize=12, fontweight='bold')
    _axes_heatmap[0, 0].set_xlabel('Hour of Day', fontsize=10)
    _axes_heatmap[0, 0].set_ylabel('Day', fontsize=10)
    _axes_heatmap[0, 0].tick_params(axis='y', labelsize=6)

    # Temperature heatmap
    _heatmap_temp = _df_reshaped.pivot_table(
        values='temperature',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    sns.heatmap(_heatmap_temp, cmap='plasma', ax=_axes_heatmap[0, 1], cbar_kws={'label': 'Temperature (°C)'})
    _axes_heatmap[0, 1].set_title('Temperature: Days vs Hours of Day', fontsize=12, fontweight='bold')
    _axes_heatmap[0, 1].set_xlabel('Hour of Day', fontsize=10)
    _axes_heatmap[0, 1].set_ylabel('Day', fontsize=10)
    _axes_heatmap[0, 1].tick_params(axis='y', labelsize=6)

    # Wind Speed heatmap
    _heatmap_wind = _df_reshaped.pivot_table(
        values='wind_speed',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    sns.heatmap(_heatmap_wind, cmap='coolwarm', ax=_axes_heatmap[1, 0], cbar_kws={'label': 'Wind Speed (m/s)'})
    _axes_heatmap[1, 0].set_title('Wind Speed: Days vs Hours of Day', fontsize=12, fontweight='bold')
    _axes_heatmap[1, 0].set_xlabel('Hour of Day', fontsize=10)
    _axes_heatmap[1, 0].set_ylabel('Day', fontsize=10)
    _axes_heatmap[1, 0].tick_params(axis='y', labelsize=6)

    # Rain Hours heatmap
    _heatmap_rain = _df_reshaped.pivot_table(
        values='rain_hours',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    sns.heatmap(_heatmap_rain, cmap='Blues', ax=_axes_heatmap[1, 1], cbar_kws={'label': 'Rain Hours'})
    _axes_heatmap[1, 1].set_title('Rain Hours: Days vs Hours of Day', fontsize=12, fontweight='bold')
    _axes_heatmap[1, 1].set_xlabel('Hour of Day', fontsize=10)
    _axes_heatmap[1, 1].set_ylabel('Day', fontsize=10)
    _axes_heatmap[1, 1].tick_params(axis='y', labelsize=6)

    # Pressure heatmap
    _heatmap_pres = _df_reshaped.pivot_table(
        values='pressure',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    sns.heatmap(_heatmap_pres, cmap='RdYlBu_r', ax=_axes_heatmap[2, 0], cbar_kws={'label': 'Pressure (hPa)'})
    _axes_heatmap[2, 0].set_title('Pressure: Days vs Hours of Day', fontsize=12, fontweight='bold')
    _axes_heatmap[2, 0].set_xlabel('Hour of Day', fontsize=10)
    _axes_heatmap[2, 0].set_ylabel('Day', fontsize=10)
    _axes_heatmap[2, 0].tick_params(axis='y', labelsize=6)

    # Hide the last subplot
    _axes_heatmap[2, 1].axis('off')

    plt.tight_layout()
    _fig_heatmap
    return


@app.cell
def _(df, plt):
    # Expanded to 3x3 grid to include all comparisons including pressure
    _fig_overview, _axes_overview = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

    # Time series plots (top row)
    _axes_overview[0, 0].plot(df.index, df['pm25'], linewidth=0.5, alpha=0.7)
    _axes_overview[0, 0].set_title('PM2.5 over time')
    _axes_overview[0, 0].set_ylabel('PM2.5 (μg/m³)')
    _axes_overview[0, 0].grid(True, alpha=0.3)

    _axes_overview[0, 1].plot(df.index, df['temperature'], linewidth=0.5, alpha=0.7, color='orange')
    _axes_overview[0, 1].set_title('Temperature over time')
    _axes_overview[0, 1].set_ylabel('Temperature (°C)')
    _axes_overview[0, 1].grid(True, alpha=0.3)

    _axes_overview[0, 2].plot(df.index, df['pressure'], linewidth=0.5, alpha=0.7, color='purple')
    _axes_overview[0, 2].set_title('Pressure over time')
    _axes_overview[0, 2].set_ylabel('Pressure (hPa)')
    _axes_overview[0, 2].grid(True, alpha=0.3)

    # Scatter plots: PM2.5 vs predictors (middle and bottom rows)
    _axes_overview[1, 0].scatter(df['temperature'], df['pm25'], s=0.5, alpha=0.3)
    _axes_overview[1, 0].set_xlabel('Temperature (°C)')
    _axes_overview[1, 0].set_ylabel('PM2.5 (μg/m³)')
    _axes_overview[1, 0].set_title('PM2.5 vs Temperature')
    _axes_overview[1, 0].grid(True, alpha=0.3)

    _axes_overview[1, 1].scatter(df['dewpoint'], df['pm25'], s=0.5, alpha=0.3, color='cyan')
    _axes_overview[1, 1].set_xlabel('Dewpoint (°C)')
    _axes_overview[1, 1].set_ylabel('PM2.5 (μg/m³)')
    _axes_overview[1, 1].set_title('PM2.5 vs Dewpoint')
    _axes_overview[1, 1].grid(True, alpha=0.3)

    _axes_overview[1, 2].scatter(df['pressure'], df['pm25'], s=0.5, alpha=0.3, color='purple')
    _axes_overview[1, 2].set_xlabel('Pressure (hPa)')
    _axes_overview[1, 2].set_ylabel('PM2.5 (μg/m³)')
    _axes_overview[1, 2].set_title('PM2.5 vs Pressure')
    _axes_overview[1, 2].grid(True, alpha=0.3)

    _axes_overview[2, 0].scatter(df['wind_speed'], df['pm25'], s=0.5, alpha=0.3, color='green')
    _axes_overview[2, 0].set_xlabel('Wind Speed (m/s)')
    _axes_overview[2, 0].set_ylabel('PM2.5 (μg/m³)')
    _axes_overview[2, 0].set_title('PM2.5 vs Wind Speed')
    _axes_overview[2, 0].grid(True, alpha=0.3)

    _axes_overview[2, 1].scatter(df['rain_hours'], df['pm25'], s=0.5, alpha=0.3, color='blue')
    _axes_overview[2, 1].set_xlabel('Rain Hours')
    _axes_overview[2, 1].set_ylabel('PM2.5 (μg/m³)')
    _axes_overview[2, 1].set_title('PM2.5 vs Rain Hours')
    _axes_overview[2, 1].grid(True, alpha=0.3)

    # Hide the last subplot (empty)
    _axes_overview[2, 2].axis('off')

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Configure Model and Split Data
    """)
    return


@app.cell
def _(mo):
    train_start = mo.ui.text(value='2012-01-01', label='Training start date')
    train_end = mo.ui.text(value='2013-12-31', label='Training end date')
    test_start = mo.ui.text(value='2014-01-01', label='Test start date')
    test_end = mo.ui.text(value='2014-03-31', label='Test end date')
    take_log = mo.ui.switch(label='Take log of target data', value=True)
    use_ar = mo.ui.switch(label='Use AR model', value=True)
    use_outlier = mo.ui.switch(label='Use outlier detector', value=False)
    outlier_reg_weight = mo.ui.slider(0.0001, 2.0, step=0.0001, value=0.002, label='Outlier L1 reg weight')
    use_temp = mo.ui.switch(label='Use temperature', value=True)
    use_dewpoint = mo.ui.switch(label='Use dewpoint', value=True)
    use_wind = mo.ui.switch(label='Use wind speed', value=True)
    use_pressure = mo.ui.switch(label='Use pressure', value=True)
    use_rain = mo.ui.switch(label='Use rain hours', value=True)
    solver_select = mo.ui.dropdown(['CLARABEL', 'MOSEK'], value='CLARABEL', label='Solver')
    verbose = mo.ui.switch(label='Verbose solver output', value=False)
    debug = mo.ui.switch(label='Debug mode', value=False)

    mo.vstack([
        mo.hstack([train_start, train_end]),
        mo.hstack([test_start, test_end]),
        mo.hstack([take_log, use_ar, use_outlier]),
        mo.hstack([outlier_reg_weight]),
        mo.md("**Exogenous Variables:**"),
        mo.hstack([use_temp, use_dewpoint, use_wind]),
        mo.hstack([use_pressure, use_rain]),
        mo.hstack([solver_select]),
        mo.hstack([verbose, debug])
    ])
    return (
        debug,
        outlier_reg_weight,
        solver_select,
        take_log,
        test_end,
        test_start,
        train_end,
        train_start,
        use_ar,
        use_dewpoint,
        use_outlier,
        use_pressure,
        use_rain,
        use_temp,
        use_wind,
        verbose,
    )


@app.cell
def _(
    df,
    pd,
    test_end,
    test_start,
    train_end,
    train_start,
    use_dewpoint,
    use_pressure,
    use_rain,
    use_temp,
    use_wind,
):
    # Split data
    df_train = df[train_start.value:train_end.value].copy()
    df_test = df[test_start.value:test_end.value].copy()

    print(f"Training data: {len(df_train)} samples")
    print(f"Test data: {len(df_test)} samples")

    # Prepare target (log transform if requested)
    y_train = df_train['pm25'].values
    y_test = df_test['pm25'].values

    # Prepare exogenous variables based on selection
    exog_dict_train = {}
    exog_dict_test = {}

    if use_temp.value:
        exog_dict_train['temperature'] = df_train['temperature'].values
        exog_dict_test['temperature'] = df_test['temperature'].values

    if use_dewpoint.value:
        exog_dict_train['dewpoint'] = df_train['dewpoint'].values
        exog_dict_test['dewpoint'] = df_test['dewpoint'].values

    if use_wind.value:
        exog_dict_train['wind_speed'] = df_train['wind_speed'].values
        exog_dict_test['wind_speed'] = df_test['wind_speed'].values

    if use_pressure.value:
        exog_dict_train['pressure'] = df_train['pressure'].values
        exog_dict_test['pressure'] = df_test['pressure'].values

    if use_rain.value:
        exog_dict_train['rain_hours'] = df_train['rain_hours'].values
        exog_dict_test['rain_hours'] = df_test['rain_hours'].values

    # Create DataFrames - handle empty case
    if len(exog_dict_train) == 0:
        X_train = pd.DataFrame(index=df_train.index)
        X_test = pd.DataFrame(index=df_test.index)
    else:
        X_train = pd.DataFrame(exog_dict_train, index=df_train.index)
        X_test = pd.DataFrame(exog_dict_test, index=df_test.index)

    print(f"Selected exogenous variables: {list(X_train.columns)}")
    print(f"X_train shape: {X_train.shape}")
    return X_test, X_train, df_test, df_train, y_test, y_train


@app.cell
def _(mo):
    fit_model = mo.ui.run_button(label='Fit Model')
    fit_model
    return (fit_model,)


@app.cell
def _(
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
    TsgamArConfig,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    X_train,
    debug,
    fit_model,
    mo,
    np,
    outlier_reg_weight,
    solver_select,
    take_log,
    use_ar,
    use_outlier,
    verbose,
    y_train,
):
    mo.stop(not fit_model.value)

    # Apply log transform if requested
    if take_log.value:
        y_train_log = np.log(y_train + 1.0)
    else:
        y_train_log = y_train.copy()

    # Multi-harmonic Fourier configuration
    # Using 30-day period instead of yearly for shorter time series
    # periods must match num_harmonics in length
    _periods_list = [PERIOD_HOURLY_YEARLY, float(30*24.0), float(PERIOD_HOURLY_WEEKLY), float(PERIOD_HOURLY_DAILY)]
    _num_harmonics_list = [4, 4, 8, 4]

    # Verify configuration before creating
    if len(_num_harmonics_list) != len(_periods_list):
        raise ValueError(
            f"num_harmonics ({len(_num_harmonics_list)}) and "
            f"periods ({len(_periods_list)}) must have the same length"
        )

    print(f"Multi-harmonic config: {_num_harmonics_list} harmonics for periods {_periods_list}")

    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=_num_harmonics_list,
        periods=_periods_list,
        reg_weight=6e-5
    )

    # Exogenous variables configuration
    # Note: Only non-negative lags (0, 1, 2, ...) are used for forecasting
    # Negative lags would use future data, which is not available for forecasting
    # Order must match the order of columns in X_train/X_test

    # Check if X_train has any columns
    if X_train.shape[1] == 0:
        exog_config = None
    else:
        exog_config = []

    # Define configurations for each variable (in order they appear in X_train/X_test)
    var_configs = {
        'temperature': TsgamSplineConfig(
            n_knots=8,
            lags=[0, 1, 2],  # Current, 1-2 hours back
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'dewpoint': TsgamSplineConfig(
            n_knots=10,
            lags=[0, 1],  # Current, 1 hour back
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'wind_speed': TsgamSplineConfig(
            n_knots=8,
            lags=[0, 1],  # Current, 1 hour back
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'pressure': TsgamSplineConfig(
            n_knots=8,
            lags=[0],  # Current only
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'rain_hours': TsgamSplineConfig(
            n_knots=6,
            lags=[0],  # Current only
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
    }

    # Build exog_config based on selected variables (order matches X_train columns)
    if exog_config is not None:
        print(f"X_train columns: {list(X_train.columns)}")
        print(f"X_train shape: {X_train.shape}")

        for var_name in X_train.columns:
            if var_name in var_configs:
                exog_config.append(var_configs[var_name])
            else:
                # Fallback configuration if variable not in predefined configs
                exog_config.append(TsgamSplineConfig(
                    n_knots=8,
                    lags=[0],
                    reg_weight=6e-5,
                    diff_reg_weight=0.5
                ))

        print(f"exog_config length: {len(exog_config)}")
    else:
        print("No exogenous variables selected - exog_config is None")

    # AR configuration
    ar_config = None
    if use_ar.value:
        ar_config = TsgamArConfig(
            lags=[1, 2, 3, 4],
            l1_constraint=0.97
        )
    else:
        ar_config = None

    # Outlier detector configuration
    if use_outlier.value:
        outlier_config = TsgamOutlierConfig(
            reg_weight=outlier_reg_weight.value,
            period_hours=24.0  # Daily outliers
        )
    else:
        outlier_config = None

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver=solver_select.value,
        verbose=verbose.value
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=exog_config,
        ar_config=ar_config,
        outlier_config=outlier_config,
        solver_config=solver_config,
        random_state=42,
        debug=debug.value
    )

    # Create and fit estimator
    print("Creating and fitting TSGAM estimator...")
    estimator = TsgamEstimator(config=config)

    print("Fitting model (this may take a few minutes)...")
    estimator.fit(X_train, y_train_log)

    print("\nModel fitting complete!")
    print(f"Problem status: {estimator.problem_.status}")
    if estimator.problem_.status in ["optimal", "optimal_inaccurate"]:
        print(f"Optimal value: {estimator.problem_.value:.6e}")

    if estimator.ar_coef_ is not None:
        print("\nAR model fitted successfully:")
        print(f"  AR coefficients: {estimator.ar_coef_}")
        print(f"  AR intercept: {estimator.ar_intercept_:.6f}")
    return (estimator,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Ablation Study

    Test different combinations of exogenous variables to see their individual contributions.
    All models use CLARABEL solver.
    """)
    return


@app.cell
def _(mo):
    run_ablation = mo.ui.run_button(label='Run Ablation Study')
    run_ablation
    return (run_ablation,)


@app.cell(hide_code=True)
def _(
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    X_test,
    X_train,
    mo,
    np,
    pd,
    plt,
    run_ablation,
    solver_select,
    take_log,
    y_test,
    y_train,
):
    mo.stop(not run_ablation.value)

    # Apply log transform if requested
    if take_log.value:
        _y_train_log = np.log(y_train + 1.0)
        _y_test_log = np.log(y_test + 1.0)
    else:
        _y_train_log = y_train.copy()
        _y_test_log = y_test.copy()

    # Multi-harmonic config
    _periods_list = [PERIOD_HOURLY_YEARLY, float(30*24.0), float(PERIOD_HOURLY_WEEKLY), float(PERIOD_HOURLY_DAILY)]
    _num_harmonics_list = [4, 4, 8, 4]
    _multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=_num_harmonics_list,
        periods=_periods_list,
        reg_weight=6e-5
    )

    # Variable configs mapping
    # Note: n_knots=10+ fails with CLARABEL for temperature when used alone or with dewpoint/wind_speed
    # Testing showed: n_knots=8 with lags=[0,1,2] works reliably for all combinations
    _var_configs = {
        'temperature': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),  # Using n_knots=8 to avoid solver failure
        'dewpoint': TsgamSplineConfig(n_knots=10, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        'wind_speed': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        'pressure': TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5),
        'rain_hours': TsgamSplineConfig(n_knots=6, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5),
    }

    # Get available variables from X_train
    _available_vars = list(X_train.columns)

    # Test configurations - different combinations of variables
    # Only include variables that are actually available in X_train
    _all_test_configs = [
        {'name': 'Baseline (Seasonal Only)', 'vars': []},
        {'name': 'Temperature', 'vars': ['temperature']},
        {'name': 'Dewpoint', 'vars': ['dewpoint']},
        {'name': 'Wind Speed', 'vars': ['wind_speed']},
        {'name': 'Pressure', 'vars': ['pressure']},
        {'name': 'Rain Hours', 'vars': ['rain_hours']},
        {'name': 'Temperature + Dewpoint', 'vars': ['temperature', 'dewpoint']},
        {'name': 'Temperature + Wind Speed', 'vars': ['temperature', 'wind_speed']},
        {'name': 'Temperature + Pressure', 'vars': ['temperature', 'pressure']},
        {'name': 'Dewpoint + Wind Speed', 'vars': ['dewpoint', 'wind_speed']},
        {'name': 'Temperature + Dewpoint + Wind Speed', 'vars': ['temperature', 'dewpoint', 'wind_speed']},
        {'name': 'Temperature + Dewpoint + Wind Speed + Pressure', 'vars': ['temperature', 'dewpoint', 'wind_speed', 'pressure']},
        {'name': 'All Variables', 'vars': ['temperature', 'dewpoint', 'wind_speed', 'pressure', 'rain_hours']},
    ]

    # Filter to only include configs where all variables are available
    _test_configs = []
    for _config in _all_test_configs:
        if all(_var in _available_vars for _var in _config['vars']):
            _test_configs.append(_config)

    print(f"Available variables: {_available_vars}")
    print(f"Testing {len(_test_configs)} configurations (filtered from {len(_all_test_configs)} total)")

    _ablation_results = {}
    _y_test_orig = None

    print("Running ablation study...")
    print("="*60)
    print(f"Testing {len(_test_configs)} different variable combinations...\n")

    # Helper function to fit model (always use CLARABEL since no MOSEK license)
    def _fit_model(_name, _vars, _X_train, _X_test):
        # Use var_configs directly - n_knots=8 works for all combinations
        _exog_config = [_var_configs[v] for v in _vars] if _vars else None

        # Always use CLARABEL (no MOSEK license available)
        try:
            _config = TsgamEstimatorConfig(
                multi_harmonic_config=_multi_harmonic_config,
                exog_config=_exog_config,
                ar_config=None,
                solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
                random_state=42,
                debug=False
            )
            _estimator = TsgamEstimator(config=_config)
            _estimator.fit(_X_train, _y_train_log)
            _pred_log = _estimator.predict(_X_test)
            return _pred_log, None
        except Exception as _e:
            return None, str(_e)

    for _idx, _config in enumerate(_test_configs, 1):
        _name = _config['name']
        _vars = _config['vars']

        print(f"{_idx}. Testing: {_name}...")

        # Prepare X for this configuration
        if len(_vars) == 0:
            _X_train_ablation = pd.DataFrame(index=X_train.index)
            _X_test_ablation = pd.DataFrame(index=X_test.index)
        else:
            _X_train_ablation = X_train[_vars].copy()
            _X_test_ablation = X_test[_vars].copy()

        # Fit model (always uses CLARABEL)
        _pred_log, _error = _fit_model(_name, _vars, _X_train_ablation, _X_test_ablation)

        if _pred_log is not None:
            # Convert back from log space (only compute once)
            if _y_test_orig is None:
                if take_log.value:
                    _y_test_orig = np.exp(_y_test_log) - 1.0
                else:
                    _y_test_orig = _y_test_log.copy()

            if take_log.value:
                # Clip predictions to avoid overflow in exp
                _pred_log_clipped = np.clip(_pred_log, -10, 20)  # Reasonable range for log space
                _pred_orig = np.exp(_pred_log_clipped) - 1.0
            else:
                _pred_orig = _pred_log.copy()

            # Calculate metrics
            _valid = np.isfinite(_pred_orig) & np.isfinite(_y_test_orig) & (_y_test_orig > 0) & (_pred_orig > 0)
            if np.any(_valid):
                _mae = np.mean(np.abs(_pred_orig[_valid] - _y_test_orig[_valid]))
                _rmse = np.sqrt(np.mean((_pred_orig[_valid] - _y_test_orig[_valid])**2))
                _mape = np.mean(np.abs((_pred_orig[_valid] - _y_test_orig[_valid]) / _y_test_orig[_valid])) * 100
                _ablation_results[_name] = {
                    'rmse': _rmse,
                    'mae': _mae,
                    'mape': _mape,
                    'success': True
                }
                print(f"   ✓ RMSE: {_rmse:.2f} μg/m³, MAE: {_mae:.2f} μg/m³, MAPE: {_mape:.2f}%")
            else:
                _ablation_results[_name] = {
                    'rmse': np.nan,
                    'mae': np.nan,
                    'mape': np.nan,
                    'success': False,
                    'error': 'No valid predictions'
                }
                print("   ✗ Failed: No valid predictions")
        else:
            _ablation_results[_name] = {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'success': False,
                'error': _error
            }
            print(f"   ✗ Failed: {_error}")

    print("\n" + "="*60)
    print("Ablation Study Complete")
    print("="*60)

    # Create visualization
    if not _ablation_results or len(_ablation_results) == 0:
        _fig_ablation = plt.figure(figsize=(10, 6))
        _ax = _fig_ablation.add_subplot(111)
        _ax.text(0.5, 0.5, 'Run the ablation study first to see results.',
                ha='center', va='center', fontsize=14, transform=_ax.transAxes)
        _ax.axis('off')
    else:
        # Sort by number of variables
        def _count_vars(name):
            if 'Baseline' in name or 'Seasonal Only' in name:
                return 0
            if 'All Variables' in name:
                return 5
            return name.count('+') + 1

        _sorted_names = sorted(_ablation_results.keys(), key=_count_vars)
        _rmses = [_ablation_results[n]['rmse'] for n in _sorted_names]
        _maes = [_ablation_results[n]['mae'] for n in _sorted_names]
        _success = [_ablation_results[n]['success'] for n in _sorted_names]
        _solvers = [_ablation_results[n].get('solver', 'Unknown') for n in _sorted_names]

        # Create labels with solver info
        _labels = []
        for _n in _sorted_names:
            _label = _n.replace(' + ', ' +\n')
            if _ablation_results[_n].get('solver') and _ablation_results[_n]['solver'] != solver_select.value:
                _label += '\n(CLARABEL)'
            if not _ablation_results[_n]['success']:
                _label += '\n(FAILED)'
            _labels.append(_label)

        # Create figure with subplots
        _fig_ablation, _axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

        # RMSE plot
        _ax1 = _axes[0]
        _colors = ['steelblue' if 'Baseline' in n or 'Seasonal Only' in n
                  else 'coral' if 'All Variables' in n
                  else 'mediumseagreen' if _ablation_results[n]['success']
                  else 'red' for n in _sorted_names]
        _bars1 = _ax1.bar(range(len(_sorted_names)), _rmses, color=_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        _ax1.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
        _ax1.set_ylabel('RMSE (μg/m³)', fontsize=11, fontweight='bold')
        _ax1.set_title('RMSE: Different Variable Combinations', fontsize=12, fontweight='bold')
        _ax1.set_xticks(range(len(_sorted_names)))
        _ax1.set_xticklabels(_labels, rotation=45, ha='right', fontsize=8)
        _ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        for _i, (_bar, _val) in enumerate(zip(_bars1, _rmses)):
            if not np.isnan(_val):
                _height = _bar.get_height()
                _ax1.text(_bar.get_x() + _bar.get_width()/2., _height,
                         f'{_height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # MAE plot
        _ax2 = _axes[1]
        _bars2 = _ax2.bar(range(len(_sorted_names)), _maes, color=_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        _ax2.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
        _ax2.set_ylabel('MAE (μg/m³)', fontsize=11, fontweight='bold')
        _ax2.set_title('MAE: Different Variable Combinations', fontsize=12, fontweight='bold')
        _ax2.set_xticks(range(len(_sorted_names)))
        _ax2.set_xticklabels(_labels, rotation=45, ha='right', fontsize=8)
        _ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        for _i, (_bar, _val) in enumerate(zip(_bars2, _maes)):
            if not np.isnan(_val):
                _height = _bar.get_height()
                _ax2.text(_bar.get_x() + _bar.get_width()/2., _height,
                         f'{_height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Calculate and display improvement over baseline
        _baseline_rmse = _rmses[0] if len(_rmses) > 0 and not np.isnan(_rmses[0]) else None
        if _baseline_rmse:
            _valid_rmses = [_r for _r in _rmses if not np.isnan(_r)]
            if _valid_rmses:
                _best_idx = _rmses.index(min(_valid_rmses))
                _best_rmse = _rmses[_best_idx]
                _improvement = ((_baseline_rmse - _best_rmse) / _baseline_rmse) * 100
                _fig_ablation.suptitle(f'Ablation Study: Variable Combinations Impact\n'
                                      f'Best: {_sorted_names[_best_idx]} (RMSE: {_best_rmse:.1f} μg/m³, '
                                      f'Improvement: {_improvement:.1f}% over baseline)',
                                      fontsize=13, fontweight='bold', y=0.98)
            else:
                _fig_ablation.suptitle('Ablation Study: Variable Combinations Impact',
                                      fontsize=13, fontweight='bold', y=0.98)
        else:
            _fig_ablation.suptitle('Ablation Study: Variable Combinations Impact',
                                  fontsize=13, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

    _fig_ablation
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Predictions and Evaluation
    """)
    return


@app.cell
def _(X_test, estimator, np, take_log, y_test):
    # Make predictions
    predictions_log = estimator.predict(X_test)

    # Convert back from log space if needed
    if take_log.value:
        # y_test was already converted to log space, so convert back
        y_test_log = np.log(y_test + 1.0)
        y_test_original = np.exp(y_test_log) - 1.0
        predictions_original = np.exp(predictions_log) - 1.0
    else:
        y_test_original = y_test.copy()
        predictions_original = predictions_log.copy()

    # Calculate metrics
    valid_mask = np.isfinite(predictions_original) & np.isfinite(y_test_original) & (y_test_original > 0)
    if np.any(valid_mask):
        mae = np.mean(np.abs(predictions_original[valid_mask] - y_test_original[valid_mask]))
        rmse = np.sqrt(np.mean((predictions_original[valid_mask] - y_test_original[valid_mask])**2))
        mape = np.mean(np.abs((predictions_original[valid_mask] - y_test_original[valid_mask]) / y_test_original[valid_mask])) * 100
    else:
        mae = rmse = mape = np.nan

    print("Forecast Performance Metrics:")
    print(f"  MAE:  {mae:.2f} μg/m³")
    print(f"  RMSE: {rmse:.2f} μg/m³")
    print(f"  MAPE: {mape:.2f}%")
    return predictions_log, predictions_original, y_test_original


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualizations
    """)
    return


@app.cell
def _(
    np,
    plt,
    predictions_log,
    predictions_original,
    take_log,
    y_test_original,
):
    # Model fit plot
    _fig_fit, _axes_fit = plt.subplots(ncols=2, figsize=(14, 5))

    # Transformed space
    if take_log.value:
        _y_test_log = np.log(y_test_original + 1.0)
        _axes_fit[0].scatter(predictions_log, _y_test_log, marker='.', s=1, alpha=0.5)
        _axes_fit[0].set_xlabel('predicted (log space)')
        _axes_fit[0].set_ylabel('actual (log space)')
    else:
        _axes_fit[0].scatter(predictions_original, y_test_original, marker='.', s=1, alpha=0.5)
        _axes_fit[0].set_xlabel('predicted')
        _axes_fit[0].set_ylabel('actual')

    _xlim = _axes_fit[0].get_xlim()
    _ylim = _axes_fit[0].get_ylim()
    _axes_fit[0].plot([-1e4, 1e4], [-1e4, 1e4], color='red', ls='--', linewidth=1)
    _axes_fit[0].set_xlim(_xlim)
    _axes_fit[0].set_ylim(_ylim)
    _axes_fit[0].set_title('Model Fit (transformed space)')
    _axes_fit[0].grid(True, alpha=0.3)

    # Original space
    _axes_fit[1].scatter(predictions_original, y_test_original, marker='.', s=1, alpha=0.5)
    _xlim = _axes_fit[1].get_xlim()
    _ylim = _axes_fit[1].get_ylim()
    _axes_fit[1].plot([-1e4, 1e4], [-1e4, 1e4], color='red', ls='--', linewidth=1)
    _axes_fit[1].set_xlim(_xlim)
    _axes_fit[1].set_ylim(_ylim)
    _axes_fit[1].set_xlabel('predicted (μg/m³)')
    _axes_fit[1].set_ylabel('actual (μg/m³)')
    _axes_fit[1].set_title('Model Fit (original space)')
    _axes_fit[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(df_test, df_train, plt, predictions_original, y_test_original, y_train):
    # Time series plot
    _fig_ts, _axes_ts = plt.subplots(nrows=2, figsize=(16, 8), sharex=False)

    # Full time series
    _axes_ts[0].plot(df_train.index, y_train, 'b-', alpha=0.6, label='Training data', linewidth=0.5)
    test_idx = df_test.index[:len(y_test_original)]
    _axes_ts[0].plot(test_idx, y_test_original, 'g-', alpha=0.7, label='Actual (test)', linewidth=1)
    _axes_ts[0].plot(test_idx, predictions_original, 'r-', label='Forecast', linewidth=1.5)
    _axes_ts[0].set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    _axes_ts[0].set_title('Air Quality Forecast: PM2.5', fontsize=12, fontweight='bold')
    _axes_ts[0].legend(loc='upper left', fontsize=9)
    _axes_ts[0].grid(True, alpha=0.3, linestyle='--')

    # Zoomed view of first month
    month_len = min(24*30, len(y_test_original))
    month_idx = test_idx[:month_len]
    month_actual = y_test_original[:month_len]
    month_pred = predictions_original[:month_len]

    _axes_ts[1].plot(month_idx, month_actual, 'g-', alpha=0.7, label='Actual', linewidth=1.5)
    _axes_ts[1].plot(month_idx, month_pred, 'r-', label='Forecast', linewidth=1.5)
    # Set x-axis limits to just the selected month range
    if len(month_idx) > 0:
        _axes_ts[1].set_xlim([month_idx[0], month_idx[-1]])
    _axes_ts[1].set_xlabel('Date', fontsize=11, fontweight='bold')
    _axes_ts[1].set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    _axes_ts[1].set_title('Zoomed View: First Month', fontsize=12, fontweight='bold')
    _axes_ts[1].legend(loc='upper left', fontsize=9)
    _axes_ts[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(np, plt, predictions_original, stats, y_test_original):
    # Residuals
    residuals = y_test_original - predictions_original

    # Residual distribution
    _fig_resid, _axes_resid = plt.subplots(nrows=2, figsize=(12, 10))

    # Histogram with fitted distributions
    _r = residuals
    _s = ~np.isnan(_r)
    _axes_resid[0].hist(_r[_s], bins=200, density=True, alpha=0.7)
    _xs = np.linspace(np.min(_r[_s]), np.max(_r[_s]), 1001)
    lap_loc, lap_scale = stats.laplace.fit(_r[_s])
    nor_loc, nor_scale = stats.norm.fit(_r[_s])
    _axes_resid[0].plot(_xs, stats.laplace.pdf(_xs, lap_loc, lap_scale), label='Laplace fit', linewidth=2, color='dodgerblue')
    _axes_resid[0].plot(_xs, stats.norm.pdf(_xs, nor_loc, nor_scale), label='Normal fit', linewidth=2, color='lime')
    _axes_resid[0].axvline(np.nanquantile(_r, .025), color='orange', ls='--', label='95% confidence bounds', linewidth=1)
    _axes_resid[0].axvline(np.nanquantile(_r, .975), color='orange', ls='--', linewidth=1)
    _axes_resid[0].set_xlabel('Residual (μg/m³)', fontsize=11)
    _axes_resid[0].set_ylabel('Density', fontsize=11)
    _axes_resid[0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    _axes_resid[0].legend()
    _axes_resid[0].grid(True, alpha=0.3)

    # Residuals over time
    _axes_resid[1].plot(_r, 'k-', alpha=0.6, linewidth=0.5)
    _axes_resid[1].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    _axes_resid[1].set_xlabel('Time Index', fontsize=11)
    _axes_resid[1].set_ylabel('Residual (μg/m³)', fontsize=11)
    _axes_resid[1].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    _axes_resid[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Outlier Detection Results
    """)
    return


@app.cell
def _(df_train, estimator, np, outlier_reg_weight, pd, plt, use_outlier):
    # Plot outlier detection results if outlier detector was used
    if not hasattr(estimator, 'variables_'):
        _fig_outlier = plt.figure(figsize=(10, 6))
        _ax = _fig_outlier.add_subplot(111)
        _ax.text(0.5, 0.5, 'Fit the model first to see outlier detection results.',
                ha='center', va='center', fontsize=14, transform=_ax.transAxes)
        _ax.axis('off')
    elif use_outlier.value and 'outlier' in estimator.variables_:
        detected_outlier = estimator.variables_['outlier'].value
        if detected_outlier is not None:
            # Get training timestamps to map days to dates
            timestamps = df_train.index
            n_days = len(detected_outlier)

            # Calculate day indices (assuming hourly data)
            # Each day has 24 hours, so we need to map days to dates
            day_dates = []
            for day_idx in range(n_days):
                hour_idx = day_idx * 24
                if hour_idx < len(timestamps):
                    day_dates.append(timestamps[hour_idx])
                else:
                    # Extend with last date if needed
                    day_dates.append(timestamps[-1] + pd.Timedelta(days=day_idx - n_days + 1))

            day_dates = pd.DatetimeIndex(day_dates)

            # Create figure with subplots
            _fig_outlier, _axes_outlier = plt.subplots(nrows=2, figsize=(16, 10))

            # Plot 1: Outlier values over time
            _ax1 = _axes_outlier[0]
            _ax1.plot(day_dates, detected_outlier, 'b-', linewidth=1.5, marker='o', markersize=4, label='Detected outlier values')
            _ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            _ax1.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (|outlier| > 0.1)')
            _ax1.axhline(y=-0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)

            # Highlight significant outliers
            significant_mask = np.abs(detected_outlier) > 0.1
            if np.any(significant_mask):
                _ax1.scatter(day_dates[significant_mask], detected_outlier[significant_mask],
                           color='red', s=100, zorder=5, label='Significant outliers (|outlier| > 0.1)', marker='s')

            _ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
            _ax1.set_ylabel('Outlier Value (log space)', fontsize=11, fontweight='bold')
            _ax1.set_title('Detected Outlier Values Over Time', fontsize=12, fontweight='bold')
            _ax1.legend(loc='best', fontsize=9)
            _ax1.grid(True, alpha=0.3, linestyle='--')
            _ax1.tick_params(axis='x', rotation=45)

            # Plot 2: Histogram of outlier values
            _ax2 = _axes_outlier[1]
            _ax2.hist(detected_outlier, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
            _ax2.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
            _ax2.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (|outlier| > 0.1)')
            _ax2.axvline(x=-0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            _ax2.set_xlabel('Outlier Value (log space)', fontsize=11, fontweight='bold')
            _ax2.set_ylabel('Frequency (number of days)', fontsize=11, fontweight='bold')
            _ax2.set_title('Distribution of Outlier Values', fontsize=12, fontweight='bold')
            _ax2.legend(loc='best', fontsize=9)
            _ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

            # Add statistics text
            n_significant = np.sum(significant_mask)
            sparsity = np.sum(np.abs(detected_outlier) < 0.1) / len(detected_outlier) * 100
            stats_text = f'Total days: {n_days}\n'
            stats_text += f'Significant outliers (|outlier| > 0.1): {n_significant}\n'
            stats_text += f'Sparsity: {sparsity:.1f}% (days with |outlier| < 0.1)\n'
            stats_text += f'Min: {np.min(detected_outlier):.4f}\n'
            stats_text += f'Max: {np.max(detected_outlier):.4f}\n'
            stats_text += f'Mean: {np.mean(detected_outlier):.4f}\n'
            stats_text += f'Std: {np.std(detected_outlier):.4f}'

            # Print outlier days with values
            if n_significant > 0:
                print(f"\nDetected {n_significant} outlier days (threshold: |outlier| > 0.05):")
                for day_idx in np.where(significant_mask)[0]:
                    outlier_val = detected_outlier[day_idx]
                    multiplier = np.exp(outlier_val)
                    print(f"  Day {day_idx} ({day_dates[day_idx].strftime('%Y-%m-%d')}): {outlier_val:.4f} (multiplier: {multiplier:.3f}x)")
            else:
                print("\nNo outliers detected with threshold |outlier| > 0.05")
                print(f"Outlier value range: [{np.min(detected_outlier):.6f}, {np.max(detected_outlier):.6f}]")
                print(f"Outlier value std: {np.std(detected_outlier):.6f}")
                print(f"Current regularization weight: {outlier_reg_weight.value:.6f}")
                if outlier_reg_weight.value > 0.002:
                    print("Try lowering the regularization weight (can go as low as 0.0001)")
                elif outlier_reg_weight.value < 0.001:
                    print("Try increasing the regularization weight (can go up to 2.0) to reduce false positives")
                else:
                    print("Regularization weight is in a reasonable range. The data may not have strong outliers,")
                    print("or the model may be explaining the variation through other components (trend, seasonality, etc.)")

            _ax2.text(0.98, 0.98, stats_text, transform=_ax2.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
        else:
            _fig_outlier = plt.figure(figsize=(10, 6))
            _ax = _fig_outlier.add_subplot(111)
            _ax.text(0.5, 0.5, 'Outlier detector was enabled but no outlier values were computed.\nModel may not have converged.',
                    ha='center', va='center', fontsize=14, transform=_ax.transAxes)
            _ax.axis('off')
    else:
        _fig_outlier = plt.figure(figsize=(10, 6))
        _ax = _fig_outlier.add_subplot(111)
        _ax.text(0.5, 0.5, 'Enable outlier detector in model configuration to see results.',
                ha='center', va='center', fontsize=14, transform=_ax.transAxes)
        _ax.axis('off')

    plt.gcf()
    return


@app.cell
def _(mo):
    res_dist = mo.ui.switch(label='Residual distribution: Normal <> Laplace', value=False)
    res_dist
    return (res_dist,)


@app.cell
def _(np, plt, predictions_original, res_dist, sm, stats, y_test_original):
    # Q-Q and P-P plots
    _r = y_test_original - predictions_original
    _s = ~np.isnan(_r)

    if not res_dist.value:
        _d = stats.norm
        _t = "Normal"
    else:
        _d = stats.laplace
        _t = "Laplace"

    _fig_ppqq, _axes_ppqq = plt.subplots(ncols=2, figsize=(12, 5))

    _pplot = sm.ProbPlot(_r[_s], _d, fit=True)
    _pplot.ppplot(line="45", ax=_axes_ppqq[0])
    _axes_ppqq[0].set_title(f"P-P plot for {_t}", fontsize=12, fontweight='bold')
    _axes_ppqq[0].grid(True, alpha=0.3)

    _pplot2 = sm.ProbPlot(_r[_s], _d, fit=True)
    _pplot2.qqplot(line="45", ax=_axes_ppqq[1])
    _axes_ppqq[1].set_title(f"Q-Q plot for {_t}", fontsize=12, fontweight='bold')
    _axes_ppqq[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Response Functions
    """)
    return


@app.cell
def _(X_train, estimator, mo, np):
    # Diagnostic: Check response function values
    _log_response = None
    _correction_factor = None
    if hasattr(estimator, 'variables_') and 'exog_coef_0' in estimator.variables_:
        _exog_coef = estimator.variables_['exog_coef_0'].value
        if _exog_coef is not None and estimator.exog_knots_ and len(estimator.exog_knots_) > 0:
            _knots = estimator.exog_knots_[0]
            if 'temperature' in X_train.columns:
                _x = X_train['temperature'].values
                _H = estimator._make_H(_x, _knots, include_offset=False)
                _log_response = _H @ _exog_coef[:, 0]
                _correction_factor = np.exp(_log_response)

            mo.md(f"""
            ### Response Function Diagnostics

            **Why correction factors are large (1e61):**

            The model is trained on **log-transformed targets**: `log(PM2.5 + 1)`

            The response function shows: `correction_factor = exp(H @ exog_coef)`

            For temperature (example):
            {f"- Log-space response range: [{np.min(_log_response):.2f}, {np.max(_log_response):.2f}]" if _log_response is not None else "- Temperature variable not selected"}
            {f"- Correction factor range: [{np.min(_correction_factor):.2e}, {np.max(_correction_factor):.2e}]" if _correction_factor is not None else ""}
            {f"- Mean correction factor: {np.mean(_correction_factor):.2e}" if _correction_factor is not None else ""}

            **Explanation:**
            - If log-space response = 140, then exp(140) ≈ 1e61
            - This suggests the coefficients are very large in log space
            - This can happen if the model is trying to capture very large multiplicative effects
            - **The correction factors should typically be close to 1.0** (multiplicative factors)
            - Values >> 1 suggest the model may be overfitting or there's a scaling issue

            **Possible causes:**
            1. The exogenous variables might need normalization/scaling
            2. The regularization weights might be too small
            3. The model might be overfitting to the training data
            4. There might be numerical instability in the optimization
            """)
    return


@app.cell
def _(X_train, estimator, np, plt):
    # Response functions - 2x3 subplot grid (5 variables)
    # Plot in log space to better visualize the response
    _fig_resp, _axes_resp = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    _axes_resp = _axes_resp.flatten()

    # Configuration for each response function (metadata only - indices will be computed dynamically)
    _response_configs = [
        {
            'var_name': 'temperature',
            'title': 'Temperature Response',
            'xlabel': 'Temperature (°C)',
            'color': None
        },
        {
            'var_name': 'dewpoint',
            'title': 'Dewpoint Response',
            'xlabel': 'Dewpoint (°C)',
            'color': 'green'
        },
        {
            'var_name': 'wind_speed',
            'title': 'Wind Speed Response',
            'xlabel': 'Wind Speed (m/s)',
            'color': 'orange'
        },
        {
            'var_name': 'pressure',
            'title': 'Pressure Response',
            'xlabel': 'Pressure (hPa)',
            'color': 'purple'
        },
        {
            'var_name': 'rain_hours',
            'title': 'Rain Hours Response',
            'xlabel': 'Rain Hours',
            'color': 'blue'
        }
    ]

    # Create mapping from variable names to their indices in X_train
    # This maps to exog_coef_{idx} and exog_knots_[idx]
    _var_to_idx = {var_name: idx for idx, var_name in enumerate(X_train.columns)} if X_train.shape[1] > 0 else {}

    # Plot each response function
    for _idx, _config in enumerate(_response_configs):
        _ax = _axes_resp[_idx]

        # First check if variable exists in X_train
        if _config['var_name'] not in X_train.columns:
            # Variable not selected - show message
            _ax.text(0.5, 0.5, f"Variable '{_config['var_name']}' not selected",
                    ha='center', va='center', transform=_ax.transAxes, fontsize=10)
        elif _config['var_name'] in _var_to_idx:
            # Get the actual index for this variable
            _var_idx = _var_to_idx[_config['var_name']]
            _var_key = f'exog_coef_{_var_idx}'

            if hasattr(estimator, 'variables_') and _var_key in estimator.variables_:
                _exog_coef = estimator.variables_[_var_key].value
                if _exog_coef is not None:
                    _knots = estimator.exog_knots_[_var_idx] if estimator.exog_knots_ and len(estimator.exog_knots_) > _var_idx else None
                    if _knots is not None:
                        _x = X_train[_config['var_name']].values
                        _H = estimator._make_H(_x, _knots, include_offset=False)
                        _log_response = _H @ _exog_coef[:, 0]
                        _correction_factor = np.exp(_log_response)

                        # Plot in log space (more interpretable)
                        _scatter_kwargs = {'s': 1, 'alpha': 0.5}
                        if _config['color']:
                            _scatter_kwargs['color'] = _config['color']
                        _ax.scatter(_x, _log_response, **_scatter_kwargs)
                        _ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No effect (log=0)')
                    else:
                        # Knots not available
                        _ax.text(0.5, 0.5, f"No knots available for '{_config['var_name']}'",
                                ha='center', va='center', transform=_ax.transAxes, fontsize=10)
                else:
                    # Coefficients not available
                    _ax.text(0.5, 0.5, f"No coefficients available for '{_config['var_name']}'",
                            ha='center', va='center', transform=_ax.transAxes, fontsize=10)
            else:
                # Variable key not in estimator (variable not fitted)
                _ax.text(0.5, 0.5, f"Variable '{_config['var_name']}' not fitted (key: {_var_key})",
                        ha='center', va='center', transform=_ax.transAxes, fontsize=10)
        else:
            # Variable not in mapping (shouldn't happen, but handle gracefully)
            _ax.text(0.5, 0.5, f"Variable '{_config['var_name']}' mapping error",
                    ha='center', va='center', transform=_ax.transAxes, fontsize=10)

        _ax.set_title(f"Inferred {_config['title']} (Log Space)", fontsize=11, fontweight='bold')
        _ax.set_xlabel(_config['xlabel'], fontsize=10)
        _ax.set_ylabel('Log-space Response', fontsize=10)
        _ax.grid(True, alpha=0.3)
        # Only add legend if there are artists with labels
        if _idx == 0 and len(_ax.get_legend_handles_labels()[0]) > 0:
            _ax.legend(fontsize=8)

    # Hide the 6th subplot (empty)
    _axes_resp[5].axis('off')

    plt.tight_layout()
    _fig_resp
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    This notebook demonstrates:
    - Loading and preprocessing air quality data
    - Fitting TSGAM models with seasonal patterns and exogenous variables
    - Evaluating forecast performance
    - Visualizing model fits, residuals, and response functions

    The model captures:
    - Daily, weekly, and monthly seasonal patterns
    - Non-linear relationships with temperature, dewpoint, wind speed, and pressure
    - Temporal dependencies through AR modeling
    """)
    return


if __name__ == "__main__":
    app.run()
