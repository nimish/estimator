# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Marimo Notebook: Los Angeles Energy Forecasting with TSGAM

This notebook demonstrates forecasting Los Angeles energy demand using:
- Multi-harmonic Fourier basis for seasonal patterns (daily, weekly, yearly)
- Weather variables (temperature, humidity, solar irradiance) as exogenous variables with spline basis
- Autoregressive (AR) modeling of residuals

The notebook uses real energy and weather data from Los Angeles, CA (2018).
Since there's only one year of data, we use a strategy of holding out the last
week of every month for validation, creating gaps in the training data to test
the estimator's ability to handle missing time periods.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Los Angeles Energy Forecasting with TSGAM

    This notebook demonstrates forecasting Los Angeles energy demand using TSGAM.

    ## Strategy
    Since we only have one year of data (2018), we use a cross-validation strategy
    that holds out the **last week of every month** for validation. This creates
    gaps in the training data, testing the estimator's ability to handle missing
    time periods.

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
    from pathlib import Path
    import sys
    from datetime import timedelta
    from scipy import stats

    # Add parent directory to path to import tsgam_estimator
    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

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
    from spcqe import make_basis_matrix
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
        make_basis_matrix,
        mo,
        np,
        pd,
        plt,
        sns,
        stats,
        timedelta,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load and Prepare Data
    """)
    return


@app.cell
def _(Path, mo):
    # File selectors for data files
    _project_root = Path(__file__).parent.parent
    _nerc_dir = _project_root / "nerc"

    # Get list of available CSV files in nerc directory
    _csv_files = sorted([f.name for f in _nerc_dir.glob("*.csv")])

    # Default files
    _default_weather = "weather_CA_Los Angeles.csv"
    _default_energy = "CA_Los Angeles_R.csv"

    # File selectors
    weather_file = mo.ui.dropdown(
        options=_csv_files,
        value=_default_weather if _default_weather in _csv_files else _csv_files[0] if _csv_files else "",
        label='Weather data file (X)'
    )
    energy_file = mo.ui.dropdown(
        options=_csv_files,
        value=_default_energy if _default_energy in _csv_files else _csv_files[0] if _csv_files else "",
        label='Energy data file (Y)'
    )

    mo.vstack([weather_file, energy_file])
    return energy_file, weather_file


@app.cell
def _(Path, energy_file, pd, weather_file):
    # Load data files
    _project_root = Path(__file__).parent.parent
    _nerc_dir = _project_root / "nerc"

    # Load weather data (X - features)
    _weather_file = _nerc_dir / weather_file.value
    df_weather = pd.read_csv(_weather_file)
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather = df_weather.set_index('timestamp')

    # Load energy data (Y - target)
    _energy_file = _nerc_dir / energy_file.value
    df_energy = pd.read_csv(_energy_file)
    df_energy['timestamp'] = pd.to_datetime(df_energy['timestamp'])
    df_energy = df_energy.set_index('timestamp')

    print(f"Weather data file: {weather_file.value}")
    print(f"Weather data shape: {df_weather.shape}")
    print(f"Weather date range: {df_weather.index.min()} to {df_weather.index.max()}")
    print(f"\nEnergy data file: {energy_file.value}")
    print(f"Energy data shape: {df_energy.shape}")
    print(f"Energy date range: {df_energy.index.min()} to {df_energy.index.max()}")
    print(f"\nEnergy columns: {list(df_energy.columns)}")
    return df_energy, df_weather


@app.cell
def _(df_energy, df_weather, pd):
    # Merge dataframes on timestamp
    df = pd.merge(df_weather, df_energy, left_index=True, right_index=True, how='inner')

    # Sort by timestamp
    df = df.sort_index()

    print(f"Merged data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")

    # Check for missing values
    print("\nMissing values per column:")
    for col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100
        print(f"  {col}: {missing} ({pct:.1f}%)")
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Select Target Variable
    """)
    return


@app.cell
def _(mo):
    target_var = mo.ui.dropdown(
        options=['elec_total_MW', 'elec_net_MW', 'nonelec_total_MW'],
        value='elec_total_MW',
        label='Target variable'
    )
    target_var
    return (target_var,)


@app.cell
def _(df, np, target_var):
    # Extract target variable
    y_full = df[target_var.value].values
    y_full = y_full[~np.isnan(y_full)]  # Remove NaN values

    # Get corresponding timestamps
    timestamps_full = df.index[~np.isnan(df[target_var.value])]

    print(f"Target variable: {target_var.value}")
    print(f"Valid samples: {len(y_full)}")
    print(f"Value range: {y_full.min():.2f} to {y_full.max():.2f} MW")
    print(f"Mean: {y_full.mean():.2f} MW")
    print(f"Std: {y_full.std():.2f} MW")
    return timestamps_full, y_full


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualize Data
    """)
    return


@app.cell
def _(df):
    df_plt = df.copy()
    df_plt['timestamp'] = df_plt.index
    df_plt
    return


@app.cell
def _(df, plt, target_var, timestamps_full, y_full):
    # Visualize full dataset before train/test split
    fig_data, axes_data = plt.subplots(nrows=3, ncols=1, figsize=(14, 10))

    # Plot 1: Target variable over time
    ax1_data = axes_data[0]
    ax1_data.plot(timestamps_full, y_full, 'b-', linewidth=0.5, alpha=0.7, label='Energy demand')
    ax1_data.set_ylabel(f'{target_var.value} (MW)', fontsize=10)
    ax1_data.set_title('Target Variable Over Time (Full Dataset)', fontsize=12, fontweight='bold')
    ax1_data.legend()
    ax1_data.grid(True, alpha=0.3)

    # Plot 2: Temperature over time
    ax2_data = axes_data[1]
    _temp_valid = ~df['temperature_degF'].isna()
    ax2_data.plot(df.index[_temp_valid], df['temperature_degF'][_temp_valid], 'r-', linewidth=0.5, alpha=0.7, label='Temperature')
    ax2_data.set_ylabel('Temperature (°F)', fontsize=10)
    ax2_data.set_title('Temperature Over Time', fontsize=12, fontweight='bold')
    ax2_data.legend()
    ax2_data.grid(True, alpha=0.3)

    # Plot 3: Humidity over time
    ax3_data = axes_data[2]
    _humidity_valid = ~df['humidity_pc'].isna()
    ax3_data.plot(df.index[_humidity_valid], df['humidity_pc'][_humidity_valid], 'g-', linewidth=0.5, alpha=0.7, label='Humidity')
    ax3_data.set_xlabel('Date', fontsize=10)
    ax3_data.set_ylabel('Humidity (%)', fontsize=10)
    ax3_data.set_title('Humidity Over Time', fontsize=12, fontweight='bold')
    ax3_data.legend()
    ax3_data.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(df, plt, sns):
    # Create heatmap visualizations for weather data
    # Reshape data for heatmap: days vs hours of day
    _df_reshaped = df.copy()
    _df_reshaped['day'] = _df_reshaped.index.date
    _df_reshaped['hour'] = _df_reshaped.index.hour

    # Create figure with multiple heatmaps (1x2 grid for 2 weather variables)
    fig_heatmap, axes_heatmap = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Weather variables to plot
    weather_vars = [
        ('temperature_degF', 'Temperature (°F)', 'plasma'),
        ('humidity_pc', 'Humidity (%)', 'viridis'),
    ]

    # Plot weather variable heatmaps
    for idx, (_weather_var_name, _weather_var_label, _weather_cmap) in enumerate(weather_vars):
        ax = axes_heatmap[idx]

        # Pivot to create day x hour matrix
        _heatmap_data = _df_reshaped.pivot_table(
            values=_weather_var_name,
            index='day',
            columns='hour',
            aggfunc='mean'
        )

        sns.heatmap(_heatmap_data, cmap=_weather_cmap, ax=ax, cbar_kws={'label': _weather_var_label})
        ax.set_title(f'{_weather_var_label}: Days vs Hours of Day', fontsize=11, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=9)
        ax.set_ylabel('Day', fontsize=9)
        ax.tick_params(axis='y', labelsize=6)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(df, plt, sns, target_var):
    # Create heatmap for target variable
    _df_reshaped = df.copy()
    _df_reshaped['day'] = _df_reshaped.index.date
    _df_reshaped['hour'] = _df_reshaped.index.hour

    # Pivot to create day x hour matrix for target
    _heatmap_target = _df_reshaped.pivot_table(
        values=target_var.value,
        index='day',
        columns='hour',
        aggfunc='mean'
    )

    fig_target_heatmap, ax_target_heatmap = plt.subplots(figsize=(14, 8))
    sns.heatmap(_heatmap_target, cmap='viridis', ax=ax_target_heatmap, cbar_kws={'label': f'{target_var.value} (MW)'})
    ax_target_heatmap.set_title(f'{target_var.value}: Days vs Hours of Day', fontsize=12, fontweight='bold')
    ax_target_heatmap.set_xlabel('Hour of Day', fontsize=10)
    ax_target_heatmap.set_ylabel('Day', fontsize=10)
    ax_target_heatmap.tick_params(axis='y', labelsize=6)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Create Train/Test Split (Hold Out Last Week of Each Month)
    """)
    return


@app.cell
def _(pd, timedelta):
    def create_train_test_split_with_gaps(df, target_col, holdout_days=7):
        """
        Create train/test split holding out the last N days of each month.
        This creates gaps in the training data to test the estimator's ability
        to handle missing time periods.

        Parameters
        ----------
        df : DataFrame
            Data with DatetimeIndex
        target_col : str
            Name of target column
        holdout_days : int
            Number of days to hold out at the end of each month (default: 7)

        Returns
        -------
        train_mask : Series
            Boolean mask for training data
        test_mask : Series
            Boolean mask for test data
        """
        train_mask = pd.Series(True, index=df.index)
        test_mask = pd.Series(False, index=df.index)

        # Get unique year-month combinations
        df_with_month = df.copy()
        df_with_month['year_month'] = df_with_month.index.to_period('M')

        for year_month in df_with_month['year_month'].unique():
            # Get data for this month
            month_data = df_with_month[df_with_month['year_month'] == year_month]

            if len(month_data) == 0:
                continue

            # Get the last N days of the month
            month_end = month_data.index.max()
            holdout_start = month_end - timedelta(days=holdout_days-1)
            holdout_start = holdout_start.replace(hour=0, minute=0, second=0, microsecond=0)

            # Mark holdout period as test
            month_mask = (df.index >= holdout_start) & (df.index <= month_end) & (df_with_month['year_month'] == year_month)
            test_mask[month_mask] = True
            train_mask[month_mask] = False

        return train_mask, test_mask
    return (create_train_test_split_with_gaps,)


@app.cell
def _(create_train_test_split_with_gaps, df, pd, target_var):
    # Create train/test split
    train_mask, test_mask = create_train_test_split_with_gaps(df, target_var.value, holdout_days=7)

    # Apply masks
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print(f"\nTraining date range: {df_train.index.min()} to {df_train.index.max()}")
    print(f"Test date range: {df_test.index.min()} to {df_test.index.max()}")

    # Check for gaps in training data
    _expected_freq = 'h'
    _date_range_full = pd.date_range(start=df.index.min(), end=df.index.max(), freq=_expected_freq)
    _missing_in_train = _date_range_full.difference(df_train.index)
    print(f"\nMissing timestamps in training data: {len(_missing_in_train)}")
    print("  (This includes the held-out test periods)")

    # Show some example gaps
    if len(_missing_in_train) > 0:
        print("\nExample missing periods (first 5):")
        for i, ts in enumerate(_missing_in_train[:5]):
            print(f"  {ts}")
    return df_test, df_train


@app.cell
def _(df_train, mo):
    # Get available weather columns (exclude timestamp and energy columns)
    _energy_cols = ['elec_total_MW', 'elec_net_MW', 'nonelec_total_MW', 'timestamp']
    _available_weather_cols = [col for col in df_train.columns if col not in _energy_cols]

    # Default selection
    _default_exog = ['temperature_degF', 'humidity_pc']
    _default_exog = [col for col in _default_exog if col in _available_weather_cols]

    # Multi-select for exogenous variables
    exog_vars = mo.ui.multiselect(
        options=_available_weather_cols,
        value=_default_exog if _default_exog else _available_weather_cols[:2] if len(_available_weather_cols) >= 2 else _available_weather_cols,
        label='Exogenous variables (X)'
    )
    exog_vars
    return (exog_vars,)


@app.cell
def _(df_test, df_train, exog_vars, target_var):
    # Prepare exogenous variables (weather features)
    # Use selected weather columns
    weather_cols = exog_vars.value if len(exog_vars.value) > 0 else ['temperature_degF', 'humidity_pc']

    # Ensure we have at least one column
    if len(weather_cols) == 0:
        print("Warning: No exogenous variables selected. Using temperature_degF as default.")
        weather_cols = ['temperature_degF'] if 'temperature_degF' in df_train.columns else list(df_train.columns)[:1]

    # Create X_train and X_test
    X_train = df_train[weather_cols].copy()
    X_test = df_test[weather_cols].copy()

    # Find valid samples (no NaN in target or weather data)
    train_valid = ~(df_train[target_var.value].isna() | X_train.isna().any(axis=1))
    test_valid = ~(df_test[target_var.value].isna() | X_test.isna().any(axis=1))

    # Filter to valid samples
    X_train = X_train[train_valid]
    X_test = X_test[test_valid]

    y_train_aligned = df_train[target_var.value].loc[X_train.index].values
    y_test_aligned = df_test[target_var.value].loc[X_test.index].values

    timestamps_train_aligned = X_train.index
    timestamps_test_aligned = X_test.index

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train aligned: {len(y_train_aligned)} samples")
    print(f"y_test aligned: {len(y_test_aligned)} samples")
    print(f"\nExogenous variables: {list(X_train.columns)}")
    return (
        X_test,
        X_train,
        timestamps_test_aligned,
        timestamps_train_aligned,
        weather_cols,
        y_test_aligned,
        y_train_aligned,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualize Train/Test Split and Gaps
    """)
    return


@app.cell
def _(
    X_train,
    df,
    df_test,
    pd,
    plt,
    target_var,
    timestamps_train_aligned,
    y_train_aligned,
):
    # Visualize training data with gaps and test data
    fig_gaps, axes_gaps = plt.subplots(nrows=3, ncols=1, figsize=(14, 10))

    # Plot 1: Full timeline showing gaps (training vs test)
    ax1_gaps = axes_gaps[0]
    _date_range_full = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    _is_train = _date_range_full.isin(X_train.index)
    _is_test = _date_range_full.isin(df_test.index)
    ax1_gaps.plot(_date_range_full, _is_train.astype(int), 'b-', linewidth=0.5, alpha=0.7, label='Training data')
    ax1_gaps.plot(_date_range_full, _is_test.astype(int) * 0.5, 'r-', linewidth=0.5, alpha=0.7, label='Test data (held out)')
    ax1_gaps.fill_between(_date_range_full, 0, _is_train.astype(int), alpha=0.3, color='blue')
    ax1_gaps.fill_between(_date_range_full, 0, _is_test.astype(int) * 0.5, alpha=0.3, color='red')
    ax1_gaps.set_ylabel('Data Available', fontsize=10)
    ax1_gaps.set_title('Training vs Test Data Split (Gaps = Held-Out Last Week of Each Month)', fontsize=12, fontweight='bold')
    ax1_gaps.legend()
    ax1_gaps.grid(True, alpha=0.3)

    # Plot 2: Training target variable over time
    ax2_gaps = axes_gaps[1]
    ax2_gaps.plot(timestamps_train_aligned, y_train_aligned, 'b-', linewidth=0.5, alpha=0.7, label='Training data')
    ax2_gaps.set_ylabel(f'{target_var.value} (MW)', fontsize=10)
    ax2_gaps.set_title('Training Target Variable Over Time (with gaps)', fontsize=12, fontweight='bold')
    ax2_gaps.legend()
    ax2_gaps.grid(True, alpha=0.3)

    # Plot 3: Training temperature over time
    ax3_gaps = axes_gaps[2]
    ax3_gaps.plot(X_train.index, X_train['temperature_degF'], 'r-', linewidth=0.5, alpha=0.7, label='Temperature (training)')
    ax3_gaps.set_xlabel('Date', fontsize=10)
    ax3_gaps.set_ylabel('Temperature (°F)', fontsize=10)
    ax3_gaps.set_title('Training Temperature Over Time (with gaps)', fontsize=12, fontweight='bold')
    ax3_gaps.legend()
    ax3_gaps.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Configure and Fit Model
    """)
    return


@app.cell
def _(mo):
    take_log = mo.ui.switch(label='Take log of target data', value=True)
    use_ar = mo.ui.switch(label='Use AR model', value=True)
    use_outlier = mo.ui.switch(label='Use outlier detector', value=False)
    outlier_reg_weight = mo.ui.slider(0.0001, 2.0, step=0.0001, value=1e-4, label='Outlier L1 reg weight')
    outlier_threshold = mo.ui.slider(0.001, 0.5, step=0.001, value=0.05, label='Outlier threshold (log space)')
    solver_select = mo.ui.dropdown(['CLARABEL', 'MOSEK'], value='CLARABEL', label='Solver')
    verbose = mo.ui.switch(label='Verbose solver output', value=False)

    mo.vstack([
        mo.hstack([take_log, use_ar, use_outlier]),
        mo.hstack([outlier_reg_weight, outlier_threshold]),
        mo.hstack([solver_select, verbose])
    ])
    return (
        outlier_reg_weight,
        outlier_threshold,
        solver_select,
        take_log,
        use_ar,
        use_outlier,
        verbose,
    )


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
    fit_model,
    mo,
    np,
    outlier_reg_weight,
    solver_select,
    take_log,
    use_ar,
    use_outlier,
    verbose,
    y_train_aligned,
):
    mo.stop(not fit_model.value)

    # Apply log transform if requested
    if take_log.value:
        y_train_log = np.log(y_train_aligned + 1.0)
        print("Applied log transform to target")
    else:
        y_train_log = y_train_aligned.copy()

    # Multi-harmonic Fourier configuration
    # For one year of data, we use yearly, weekly, and daily periods
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[4, 4, 6],  # yearly, weekly, daily
        periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
        reg_weight=6e-5
    )

    print(f"Multi-harmonic config: {multi_harmonic_config.num_harmonics} harmonics for periods {multi_harmonic_config.periods}")

    # Exogenous variables configuration
    # Order must match the order of columns in X_train/X_test
    exog_config = []

    # Define configurations for each weather variable
    var_configs = {
        'temperature_degF': TsgamSplineConfig(
            n_knots=10,
            lags=[0, 1, 2, 3],  # Current and 1-3 hours back
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'humidity_pc': TsgamSplineConfig(
            n_knots=8,
            lags=[0, 1, 2],  # Current and 1-2 hours back
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'global_Wpms': TsgamSplineConfig(
            n_knots=8,
            lags=[0, 1],
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'direct_Wpms': TsgamSplineConfig(
            n_knots=8,
            lags=[0, 1],
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        'diffuse_Wpms': TsgamSplineConfig(
            n_knots=8,
            lags=[0, 1],
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
    }

    # Build exog_config based on X_train columns
    print(f"X_train columns: {list(X_train.columns)}")
    for var_name in X_train.columns:
        if var_name in var_configs:
            exog_config.append(var_configs[var_name])
        else:
            # Fallback configuration for unknown variables
            exog_config.append(TsgamSplineConfig(
                n_knots=8,
                lags=[0],
                reg_weight=6e-5,
                diff_reg_weight=0.5
            ))

    print(f"exog_config length: {len(exog_config)}")

    # AR configuration
    ar_config = None
    if use_ar.value:
        ar_config = TsgamArConfig(
            lags=[1, 2, 3, 4],
            l1_constraint=0.97
        )

    # Outlier detector configuration
    outlier_config = None
    if use_outlier.value:
        outlier_config = TsgamOutlierConfig(
            reg_weight=outlier_reg_weight.value,
            period_hours=24.0  # Daily outliers
        )

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
        random_state=42
    )

    # Create and fit estimator
    print("\nCreating and fitting TSGAM estimator...")
    print(f"Training data has {len(X_train)} samples with gaps (missing last week of each month)")
    estimator = TsgamEstimator(config=config)

    print("Fitting model (this may take a few minutes)...")
    estimator.fit(X_train, y_train_log)

    print("\nModel fitting complete!")
    print(f"Problem status: {estimator.problem_.status}")
    if estimator.problem_.status in ["optimal", "optimal_inaccurate"]:
        print(f"Optimal value: {estimator.problem_.value:.6e}")

    if getattr(estimator, "ar_coef_", None) is not None:
        print("\nAR model fitted successfully:")
        print(f"  AR coefficients: {estimator.ar_coef_}")
        print(f"  AR intercept: {estimator.ar_intercept_:.6f}")
    return estimator, y_train_log


@app.cell
def _(mo):
    mo.md(r"""
    ## Make Predictions
    """)
    return


@app.cell
def _(X_test, estimator, np, take_log):
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred_log = estimator.predict(X_test)

    # Transform back if log was used
    if take_log.value:
        y_pred = np.exp(y_pred_log) - 1.0
    else:
        y_pred = y_pred_log

    print(f"Predictions complete: {len(y_pred)} samples")
    print(f"Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f} MW")
    return (y_pred,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Evaluate Model Performance
    """)
    return


@app.cell
def _(np, y_pred, y_test_aligned):
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_test_aligned))
    rmse = np.sqrt(np.mean((y_pred - y_test_aligned) ** 2))
    mape = np.mean(np.abs((y_pred - y_test_aligned) / (y_test_aligned + 1e-6))) * 100

    # Calculate R-squared
    ss_res = np.sum((y_test_aligned - y_pred) ** 2)
    ss_tot = np.sum((y_test_aligned - np.mean(y_test_aligned)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("Model Performance Metrics:")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    return mae, mape, r2, rmse


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualize Results
    """)
    return


@app.cell
def _(plt, timestamps_test_aligned, y_pred, y_test_aligned):
    # Create comprehensive visualization
    fig_results, axes_results = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))

    # Plot 1: Predictions vs Actual over time
    ax1_results = axes_results[0]
    ax1_results.plot(timestamps_test_aligned, y_test_aligned, 'b-', linewidth=1, alpha=0.7, label='Actual')
    ax1_results.plot(timestamps_test_aligned, y_pred, 'r-', linewidth=1, alpha=0.7, label='Predicted')
    ax1_results.set_ylabel('Energy (MW)', fontsize=10)
    ax1_results.set_title('Predictions vs Actual Over Time', fontsize=12, fontweight='bold')
    ax1_results.legend()
    ax1_results.grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    ax2_results = axes_results[1]
    ax2_results.scatter(y_test_aligned, y_pred, alpha=0.5, s=10)
    _min_val = min(y_test_aligned.min(), y_pred.min())
    _max_val = max(y_test_aligned.max(), y_pred.max())
    ax2_results.plot([_min_val, _max_val], [_min_val, _max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax2_results.set_xlabel('Actual (MW)', fontsize=10)
    ax2_results.set_ylabel('Predicted (MW)', fontsize=10)
    ax2_results.set_title('Predictions vs Actual (Scatter)', fontsize=12, fontweight='bold')
    ax2_results.legend()
    ax2_results.grid(True, alpha=0.3)

    # Plot 3: Residuals over time
    ax3_results = axes_results[2]
    residuals = y_test_aligned - y_pred
    ax3_results.plot(timestamps_test_aligned, residuals, 'g-', linewidth=0.5, alpha=0.7)
    ax3_results.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3_results.set_xlabel('Date', fontsize=10)
    ax3_results.set_ylabel('Residuals (MW)', fontsize=10)
    ax3_results.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax3_results.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Components Analysis
    """)
    return


@app.cell
def _(X_train, estimator, np, plt, weather_cols):
    def _():
        # Visualize exogenous variable response functions
        # Create a grid of subplots for each weather variable
        n_vars = len(weather_cols)
        n_cols = 2
        n_rows = 1

        fig_exog, axes_exog = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 5))
        axes_exog = axes_exog.flatten() if n_vars > 1 else [axes_exog]

        for var_idx, _weather_var_name in enumerate(weather_cols):
            _ax_exog = axes_exog[var_idx]
            var_key = f'exog_coef_{var_idx}'

            if var_key in estimator.variables_:
                _exog_coef_vis = estimator.variables_[var_key].value
                if _exog_coef_vis is not None:
                    knots = estimator.exog_knots_[var_idx] if estimator.exog_knots_ and len(estimator.exog_knots_) > var_idx else None
                    if knots is not None:
                        # Get variable values from training data
                        x_vals = X_train[_weather_var_name].values

                        # Build basis matrix for visualization
                        # Use lag 0 (current) for main response
                        H = estimator._make_H(x_vals, knots, include_offset=False)

                        # Get response for lag 0 (main effect)
                        if _exog_coef_vis.shape[1] > 0:
                            log_response = H @ _exog_coef_vis[:, 0]  # Use lag 0

                            # Create smooth curve for visualization
                            x_smooth = np.linspace(x_vals.min(), x_vals.max(), 200)
                            H_smooth = estimator._make_H(x_smooth, knots, include_offset=False)
                            log_response_smooth = H_smooth @ _exog_coef_vis[:, 0]

                            # Plot
                            _ax_exog.scatter(x_vals, log_response, s=1, alpha=0.3, color='blue', label='Data points')
                            _ax_exog.plot(x_smooth, log_response_smooth, 'r-', linewidth=2, label='Spline fit')
                            _ax_exog.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                            _ax_exog.set_xlabel(f'{_weather_var_name}', fontsize=10)
                            _ax_exog.set_ylabel('Log Response', fontsize=10)
                            _ax_exog.set_title(f'{_weather_var_name} Response Function', fontsize=11, fontweight='bold')
                            _ax_exog.legend(fontsize=8)
                            _ax_exog.grid(True, alpha=0.3)
                        else:
                            _ax_exog.text(0.5, 0.5, 'No lags configured', ha='center', va='center', transform=_ax_exog.transAxes)
                    else:
                        _ax_exog.text(0.5, 0.5, 'No knots available', ha='center', va='center', transform=_ax_exog.transAxes)
                else:
                    _ax_exog.text(0.5, 0.5, 'No coefficients available', ha='center', va='center', transform=_ax_exog.transAxes)
            else:
                _ax_exog.text(0.5, 0.5, 'Variable not fitted', ha='center', va='center', transform=_ax_exog.transAxes)

        # No unused subplots to hide (exactly 2 variables, 2 subplots)

        plt.tight_layout()
        return plt.gcf()


    _()
    return


@app.cell
def _(estimator, make_basis_matrix, np, plt, timestamps_train_aligned):
    def _():
        # Visualize Fourier/harmonic basis functions
        if estimator.config.multi_harmonic_config and 'fourier_coef' in estimator.variables_:
            fourier_coef = estimator.variables_['fourier_coef'].value
            if fourier_coef is not None:
                # Reconstruct Fourier contribution over time
                max_idx = int(np.max(estimator.time_indices_))
                F_full = make_basis_matrix(
                    num_harmonics=estimator.config.multi_harmonic_config.num_harmonics,
                    length=max_idx + 1,
                    periods=estimator.config.multi_harmonic_config.periods
                )
                F = F_full[estimator.time_indices_.astype(int), 1:]  # Drop constant column
                fourier_contribution = F @ fourier_coef

                # Get period labels
                periods = estimator.config.multi_harmonic_config.periods
                num_harmonics = estimator.config.multi_harmonic_config.num_harmonics
                period_labels = []
                for period_hours in periods:
                    if period_hours >= 8000:  # Yearly
                        period_labels.append(f'Yearly ({period_hours/24:.1f} days)')
                    elif period_hours >= 160:  # Weekly
                        period_labels.append(f'Weekly ({period_hours/24:.1f} days)')
                    elif period_hours >= 20:  # Daily
                        period_labels.append(f'Daily ({period_hours:.1f} hours)')
                    else:
                        period_labels.append(f'{period_hours:.1f} hours')

                # Create visualization with combined and individual period contributions
                n_periods = len(periods)
                fig_fourier, axes_fourier = plt.subplots(nrows=n_periods + 2, ncols=1, figsize=(14, 4 * (n_periods + 2)))

                # Plot 1: Combined Fourier contribution over full time
                ax_combined = axes_fourier[0]
                ax_combined.plot(timestamps_train_aligned[:len(fourier_contribution)], fourier_contribution, 'b-', linewidth=0.5, alpha=0.7)
                ax_combined.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                ax_combined.set_ylabel('Fourier Contribution (log space)', fontsize=10)
                ax_combined.set_title('Combined Fourier/Harmonic Contribution Over Time', fontsize=12, fontweight='bold')
                ax_combined.grid(True, alpha=0.3)

                # Plot individual period contributions
                # The basis matrix structure: coefficients are ordered by period, then by harmonic
                # For each period with k harmonics, we have 2*k coefficients (sin and cos pairs)
                coef_idx = 0
                for period_idx, (n_harm, period_hours) in enumerate(zip(num_harmonics, periods)):
                    # Each harmonic has 2 coefficients (sin and cos)
                    n_coefs_per_period = 2 * n_harm
                    period_coefs = fourier_coef[coef_idx:coef_idx + n_coefs_per_period]

                    # Create basis matrix for just this period to extract its contribution
                    F_period = make_basis_matrix(
                        num_harmonics=[n_harm],
                        length=max_idx + 1,
                        periods=[period_hours]
                    )
                    F_period_subset = F_period[estimator.time_indices_.astype(int), 1:]  # Drop constant
                    period_contribution = F_period_subset @ period_coefs

                    ax_period = axes_fourier[period_idx + 1]
                    ax_period.plot(timestamps_train_aligned[:len(period_contribution)], period_contribution, 'g-', linewidth=0.5, alpha=0.7)
                    ax_period.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                    ax_period.set_ylabel('Contribution (log space)', fontsize=10)
                    ax_period.set_title(f'{period_labels[period_idx]} Pattern ({n_harm} harmonics)', fontsize=11, fontweight='bold')
                    ax_period.grid(True, alpha=0.3)

                    coef_idx += n_coefs_per_period

                # Plot zoomed view for daily pattern (last week of data)
                ax_zoom = axes_fourier[-1]
                zoom_days = 7
                zoom_hours = zoom_days * 24
                if len(fourier_contribution) > zoom_hours:
                    zoom_start = len(fourier_contribution) - zoom_hours
                    zoom_timestamps = timestamps_train_aligned[zoom_start:len(fourier_contribution)]
                    zoom_contribution = fourier_contribution[zoom_start:]
                    ax_zoom.plot(zoom_timestamps, zoom_contribution, 'b-', linewidth=1, alpha=0.7)
                    ax_zoom.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                    ax_zoom.set_ylabel('Fourier Contribution (log space)', fontsize=10)
                    ax_zoom.set_xlabel('Date', fontsize=10)
                    ax_zoom.set_title(f'Zoomed View: Last {zoom_days} Days (Daily Pattern)', fontsize=11, fontweight='bold')
                    ax_zoom.grid(True, alpha=0.3)
                else:
                    ax_zoom.axis('off')

                plt.tight_layout()
                # Return first figure (combined and period contributions)
                return plt.gcf()
            else:
                print("Fourier coefficients not available")
                return None
        else:
            print("Fourier/harmonic basis not configured")
            return None

    _()
    return


@app.cell
def _(estimator, make_basis_matrix, np, plt):
    def _():
        # Visualize individual harmonic basis functions
        if estimator.config.multi_harmonic_config and 'fourier_coef' in estimator.variables_:
            fourier_coef = estimator.variables_['fourier_coef'].value
            if fourier_coef is not None:
                periods = estimator.config.multi_harmonic_config.periods
                num_harmonics = estimator.config.multi_harmonic_config.num_harmonics
                period_labels = []
                for period_hours in periods:
                    if period_hours >= 8000:  # Yearly
                        period_labels.append(f'Yearly ({period_hours/24:.1f} days)')
                    elif period_hours >= 160:  # Weekly
                        period_labels.append(f'Weekly ({period_hours/24:.1f} days)')
                    elif period_hours >= 20:  # Daily
                        period_labels.append(f'Daily ({period_hours:.1f} hours)')
                    else:
                        period_labels.append(f'{period_hours:.1f} hours')

                # Now plot individual basis functions
                # Create a separate figure for individual harmonics
                total_harmonics = sum(num_harmonics)
                n_cols = 3
                n_rows = (total_harmonics + n_cols - 1) // n_cols
                fig_basis, axes_basis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4 * n_rows))
                axes_basis = axes_basis.flatten() if total_harmonics > 1 else [axes_basis]

                coef_idx = 0
                plot_idx = 0
                for period_idx, (n_harm, period_hours) in enumerate(zip(num_harmonics, periods)):
                    period_coefs = fourier_coef[coef_idx:coef_idx + 2 * n_harm]

                    # Create a full period cycle for visualization
                    # For hourly data, show one full cycle
                    n_samples_per_cycle = int(period_hours)
                    time_cycle = np.arange(n_samples_per_cycle)

                    # Create basis matrix for one cycle
                    F_cycle = make_basis_matrix(
                        num_harmonics=[n_harm],
                        length=n_samples_per_cycle,
                        periods=[period_hours]
                    )
                    F_cycle_subset = F_cycle[:, 1:]  # Drop constant

                    # Plot each harmonic pair (sin and cos)
                    for harm_idx in range(n_harm):
                        sin_coef_idx = 2 * harm_idx
                        cos_coef_idx = 2 * harm_idx + 1

                        # Get coefficients for this harmonic
                        sin_coef = period_coefs[sin_coef_idx]
                        cos_coef = period_coefs[cos_coef_idx]

                        # Get basis functions (sin and cos columns)
                        sin_basis = F_cycle_subset[:, sin_coef_idx]
                        cos_basis = F_cycle_subset[:, cos_coef_idx]

                        # Compute contribution of this harmonic
                        harmonic_contribution = sin_coef * sin_basis + cos_coef * cos_basis

                        # Plot
                        ax_basis = axes_basis[plot_idx]
                        ax_basis.plot(time_cycle, harmonic_contribution, 'b-', linewidth=2, label=f'Harmonic {harm_idx + 1}')
                        ax_basis.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                        ax_basis.set_xlabel('Time (hours)', fontsize=9)
                        ax_basis.set_ylabel('Contribution', fontsize=9)
                        ax_basis.set_title(f'{period_labels[period_idx]}, Harmonic {harm_idx + 1}\n(sin_coef={sin_coef:.4f}, cos_coef={cos_coef:.4f})', fontsize=10, fontweight='bold')
                        ax_basis.legend(fontsize=8)
                        ax_basis.grid(True, alpha=0.3)

                        plot_idx += 1

                    coef_idx += 2 * n_harm

                # Hide unused subplots
                for idx in range(plot_idx, len(axes_basis)):
                    axes_basis[idx].axis('off')

                plt.tight_layout()
                return plt.gcf()
            else:
                print("Fourier coefficients not available")
                return None
        else:
            print("Fourier/harmonic basis not configured")
            return None

    _()
    return


@app.cell
def _(estimator, np, outlier_threshold, plt, timestamps_train_aligned):
    def _():
        # Visualize outlier detector if present
        # Check if outlier is configured
        has_outlier_config = hasattr(estimator, 'config') and estimator.config.outlier_config is not None
        has_outlier_var = 'outlier' in estimator.variables_
        has_outlier_value = has_outlier_var and estimator.variables_['outlier'].value is not None

        if not has_outlier_config:
            print("Outlier detector not configured in estimator.config.outlier_config")
            return None

        if not has_outlier_var:
            print("Outlier variable not found in estimator.variables_")
            return None

        if not has_outlier_value:
            print("Outlier variable value is None")
            return None

        _outlier_values_vis = estimator.variables_['outlier'].value

        # Map outlier values to timestamps using outlier_T_matrix
        if hasattr(estimator, 'outlier_T_matrix_') and estimator.outlier_T_matrix_ is not None:
            # Outlier values are per period, need to map to samples
            _T_outlier = estimator.outlier_T_matrix_
            _outlier_per_sample_vis = _T_outlier @ _outlier_values_vis

            # Get timestamps for training data
            outlier_timestamps = timestamps_train_aligned[:len(_outlier_per_sample_vis)]

            # Use threshold from UI (in log space)
            # 0.05 in log space ≈ 5% multiplicative effect (exp(0.05) ≈ 1.05)
            # 0.1 in log space ≈ 10% multiplicative effect (exp(0.1) ≈ 1.10)
            threshold_value = outlier_threshold.value

            # Calculate statistics with different thresholds
            # Very small threshold for "non-zero" (essentially any non-zero value)
            non_zero_mask = np.abs(_outlier_values_vis) > 1e-6
            non_zero_outliers = _outlier_values_vis[non_zero_mask]

            # Threshold for "significant" outliers (from UI)
            significant_mask = np.abs(_outlier_values_vis) > threshold_value
            significant_outliers = _outlier_values_vis[significant_mask]

            n_periods = len(_outlier_values_vis)
            n_non_zero = len(non_zero_outliers)
            n_significant = len(significant_outliers)

            outlier_period_hours = estimator.outlier_period_hours_ if hasattr(estimator, 'outlier_period_hours_') else None
            period_label = f'{outlier_period_hours/24:.1f} days' if outlier_period_hours else 'period'

            # Find outlier period indices (periods with significant outliers)
            outlier_period_indices = np.where(significant_mask)[0]
            non_zero_period_indices = np.where(non_zero_mask)[0]

            # Calculate number of outlier days
            # If period_hours is 24, then each period is a day
            if outlier_period_hours == 24.0:
                n_outlier_days_non_zero = n_non_zero
                n_outlier_days_significant = n_significant
                day_label = "days"
            elif outlier_period_hours is not None:
                # Convert periods to days
                n_outlier_days_non_zero = n_non_zero * (outlier_period_hours / 24.0)
                n_outlier_days_significant = n_significant * (outlier_period_hours / 24.0)
                day_label = "day-equivalents"
            else:
                n_outlier_days_non_zero = n_non_zero
                n_outlier_days_significant = n_significant
                day_label = "periods"

            fig_outlier, axes_outlier = plt.subplots(nrows=3, ncols=1, figsize=(14, 10))

            # Plot 1: Outlier values over time (per sample)
            ax1_outlier = axes_outlier[0]
            ax1_outlier.plot(outlier_timestamps, _outlier_per_sample_vis, 'b-', linewidth=0.5, alpha=0.7, label='Outlier correction')
            ax1_outlier.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax1_outlier.set_ylabel('Outlier Correction (log space)', fontsize=10)
            ax1_outlier.set_title('Outlier Detector: Corrections Over Time (per sample)', fontsize=12, fontweight='bold')
            ax1_outlier.legend(fontsize=9)
            ax1_outlier.grid(True, alpha=0.3)

            # Plot 2: Outlier values per period
            ax2_outlier = axes_outlier[1]
            period_indices = np.arange(len(_outlier_values_vis))
            ax2_outlier.plot(period_indices, _outlier_values_vis, 'go-', markersize=3, linewidth=0.5, alpha=0.7, label='Outlier per period')
            ax2_outlier.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            # Add threshold lines
            ax2_outlier.axhline(y=threshold_value, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Threshold: ±{threshold_value:.3f}')
            ax2_outlier.axhline(y=-threshold_value, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            # Highlight significant outliers
            if n_significant > 0:
                significant_periods = period_indices[significant_mask]
                significant_vals = _outlier_values_vis[significant_mask]
                ax2_outlier.scatter(significant_periods, significant_vals, s=50, c='red', marker='x', linewidths=2, label=f'Significant outliers (n={n_significant})', zorder=5)
            ax2_outlier.set_ylabel('Outlier Correction (log space)', fontsize=10)
            ax2_outlier.set_xlabel(f'Period Index ({period_label} per period)', fontsize=10)
            ax2_outlier.set_title(f'Outlier Detector: Corrections Per Period (Non-zero: {n_non_zero}/{n_periods}, Significant: {n_significant}/{n_periods})', fontsize=12, fontweight='bold')
            ax2_outlier.legend(fontsize=9)
            ax2_outlier.grid(True, alpha=0.3)

            # Plot 3: Histogram of outlier values
            ax3_outlier = axes_outlier[2]
            ax3_outlier.hist(_outlier_values_vis, bins=50, alpha=0.7, edgecolor='black', color='steelblue', label='All periods')
            ax3_outlier.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            # Add threshold lines
            ax3_outlier.axvline(x=threshold_value, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Threshold: ±{threshold_value:.3f}')
            ax3_outlier.axvline(x=-threshold_value, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            if len(non_zero_outliers) > 0:
                ax3_outlier.axvline(x=np.mean(non_zero_outliers), color='g', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean (non-zero): {np.mean(non_zero_outliers):.4f}')
            if len(significant_outliers) > 0:
                ax3_outlier.axvline(x=np.mean(significant_outliers), color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean (significant): {np.mean(significant_outliers):.4f}')
            ax3_outlier.set_xlabel('Outlier Correction Value (log space)', fontsize=10)
            ax3_outlier.set_ylabel('Frequency', fontsize=10)
            ax3_outlier.set_title(f'Outlier Distribution (Non-zero: {n_non_zero}/{n_periods}, Significant: {n_significant}/{n_periods})', fontsize=12, fontweight='bold')
            ax3_outlier.legend(fontsize=9)
            ax3_outlier.grid(True, alpha=0.3)

            plt.tight_layout()

            # Print statistics about outlier days
            print("\nOutlier Detector Statistics:")
            print(f"  Total periods: {n_periods}")
            print(f"  Threshold for significant outliers: |outlier| > {threshold_value:.3f} (≈{100*(np.exp(threshold_value)-1):.1f}% multiplicative effect)")
            print(f"  Non-zero outlier periods (|outlier| > 1e-6): {n_non_zero} ({100*n_non_zero/n_periods:.1f}%)")
            print(f"  Significant outlier periods (|outlier| > {threshold_value:.3f}): {n_significant} ({100*n_significant/n_periods:.1f}%)")
            print(f"  Number of outlier {day_label} (non-zero): {n_outlier_days_non_zero:.1f}")
            print(f"  Number of outlier {day_label} (significant): {n_outlier_days_significant:.1f}")
            if len(non_zero_outliers) > 0:
                print("\n  Non-zero outlier statistics:")
                print(f"    Mean: {np.mean(non_zero_outliers):.4f}")
                print(f"    Max: {np.max(np.abs(_outlier_values_vis)):.4f}")
                print(f"    Min: {np.min(non_zero_outliers):.4f}")
                print(f"    Std: {np.std(non_zero_outliers):.4f}")
            if len(significant_outliers) > 0:
                print(f"\n  Significant outlier statistics (|outlier| > {threshold_value:.3f}):")
                print(f"    Count: {n_significant}")
                print(f"    Mean: {np.mean(significant_outliers):.4f}")
                print(f"    Max: {np.max(np.abs(significant_outliers)):.4f}")
                print(f"    Min: {np.min(significant_outliers):.4f}")
                print(f"    Period indices: {outlier_period_indices[:20]}{'...' if len(outlier_period_indices) > 20 else ''}")
                # Show multiplier effects for significant outliers
                multipliers = np.exp(significant_outliers)
                print(f"    Multiplier range: {np.min(multipliers):.3f}x to {np.max(multipliers):.3f}x")
            else:
                print(f"\n  No significant outliers detected (all |outlier| <= {threshold_value:.3f})")
                if len(non_zero_period_indices) > 0:
                    print(f"  Non-zero period indices: {non_zero_period_indices[:20]}{'...' if len(non_zero_period_indices) > 20 else ''}")

            return plt.gcf()
        else:
            print("Outlier T matrix not available for visualization")
            return None

    _()
    return


@app.cell
def _(estimator, np, outlier_threshold, pd, timestamps_train_aligned):
    def _():
        # Analyze if significant outlier days correspond to holidays or interesting dates
        if 'outlier' in estimator.variables_ and estimator.variables_['outlier'].value is not None:
            _outlier_values_vis = estimator.variables_['outlier'].value
            threshold_value = outlier_threshold.value
            significant_mask = np.abs(_outlier_values_vis) > threshold_value
            outlier_period_indices = np.where(significant_mask)[0]

            if len(outlier_period_indices) == 0:
                print("No significant outliers detected to analyze.")
                return

            # Get period information
            outlier_period_hours = estimator.outlier_period_hours_ if hasattr(estimator, 'outlier_period_hours_') else 24.0

            # Map period indices to dates
            # Get the first timestamp and calculate dates for each period
            first_timestamp = timestamps_train_aligned[0]
            outlier_dates = []
            outlier_values = []

            for period_idx in outlier_period_indices:
                # Calculate the date for this period
                period_date = first_timestamp + pd.Timedelta(hours=period_idx * outlier_period_hours)
                outlier_dates.append(period_date)
                outlier_values.append(_outlier_values_vis[period_idx])

            outlier_dates = pd.DatetimeIndex(outlier_dates)

            # Check for holidays (US holidays)
            import holidays
            us_holidays = holidays.UnitedStates(years=outlier_dates.year.unique().tolist())

            # Analyze each outlier date
            results = []
            holiday_count = 0
            weekend_count = 0

            for i, date in enumerate(outlier_dates):
                date_only = date.date() if hasattr(date, 'date') else pd.Timestamp(date).date()
                is_holiday = date_only in us_holidays
                holiday_name = us_holidays.get(date_only, None)
                is_weekend = date.weekday() >= 5  # Saturday = 5, Sunday = 6
                outlier_val = outlier_values[i]
                multiplier = np.exp(outlier_val)

                results.append({
                    'date': date,
                    'outlier_value': outlier_val,
                    'multiplier': multiplier,
                    'is_holiday': is_holiday,
                    'holiday_name': holiday_name,
                    'is_weekend': is_weekend,
                    'day_of_week': date.strftime('%A')
                })

                if is_holiday:
                    holiday_count += 1
                if is_weekend:
                    weekend_count += 1

            # Print summary
            print(f"\n{'='*80}")
            print(f"Outlier Day Analysis: {len(outlier_dates)} significant outlier days")
            print(f"{'='*80}")
            print("\nSummary:")
            print(f"  Total significant outliers: {len(outlier_dates)}")
            print(f"  Holidays: {holiday_count} ({100*holiday_count/len(outlier_dates):.1f}%)")
            print(f"  Weekends: {weekend_count} ({100*weekend_count/len(outlier_dates):.1f}%)")
            print(f"  Weekdays (non-holiday): {len(outlier_dates) - holiday_count - weekend_count} ({100*(len(outlier_dates) - holiday_count - weekend_count)/len(outlier_dates):.1f}%)")

            # Print detailed table
            print("\nDetailed Outlier Days:")
            print(f"{'Date':<12} {'Day':<10} {'Outlier':<10} {'Multiplier':<12} {'Holiday/Notes':<30}")
            print(f"{'-'*80}")

            for result in sorted(results, key=lambda x: x['date']):
                date_str = result['date'].strftime('%Y-%m-%d')
                day_str = result['day_of_week']
                outlier_str = f"{result['outlier_value']:.4f}"
                mult_str = f"{result['multiplier']:.3f}x"

                notes = []
                if result['is_holiday']:
                    notes.append(result['holiday_name'])
                if result['is_weekend']:
                    notes.append("Weekend")
                if not notes:
                    notes.append("Weekday")

                notes_str = ", ".join(notes)
                print(f"{date_str:<12} {day_str:<10} {outlier_str:<10} {mult_str:<12} {notes_str:<30}")

            # Additional insights
            print(f"\n{'='*80}")
            print("Insights:")
            if holiday_count > 0:
                print(f"  • {holiday_count} outlier day(s) correspond to holidays")
                holiday_outliers = [r for r in results if r['is_holiday']]
                print(f"    Average holiday outlier value: {np.mean([r['outlier_value'] for r in holiday_outliers]):.4f}")
                print(f"    Average holiday multiplier: {np.mean([r['multiplier'] for r in holiday_outliers]):.3f}x")

            if weekend_count > 0:
                print(f"  • {weekend_count} outlier day(s) are weekends")
                weekend_outliers = [r for r in results if r['is_weekend']]
                print(f"    Average weekend outlier value: {np.mean([r['outlier_value'] for r in weekend_outliers]):.4f}")
                print(f"    Average weekend multiplier: {np.mean([r['multiplier'] for r in weekend_outliers]):.3f}x")

            weekday_outliers = [r for r in results if not r['is_holiday'] and not r['is_weekend']]
            if len(weekday_outliers) > 0:
                print(f"  • {len(weekday_outliers)} outlier day(s) are regular weekdays (may indicate special events or data issues)")
                print(f"    Average weekday outlier value: {np.mean([r['outlier_value'] for r in weekday_outliers]):.4f}")
                print(f"    Average weekday multiplier: {np.mean([r['multiplier'] for r in weekday_outliers]):.3f}x")

            print(f"{'='*80}\n")

            return results
        else:
            print("Outlier detector not configured or not fitted")
            return None

    _()
    return


@app.cell
def _(
    TsgamSplineConfig,
    X_train,
    estimator,
    make_basis_matrix,
    np,
    plt,
    stats,
    timestamps_train_aligned,
    y_train_log,
):
    def _():
        # Compute residuals and fit distributions
        # Get baseline predictions (without AR)
        baseline_pred = np.full(len(y_train_log), estimator.variables_['constant'].value)

        # Add Fourier terms
        if estimator.config.multi_harmonic_config:
            max_idx = int(np.max(estimator.time_indices_))
            F_full = make_basis_matrix(
                num_harmonics=estimator.config.multi_harmonic_config.num_harmonics,
                length=max_idx + 1,
                periods=estimator.config.multi_harmonic_config.periods
            )
            F = F_full[estimator.time_indices_.astype(int), 1:]
            fourier_coef = estimator.variables_['fourier_coef'].value
            if fourier_coef is not None:
                baseline_pred += F @ fourier_coef

        # Add exogenous terms
        if estimator.config.exog_config:
            for ix, exog_cfg in enumerate(estimator.config.exog_config):
                exog_var = X_train.iloc[:, ix].values
                stored_knots = estimator.exog_knots_[ix] if isinstance(exog_cfg, TsgamSplineConfig) else None
                _, Hs = estimator._process_exog_config(exog_cfg, exog_var, knots=stored_knots)
                _exog_coef_res = estimator.variables_[f'exog_coef_{ix}'].value
                if _exog_coef_res is not None:
                    # Handle NaN in basis matrices (from lead/lag boundaries)
                    exog_pred = np.zeros(len(exog_var))
                    for lag_ix, H in enumerate(Hs):
                        H_clean = np.nan_to_num(H, nan=0.0)
                        lag_contrib = H_clean @ _exog_coef_res[:, lag_ix]
                        exog_pred += lag_contrib
                    baseline_pred += exog_pred

        # Add outlier term if present
        if 'outlier' in estimator.variables_ and estimator.variables_['outlier'].value is not None:
            if hasattr(estimator, 'outlier_T_matrix_') and estimator.outlier_T_matrix_ is not None:
                _outlier_values_res = estimator.variables_['outlier'].value
                _T_outlier_res = estimator.outlier_T_matrix_
                _outlier_per_sample_res = _T_outlier_res @ _outlier_values_res
                baseline_pred += _outlier_per_sample_res[:len(baseline_pred)]

        # Check baseline_pred for NaN before computing residuals
        if np.any(~np.isfinite(baseline_pred)):
            nan_count = np.sum(~np.isfinite(baseline_pred))
            print(f"Warning: {nan_count} non-finite values in baseline_pred. This may be from lead/lag boundaries.")
            # Replace NaN with 0 (no contribution from components with NaN)
            baseline_pred = np.nan_to_num(baseline_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # Check y_train_log for non-finite values
        if np.any(~np.isfinite(y_train_log)):
            nan_count = np.sum(~np.isfinite(y_train_log))
            print(f"Warning: {nan_count} non-finite values in y_train_log.")

        # Compute residuals
        train_residuals = y_train_log - baseline_pred

        # Check for non-finite values in residuals and filter them out
        valid_residuals_mask = np.isfinite(train_residuals)
        if not np.all(valid_residuals_mask):
            nan_count = np.sum(~valid_residuals_mask)
            print(f"Warning: {nan_count} non-finite values in residuals. Filtering them out.")
            # Filter out non-finite values
            train_residuals = train_residuals[valid_residuals_mask]
            print(f"Using {len(train_residuals)} valid residuals out of {len(y_train_log)} total samples")

        if len(train_residuals) == 0:
            raise ValueError("No valid residuals available after filtering non-finite values.")

        # Final check - ensure no non-finite values remain
        if np.any(~np.isfinite(train_residuals)):
            raise ValueError(f"train_residuals still contains {np.sum(~np.isfinite(train_residuals))} non-finite values after filtering.")

        # Fit normal distribution
        mu_norm, sigma_norm = stats.norm.fit(train_residuals)

        # Fit Laplace distribution
        loc_laplace, scale_laplace = stats.laplace.fit(train_residuals)

        # Create visualization with 5 plots (2x3 grid)
        fig_residuals, axes_residuals = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

        # Plot 1: Residuals over time
        ax1_res = axes_residuals[0, 0]
        # Use valid timestamps (aligned with filtered residuals)
        if len(valid_residuals_mask) == len(timestamps_train_aligned):
            valid_timestamps = timestamps_train_aligned[valid_residuals_mask]
        else:
            valid_timestamps = timestamps_train_aligned[:len(train_residuals)]
        ax1_res.plot(valid_timestamps, train_residuals, 'b-', linewidth=0.5, alpha=0.7)
        ax1_res.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1_res.set_xlabel('Date', fontsize=10)
        ax1_res.set_ylabel('Residuals (log space)', fontsize=10)
        ax1_res.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        ax1_res.grid(True, alpha=0.3)

        # Plot 2: Residual histogram with normal fit
        ax2_res = axes_residuals[0, 1]
        ax2_res.hist(train_residuals, bins=50, density=True, alpha=0.7, edgecolor='black', label='Residuals')
        x_norm = np.linspace(train_residuals.min(), train_residuals.max(), 200)
        ax2_res.plot(x_norm, stats.norm.pdf(x_norm, mu_norm, sigma_norm), 'r-', linewidth=2, label=f'Normal (μ={mu_norm:.3f}, σ={sigma_norm:.3f})')
        ax2_res.set_xlabel('Residual Value (log space)', fontsize=10)
        ax2_res.set_ylabel('Density', fontsize=10)
        ax2_res.set_title('Residual Distribution: Normal Fit', fontsize=12, fontweight='bold')
        ax2_res.legend()
        ax2_res.grid(True, alpha=0.3)

        # Plot 3: Residual histogram with Laplace fit
        ax3_res = axes_residuals[0, 2]
        ax3_res.hist(train_residuals, bins=50, density=True, alpha=0.7, edgecolor='black', label='Residuals')
        x_laplace = np.linspace(train_residuals.min(), train_residuals.max(), 200)
        ax3_res.plot(x_laplace, stats.laplace.pdf(x_laplace, loc_laplace, scale_laplace), 'g-', linewidth=2, label=f'Laplace (loc={loc_laplace:.3f}, scale={scale_laplace:.3f})')
        ax3_res.set_xlabel('Residual Value (log space)', fontsize=10)
        ax3_res.set_ylabel('Density', fontsize=10)
        ax3_res.set_title('Residual Distribution: Laplace Fit', fontsize=12, fontweight='bold')
        ax3_res.legend()
        ax3_res.grid(True, alpha=0.3)

        # Plot 4: Q-Q plot (normal)
        ax4_res = axes_residuals[1, 0]
        stats.probplot(train_residuals, dist="norm", plot=ax4_res)
        ax4_res.set_title('Q-Q Plot: Normal Distribution', fontsize=12, fontweight='bold')
        ax4_res.grid(True, alpha=0.3)

        # Plot 5: Q-Q plot (Laplace)
        ax5_res = axes_residuals[1, 1]
        stats.probplot(train_residuals, dist="laplace", plot=ax5_res)
        ax5_res.set_title('Q-Q Plot: Laplace Distribution', fontsize=12, fontweight='bold')
        ax5_res.grid(True, alpha=0.3)

        # Hide the last subplot (6th position)
        axes_residuals[1, 2].axis('off')

        plt.tight_layout()

        print("Residual Statistics:")
        print(f"  Mean: {np.mean(train_residuals):.6f}")
        print(f"  Std: {np.std(train_residuals):.6f}")
        print(f"  Normal fit: μ={mu_norm:.6f}, σ={sigma_norm:.6f}")
        print(f"  Laplace fit: loc={loc_laplace:.6f}, scale={scale_laplace:.6f}")

        return plt.gcf()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Ablation Test

    Test different model configurations to understand the contribution of each component.
    """)
    return


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
    X_test,
    X_train,
    np,
    pd,
    take_log,
    use_ar,
    use_outlier,
    weather_cols,
    y_test_aligned,
    y_train_aligned,
):

    # Ablation test: test different model configurations
    print("Running ablation test...")
    print("=" * 80)

    # Apply log transform if requested (same as main model)
    if take_log.value:
        y_train_log_ablation = np.log(y_train_aligned + 1.0)
        np.log(y_test_aligned + 1.0)
    else:
        y_train_log_ablation = y_train_aligned.copy()
        y_test_aligned.copy()

    # Define ablation configurations
    ablation_configs = []

    # 1. Baseline: No harmonics, no exogenous, no AR, no outliers
    ablation_configs.append({
        'name': 'Baseline (constant only)',
        'multi_harmonic': None,
        'exog': None,
        'ar': None,
        'outlier': None
    })

    # 2. Harmonics only (yearly)
    ablation_configs.append({
        'name': 'Harmonics: Yearly only',
        'multi_harmonic': TsgamMultiHarmonicConfig(
            num_harmonics=[4, 0, 0],
            periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
            reg_weight=6e-5
        ),
        'exog': None,
        'ar': None,
        'outlier': None
    })

    # 3. Harmonics only (weekly)
    ablation_configs.append({
        'name': 'Harmonics: Weekly only',
        'multi_harmonic': TsgamMultiHarmonicConfig(
            num_harmonics=[0, 4, 0],
            periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
            reg_weight=6e-5
        ),
        'exog': None,
        'ar': None,
        'outlier': None
    })

    # 4. Harmonics only (daily)
    ablation_configs.append({
        'name': 'Harmonics: Daily only',
        'multi_harmonic': TsgamMultiHarmonicConfig(
            num_harmonics=[0, 0, 6],
            periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
            reg_weight=6e-5
        ),
        'exog': None,
        'ar': None,
        'outlier': None
    })

    # 5. All harmonics (yearly + weekly + daily)
    ablation_configs.append({
        'name': 'Harmonics: All (yearly + weekly + daily)',
        'multi_harmonic': TsgamMultiHarmonicConfig(
            num_harmonics=[4, 4, 6],
            periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
            reg_weight=6e-5
        ),
        'exog': None,
        'ar': None,
        'outlier': None
    })

    # 6. Exogenous only
    if len(weather_cols) > 0 and X_train.shape[1] > 0:
        exog_config_ablation = []
        var_configs_ablation = {
            'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
            'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
            'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        }
        for _var_name_ablation in X_train.columns:
            if _var_name_ablation in var_configs_ablation:
                exog_config_ablation.append(var_configs_ablation[_var_name_ablation])
            else:
                exog_config_ablation.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

        ablation_configs.append({
            'name': 'Exogenous only',
            'multi_harmonic': None,
            'exog': exog_config_ablation,
            'ar': None,
            'outlier': None
        })

    # 7. All harmonics + Exogenous
    if len(weather_cols) > 0 and X_train.shape[1] > 0:
        exog_config_ablation = []
        var_configs_ablation = {
            'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
            'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
            'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        }
        for _var_name_ablation in X_train.columns:
            if _var_name_ablation in var_configs_ablation:
                exog_config_ablation.append(var_configs_ablation[_var_name_ablation])
            else:
                exog_config_ablation.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

        ablation_configs.append({
            'name': 'Harmonics (all) + Exogenous',
            'multi_harmonic': TsgamMultiHarmonicConfig(
                num_harmonics=[4, 4, 6],
                periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
                reg_weight=6e-5
            ),
            'exog': exog_config_ablation,
            'ar': None,
            'outlier': None
        })

    # 8. All harmonics + Exogenous + AR
    if len(weather_cols) > 0 and X_train.shape[1] > 0:
        exog_config_ablation = []
        var_configs_ablation = {
            'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
            'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
            'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        }
        for _var_name_ablation in X_train.columns:
            if _var_name_ablation in var_configs_ablation:
                exog_config_ablation.append(var_configs_ablation[_var_name_ablation])
            else:
                exog_config_ablation.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

        ablation_configs.append({
            'name': 'Harmonics (all) + Exogenous + AR',
            'multi_harmonic': TsgamMultiHarmonicConfig(
                num_harmonics=[4, 4, 6],
                periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
                reg_weight=6e-5
            ),
            'exog': exog_config_ablation,
            'ar': TsgamArConfig(lags=[1, 2, 3, 4], l1_constraint=0.97),
            'outlier': None
        })

    # 9. Harmonics (all) + Outlier detector
    ablation_configs.append({
        'name': 'Harmonics (all) + Outlier detector',
        'multi_harmonic': TsgamMultiHarmonicConfig(
            num_harmonics=[4, 4, 6],
            periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
            reg_weight=6e-5
        ),
        'exog': None,
        'ar': None,
        'outlier': TsgamOutlierConfig(reg_weight=1e-4, period_hours=24.0)
    })

    # 10. All harmonics + Exogenous + Outlier detector
    if len(weather_cols) > 0 and X_train.shape[1] > 0:
        exog_config_ablation = []
        var_configs_ablation = {
            'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
            'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
            'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        }
        for _var_name_ablation in X_train.columns:
            if _var_name_ablation in var_configs_ablation:
                exog_config_ablation.append(var_configs_ablation[_var_name_ablation])
            else:
                exog_config_ablation.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

        ablation_configs.append({
            'name': 'Harmonics (all) + Exogenous + Outlier detector',
            'multi_harmonic': TsgamMultiHarmonicConfig(
                num_harmonics=[4, 4, 6],
                periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
                reg_weight=6e-5
            ),
            'exog': exog_config_ablation,
            'ar': None,
            'outlier': TsgamOutlierConfig(reg_weight=1e-4, period_hours=24.0)
        })

    # 11. All harmonics + Exogenous + AR + Outlier detector
    if len(weather_cols) > 0 and X_train.shape[1] > 0:
        exog_config_ablation = []
        var_configs_ablation = {
            'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
            'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
            'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        }
        for _var_name_ablation in X_train.columns:
            if _var_name_ablation in var_configs_ablation:
                exog_config_ablation.append(var_configs_ablation[_var_name_ablation])
            else:
                exog_config_ablation.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

        ablation_configs.append({
            'name': 'Harmonics (all) + Exogenous + AR + Outlier detector',
            'multi_harmonic': TsgamMultiHarmonicConfig(
                num_harmonics=[4, 4, 6],
                periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
                reg_weight=6e-5
            ),
            'exog': exog_config_ablation,
            'ar': TsgamArConfig(lags=[1, 2, 3, 4], l1_constraint=0.97),
            'outlier': TsgamOutlierConfig(reg_weight=1e-4, period_hours=24.0)
        })

    # 12. Full model (all components) - matches main model config
    if len(weather_cols) > 0 and X_train.shape[1] > 0:
        exog_config_ablation = []
        var_configs_ablation = {
            'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
            'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
            'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
            'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        }
        for _var_name_ablation in X_train.columns:
            if _var_name_ablation in var_configs_ablation:
                exog_config_ablation.append(var_configs_ablation[_var_name_ablation])
            else:
                exog_config_ablation.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

        outlier_config_ablation = None
        if use_outlier.value:
            outlier_config_ablation = TsgamOutlierConfig(reg_weight=1e-4, period_hours=24.0)

        ablation_configs.append({
            'name': 'Full model (all components)',
            'multi_harmonic': TsgamMultiHarmonicConfig(
                num_harmonics=[4, 4, 6],
                periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
                reg_weight=6e-5
            ),
            'exog': exog_config_ablation,
            'ar': TsgamArConfig(lags=[1, 2, 3, 4], l1_constraint=0.97) if use_ar.value else None,
            'outlier': outlier_config_ablation
        })

    # Run ablation tests
    ablation_results = []

    for _idx_ablation, config_dict in enumerate(ablation_configs):
        print(f"\n[{_idx_ablation+1}/{len(ablation_configs)}] Testing: {config_dict['name']}")
        print("-" * 80)

        try:
            # Create config
            solver_config_ablation = TsgamSolverConfig(
                solver='CLARABEL',
                verbose=False  # Less verbose for ablation
            )

            config_ablation = TsgamEstimatorConfig(
                multi_harmonic_config=config_dict['multi_harmonic'],
                exog_config=config_dict['exog'],
                ar_config=config_dict['ar'],
                outlier_config=config_dict['outlier'],
                solver_config=solver_config_ablation,
                random_state=42
            )

            # Create and fit estimator
            estimator_ablation = TsgamEstimator(config=config_ablation)

            # Handle case where no exogenous variables are used
            if config_dict['exog'] is None or X_train.shape[1] == 0:
                # Create empty DataFrame with same index
                X_train_empty = pd.DataFrame(index=X_train.index)
                X_test_empty = pd.DataFrame(index=X_test.index)
                estimator_ablation.fit(X_train_empty, y_train_log_ablation)
                y_pred_log_ablation = estimator_ablation.predict(X_test_empty)
            else:
                estimator_ablation.fit(X_train, y_train_log_ablation)
                y_pred_log_ablation = estimator_ablation.predict(X_test)

            # Transform back if log was used
            if take_log.value:
                y_pred_ablation = np.exp(y_pred_log_ablation) - 1.0
            else:
                y_pred_ablation = y_pred_log_ablation

            # Calculate metrics
            mae_ablation = np.mean(np.abs(y_pred_ablation - y_test_aligned))
            rmse_ablation = np.sqrt(np.mean((y_pred_ablation - y_test_aligned) ** 2))
            mape_ablation = np.mean(np.abs((y_pred_ablation - y_test_aligned) / (y_test_aligned + 1e-6))) * 100
            ss_res_ablation = np.sum((y_test_aligned - y_pred_ablation) ** 2)
            ss_tot_ablation = np.sum((y_test_aligned - np.mean(y_test_aligned)) ** 2)
            r2_ablation = 1 - (ss_res_ablation / ss_tot_ablation)

            ablation_results.append({
                'name': config_dict['name'],
                'mae': mae_ablation,
                'rmse': rmse_ablation,
                'mape': mape_ablation,
                'r2': r2_ablation,
                'status': estimator_ablation.problem_.status
            })

            print(f"  Status: {estimator_ablation.problem_.status}")
            print(f"  MAE:  {mae_ablation:.2f} MW")
            print(f"  RMSE: {rmse_ablation:.2f} MW")
            print(f"  MAPE: {mape_ablation:.2f}%")
            print(f"  R²:   {r2_ablation:.4f}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            ablation_results.append({
                'name': config_dict['name'],
                'mae': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'status': 'error'
            })

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION TEST SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<40} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'Status':>15}")
    print("-" * 80)

    for result in ablation_results:
        if np.isfinite(result['mae']):
            print(f"{result['name']:<40} {result['mae']:>10.2f} {result['rmse']:>10.2f} {result['mape']:>10.2f} {result['r2']:>10.4f} {result['status']:>15}")
        else:
            print(f"{result['name']:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {result['status']:>15}")
    return (ablation_results,)


@app.cell
def _(ablation_results, mo, pd):
    # Create a DataFrame for easier viewing
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df = ablation_df.sort_values('mae')  # Sort by MAE (best first)

    mo.md(f"""
    ### Ablation Test Results (sorted by MAE)

    {ablation_df.to_markdown(index=False)}
    """)
    return


@app.cell
def _(mae, mape, mo, r2, rmse):
    mo.md(f"""
    ## Summary

    The TSGAM estimator successfully handled the data with gaps (missing last week of each month).

    **Performance Metrics:**
    - MAE: {mae:.2f} MW
    - RMSE: {rmse:.2f} MW
    - MAPE: {mape:.2f}%
    - R²: {r2:.4f}

    The model was able to learn from the training data despite the gaps, demonstrating
    that the estimator correctly handles missing time periods in the data.
    """)
    return


if __name__ == "__main__":
    app.run()
