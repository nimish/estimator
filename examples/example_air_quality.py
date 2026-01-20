# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Air Quality Forecasting with TSGAM

This example demonstrates forecasting PM2.5 air quality using:
- Multi-harmonic Fourier basis for seasonal patterns (daily, weekly, yearly)
- Temperature as an exogenous variable with spline basis
- Autoregressive (AR) modeling of residuals

The example uses real air quality data from Beijing, China, which is publicly
available and includes both PM2.5 and temperature measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import zipfile
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src directory to path to import tsgam_estimator
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSplineConfig,
    TsgamArConfig,
    TsgamSolverConfig,
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
)


def download_beijing_air_quality_data(data_dir: Path):
    """
    Download Beijing air quality dataset from UCI ML Repository.

    Dataset: Beijing PM2.5 Data Set
    Source: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

    Parameters
    ----------
    data_dir : Path
        Directory to save the data file.
    """
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

        # Extract the CSV file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the CSV file in the zip
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_files:
                zip_ref.extract(csv_files[0], data_dir)
                # Rename if needed
                extracted_file = data_dir / csv_files[0]
                if extracted_file != data_file:
                    extracted_file.rename(data_file)

        # Clean up zip file
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
    """
    Load and preprocess Beijing air quality data.

    Parameters
    ----------
    data_file : Path
        Path to the CSV file.

    Returns
    -------
    df : DataFrame
        Preprocessed dataframe with DatetimeIndex and required columns.
    """
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    # Create datetime index
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']]
    )
    df = df.set_index('datetime')

    # Select relevant columns: PM2.5 (target) and meteorological variables
    # DEWP: Dew point (humidity indicator)
    # TEMP: Temperature
    # PRES: Pressure
    # Iws: Cumulated wind speed
    # Is: Cumulated hours of snow
    # Ir: Cumulated hours of rain
    # cbwd: Wind direction (categorical - will encode)
    df = df[['pm2.5', 'TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd']].copy()

    # Rename columns for clarity
    df.columns = ['pm25', 'temperature', 'dewpoint', 'pressure', 'wind_speed', 'snow_hours', 'rain_hours', 'wind_dir']

    # Filter to reasonable ranges (remove outliers)
    df = df[(df['pm25'] > 0) | df['pm25'].isna()]  # Keep NaN for now
    df = df[(df['pm25'] < 1000) | df['pm25'].isna()]
    df = df[(df['temperature'] > -50) | df['temperature'].isna()]
    df = df[(df['temperature'] < 50) | df['temperature'].isna()]
    df = df[(df['dewpoint'] > -50) | df['dewpoint'].isna()]
    df = df[(df['dewpoint'] < 50) | df['dewpoint'].isna()]
    df = df[(df['pressure'] > 900) | df['pressure'].isna()]  # Reasonable pressure range
    df = df[(df['pressure'] < 1100) | df['pressure'].isna()]
    df = df[(df['wind_speed'] >= 0) | df['wind_speed'].isna()]
    df = df[(df['wind_speed'] < 100) | df['wind_speed'].isna()]  # Reasonable wind speed

    # Sort by datetime
    df = df.sort_index()

    # Create a regular hourly index for the date range
    date_range = pd.date_range(
        start=df.index.min().floor('h'),  # Round down to hour
        end=df.index.max().floor('h'),
        freq='h'
    )

    # Reindex to regular hourly frequency
    df = df.reindex(date_range)

    # Interpolate missing values (forward fill then backward fill for edges)
    df = df.interpolate(method='linear', limit_direction='both')

    # If there are still missing values at the edges, forward/backward fill
    df = df.ffill().bfill()

    # Final check: remove any remaining NaN (shouldn't happen after interpolation)
    df = df.dropna()

    print(f"Loaded {len(df)} samples")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"PM2.5 range: {df['pm25'].min():.1f} to {df['pm25'].max():.1f} μg/m³")
    print(f"Temperature range: {df['temperature'].min():.1f} to {df['temperature'].max():.1f} °C")
    print(f"Dewpoint range: {df['dewpoint'].min():.1f} to {df['dewpoint'].max():.1f} °C")
    print(f"Pressure range: {df['pressure'].min():.1f} to {df['pressure'].max():.1f} hPa")
    print(f"Wind speed range: {df['wind_speed'].min():.1f} to {df['wind_speed'].max():.1f} m/s")

    return df


def fit_model_with_variables(X_train, y_train, X_test, y_test, variable_names, config_overrides=None):
    """
    Fit a TSGAM model with specified exogenous variables.

    Parameters
    ----------
    X_train : DataFrame
        Training exogenous variables.
    y_train : ndarray
        Training target values.
    X_test : DataFrame
        Test exogenous variables.
    y_test : ndarray
        Test target values.
    variable_names : list of str
        Names of variables being used (for display).
    config_overrides : dict, optional
        Overrides for default config settings.

    Returns
    -------
    rmse : float
        Root mean squared error on test set.
    mae : float
        Mean absolute error on test set.
    mape : float
        Mean absolute percentage error on test set.
    """
    # Default configuration
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[8, 6, 4],
        periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
        reg_weight=6e-5
    )

    # Create exog config based on number of variables
    n_vars = len(variable_names)
    exog_config = []

    # Default configs for each variable type
    configs = {
        'temperature': TsgamSplineConfig(n_knots=12, lags=[-2, -1, 0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
        'dewpoint': TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        'wind_speed': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
        'pressure': TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5),
    }

    # Apply configs in order
    for var_name in variable_names:
        if var_name in configs:
            exog_config.append(configs[var_name])
        else:
            # Default config for unknown variables
            exog_config.append(TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5))

    # Apply overrides if provided
    if config_overrides:
        if 'reg_weight' in config_overrides:
            for cfg in exog_config:
                cfg.reg_weight = config_overrides['reg_weight']
        if 'n_knots' in config_overrides:
            for i, cfg in enumerate(exog_config):
                if isinstance(cfg, TsgamSplineConfig):
                    cfg.n_knots = config_overrides['n_knots'][i] if isinstance(config_overrides['n_knots'], list) else config_overrides['n_knots']

    # For ablation study, skip AR model to avoid solver issues and speed up
    # AR model will be used in the full model fit
    ar_config = None  # Disable AR for ablation study
    solver_config = TsgamSolverConfig(solver='CLARABEL', verbose=False)

    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=exog_config if n_vars > 0 else None,
        ar_config=ar_config,
        solver_config=solver_config,
        random_state=42,
        debug=False
    )

    estimator = TsgamEstimator(config=config)
    estimator.fit(X_train, y_train)

    # Make predictions
    predictions = estimator.predict(X_test)

    # Convert back from log space
    y_test_original = np.exp(y_test) - 1.0
    predictions_original = np.exp(predictions) - 1.0

    # Calculate metrics
    valid_mask = np.isfinite(predictions_original) & np.isfinite(y_test_original) & (y_test_original > 0)
    if np.any(valid_mask):
        mae = np.mean(np.abs(predictions_original[valid_mask] - y_test_original[valid_mask]))
        rmse = np.sqrt(np.mean((predictions_original[valid_mask] - y_test_original[valid_mask])**2))
        mape = np.mean(np.abs((predictions_original[valid_mask] - y_test_original[valid_mask]) / y_test_original[valid_mask])) * 100
    else:
        mae = rmse = mape = np.nan

    return rmse, mae, mape


def run_single_ablation_test(test_config):
    """
    Run a single ablation test. Designed to be called in parallel.

    Parameters
    ----------
    test_config : dict
        Contains: name, X_train, y_train, X_test, y_test, variable_names

    Returns
    -------
    name : str
        Name of the test configuration
    rmse : float
        Root mean squared error
    mae : float
        Mean absolute error
    mape : float
        Mean absolute percentage error
    """
    name = test_config['name']
    X_train = test_config['X_train']
    y_train = test_config['y_train']
    X_test = test_config['X_test']
    y_test = test_config['y_test']
    variable_names = test_config['variable_names']

    try:
        rmse, mae, mape = fit_model_with_variables(X_train, y_train, X_test, y_test, variable_names)
        print(f"✓ Completed: {name} - RMSE: {rmse:.2f} μg/m³")
        return name, rmse, mae, mape, None
    except Exception as e:
        print(f"✗ Failed: {name} - {str(e)}")
        return name, np.nan, np.nan, np.nan, str(e)


def run_ablation_study(X_train, y_train, X_test, y_test, n_jobs=4):
    """
    Run ablation study by testing different combinations of exogenous variables in parallel.

    Parameters
    ----------
    X_train : DataFrame
        Training exogenous variables
    y_train : ndarray
        Training target values
    X_test : DataFrame
        Test exogenous variables
    y_test : ndarray
        Test target values
    n_jobs : int
        Number of parallel jobs to run

    Returns
    -------
    results : dict
        Dictionary with variable combinations as keys and (rmse, mae, mape) as values.
    """
    print("\n" + "="*60)
    print("Running Ablation Study (Parallel)")
    print("="*60)
    print(f"Testing different combinations of exogenous variables using {n_jobs} parallel workers...\n")

    # Prepare test configurations
    test_configs = [
        {
            'name': 'None (seasonal only)',
            'X_train': pd.DataFrame(index=X_train.index),
            'y_train': y_train,
            'X_test': pd.DataFrame(index=X_test.index),
            'y_test': y_test,
            'variable_names': []
        },
        {
            'name': 'Temperature',
            'X_train': X_train[['temperature']],
            'y_train': y_train,
            'X_test': X_test[['temperature']],
            'y_test': y_test,
            'variable_names': ['temperature']
        },
        {
            'name': 'Temperature + Dewpoint',
            'X_train': X_train[['temperature', 'dewpoint']],
            'y_train': y_train,
            'X_test': X_test[['temperature', 'dewpoint']],
            'y_test': y_test,
            'variable_names': ['temperature', 'dewpoint']
        },
        {
            'name': 'Temperature + Dewpoint + Wind Speed',
            'X_train': X_train[['temperature', 'dewpoint', 'wind_speed']],
            'y_train': y_train,
            'X_test': X_test[['temperature', 'dewpoint', 'wind_speed']],
            'y_test': y_test,
            'variable_names': ['temperature', 'dewpoint', 'wind_speed']
        },
        {
            'name': 'All variables',
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'variable_names': ['temperature', 'dewpoint', 'wind_speed', 'pressure']
        },
    ]

    results = {}

    # Run tests in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_config = {executor.submit(run_single_ablation_test, config): config['name']
                           for config in test_configs}

        for future in as_completed(future_to_config):
            name, rmse, mae, mape, error = future.result()
            if error is None:
                results[name] = (rmse, mae, mape)
                print(f"   {name}: RMSE: {rmse:.2f} μg/m³, MAE: {mae:.2f} μg/m³, MAPE: {mape:.2f}%")
            else:
                print(f"   {name}: Failed with error: {error}")

    print()  # Empty line for spacing
    return results


def plot_ablation_results(results, plots_dir):
    """Plot ablation study results with improved visualizations."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Sort combinations by number of variables (from least to most)
    def count_variables(combo):
        """Count the number of variables in a combination."""
        if 'None' in combo or 'seasonal only' in combo:
            return 0
        # Count variables by splitting on common separators
        if 'All variables' in combo:
            return 4  # temperature, dewpoint, wind_speed, pressure
        # Count '+' signs + 1
        return combo.count('+') + 1

    # Sort by variable count, then alphabetically for ties
    combinations = sorted(results.keys(), key=lambda x: (count_variables(x), x))

    rmses = [results[k][0] for k in combinations]
    maes = [results[k][1] for k in combinations]
    mapes = [results[k][2] for k in combinations]

    # Create clearer labels that indicate seasonal patterns are always used
    # except we want to emphasize the baseline uses ONLY seasonal
    labels = []
    for combo in combinations:
        if 'None' in combo or 'seasonal only' in combo:
            labels.append('Seasonal Patterns\n(No Exogenous Variables)')
        else:
            labels.append(f'Seasonal + {combo}')

    # Calculate improvements relative to baseline
    baseline_rmse = results.get('None (seasonal only)', [rmses[0] if rmses else 0])[0]
    improvements = [((baseline_rmse - rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0
                    for rmse in rmses]

    # Plot 1: RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['steelblue' if 'None' in c or 'seasonal only' in c else 'coral' if 'All' in c else 'mediumseagreen' for c in combinations]
    bars1 = ax1.bar(range(len(combinations)), rmses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('RMSE (μg/m³)', fontsize=11, fontweight='bold')
    ax1.set_title('RMSE: Seasonal Patterns + Exogenous Variables', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(combinations)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Add annotation explaining seasonal patterns
    ax1.text(0.02, 0.98, 'Note: All models use seasonal patterns\n(daily, weekly, yearly cycles)',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, rmses)):
        if not np.isnan(val):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: MAE
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(combinations)), maes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MAE (μg/m³)', fontsize=11, fontweight='bold')
    ax2.set_title('MAE: Seasonal Patterns + Exogenous Variables', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(combinations)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, maes)):
        if not np.isnan(val):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: MAPE
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(range(len(combinations)), mapes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax3.set_title('MAPE: Seasonal Patterns + Exogenous Variables', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(combinations)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars3, mapes)):
        if not np.isnan(val):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Improvement over baseline (horizontal bar chart)
    ax4 = fig.add_subplot(gs[1, :2])
    # Filter out baseline itself
    combo_imp = [(c, l, imp) for c, l, imp in zip(combinations, labels, improvements)
                 if 'None' not in c and 'seasonal only' not in c.lower()]
    combo_names = [l for _, l, _ in combo_imp]
    imp_values = [imp for _, _, imp in combo_imp]
    colors_imp = ['coral' if 'All' in c else 'mediumseagreen' for c, _, _ in combo_imp]
    bars4 = ax4.barh(range(len(combo_names)), imp_values, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('RMSE Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Model Configuration', fontsize=11, fontweight='bold')
    ax4.set_title('Improvement Over Baseline\n(Baseline: Seasonal Patterns Only, No Exogenous Variables)',
                 fontsize=12, fontweight='bold')
    ax4.set_yticks(range(len(combo_names)))
    ax4.set_yticklabels(combo_names, fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    # Add vertical line at 0
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars4, imp_values)):
        if not np.isnan(val):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

    # Plot 5: Summary table as text (using sorted order)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    table_data = []
    for combo in combinations:
        rmse, mae, mape = results[combo]
        if not np.isnan(rmse):
            improvement = improvements[combinations.index(combo)]
            # Use clearer labels in table
            if 'None' in combo or 'seasonal only' in combo:
                table_label = 'Seasonal Only\n(No Exogenous)'
            else:
                table_label = f'Seasonal + {combo}'
            table_data.append([table_label, f'{rmse:.1f}', f'{mae:.1f}', f'{mape:.1f}%', f'{improvement:.1f}%'])

    # Update table headers to be clearer
    table = ax5.table(cellText=table_data,
                      colLabels=['Model Configuration', 'RMSE', 'MAE', 'MAPE', 'Improvement'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    # Style alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.suptitle('Ablation Study: Impact of Exogenous Variables on Air Quality Forecasting\n' +
                 '(All models include seasonal patterns: daily, weekly, yearly cycles)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_file = plots_dir / "example_air_quality_ablation.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved ablation plot to: {plot_file}")
    plt.close()

    # Also create a summary table (sorted by variable count)
    print("\n" + "="*60)
    print("Ablation Study Summary")
    print("="*60)
    print("Note: All models include seasonal patterns (daily, weekly, yearly cycles)")
    print("Differences are in which exogenous variables are included.")
    print("(Ordered by number of variables: from least to most)\n")
    print(f"{'Model Configuration':<50} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print("-" * 80)
    for combo in combinations:
        rmse, mae, mape = results[combo]
        if not np.isnan(rmse):
            # Create clearer label
            if 'None' in combo or 'seasonal only' in combo:
                label = 'Seasonal Patterns Only (No Exogenous Variables)'
            else:
                label = f'Seasonal + {combo}'
            print(f"{label:<50} {rmse:>10.2f} {mae:>10.2f} {mape:>10.2f}%")

    # Calculate improvements
    baseline_rmse = results.get('None (seasonal only)', [0])[0]
    if not np.isnan(baseline_rmse) and baseline_rmse > 0:
        print("\n" + "="*60)
        print("Improvement Over Baseline")
        print("Baseline: Seasonal Patterns Only (No Exogenous Variables)")
        print("="*60)
        for combo in combinations:
            if combo != 'None (seasonal only)':
                rmse, mae, mape = results[combo]
                if not np.isnan(rmse):
                    improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
                    label = f'Seasonal + {combo}'
                    print(f"{label:<50} RMSE improvement: {improvement:>6.1f}%")


def main():
    """Main example function."""
    # Setup paths
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data" / "air_quality"
    plots_dir = examples_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Download and load data
    data_file = download_beijing_air_quality_data(data_dir)
    df = load_beijing_air_quality(data_file)

    # Use data from 2012-2014 for training, predict into 2014
    # Split: train on 2012-2013 (more data), test on first part of 2014
    train_start = '2012-01-01'
    train_end = '2013-12-31'
    test_start = '2014-01-01'
    test_end = '2014-03-31'  # Predict first quarter of 2014

    df_train = df[train_start:train_end].copy()
    df_test = df[test_start:test_end].copy()

    print(f"\nTraining data: {len(df_train)} samples ({train_start} to {train_end})")
    print(f"Test data: {len(df_test)} samples ({test_start} to {test_end})")

    # Prepare data for modeling
    # Use log transformation - helps with skewed data and multiplicative relationships
    # PM2.5 data is right-skewed (skewness ~1.8), so log helps stabilize variance
    # Note: Log transformation is beneficial but not strictly necessary for TSGAM
    # Raw values can work, but log typically provides better numerical stability
    y_train = np.log(df_train['pm25'].values + 1.0)

    # Use multiple meteorological variables for better predictions
    # Temperature, dewpoint (humidity), and wind speed are key predictors
    X_train = pd.DataFrame({
        'temperature': df_train['temperature'].values,
        'dewpoint': df_train['dewpoint'].values,
        'wind_speed': df_train['wind_speed'].values,
        'pressure': df_train['pressure'].values,
    }, index=df_train.index)

    y_test = np.log(df_test['pm25'].values + 1.0)
    X_test = pd.DataFrame({
        'temperature': df_test['temperature'].values,
        'dewpoint': df_test['dewpoint'].values,
        'wind_speed': df_test['wind_speed'].values,
        'pressure': df_test['pressure'].values,
    }, index=df_test.index)

    # Run ablation study first (in parallel)
    import multiprocessing
    n_jobs = min(4, multiprocessing.cpu_count())  # Use up to 4 cores
    ablation_results = run_ablation_study(X_train, y_train, X_test, y_test, n_jobs=n_jobs)

    # Plot ablation results
    plot_ablation_results(ablation_results, plots_dir)

    # Now fit the full model for detailed analysis
    print("\n" + "="*60)
    print("Fitting Full Model (All Variables)")
    print("="*60)

    # Multi-harmonic Fourier basis for seasonal patterns
    # Daily (24h), weekly (168h), yearly (8766h) patterns
    # Increased harmonics for better seasonal pattern capture
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[8, 6, 4],  # Balanced: 8 for daily, 6 for weekly, 4 for yearly
        periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
        reg_weight=6e-5  # Higher regularization for stability with multiple variables
    )

    # Spline configuration for exogenous variables
    # Use conservative settings to avoid numerical issues while capturing key relationships
    exog_config = [
        TsgamSplineConfig(
            n_knots=12,  # Temperature: 12 knots (reduced from 15 for stability)
            lags=[-2, -1, 0, 1, 2],  # Temperature effects with moderate lags
            reg_weight=6e-5,  # Higher regularization for stability
            diff_reg_weight=0.5
        ),
        TsgamSplineConfig(
            n_knots=10,  # Dewpoint: 10 knots for humidity effects
            lags=[-1, 0, 1],  # Dewpoint effects with some lag
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        TsgamSplineConfig(
            n_knots=8,  # Wind speed: 8 knots for dispersion effects
            lags=[0, 1],  # Wind speed effects
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
        TsgamSplineConfig(
            n_knots=8,  # Pressure: 8 knots for atmospheric stability effects
            lags=[0],  # Pressure typically immediate effect
            reg_weight=6e-5,
            diff_reg_weight=0.5
        ),
    ]

    # AR model for residual temporal dependencies
    # More AR lags to capture longer-term dependencies
    ar_config = TsgamArConfig(
        lags=[1, 2, 3, 4],  # AR(4) model for better residual modeling
        l1_constraint=0.97  # Balanced constraint
    )

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver='CLARABEL',
        verbose=False  # Set to True for debugging
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=exog_config,
        ar_config=ar_config,
        solver_config=solver_config,
        random_state=42,
        debug=False
    )

    # Create and fit estimator
    print("\nCreating and fitting TSGAM estimator...")
    estimator = TsgamEstimator(config=config)

    print("Fitting model (this may take a few minutes)...")
    estimator.fit(X_train, y_train)

    print("\nModel fitting complete!")
    print(f"Problem status: {estimator.problem_.status}")
    if estimator.problem_.status in ["optimal", "optimal_inaccurate"]:
        print(f"Optimal value: {estimator.problem_.value:.6e}")

    # Check AR model
    if estimator.ar_coef_ is not None:
        print("\nAR model fitted successfully:")
        print(f"  AR coefficients: {estimator.ar_coef_}")
        print(f"  AR intercept: {estimator.ar_intercept_:.6f}")
    else:
        print("\nAR model not fitted (insufficient data or convergence issue)")

    # Make predictions
    print("\n" + "="*60)
    print("Making predictions on test data...")
    print("="*60)

    predictions = estimator.predict(X_test)

    # Convert back from log space
    y_test_original = np.exp(y_test) - 1.0
    predictions_original = np.exp(predictions) - 1.0

    # Ensure same length (in case of any indexing issues)
    min_len = min(len(predictions_original), len(y_test_original))
    predictions_original = predictions_original[:min_len]
    y_test_original = y_test_original[:min_len]

    # Generate sample paths for uncertainty quantification
    print("\nGenerating sample paths for uncertainty quantification...")
    n_samples = 100
    samples = estimator.sample(X_test, n_samples=n_samples, random_state=42)
    samples_original = np.exp(samples) - 1.0  # Convert from log space

    # Ensure samples match length
    if samples_original.shape[1] != min_len:
        samples_original = samples_original[:, :min_len]

    # Compute percentiles
    p5 = np.percentile(samples_original, 5, axis=0)
    p25 = np.percentile(samples_original, 25, axis=0)
    p75 = np.percentile(samples_original, 75, axis=0)
    p95 = np.percentile(samples_original, 95, axis=0)

    # Calculate metrics (handle any NaN or inf values)
    valid_mask = np.isfinite(predictions_original) & np.isfinite(y_test_original) & (y_test_original > 0)
    if np.any(valid_mask):
        mae = np.mean(np.abs(predictions_original[valid_mask] - y_test_original[valid_mask]))
        rmse = np.sqrt(np.mean((predictions_original[valid_mask] - y_test_original[valid_mask])**2))
        mape = np.mean(np.abs((predictions_original[valid_mask] - y_test_original[valid_mask]) / y_test_original[valid_mask])) * 100
    else:
        mae = rmse = mape = np.nan

    print("\nForecast Performance Metrics:")
    print(f"  MAE:  {mae:.2f} μg/m³")
    print(f"  RMSE: {rmse:.2f} μg/m³")
    print(f"  MAPE: {mape:.2f}%")

    # Create visualization with ablation comparison
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, width_ratios=[2, 1])

    # Plot 1: Full time series with predictions (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_train.index, np.exp(y_train) - 1.0, 'b-', alpha=0.6, label='Training data', linewidth=0.5)
    test_idx = df_test.index[:min_len]
    ax1.plot(test_idx, y_test_original, 'g-', alpha=0.7, label='Actual (test)', linewidth=1)
    ax1.plot(test_idx, predictions_original, 'r-', label='Forecast', linewidth=1.5)
    ax1.fill_between(test_idx, p5, p95, alpha=0.2, color='red', label='90% prediction interval')
    ax1.fill_between(test_idx, p25, p75, alpha=0.3, color='red', label='50% prediction interval')
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax1.set_title('Air Quality Forecast: PM2.5 with Uncertainty Intervals', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Zoomed view of first month
    ax2 = fig.add_subplot(gs[1, 0])
    month_len = min(24*30, min_len)  # First 30 days
    month_idx = test_idx[:month_len]
    month_actual = y_test_original[:month_len]
    month_pred = predictions_original[:month_len]
    month_p5 = p5[:month_len]
    month_p95 = p95[:month_len]

    ax2.plot(month_idx, month_actual, 'g-', alpha=0.7, label='Actual', linewidth=1.5)
    ax2.plot(month_idx, month_pred, 'r-', label='Forecast', linewidth=1.5)
    ax2.fill_between(month_idx, month_p5, month_p95, alpha=0.2, color='red', label='90% interval')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax2.set_title('Zoomed View: First Month', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Residuals
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = y_test_original - predictions_original
    ax3.plot(test_idx, residuals, 'k-', alpha=0.6, linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Residual (μg/m³)', fontsize=11, fontweight='bold')
    ax3.set_title('Forecast Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Ablation comparison (RMSE comparison)
    ax4 = fig.add_subplot(gs[2, :])
    if ablation_results:
        combinations = list(ablation_results.keys())
        rmses_ablation = [ablation_results[k][0] for k in combinations]
        colors_ablation = ['steelblue' if 'None' in c else 'coral' if 'All' in c else 'mediumseagreen'
                          for c in combinations]
        bars = ax4.bar(range(len(combinations)), rmses_ablation, color=colors_ablation,
                       alpha=0.7, edgecolor='black', linewidth=1)
        # Add current model RMSE as horizontal line
        ax4.axhline(y=rmse, color='red', linestyle='--', linewidth=2,
                   label=f'Full Model (Seasonal + All Variables + AR): {rmse:.2f} μg/m³')
        # Create clearer labels
        labels_ablation = []
        for c in combinations:
            if 'None' in c or 'seasonal only' in c:
                labels_ablation.append('Seasonal Only\n(No Exogenous)')
            else:
                labels_ablation.append(f'Seasonal + {c}')
        ax4.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
        ax4.set_ylabel('RMSE (μg/m³)', fontsize=11, fontweight='bold')
        ax4.set_title('Ablation Study: RMSE Comparison\n(All models use seasonal patterns)',
                     fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(combinations)))
        ax4.set_xticklabels(labels_ablation, rotation=45, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax4.legend(loc='upper right', fontsize=9)
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, rmses_ablation)):
            if not np.isnan(val):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_file = plots_dir / "example_air_quality.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_file}")
    plt.close()

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Trained on {len(df_train)} hourly samples from {train_start[:4]}-{train_end[:4]}")
    print(f"  - Forecasted {len(df_test)} hourly samples for Q1 2014")
    print("  - Model captures daily, weekly, and yearly seasonal patterns")
    print("  - Multiple meteorological variables: temperature, dewpoint, wind speed, pressure")
    print("  - Temperature: 12 knots, 5 lags | Dewpoint: 10 knots, 3 lags")
    print("  - Wind speed: 8 knots, 2 lags | Pressure: 8 knots, 1 lag")
    print("  - AR(4) model captures residual temporal dependencies")
    baseline_rmse = ablation_results.get('None (seasonal only)', [rmse])[0]
    improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
    print(f"  - Forecast RMSE: {rmse:.2f} μg/m³")
    print(f"  - Improvement over seasonal-only baseline: {improvement:.1f}%")


if __name__ == "__main__":
    main()

