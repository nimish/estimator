#!/usr/bin/env python3
"""
Script version of general_tsgam_analysis_3.py using TsgamEstimator.

This script replicates the notebook functionality but uses the TsgamEstimator
class instead of manual CVXPY code. Easier to debug than the marimo notebook.
"""

import sys
from pathlib import Path

# Add src directory to path
_project_root = Path(__file__).parent.parent.parent.parent
_src_dir = _project_root / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import click
from sklearn.metrics import r2_score
from solardatatools import DataHandler

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSplineConfig,
    TsgamSolverConfig,
    TsgamTrendConfig,
    TrendType,
)


def load_file(file_path):
    """Load CSV file(s) into a DataFrame."""
    if isinstance(file_path, (list, tuple)):
        dfs = [pd.read_csv(f, parse_dates=[0], index_col=0) for f in file_path]
        return pd.concat(dfs, axis=1)
    else:
        return pd.read_csv(file_path, parse_dates=[0], index_col=0)


@click.command()
@click.option(
    '--data-file',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to CSV data file. Defaults to 2107_data_combined.csv in script directory.'
)
@click.option(
    '--primary-col',
    type=str,
    default='inv_03_ac_power_inv_149593',
    help='Primary column name (power data). Defaults to first column if not specified.'
)
@click.option(
    '--module-temp-col',
    type=str,
    default='ambient_temperature_o_149575',
    help='Module temperature column name.'
)
@click.option(
    '--irrad-col',
    type=str,
    default='poa_irradiance_o_149574',
    help='Irradiance column name.'
)
@click.option(
    '--linearity-threshold',
    type=float,
    default=0.1,
    help='Linearity threshold for DataHandler. Default: 0.1'
)
@click.option(
    '--fix-dst/--no-fix-dst',
    default=True,
    help='Fix daylight saving time transitions. Default: True'
)
@click.option(
    '--max-val',
    type=float,
    default=2000.0,
    help='Maximum value filter. Default: 2000'
)
@click.option(
    '--data-start',
    type=int,
    default=0,
    help='Start day index. Default: 0'
)
@click.option(
    '--data-end',
    type=int,
    default=None,
    help='End day index (None = use all). Default: None'
)
@click.option(
    '--take-log/--no-take-log',
    default=True,
    help='Take log of target data. Default: True'
)
@click.option(
    '--target-filter',
    type=float,
    default=0.0,
    help='Minimum value filter (0-1). Default: 0.0'
)
@click.option(
    '--solver',
    type=click.Choice(['CLARABEL', 'MOSEK'], case_sensitive=False),
    default='CLARABEL',
    help='Solver to use. Default: CLARABEL'
)
@click.option(
    '--trend-type',
    type=click.Choice(['none', 'linear', 'nonlinear'], case_sensitive=False),
    default='nonlinear',
    help='Trend type to use. Default: nonlinear'
)
@click.option(
    '--run-comparison',
    is_flag=True,
    help='Run comparison across all three trend types (none, linear, nonlinear)'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory for plots. Defaults to Archive/plots'
)
def main(
    data_file,
    primary_col,
    module_temp_col,
    irrad_col,
    linearity_threshold,
    fix_dst,
    max_val,
    data_start,
    data_end,
    take_log,
    target_filter,
    solver,
    trend_type,
    run_comparison,
    output_dir,
):
    """
    TSGAM Analysis Script - Analyze time series data using TsgamEstimator.

    This script replicates the notebook functionality but uses the TsgamEstimator
    class instead of manual CVXPY code. Easier to debug than the marimo notebook.
    """
    # ============================================================================
    # Configuration
    # ============================================================================

    # Data file path (adjust to your data location)
    if data_file is None:
        data_file = Path(__file__).parent / '2107_data_combined.csv'

    if not data_file.exists():
        click.echo(f"Error: Data file not found at {data_file}", err=True)
        click.echo("Please specify a valid data file with --data-file", err=True)
        return

    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # Load and process data
    # ============================================================================

    print("Loading data...")
    df = load_file(data_file)
    df = df.resample('15min').mean()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Auto-detect columns if not specified
    if primary_col is None:
        primary_col = df.columns[0]
    if module_temp_col is None:
        # Try to find temperature column
        temp_candidates = [c for c in df.columns if 'temp' in c.lower() or 'temp' in c.lower()]
        module_temp_col = temp_candidates[0] if temp_candidates else df.columns[1] if len(df.columns) > 1 else None
    if irrad_col is None:
        # Try to find irradiance column
        irrad_candidates = [c for c in df.columns if 'irrad' in c.lower() or 'poa' in c.lower()]
        irrad_col = irrad_candidates[0] if irrad_candidates else df.columns[2] if len(df.columns) > 2 else None

    print("Using columns:")
    print(f"  Primary: {primary_col}")
    print(f"  Module temp: {module_temp_col}")
    print(f"  Irradiance: {irrad_col}")

    # Run DataHandler pipeline
    print("\nRunning DataHandler pipeline...")
    dh = DataHandler(df)
    if fix_dst:
        dh.fix_dst()

    extra_cols = [c for c in [module_temp_col, irrad_col] if c is not None and c != primary_col]
    if len(extra_cols) == 0:
        dh.run_pipeline(power_col=primary_col, max_val=max_val, linearity_threshold=linearity_threshold)
    else:
        dh.run_pipeline(
            power_col=primary_col,
            max_val=max_val,
            extra_cols=extra_cols,
            linearity_threshold=linearity_threshold
        )

    print(f"DataHandler matrix shape: {dh.raw_data_matrix.shape}")

    # Clean up extra matrices
    if module_temp_col:
        temp_mat = dh.extra_matrices[module_temp_col]
        temp_mat[temp_mat > 140] = np.nan
    if irrad_col:
        irrad_mat = dh.extra_matrices[irrad_col]
        irrad_mat[irrad_mat < 0] = 0

    # ============================================================================
    # Prepare data for modeling
    # ============================================================================

    print("\nPreparing data for modeling...")

    # Determine data range
    if data_end is None:
        data_end = dh.raw_data_matrix.shape[1] - 1
    _data_select = np.s_[data_start:data_end + 1]

    # Prepare target (y)
    y = np.copy(dh.raw_data_matrix)
    y_max = np.nanmax(y)
    y[:, ~dh.daily_flags.no_errors] = np.nan
    y[~dh.boolean_masks.daytime] = np.nan
    y[y < 0.01 * np.nanmax(y)] = np.nan
    y = y[:, _data_select].ravel(order='F')
    y /= y_max
    y[y < target_filter] = np.nan
    if take_log:
        y = np.log(y)

    # Prepare exogenous variables
    # x1 is module temperature
    x1 = np.copy(dh.extra_matrices[module_temp_col][:, _data_select].ravel(order='F'))
    x1_avail = ~np.isnan(x1)
    x1[~x1_avail] = 0
    x1_max = np.max(x1)
    x1 /= x1_max

    # x2 is POA irradiance
    x2 = np.copy(dh.extra_matrices[irrad_col][:, _data_select].ravel(order='F'))
    x2[x2 < 0] = 0
    x2_avail = np.logical_and(~np.isnan(x2), x2 > 0.02 * np.nanquantile(x2, 0.98))
    x2[~x2_avail] = 0
    x2_max = np.max(x2)
    x2 /= x2_max

    # Filter out NaN in y
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    x1 = x1[valid_mask]
    x2 = x2[valid_mask]

    print(f"Valid samples: {len(y)} out of {len(valid_mask)}")

    # ============================================================================
    # Create timestamps
    # ============================================================================

    print("\nCreating timestamps...")

    # Get start time from original dataframe
    _start_time = df.index[0]
    _m, _n = dh.raw_data_matrix.shape
    _n_selected = data_end - data_start + 1

    # Account for data_start offset
    _start_time = _start_time + pd.Timedelta(days=data_start)

    # Create regularly spaced timestamps for filtered data
    _valid_len = len(y)
    timestamps = pd.date_range(
        start=_start_time,
        periods=_valid_len,
        freq='15min'
    )

    # Verify frequency can be inferred
    inferred_freq = pd.infer_freq(timestamps)
    print(f"Inferred frequency: {inferred_freq}")
    if inferred_freq is None:
        print("Warning: Could not infer frequency, forcing '15min'")
        timestamps = timestamps.asfreq('15min')
        inferred_freq = pd.infer_freq(timestamps)
        print(f"Frequency after forcing: {inferred_freq}")

    # Create DataFrame with exogenous variables
    X = pd.DataFrame({
        'temp': x1,
        'irrad': x2
    }, index=timestamps)

    print(f"X shape: {X.shape}")
    print(f"X index range: {X.index[0]} to {X.index[-1]}")
    print(f"X frequency: {pd.infer_freq(X.index)}")

    print(f"\nPlots will be saved to: {output_dir}")

    # ============================================================================
    # Configure and fit model(s)
    # ============================================================================

    if run_comparison:
        print("\n" + "="*70)
        print("Running trend comparison: none, linear, and nonlinear")
        print("="*70)
        trend_types_to_test = ['none', 'linear', 'nonlinear']
    else:
        trend_types_to_test = [trend_type]

    results = {}

    for trend_type in trend_types_to_test:
        print(f"\n{'='*70}")
        print(f"Fitting model with trend_type = '{trend_type}'")
        print(f"{'='*70}")

        _fit_and_visualize_trend_model(
            trend_type=trend_type,
            solver=solver,
            dh=dh,
            X=X,
            y=y,
            y_max=y_max,
            take_log=take_log,
            x1=x1,
            x1_max=x1_max,
            x2=x2,
            x2_max=x2_max,
            output_dir=output_dir,
            results=results,
            timestamps=X.index  # Pass timestamps for plotting
        )

    # Print comparison summary
    if run_comparison and len(results) > 1:
        print("\n" + "="*70)
        print("TREND COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Trend Type':<15} {'Problem Status':<20} {'Optimal Value':<20} {'Trend Slope':<15}")
        print("-"*70)
        for trend_type, result in results.items():
            status = result.get('status', 'N/A')
            opt_val = result.get('optimal_value', 'N/A')
            if isinstance(opt_val, float):
                opt_val = f"{opt_val:.6e}"
            slope = result.get('trend_slope', 'N/A')
            if isinstance(slope, float):
                slope = f"{slope:.8f}"
            print(f"{trend_type:<15} {status:<20} {opt_val:<20} {slope:<15}")
        print("="*70)

    print("\nDone!")


def _fit_and_visualize_trend_model(
    trend_type,
    solver,
    dh,
    X,
    y,
    y_max,
    take_log,
    x1,
    x1_max,
    x2,
    x2_max,
    output_dir,
    results,
    timestamps=None
):
    """Fit a single model with specified trend type and save results."""

    print(f"\nConfiguring model with trend_type='{trend_type}'...")

    # Multi-harmonic Fourier configuration
    _period_daily_hours = 24.0  # daily period in hours
    _period_yearly_hours = 365.2425 * 24.0  # yearly period in hours

    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[6, 10],
        periods=[_period_yearly_hours, _period_daily_hours],
        reg_weight=1e-2
    )

    # Exogenous variables: temperature and irradiance splines
    exog_config = [
        TsgamSplineConfig(
            n_knots=10,
            lags=[0],
            reg_weight=1e-4
        ),
        TsgamSplineConfig(
            n_knots=10,
            lags=[0],
            reg_weight=1e-4
        )
    ]

    # Trend configuration
    if trend_type == 'none':
        trend_config = None
    else:
        # Map string to TrendType enum
        trend_type_enum = TrendType.LINEAR if trend_type == 'linear' else TrendType.NONLINEAR
        trend_config = TsgamTrendConfig(
            trend_type=trend_type_enum,
            grouping=24.0,  # Daily trend (period in hours)
            reg_weight=10.0
        )

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver=solver,
        verbose=True
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=exog_config,
        trend_config=trend_config,
        solver_config=solver_config
    )

    print("Creating estimator...")
    estimator = TsgamEstimator(config=config)

    print("Fitting model...")
    try:
        estimator.fit(X, y)
        status = estimator.problem_.status
        opt_val = estimator.problem_.value if hasattr(estimator.problem_, 'value') else None
        print(f"Fit complete! Problem status: {status}")
        if opt_val is not None:
            print(f"Optimal value: {opt_val:.6e}")
    except Exception as e:
        print(f"Error during fit: {e}")
        import traceback
        traceback.print_exc()
        results[trend_type] = {'status': 'ERROR', 'error': str(e)}
        return

    # Get predictions
    print("Getting predictions...")
    model = estimator.predict(X)

    # Store results
    result = {
        'status': status,
        'optimal_value': opt_val,
        'estimator': estimator,
        'model': model
    }

    # Get trend slope if available
    if hasattr(estimator, 'variables_') and 'trend_slope' in estimator.variables_:
        slope = estimator.variables_['trend_slope'].value
        result['trend_slope'] = slope
    elif hasattr(estimator, 'variables_') and 'trend' in estimator.variables_:
        trend = estimator.variables_['trend'].value
        if trend is not None and len(trend) > 1:
            # Calculate average slope from differences
            result['trend_slope'] = np.mean(np.diff(trend))

    results[trend_type] = result

    # Visualize this model's results
    _visualize_model_results(
        estimator=estimator,
        model=model,
        y=y,
        y_max=y_max,
        take_log=take_log,
        x1=x1,
        x1_max=x1_max,
        x2=x2,
        x2_max=x2_max,
        trend_type=trend_type,
        output_dir=output_dir,
        X=X,
        timestamps=timestamps if timestamps is not None else (X.index if hasattr(X, 'index') else None)
    )


def _visualize_model_results(
    estimator,
    model,
    y,
    y_max,
    take_log,
    x1,
    x1_max,
    x2,
    x2_max,
    trend_type,
    output_dir,
    X=None,
    timestamps=None
):
    """Generate visualizations for a fitted model."""

    print(f"\nGenerating visualizations for trend_type='{trend_type}'...")

    # Model fit plot
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(model, y, marker='.', linewidth=0.5, markersize=1, alpha=0.5)
    if take_log:
        ax[1].plot(y_max * np.exp(model), y_max * np.exp(y), marker='.', linewidth=0.5, markersize=1, alpha=0.5)
    else:
        ax[1].plot(y_max * model, y_max * y, marker='.', linewidth=0.5, markersize=1, alpha=0.5)
    for _ix in range(2):
        _xlim = ax[_ix].get_xlim()
        _ylim = ax[_ix].get_ylim()
        ax[_ix].plot([-1e4, 1e4], [-1e4, 1e4], color='red', ls='--', linewidth=1)
        ax[_ix].set_xlim(_xlim)
        ax[_ix].set_ylim(_ylim)
        ax[_ix].set_ylabel('actual')
        ax[_ix].set_xlabel('predicted')
    ax[0].set_title('transformed data')
    ax[1].set_title('original data')
    plt.tight_layout()
    plot_path = output_dir / f'model_fit_{trend_type}.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    # Residuals
    if take_log:
        residuals = y_max * (np.exp(y) - np.exp(model))
    else:
        residuals = y_max * (y - model)

    # Residual distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    _r = residuals
    _s = ~np.isnan(_r)
    ax.hist(_r, bins=200, density=True)
    _xs = np.linspace(np.min(_r[_s]), np.max(_r[_s]), 1001)
    lap_loc, lap_scale = stats.laplace.fit(_r[_s])
    nor_loc, nor_scale = stats.norm.fit(_r[_s])
    ax.plot(_xs, stats.laplace.pdf(_xs, lap_loc, lap_scale), label='laplace fit', linewidth=1, color='dodgerblue')
    ax.plot(_xs, stats.norm.pdf(_xs, nor_loc, nor_scale), label='normal fit', linewidth=1, color='lime')
    ax.axvline(np.nanquantile(_r, .025), color='orange', ls='--', label='95% confidence bounds', linewidth=0.5)
    ax.axvline(np.nanquantile(_r, .975), color='orange', ls='--', linewidth=0.5)
    ax.set_xlabel('residual')
    ax.legend()
    ax.set_title('distribution of residuals')
    plt.tight_layout()
    plot_path = output_dir / f'residual_distribution_{trend_type}.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    # Trend plot
    if hasattr(estimator, 'variables_') and 'trend' in estimator.variables_:
        trend = estimator.variables_['trend'].value
        if trend is not None:
            # Print trend statistics for debugging
            print("\nTrend statistics:")
            print(f"  Length: {len(trend)}")
            print(f"  First value: {trend[0]:.6f}")
            print(f"  Last value: {trend[-1]:.6f}")
            print(f"  Range: [{np.min(trend):.6f}, {np.max(trend):.6f}]")

            # Calculate differences to check linearity
            trend_diff = np.diff(trend)
            # Get trend type from estimator config
            est_trend_type = estimator.config.trend_config.trend_type.value if estimator.config.trend_config else 'none'
            if est_trend_type == 'linear':
                print("  Differences (should be constant for linear):")
            elif est_trend_type == 'nonlinear':
                print("  Differences (nonlinear - should be <= 0 and variable):")
            else:
                print("  Differences:")
            print(f"    Mean: {np.mean(trend_diff):.6f}")
            print(f"    Std: {np.std(trend_diff):.6f}")
            print(f"    Min: {np.min(trend_diff):.6f}")
            print(f"    Max: {np.max(trend_diff):.6f}")

            if 'trend_slope' in estimator.variables_:
                slope = estimator.variables_['trend_slope'].value
                if slope is not None:
                    print(f"  Fitted slope: {slope:.6f}")

            # Plot both log space and original space
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10))

            # Plot 1: Trend in log space - MAIN PLOT FOR INSPECTION
            with sns.axes_style('whitegrid'):
                period_indices = np.arange(len(trend))
                ax1.plot(period_indices, trend, marker='o', markersize=3, linewidth=1.5, label='Trend values')
                ax1.set_xlabel('Period index', fontsize=12)
                ax1.set_ylabel('Trend (log space)', fontsize=12)

                # Set appropriate title based on trend type
                est_trend_type = estimator.config.trend_config.trend_type.value if estimator.config.trend_config else 'none'
                if est_trend_type == 'linear':
                    ax1.set_title('Trend in log space (linear - should be perfectly straight)', fontsize=14, fontweight='bold')
                elif est_trend_type == 'nonlinear':
                    ax1.set_title('Trend in log space (nonlinear - monotonic decreasing)', fontsize=14, fontweight='bold')
                else:
                    ax1.set_title('Trend in log space', fontsize=14, fontweight='bold')

                ax1.grid(True, alpha=0.3)

                # Add linear fit line for comparison (only for linear trend)
                est_trend_type = estimator.config.trend_config.trend_type.value if estimator.config.trend_config else 'none'
                if len(trend) > 1 and est_trend_type == 'linear':
                    if 'trend_slope' in estimator.variables_ and estimator.variables_['trend_slope'].value is not None:
                        slope = estimator.variables_['trend_slope'].value
                        y_fit = slope * period_indices  # trend[0] == 0 by constraint
                        ax1.plot(period_indices, y_fit, 'r--', linewidth=2,
                                label=f'Expected linear (slope={slope:.8f})', alpha=0.7)

                        # Calculate and display R² to show how linear it is
                        r2 = r2_score(trend, y_fit)
                        ax1.text(0.02, 0.98, f'R² = {r2:.10f}\nSlope = {slope:.8f}',
                                transform=ax1.transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                elif est_trend_type == 'nonlinear':
                    # For nonlinear, show that differences should be <= 0
                    ax1.text(0.02, 0.98, f'Monotonic decreasing\nMean diff: {np.mean(np.diff(trend)):.8f}',
                            transform=ax1.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax1.legend(loc='best')

                # Add zoomed inset to see detail
                if len(trend) > 100:
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    axins = inset_axes(ax1, width="40%", height="30%", loc='upper right', borderpad=2)
                    # Show last 50 points
                    zoom_start = max(0, len(trend) - 50)
                    axins.plot(period_indices[zoom_start:], trend[zoom_start:], marker='o', markersize=2, linewidth=1)
                    if est_trend_type == 'linear' and 'trend_slope' in estimator.variables_ and estimator.variables_['trend_slope'].value is not None:
                        slope = estimator.variables_['trend_slope'].value
                        axins.plot(period_indices[zoom_start:], slope * period_indices[zoom_start:],
                                  'r--', linewidth=1.5, alpha=0.7)
                    axins.grid(True, alpha=0.3)
                    axins.set_title('Last 50 periods (zoom)', fontsize=8)

            # Plot 2: Trend in original space (exp(trend))
            with sns.axes_style('whitegrid'):
                years = np.arange(len(trend)) / 365
                ax2.plot(years, np.exp(trend), linewidth=2)
                ax2.set_xlabel('Years', fontsize=12)
                ax2.set_ylabel('exp(trend) - Degradation factor', fontsize=12)
                ax2.set_title('Degradation term over time (original space)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([np.min(np.exp(trend)) * 0.99, np.max(np.exp(trend)) * 1.01])

            plt.tight_layout()
            plot_path = output_dir / f'trend_{trend_type}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {plot_path}")

            # Also save a separate large plot of just the log space trend for detailed inspection
            fig2, ax = plt.subplots(figsize=(14, 6))
            with sns.axes_style('whitegrid'):
                ax.plot(period_indices, trend, marker='o', markersize=2, linewidth=1, label='Trend values')
                est_trend_type = estimator.config.trend_config.trend_type.value if estimator.config.trend_config else 'none'
                if est_trend_type == 'linear' and 'trend_slope' in estimator.variables_ and estimator.variables_['trend_slope'].value is not None:
                    slope = estimator.variables_['trend_slope'].value
                    y_fit = slope * period_indices
                    ax.plot(period_indices, y_fit, 'r--', linewidth=2,
                           label=f'Expected linear (slope={slope:.8f})', alpha=0.7)
                    r2 = r2_score(trend, y_fit)
                    ax.text(0.02, 0.98, f'R² = {r2:.10f}\nSlope = {slope:.8f}\nStd of differences = {np.std(np.diff(trend)):.10f}',
                           transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                elif est_trend_type == 'nonlinear':
                    ax.text(0.02, 0.98, f'Nonlinear (monotonic decreasing)\nMean diff: {np.mean(np.diff(trend)):.8f}\nStd of differences = {np.std(np.diff(trend)):.10f}',
                           transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_xlabel('Period index', fontsize=12)
                ax.set_ylabel('Trend (log space)', fontsize=12)

                # Set appropriate title based on trend type
                est_trend_type = estimator.config.trend_config.trend_type.value if estimator.config.trend_config else 'none'
                if est_trend_type == 'linear':
                    ax.set_title('Trend in log space - Detailed View (linear - should be perfectly straight)', fontsize=14, fontweight='bold')
                elif est_trend_type == 'nonlinear':
                    ax.set_title('Trend in log space - Detailed View (nonlinear - monotonic decreasing)', fontsize=14, fontweight='bold')
                else:
                    ax.set_title('Trend in log space - Detailed View', fontsize=14, fontweight='bold')

                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
            plt.tight_layout()
            plot_path = output_dir / f'trend_logspace_{trend_type}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {plot_path} (detailed log space view)")

            # Plot trend term for every timestep (stairstep pattern)
            if hasattr(estimator, 'trend_T_matrix_') and estimator.trend_T_matrix_ is not None:
                T = estimator.trend_T_matrix_
                # Expand trend to all timesteps: trend_term = T @ trend
                trend_term_timesteps = T @ trend

                # Debug: Print information about T matrix and periods
                print("\nT matrix debug info:")
                print(f"  T shape: {T.shape}")
                print(f"  Trend length (periods): {len(trend)}")
                print(f"  Trend term timesteps length: {len(trend_term_timesteps)}")
                print(f"  Expected samples per period: {len(trend_term_timesteps) / len(trend):.1f}")

                # Count samples per period
                period_changes_debug = np.where(np.diff(trend_term_timesteps) != 0)[0]
                if len(period_changes_debug) > 0:
                    samples_per_period_list = []
                    prev_idx = 0
                    for change_idx in period_changes_debug[:10]:  # First 10 periods
                        samples_per_period_list.append(change_idx - prev_idx + 1)
                        prev_idx = change_idx + 1
                    print(f"  Samples per period (first 10): {samples_per_period_list}")
                    print(f"  Mean samples per period: {np.mean(samples_per_period_list):.1f}")

                # Check time_indices to understand period assignment
                if hasattr(estimator, 'time_indices_'):
                    time_indices = estimator.time_indices_
                    period_hours = estimator.trend_period_hours_ if hasattr(estimator, 'trend_period_hours_') else 24.0
                    period_indices = (time_indices / period_hours).astype(int)
                    print(f"  Period hours: {period_hours}")
                    print(f"  Time indices range: [{time_indices[0]:.2f}, {time_indices[-1]:.2f}] hours")
                    print(f"  Period indices range: [{period_indices[0]}, {period_indices[-1]}]")
                    print(f"  Unique periods: {len(np.unique(period_indices))}")

                    # Check time gaps
                    time_diffs = np.diff(time_indices)
                    print(f"  Time differences - mean: {np.mean(time_diffs):.2f} hours, median: {np.median(time_diffs):.2f} hours")
                    print(f"  Time differences - min: {np.min(time_diffs):.2f} hours, max: {np.max(time_diffs):.2f} hours")

                # Get timestamps for all timesteps
                if timestamps is not None:
                    timestamps_trend = timestamps
                elif X is not None and hasattr(X, 'index'):
                    timestamps_trend = X.index
                else:
                    timestamps_trend = np.arange(len(trend_term_timesteps))

                fig3, ax = plt.subplots(figsize=(14, 6))
                with sns.axes_style('whitegrid'):
                    # Plot stairstep pattern
                    ax.plot(timestamps_trend, trend_term_timesteps, linewidth=1, label='Trend term (stairstep)', drawstyle='steps-post')
                    ax.set_xlabel('Time', fontsize=12)
                    ax.set_ylabel('Trend term (log space)', fontsize=12)
                    est_trend_type = estimator.config.trend_config.trend_type.value if estimator.config.trend_config else 'none'
                    ax.set_title(f'Trend term for every timestep (stairstep pattern) - {est_trend_type}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')

                    # Find where periods change (where trend value changes)
                    period_changes = None
                    if len(trend) > 0:
                        period_changes = np.where(np.diff(trend_term_timesteps) != 0)[0]
                        if len(period_changes) > 0:
                            # Show first 20 period boundaries on main plot
                            for idx in period_changes[:20]:
                                if idx < len(timestamps_trend):
                                    if isinstance(timestamps_trend, pd.DatetimeIndex):
                                        ax.axvline(timestamps_trend[idx], color='red', linestyle='--', alpha=0.2, linewidth=0.5)
                                    else:
                                        ax.axvline(idx, color='red', linestyle='--', alpha=0.2, linewidth=0.5)

                    # Add zoomed inset to show stairstep pattern clearly
                    if len(trend_term_timesteps) > 200 and len(trend) > 0:
                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                        # Create inset in upper right corner to avoid overlapping main plot
                        axins = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=3)

                        # Estimate samples per period from period changes or data structure
                        if period_changes is not None and len(period_changes) > 0:
                            # Use first few period changes to estimate
                            if len(period_changes) >= 2:
                                samples_per_period = period_changes[1] - period_changes[0]
                            else:
                                samples_per_period = period_changes[0] + 1 if period_changes[0] > 0 else len(trend_term_timesteps) // len(trend)
                        else:
                            samples_per_period = len(trend_term_timesteps) // len(trend) if len(trend) > 0 else 96

                        # Show exactly 2 periods worth of data for clear stairstep visualization
                        # Use the actual samples_per_period from the data
                        zoom_window = 2 * samples_per_period
                        # Start at a point where we have clear period boundaries (after a few periods)
                        # Make sure we start at a period boundary
                        if period_changes is not None and len(period_changes) > 2:
                            # Start at the 3rd period boundary to ensure we have full periods
                            zoom_start = period_changes[2] + 1  # Start right after 3rd period boundary
                        else:
                            zoom_start = max(samples_per_period * 2, 100)
                        zoom_end = min(zoom_start + zoom_window, len(trend_term_timesteps))

                        print(f"  Zoom window: {zoom_window} samples ({zoom_window/samples_per_period:.1f} periods)")
                        print(f"  Zoom range: [{zoom_start}, {zoom_end}]")
                        print(f"  Samples in zoom: {zoom_end - zoom_start}")

                        # Plot zoomed view with stairstep - make it very clear
                        # Show all points to verify we have 96 per period
                        zoom_trend = trend_term_timesteps[zoom_start:zoom_end]

                        # Count actual samples per period in the zoom first
                        zoom_period_changes = period_changes[(period_changes >= zoom_start) & (period_changes < zoom_end)] if period_changes is not None and len(period_changes) > 0 else []
                        zoom_samples_per_period = []
                        if len(zoom_period_changes) > 0:
                            prev_idx = zoom_start
                            for change_idx in zoom_period_changes:
                                zoom_samples_per_period.append(change_idx - prev_idx + 1)
                                prev_idx = change_idx + 1
                            # Last period
                            if zoom_end > prev_idx:
                                zoom_samples_per_period.append(zoom_end - prev_idx)
                        else:
                            # Estimate from samples_per_period
                            num_periods_in_zoom = int((zoom_end - zoom_start) / samples_per_period)
                            zoom_samples_per_period = [samples_per_period] * num_periods_in_zoom

                        print(f"  Actual samples per period in zoom: {zoom_samples_per_period}")

                        # Plot with steps and overlay scatter to show ALL individual points clearly
                        if isinstance(timestamps_trend, pd.DatetimeIndex):
                            zoom_times = timestamps_trend[zoom_start:zoom_end]
                            # First plot the stairstep line (thinner, lighter)
                            axins.plot(zoom_times, zoom_trend,
                                     linewidth=1.5, drawstyle='steps-post', color='lightblue',
                                     label='Trend term (line)', zorder=1, alpha=0.5)
                            # Then overlay scatter to show ALL individual points - this is key!
                            axins.scatter(zoom_times, zoom_trend,
                                        s=12, color='blue', alpha=0.8, zorder=3,
                                        edgecolors='darkblue', linewidths=0.5, marker='o',
                                        label='All samples')
                            # Format x-axis for readability
                            axins.tick_params(axis='x', labelsize=7, rotation=45)
                        else:
                            zoom_indices = np.arange(zoom_start, zoom_end)
                            # First plot the stairstep line (thinner, lighter)
                            axins.plot(zoom_indices, zoom_trend,
                                     linewidth=1.5, drawstyle='steps-post', color='lightblue',
                                     label='Trend term (line)', zorder=1, alpha=0.5)
                            # Then overlay scatter to show ALL individual points - this is key!
                            axins.scatter(zoom_indices, zoom_trend,
                                        s=12, color='blue', alpha=0.8, zorder=3,
                                        edgecolors='darkblue', linewidths=0.5, marker='o',
                                        label='All samples')
                            axins.set_xlabel('Timestep index', fontsize=9)

                        # Add text showing samples per period
                        if len(zoom_samples_per_period) > 0:
                            # Filter out zero values
                            valid_samples = [s for s in zoom_samples_per_period if s > 0]
                            if len(valid_samples) > 0:
                                avg_samples = np.mean(valid_samples)
                                min_samples = np.min(valid_samples)
                                max_samples = np.max(valid_samples)
                                axins.text(0.02, 0.98,
                                         f'Samples/period: {min_samples}-{max_samples} (avg: {avg_samples:.0f})\nTotal points: {len(zoom_trend)}',
                                         transform=axins.transAxes, fontsize=8,
                                         verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

                        axins.set_ylabel('Trend term (log space)', fontsize=9, fontweight='bold')
                        axins.set_title('ZOOMED: Stairstep Pattern\n(2 periods shown)', fontsize=11, fontweight='bold', pad=10)
                        axins.grid(True, alpha=0.5, linestyle='--', linewidth=0.8, zorder=1)

                        # Mark period boundaries in zoom with clear vertical lines
                        if period_changes is not None and len(period_changes) > 0:
                            zoom_period_changes = period_changes[(period_changes >= zoom_start) & (period_changes < zoom_end)]
                            for i, idx in enumerate(zoom_period_changes):
                                if isinstance(timestamps_trend, pd.DatetimeIndex):
                                    axins.axvline(timestamps_trend[idx], color='red', linestyle='--',
                                                 alpha=0.8, linewidth=2, zorder=2,
                                                 label='Period boundary' if i == 0 else '')
                                else:
                                    axins.axvline(idx, color='red', linestyle='--',
                                                 alpha=0.8, linewidth=2, zorder=2,
                                                 label='Period boundary' if i == 0 else '')

                        # Add legend for zoom
                        if len(zoom_period_changes) > 0:
                            axins.legend(loc='lower right', fontsize=8, framealpha=0.9, edgecolor='black', frameon=True)

                        # Make the inset stand out with a border
                        for spine in axins.spines.values():
                            spine.set_edgecolor('black')
                            spine.set_linewidth(2)

                plt.tight_layout()
                plot_path = output_dir / f'trend_timesteps_{trend_type}.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {plot_path} (trend term for every timestep - stairstep)")

    # Response functions
    if hasattr(estimator, 'variables_') and 'exog_coef_0' in estimator.variables_:
        exog_coef = estimator.variables_['exog_coef_0'].value
        if exog_coef is not None:
            knots = estimator.exog_knots_[0] if estimator.exog_knots_ and len(estimator.exog_knots_) > 0 else None
            if knots is not None:
                H1 = estimator._make_H(x1, knots, include_offset=False)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x1 * x1_max, np.exp(H1 @ exog_coef[:, 0]), ls='none', marker='.', markersize=1)
                ax.set_title('Inferred temperature response')
                ax.set_xlabel('module temp [deg C]')
                ax.set_ylabel('correction factor [1]')
                plt.tight_layout()
                plot_path = output_dir / f'temp_response_{trend_type}.png'
                plt.savefig(plot_path, dpi=150)
                print(f"Saved: {plot_path}")

    if hasattr(estimator, 'variables_') and 'exog_coef_1' in estimator.variables_:
        exog_coef = estimator.variables_['exog_coef_1'].value
        if exog_coef is not None:
            knots = estimator.exog_knots_[1] if estimator.exog_knots_ and len(estimator.exog_knots_) > 1 else None
            if knots is not None:
                H2 = estimator._make_H(x2, knots, include_offset=False)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x2 * x2_max, np.exp(H2 @ exog_coef[:, 0]), ls='none', marker='.', markersize=1)
                ax.set_title('Inferred irradiance response')
                ax.set_xlabel('POA irradiance [W/m^2]')
                ax.set_ylabel('correction factor [1]')
                plt.tight_layout()
                plot_path = output_dir / f'irrad_response_{trend_type}.png'
                plt.savefig(plot_path, dpi=150)
                print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
