# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Outlier Detector with Synthetic Data

This example demonstrates the outlier detector component using synthetic data
with known outlier days. The example:

1. Generates synthetic hourly time series with:
   - Known seasonal patterns (daily, weekly cycles)
   - Known outlier days with multiplicative corrections (e.g., 0.2x, 2.0x)
   - Some noise

2. Fits model with outlier detector enabled

3. Verifies that:
   - Outlier detector identifies the injected outlier days
   - Detected outlier values match expected multiplicative corrections (in log space)
   - Outlier values are sparse (mostly near 0, with clear spikes for outlier days)
   - Predictions correctly incorporate outlier corrections

4. Includes visualization:
   - Plot of time series with outlier days highlighted
   - Plot of detected outlier values over time
   - Comparison of predictions with/without outlier detector
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import tsgam_estimator
sys.path.insert(0, str(Path(__file__).parent.parent))

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
)


def generate_synthetic_data(
    n_days=60,
    outlier_days=None,
    outlier_multipliers=None,
    noise_scale=0.1,
    random_state=42
):
    """
    Generate synthetic hourly time series data with known outliers.

    Parameters
    ----------
    n_days : int, default=60
        Number of days of hourly data to generate.
    outlier_days : list of int or None, default=None
        List of day indices (0-indexed) that are outliers.
        If None, defaults to [10, 25, 40] (days 11, 26, 41).
    outlier_multipliers : list of float or None, default=None
        Multiplicative corrections for outlier days (in original scale).
        If None, defaults to [0.2, 2.0, 0.5] (20%, 200%, 50% of normal).
    noise_scale : float, default=0.1
        Standard deviation of Gaussian noise (in log space).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    timestamps : DatetimeIndex
        Hourly timestamps.
    y : ndarray
        Target values in log space.
    y_original : ndarray
        Target values in original scale.
    true_outlier_values : ndarray
        True outlier values per day (in log space, mostly 0).
    """
    np.random.seed(random_state)

    n_hours = n_days * 24
    timestamps = pd.date_range('2020-01-01', periods=n_hours, freq='h')

    # Default outlier days and multipliers
    if outlier_days is None:
        outlier_days = [10, 25, 40]
    if outlier_multipliers is None:
        outlier_multipliers = [0.2, 2.0, 0.5]

    # Generate base signal with seasonal patterns
    time_hours = np.arange(n_hours)

    # Daily pattern (24-hour cycle)
    daily_pattern = 2.0 + 1.5 * np.sin(2 * np.pi * time_hours / 24 - np.pi/2)

    # Weekly pattern (168-hour cycle)
    weekly_pattern = 0.5 * np.sin(2 * np.pi * time_hours / 168)

    # Long-term trend
    trend = 0.01 * time_hours / 24  # Small daily increase

    # Combine patterns (in log space, so additive)
    y_log = np.log(10.0) + daily_pattern + weekly_pattern + trend

    # Add outlier corrections (multiplicative in original scale = additive in log space)
    true_outlier_values = np.zeros(n_days)
    for day_idx, multiplier in zip(outlier_days, outlier_multipliers):
        if 0 <= day_idx < n_days:
            # Convert multiplier to log space correction
            log_correction = np.log(multiplier)
            true_outlier_values[day_idx] = log_correction
            # Apply to all hours in that day
            day_start = day_idx * 24
            day_end = day_start + 24
            y_log[day_start:day_end] += log_correction

    # Add noise
    y_log += np.random.normal(0, noise_scale, n_hours)

    # Convert to original scale
    y_original = np.exp(y_log)

    return timestamps, y_log, y_original, true_outlier_values


def fit_model_with_outlier_detector(X, y, reg_weight=1.0):
    """
    Fit TSGAM model with outlier detector enabled.

    Parameters
    ----------
    X : DataFrame
        Input data with timestamps.
    y : ndarray
        Target values in log space.
    reg_weight : float, default=1.0
        L1 regularization weight for outlier detector.

    Returns
    -------
    estimator : TsgamEstimator
        Fitted estimator.
    """
    # Multi-harmonic configuration for seasonal patterns
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[4, 3],
        periods=[PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
        reg_weight=1e-4
    )

    # Outlier detector configuration
    outlier_config = TsgamOutlierConfig(
        reg_weight=reg_weight,
        period_hours=24.0  # Daily outliers
    )

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver='CLARABEL',
        verbose=False
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=None,
        ar_config=None,
        trend_config=None,
        outlier_config=outlier_config,
        solver_config=solver_config,
        random_state=None,
        debug=False
    )

    # Create and fit estimator
    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)

    return estimator


def fit_model_without_outlier_detector(X, y):
    """
    Fit TSGAM model without outlier detector for comparison.

    Parameters
    ----------
    X : DataFrame
        Input data with timestamps.
    y : ndarray
        Target values in log space.

    Returns
    -------
    estimator : TsgamEstimator
        Fitted estimator.
    """
    # Multi-harmonic configuration for seasonal patterns
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[4, 3],
        periods=[PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
        reg_weight=1e-4
    )

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver='CLARABEL',
        verbose=False
    )

    # Create main config (no outlier detector)
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=None,
        ar_config=None,
        trend_config=None,
        outlier_config=None,
        solver_config=solver_config,
        random_state=None,
        debug=False
    )

    # Create and fit estimator
    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)

    return estimator


def plot_results(
    timestamps,
    y_original,
    true_outlier_values,
    estimator_with_outlier,
    estimator_without_outlier,
    outlier_days,
    save_path=None
):
    """
    Create visualization plots for outlier detector results.

    Parameters
    ----------
    timestamps : DatetimeIndex
        Timestamps for the data.
    y_original : ndarray
        Original scale target values.
    true_outlier_values : ndarray
        True outlier values per day.
    estimator_with_outlier : TsgamEstimator
        Fitted estimator with outlier detector.
    estimator_without_outlier : TsgamEstimator
        Fitted estimator without outlier detector.
    outlier_days : list of int
        List of outlier day indices.
    save_path : Path or None, default=None
        Path to save the plot. If None, displays interactively.
    """
    # Get detected outlier values
    detected_outlier = estimator_with_outlier.variables_['outlier'].value
    n_days = len(detected_outlier)

    # Create predictions
    X = pd.DataFrame(index=timestamps)
    pred_with_outlier = estimator_with_outlier.predict(X)
    pred_without_outlier = estimator_without_outlier.predict(X)

    # Convert predictions to original scale
    pred_with_outlier_original = np.exp(pred_with_outlier)
    pred_without_outlier_original = np.exp(pred_without_outlier)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Outlier Detector Example: Synthetic Data', fontsize=16, fontweight='bold')

    # Plot 1: Time series with outlier days highlighted
    ax1 = axes[0]
    ax1.plot(timestamps, y_original, 'b-', alpha=0.6, linewidth=0.5, label='Observed')
    ax1.plot(timestamps, pred_with_outlier_original, 'r-', linewidth=1.5, label='Predicted (with outlier detector)')
    ax1.plot(timestamps, pred_without_outlier_original, 'g--', linewidth=1.5, label='Predicted (without outlier detector)')

    # Highlight outlier days
    for day_idx in outlier_days:
        day_start = timestamps[day_idx * 24]
        day_end = timestamps[min((day_idx + 1) * 24 - 1, len(timestamps) - 1)]
        ax1.axvspan(day_start, day_end, alpha=0.2, color='yellow', label='Outlier days' if day_idx == outlier_days[0] else '')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value (original scale)')
    ax1.set_title('Time Series with Outlier Days Highlighted')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Detected vs True outlier values
    ax2 = axes[1]
    day_indices = np.arange(n_days)
    ax2.plot(day_indices, true_outlier_values, 'b-', linewidth=2, marker='o', markersize=6, label='True outliers')
    ax2.plot(day_indices, detected_outlier, 'r--', linewidth=2, marker='s', markersize=6, label='Detected outliers')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('Day Index')
    ax2.set_ylabel('Outlier Value (log space)')
    ax2.set_title('True vs Detected Outlier Values')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals comparison
    ax3 = axes[2]
    residuals_with = y_original - pred_with_outlier_original
    residuals_without = y_original - pred_without_outlier_original
    ax3.plot(timestamps, residuals_with, 'r-', alpha=0.6, linewidth=0.5, label='Residuals (with outlier detector)')
    ax3.plot(timestamps, residuals_without, 'g-', alpha=0.6, linewidth=0.5, label='Residuals (without outlier detector)')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Highlight outlier days
    for day_idx in outlier_days:
        day_start = timestamps[day_idx * 24]
        day_end = timestamps[min((day_idx + 1) * 24 - 1, len(timestamps) - 1)]
        ax3.axvspan(day_start, day_end, alpha=0.2, color='yellow')

    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residual (original scale)')
    ax3.set_title('Residuals Comparison')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def verify_outlier_detection(
    true_outlier_values,
    detected_outlier,
    outlier_days,
    tolerance=0.3
):
    """
    Verify that outlier detector correctly identifies outliers.

    Parameters
    ----------
    true_outlier_values : ndarray
        True outlier values per day (in log space).
    detected_outlier : ndarray
        Detected outlier values per day (in log space).
    outlier_days : list of int
        List of outlier day indices.
    tolerance : float, default=0.3
        Tolerance for matching outlier values (in log space).

    Returns
    -------
    results : dict
        Dictionary with verification results.
    """
    results = {
        'n_outliers_true': np.sum(np.abs(true_outlier_values) > 1e-6),
        'n_outliers_detected': np.sum(np.abs(detected_outlier) > tolerance),
        'sparsity': np.sum(np.abs(detected_outlier) < tolerance) / len(detected_outlier),
        'matches': [],
        'errors': []
    }

    # Check if detected outliers match true outliers
    for day_idx in outlier_days:
        true_val = true_outlier_values[day_idx]
        detected_val = detected_outlier[day_idx]
        error = abs(true_val - detected_val)
        results['matches'].append({
            'day': day_idx,
            'true': true_val,
            'detected': detected_val,
            'error': error
        })
        results['errors'].append(error)

    results['mean_error'] = np.mean(results['errors'])
    results['max_error'] = np.max(results['errors'])

    return results


def main():
    """Main function to run the example."""
    print("=" * 70)
    print("Outlier Detector Example: Synthetic Data")
    print("=" * 70)
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    timestamps, y_log, y_original, true_outlier_values = generate_synthetic_data(
        n_days=60,
        outlier_days=[10, 25, 40],
        outlier_multipliers=[0.2, 2.0, 0.5],
        noise_scale=0.1,
        random_state=42
    )
    print(f"Generated {len(timestamps)} hours of data ({len(timestamps) // 24} days)")
    print(f"True outlier days: {[10, 25, 40]}")
    print(f"True outlier multipliers: {[0.2, 2.0, 0.5]} (original scale)")
    print(f"True outlier values (log space): {[np.log(0.2), np.log(2.0), np.log(0.5)]}")
    print()

    # Prepare X (empty DataFrame with timestamps)
    X = pd.DataFrame(index=timestamps)

    # Fit model with outlier detector
    print("Fitting model with outlier detector...")
    estimator_with_outlier = fit_model_with_outlier_detector(X, y_log, reg_weight=0.01)
    print(f"Model status: {estimator_with_outlier.problem_.status}")
    print()

    # Fit model without outlier detector for comparison
    print("Fitting model without outlier detector (for comparison)...")
    estimator_without_outlier = fit_model_without_outlier_detector(X, y_log)
    print(f"Model status: {estimator_without_outlier.problem_.status}")
    print()

    # Get detected outlier values
    detected_outlier = estimator_with_outlier.variables_['outlier'].value
    print("Outlier Detection Results:")
    print(f"  Number of days: {len(detected_outlier)}")
    print(f"  Detected outlier values (log space):")
    print(f"  Min: {np.min(detected_outlier):.6f}, Max: {np.max(detected_outlier):.6f}, Mean: {np.mean(detected_outlier):.6f}")
    print(f"  Non-zero values (|outlier| > 0.01):")
    non_zero_count = 0
    for i, val in enumerate(detected_outlier):
        if abs(val) > 0.01:  # Show non-zero values
            multiplier = np.exp(val)
            print(f"    Day {i}: {val:.4f} (multiplier: {multiplier:.3f}x)")
            non_zero_count += 1
    if non_zero_count == 0:
        print("    (none)")
    print()

    # Verify detection
    print("Verifying outlier detection...")
    verification = verify_outlier_detection(
        true_outlier_values,
        detected_outlier,
        outlier_days=[10, 25, 40],
        tolerance=0.3
    )
    print(f"  True outliers: {verification['n_outliers_true']}")
    print(f"  Detected outliers: {verification['n_outliers_detected']}")
    print(f"  Sparsity: {verification['sparsity']:.2%} (fraction of days with |outlier| < 0.3)")
    print(f"  Mean error: {verification['mean_error']:.4f}")
    print(f"  Max error: {verification['max_error']:.4f}")
    print()

    # Print detailed matches
    print("Detailed outlier matches:")
    for match in verification['matches']:
        print(f"  Day {match['day']}: true={match['true']:.4f}, detected={match['detected']:.4f}, error={match['error']:.4f}")
    print()

    # Create visualizations
    print("Creating visualizations...")
    plots_dir = Path(__file__).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / 'example_outlier_detector.png'

    plot_results(
        timestamps,
        y_original,
        true_outlier_values,
        estimator_with_outlier,
        estimator_without_outlier,
        outlier_days=[10, 25, 40],
        save_path=plot_path
    )
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

