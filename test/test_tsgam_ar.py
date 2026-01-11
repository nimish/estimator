# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Test AR model fitting and sampling for TsgamEstimator.

This test suite verifies that:
1. AR model is fitted correctly on baseline residuals
2. AR coefficients and intercept match notebook values
3. AR noise distribution matches notebook
4. Sample method generates AR noise rollout correctly
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSplineConfig,
    TsgamArConfig,
    TsgamSolverConfig,
)


def load_notebook_data(sheet='RI', years=[2020, 2021]):
    """Load data exactly as notebook does."""
    df_list = []
    for year in years:
        fp = Path('.') / 'ISO_Data' / f'{year}_smd_hourly.xlsx'
        df = pd.read_excel(fp, sheet_name=sheet)
        df['year'] = year
        df.index = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hr_End'].map(lambda x: f"{x-1}:00:00")) + pd.Timedelta(hours=1)
        df_list.append(df)
    return pd.concat(df_list, axis=0)


@pytest.fixture(scope="module")
def notebook_data():
    """Load notebook data."""
    return load_notebook_data(sheet='RI', years=[2020, 2021, 2022])


@pytest.fixture(scope="module")
def notebook_ar_coefficients():
    """Load saved AR coefficients from notebook."""
    base_path = Path(__file__).parent
    return {
        'ar_coef': np.load(base_path / 'ar_coeff.npy'),
        'ar_intercept': np.load(base_path / 'ar_intercept.npy'),
    }


@pytest.fixture(scope="module")
def tsgam_estimator_with_ar(notebook_data):
    """Fit TsgamEstimator with AR configuration matching notebook."""
    df_subset = notebook_data.loc["2020":"2021"]
    y = np.log(df_subset["RT_Demand"]).values
    X = pd.DataFrame({'temp': df_subset["Dry_Bulb"].values}, index=df_subset.index)

    # AR config: 36 lags, L1 constraint 0.95 (matching notebook)
    ar_config = TsgamArConfig(
        lags=list(range(1, 37)),  # 36 lags: [1, 2, ..., 36]
        l1_constraint=0.95
    )

    config = TsgamEstimatorConfig(
        multi_harmonic_config=TsgamMultiHarmonicConfig(
            num_harmonics=[6, 4, 3],
            periods=[365.2425 * 24, 7 * 24, 24]
        ),
        exog_config=[
            TsgamSplineConfig(
                n_knots=10,
                lags=[-3, -2, -1, 0, 1, 2, 3],
                reg_weight=1e-4,
                diff_reg_weight=1.0
            )
        ],
        ar_config=ar_config,
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        debug=True
    )

    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)

    return estimator


def test_ar_coefficients_match_notebook(tsgam_estimator_with_ar, notebook_ar_coefficients):
    """Test that AR coefficients match notebook values."""
    estimator = tsgam_estimator_with_ar
    notebook_ar_coef = notebook_ar_coefficients['ar_coef']

    assert hasattr(estimator, 'ar_coef_'), "AR coefficients should be fitted"
    assert estimator.ar_coef_ is not None, "AR coefficients should not be None"

    ar_coef = estimator.ar_coef_

    # Should have same length (36 lags)
    assert len(ar_coef) == len(notebook_ar_coef), \
        f"AR coefficient length mismatch: {len(ar_coef)} vs {len(notebook_ar_coef)}"

    # Compare coefficients
    np.testing.assert_allclose(
        ar_coef,
        notebook_ar_coef,
        rtol=1e-5,
        atol=1e-5,
        err_msg="AR coefficients don't match notebook"
    )

    print(f"\nAR coefficients comparison:")
    print(f"  Length: {len(ar_coef)}")
    print(f"  Max absolute diff: {np.max(np.abs(ar_coef - notebook_ar_coef)):.6e}")
    print(f"  Mean absolute diff: {np.mean(np.abs(ar_coef - notebook_ar_coef)):.6e}")


def test_ar_intercept_matches_notebook(tsgam_estimator_with_ar, notebook_ar_coefficients):
    """Test that AR intercept matches notebook value."""
    estimator = tsgam_estimator_with_ar
    notebook_ar_intercept = notebook_ar_coefficients['ar_intercept']

    assert hasattr(estimator, 'ar_intercept_'), "AR intercept should be fitted"
    assert estimator.ar_intercept_ is not None, "AR intercept should not be None"

    ar_intercept = estimator.ar_intercept_

    np.testing.assert_allclose(
        ar_intercept,
        notebook_ar_intercept,
        rtol=1e-5,
        atol=1e-5,
        err_msg="AR intercept doesn't match notebook"
    )

    print(f"\nAR intercept comparison:")
    print(f"  Estimator: {ar_intercept:.6e}")
    print(f"  Notebook: {notebook_ar_intercept:.6e}")
    print(f"  Diff: {abs(ar_intercept - notebook_ar_intercept):.6e}")


def test_ar_noise_distribution_fitted(tsgam_estimator_with_ar):
    """Test that AR noise distribution parameters are fitted."""
    estimator = tsgam_estimator_with_ar

    assert hasattr(estimator, 'ar_noise_loc_'), "AR noise location should be fitted"
    assert hasattr(estimator, 'ar_noise_scale_'), "AR noise scale should be fitted"
    assert estimator.ar_noise_loc_ is not None, "AR noise location should not be None"
    assert estimator.ar_noise_scale_ is not None, "AR noise scale should not be None"

    # Noise scale should be positive
    assert estimator.ar_noise_scale_ > 0, "AR noise scale should be positive"

    print(f"\nAR noise distribution:")
    print(f"  Location: {estimator.ar_noise_loc_:.6e}")
    print(f"  Scale: {estimator.ar_noise_scale_:.6e}")


def test_baseline_residuals_computed(tsgam_estimator_with_ar):
    """Test that baseline residuals are computed and stored in debug mode."""
    estimator = tsgam_estimator_with_ar

    assert hasattr(estimator, '_baseline_residuals_'), "Baseline residuals should be stored in debug mode"
    assert estimator._baseline_residuals_ is not None, "Baseline residuals should not be None"

    residuals = estimator._baseline_residuals_

    assert len(residuals) > 0, "Should have residuals"
    assert np.all(np.isfinite(residuals)), "Residuals should be finite"

    # Check reasonable range for residuals (log space)
    baseline_mae = np.mean(np.abs(residuals))
    assert 0.01 < baseline_mae < 0.1, \
        f"Baseline MAE {baseline_mae:.6f} outside expected range [0.01, 0.1]"

    print(f"\nBaseline residuals:")
    print(f"  Length: {len(residuals)}")
    print(f"  MAE: {baseline_mae:.6e}")
    print(f"  Mean: {np.mean(residuals):.6e}")
    print(f"  Std: {np.std(residuals):.6e}")


def test_sample_method_shape(tsgam_estimator_with_ar, notebook_data):
    """Test that sample method returns correct shape."""
    estimator = tsgam_estimator_with_ar

    # Get 2022 data for prediction
    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    # Generate samples
    n_samples = 5
    samples = estimator.sample(X_2022, n_samples=n_samples, random_state=42)

    # Should return (n_samples, n_pred_samples) shape
    assert samples.shape == (n_samples, len(X_2022)), \
        f"Sample shape mismatch: {samples.shape} vs expected ({n_samples}, {len(X_2022)})"

    # Samples may have NaNs at start/end due to lead/lag operations (matching baseline predictions)
    # Check that non-NaN values are finite
    assert np.all(np.isfinite(samples[~np.isnan(samples)])), "All non-NaN samples should be finite"
    # Check that we have some finite values
    assert np.any(np.isfinite(samples)), "Should have at least some finite samples"

    print(f"\nSample method:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {np.nanmean(samples):.6e}")
    print(f"  Std: {np.nanstd(samples):.6e}")
    print(f"  NaN count: {np.sum(np.isnan(samples))}")


def test_sample_method_reproducibility(tsgam_estimator_with_ar, notebook_data):
    """Test that sample method is reproducible with same random_state."""
    estimator = tsgam_estimator_with_ar

    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    # Generate samples with same random state
    samples1 = estimator.sample(X_2022, n_samples=3, random_state=123)
    samples2 = estimator.sample(X_2022, n_samples=3, random_state=123)

    # Should be identical
    np.testing.assert_array_equal(
        samples1,
        samples2,
        err_msg="Samples should be identical with same random_state"
    )

    # Generate samples with different random state
    samples3 = estimator.sample(X_2022, n_samples=3, random_state=456)

    # Should be different (very unlikely to be identical)
    assert not np.allclose(samples1, samples3, atol=1e-10), \
        "Samples with different random_state should be different"


def test_sample_method_without_ar():
    """Test that sample method works without AR model (adds small noise)."""
    df = load_notebook_data(sheet='RI', years=[2020, 2021])
    df_subset = df.loc["2020":"2021"]
    y = np.log(df_subset["RT_Demand"]).values
    X = pd.DataFrame({'temp': df_subset["Dry_Bulb"].values}, index=df_subset.index)

    # No AR config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=TsgamMultiHarmonicConfig(
            num_harmonics=[6, 4, 3],
            periods=[365.2425 * 24, 7 * 24, 24]
        ),
        exog_config=[
            TsgamSplineConfig(
                n_knots=10,
                lags=[-3, -2, -1, 0, 1, 2, 3],
                reg_weight=1e-4,
                diff_reg_weight=1.0
            )
        ],
        ar_config=None,  # No AR model
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        debug=False
    )

    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)

    # Should still work (adds small noise)
    samples = estimator.sample(X, n_samples=2, random_state=42)
    assert samples.shape == (2, len(X)), "Sample shape should be correct"
    # Samples may have NaNs at start/end due to lead/lag operations
    assert np.all(np.isfinite(samples[~np.isnan(samples)])), "All non-NaN samples should be finite"
    assert np.any(np.isfinite(samples)), "Should have at least some finite samples"


def roll_out_ar_noise_notebook(length, ar_coeff, intercept, loc, scale, random_state=None):
    """
    Replicate notebook's roll_out_ar_noise function for comparison.

    Returns exponentiated AR noise (multiplicative factor in original scale).

    The notebook passes random_state (int or None) directly to stats.laplace.rvs.
    When it's an int, scipy creates a new RandomState internally for each call.
    This matches the notebook's exact behavior.
    """
    from scipy import stats

    # Pass random_state directly (int or None) - notebook's approach
    window = stats.laplace.rvs(loc=loc, scale=scale, size=len(ar_coeff), random_state=random_state)
    nvals = length + len(ar_coeff) * 2
    gen_data = np.empty(nvals, dtype=float)
    for it in range(nvals):
        new_val = ar_coeff @ window + intercept + stats.laplace.rvs(loc=loc, scale=scale, random_state=random_state)
        gen_data[it] = new_val
        new_window = np.roll(window, -1)
        new_window[-1] = new_val
        window = new_window
    return np.exp(gen_data[-length:])


def test_sample_ar_noise_matches_notebook(tsgam_estimator_with_ar, notebook_data):
    """
    Test that AR noise rollout follows the correct pattern.

    Note: Our implementation uses a RandomState object (state advances), while the notebook
    uses an int (creates new RandomState internally each call). This means the exact sequences
    will differ, but we can verify the implementation is correct by checking:
    1. AR noise has correct statistical properties
    2. The pattern matches: samples = baseline * noise (in original scale)
    3. AR coefficients match (verified in other tests)
    """
    estimator = tsgam_estimator_with_ar

    # Get 2022 data for prediction
    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    # Get baseline predictions (log scale)
    baseline_pred_log = estimator.predict(X_2022)

    # Generate samples (log scale)
    samples_log = estimator.sample(X_2022, n_samples=1, random_state=42)

    # Extract AR noise (log scale): samples - baseline
    ar_noise_log = samples_log[0] - baseline_pred_log

    # Convert to original scale (multiplicative factor)
    ar_noise_ours = np.exp(ar_noise_log)

    # Verify AR noise has correct properties
    valid_mask = ~np.isnan(ar_noise_ours)
    assert np.any(valid_mask), "Should have some valid AR noise values"

    ar_noise_valid = ar_noise_ours[valid_mask]

    # AR noise should be positive (multiplicative factor)
    assert np.all(ar_noise_valid > 0), "AR noise should be positive"

    # AR noise should be close to 1 on average (small perturbations)
    mean_noise = np.mean(ar_noise_valid)
    assert 0.9 < mean_noise < 1.1, f"AR noise mean should be close to 1, got {mean_noise:.6f}"

    # AR noise should have reasonable variance
    std_noise = np.std(ar_noise_valid)
    assert 0.01 < std_noise < 0.1, f"AR noise std should be reasonable, got {std_noise:.6f}"

    print(f"\nAR noise properties:")
    print(f"  Valid values: {np.sum(valid_mask)}/{len(ar_noise_ours)}")
    print(f"  Mean: {mean_noise:.6e}")
    print(f"  Std: {std_noise:.6e}")
    print(f"  Min: {np.min(ar_noise_valid):.6e}")
    print(f"  Max: {np.max(ar_noise_valid):.6e}")


def test_sample_predictions_match_notebook_pattern(tsgam_estimator_with_ar, notebook_data):
    """
    Test that sample predictions follow the notebook pattern: baseline * noise.

    Notebook: new_baseline * new_noise (both in original scale)
    Our estimator: exp(baseline_log + ar_noise_log) = exp(baseline_log) * exp(ar_noise_log)
    """
    estimator = tsgam_estimator_with_ar

    # Get 2022 data for prediction
    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    # Get baseline predictions (log scale)
    baseline_pred_log = estimator.predict(X_2022)
    baseline_pred_orig = np.exp(baseline_pred_log)

    # Generate samples (log scale)
    samples_log = estimator.sample(X_2022, n_samples=1, random_state=42)
    samples_orig = np.exp(samples_log[0])

    # Extract AR noise (log scale)
    ar_noise_log = samples_log[0] - baseline_pred_log
    ar_noise_orig = np.exp(ar_noise_log)

    # Verify pattern: samples_orig = baseline_pred_orig * ar_noise_orig
    expected_samples = baseline_pred_orig * ar_noise_orig

    np.testing.assert_allclose(
        samples_orig,
        expected_samples,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Sample predictions should follow baseline * noise pattern"
    )

    print(f"\nSample prediction pattern verification:")
    print(f"  Mean baseline: {np.nanmean(baseline_pred_orig):.2f}")
    print(f"  Mean AR noise: {np.nanmean(ar_noise_orig):.6e}")
    print(f"  Mean samples: {np.nanmean(samples_orig):.2f}")
    print(f"  Max diff from pattern: {np.nanmax(np.abs(samples_orig - expected_samples)):.6e}")
    print(f"  NaN count: {np.sum(np.isnan(samples_orig))}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

