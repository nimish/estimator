# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Test that TsgamEstimator predictions match notebook predictions.

This test loads the actual notebook prediction output (new_baseline.npy) and
compares it with our estimator's predictions on the same 2022 data.
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
def notebook_baseline_predictions():
    """Load notebook baseline predictions for 2022."""
    base_path = Path(__file__).parent
    return np.load(base_path / 'new_baseline.npy')


@pytest.fixture(scope="module")
def tsgam_estimator_for_prediction(notebook_data):
    """Fit TsgamEstimator with baseline configuration matching notebook."""
    df_subset = notebook_data.loc["2020":"2021"]
    y = np.log(df_subset["RT_Demand"]).values
    X = pd.DataFrame({'temp': df_subset["Dry_Bulb"].values}, index=df_subset.index)

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
        ar_config=None,
        solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
        random_state=None,
        debug=False
    )

    estimator = TsgamEstimator(config=config)
    estimator.fit(X, y)

    return estimator


def test_predictions_match_notebook(tsgam_estimator_for_prediction, notebook_data, notebook_baseline_predictions):
    """
    Test that predictions on 2022 data match notebook predictions.

    The notebook:
    1. Trains on 2020-2021 data
    2. Predicts on 2022 data using indices that continue from training
    3. Returns np.exp(baseline) - predictions in original scale

    Our estimator:
    1. Trains on 2020-2021 data (with timestamps)
    2. Predicts on 2022 data (with timestamps)
    3. Returns predictions in log scale, need to exp them

    Note: Notebook predictions have NaNs at start/end due to lead/lag operations.
    """
    estimator = tsgam_estimator_for_prediction

    # Get 2022 data for prediction
    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    # Get our predictions (in log space)
    predictions_log = estimator.predict(X_2022)

    # Convert to original scale (matching notebook)
    predictions_ours = np.exp(predictions_log)

    # Notebook predictions
    predictions_notebook = notebook_baseline_predictions

    # Both should have same length
    assert len(predictions_ours) == len(predictions_notebook), \
        f"Prediction length mismatch: ours={len(predictions_ours)}, notebook={len(predictions_notebook)}"

    # Create mask for valid (non-NaN) predictions
    # Notebook has NaNs at start/end due to lead/lag operations
    valid_mask = ~np.isnan(predictions_notebook)

    # Check that we also have NaNs in same positions (or at least that notebook has them)
    if np.any(np.isnan(predictions_notebook)):
        # Notebook has NaNs - this is expected due to lead/lag operations
        # Our estimator should handle this correctly, but let's check
        print(f"\nNotebook has {np.sum(np.isnan(predictions_notebook))} NaN predictions")
        print(f"  First NaN indices: {np.where(np.isnan(predictions_notebook))[0][:10]}")
        print(f"  Last NaN indices: {np.where(np.isnan(predictions_notebook))[0][-10:]}")

        # For now, compare only valid (non-NaN) predictions
        predictions_ours_valid = predictions_ours[valid_mask]
        predictions_notebook_valid = predictions_notebook[valid_mask]
    else:
        predictions_ours_valid = predictions_ours
        predictions_notebook_valid = predictions_notebook

    # Compare valid predictions
    np.testing.assert_allclose(
        predictions_ours_valid,
        predictions_notebook_valid,
        rtol=1e-5,
        atol=1e-3,  # Allow some tolerance for numerical differences
        err_msg="Predictions don't match notebook"
    )

    print("\nPredictions comparison:")
    print(f"  Total predictions: {len(predictions_ours)}")
    print(f"  Valid predictions: {len(predictions_ours_valid)}")
    print(f"  Max absolute diff: {np.max(np.abs(predictions_ours_valid - predictions_notebook_valid)):.6e}")
    print(f"  Mean absolute diff: {np.mean(np.abs(predictions_ours_valid - predictions_notebook_valid)):.6e}")
    print(f"  Max relative diff: {np.max(np.abs((predictions_ours_valid - predictions_notebook_valid) / predictions_notebook_valid)):.6e}")


def test_predictions_shape_matches_notebook(tsgam_estimator_for_prediction, notebook_data, notebook_baseline_predictions):
    """Test that prediction shape matches notebook."""
    estimator = tsgam_estimator_for_prediction

    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    predictions_ours = estimator.predict(X_2022)
    predictions_notebook = notebook_baseline_predictions

    assert predictions_ours.shape == predictions_notebook.shape, \
        f"Shape mismatch: ours={predictions_ours.shape}, notebook={predictions_notebook.shape}"

    print("\nPrediction shapes comparison:")
    print(f"  Shape: {predictions_ours.shape}")


def test_prediction_statistics_match_notebook(tsgam_estimator_for_prediction, notebook_data, notebook_baseline_predictions):
    """Test that prediction statistics match notebook."""
    estimator = tsgam_estimator_for_prediction

    df_2022 = notebook_data.loc["2022"]
    X_2022 = pd.DataFrame({'temp': df_2022["Dry_Bulb"].values}, index=df_2022.index)

    predictions_ours_log = estimator.predict(X_2022)
    predictions_ours = np.exp(predictions_ours_log)
    predictions_notebook = notebook_baseline_predictions

    # Compare statistics on valid predictions
    valid_mask = ~np.isnan(predictions_notebook)
    predictions_ours_valid = predictions_ours[valid_mask]
    predictions_notebook_valid = predictions_notebook[valid_mask]

    # Mean
    mean_ours = np.mean(predictions_ours_valid)
    mean_notebook = np.mean(predictions_notebook_valid)

    # Median
    median_ours = np.median(predictions_ours_valid)
    median_notebook = np.median(predictions_notebook_valid)

    # Std
    std_ours = np.std(predictions_ours_valid)
    std_notebook = np.std(predictions_notebook_valid)

    print("\nPrediction statistics comparison:")
    print(f"  Mean - Ours: {mean_ours:.2f}, Notebook: {mean_notebook:.2f}, Diff: {abs(mean_ours - mean_notebook):.2f}")
    print(f"  Median - Ours: {median_ours:.2f}, Notebook: {median_notebook:.2f}, Diff: {abs(median_ours - median_notebook):.2f}")
    print(f"  Std - Ours: {std_ours:.2f}, Notebook: {std_notebook:.2f}, Diff: {abs(std_ours - std_notebook):.2f}")

    # Check that statistics are close
    np.testing.assert_allclose(mean_ours, mean_notebook, rtol=1e-4)
    np.testing.assert_allclose(median_ours, median_notebook, rtol=1e-4)
    np.testing.assert_allclose(std_ours, std_notebook, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

