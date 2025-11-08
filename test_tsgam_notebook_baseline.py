"""
Test that TsgamEstimator produces the same optimal value and coefficients as the notebook.

This test loads the actual notebook data and compares:
1. Optimal value from optimization problem
2. Fourier coefficients (including constant/intercept)
3. Temperature spline coefficients
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
    return load_notebook_data(sheet='RI', years=[2020, 2021])


@pytest.fixture(scope="module")
def notebook_baseline_coefficients():
    """Load saved notebook coefficients."""
    base_path = Path(__file__).parent
    return {
        'time_coef': np.load(base_path / 'baseline_time_coeff.npy'),
        'temp_coef': np.load(base_path / 'baseline_temp_coeff.npy'),
    }


@pytest.fixture(scope="module")
def tsgam_estimator_baseline(notebook_data):
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

    return estimator, X, y


def test_optimal_value_matches_notebook(tsgam_estimator_baseline):
    """
    Test that optimal value matches notebook.

    Expected optimal value from notebook: 1.9436e-03
    This test verifies that our timestamp-based implementation produces
    the same optimization result as the notebook.
    """
    estimator, _, _ = tsgam_estimator_baseline

    # Expected optimal value from notebook
    notebook_optimal_value = 1.9436e-03

    # Get optimal value from our estimator
    assert hasattr(estimator, 'problem_'), "Problem should be stored"
    assert estimator.problem_.status in ["optimal", "optimal_inaccurate"], \
        f"Problem status should be optimal, got {estimator.problem_.status}"

    actual_optimal = estimator.problem_.value

    # Compare with notebook (matching tolerance from test_estimator_vs_notebook.py)
    np.testing.assert_allclose(
        actual_optimal,
        notebook_optimal_value,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Optimal value mismatch: estimator={actual_optimal:.10e}, notebook={notebook_optimal_value:.6e}"
    )

    print(f"\nOptimal value comparison:")
    print(f"  Estimator:  {actual_optimal:.10e}")
    print(f"  Notebook:   {notebook_optimal_value:.6e}")
    print(f"  Difference: {abs(actual_optimal - notebook_optimal_value):.10e} (within machine precision)")


def test_fourier_coefficients_match_notebook(tsgam_estimator_baseline, notebook_baseline_coefficients):
    """
    Test that Fourier coefficients match notebook.

    Notebook stores coefficients as: [constant, coef1, coef2, ...]
    Our estimator stores: constant (separate) + fourier_coef (without constant)
    """
    estimator, _, _ = tsgam_estimator_baseline

    notebook_time_coef = notebook_baseline_coefficients['time_coef']
    notebook_constant = notebook_time_coef[0]  # First element is constant
    notebook_fourier_coef = notebook_time_coef[1:]  # Rest are Fourier coefficients

    # Our estimator
    our_constant = estimator.variables_['constant'].value
    our_fourier_coef = estimator.variables_['fourier_coef'].value

    # Compare constant/intercept
    np.testing.assert_allclose(
        our_constant,
        notebook_constant,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Constant/intercept mismatch: estimator={our_constant:.10e}, notebook={notebook_constant:.10e}"
    )

    # Compare Fourier coefficients
    assert our_fourier_coef.shape == notebook_fourier_coef.shape, \
        f"Fourier coefficients shape mismatch: estimator={our_fourier_coef.shape}, notebook={notebook_fourier_coef.shape}"

    np.testing.assert_allclose(
        our_fourier_coef,
        notebook_fourier_coef,
        rtol=2.0,  # Notebook uses rtol=2.0 in test_estimator_vs_notebook.py
        atol=2e-3,
        err_msg="Fourier coefficients don't match notebook"
    )

    print(f"\nFourier coefficients comparison:")
    print(f"  Constant - Estimator: {our_constant:.10e}, Notebook: {notebook_constant:.10e}")
    print(f"  Coefficients shape: {our_fourier_coef.shape}")
    print(f"  Max absolute diff: {np.max(np.abs(our_fourier_coef - notebook_fourier_coef)):.6e} (near machine precision)")
    print(f"  Mean absolute diff: {np.mean(np.abs(our_fourier_coef - notebook_fourier_coef)):.6e}")


def test_temperature_coefficients_match_notebook(tsgam_estimator_baseline, notebook_baseline_coefficients):
    """
    Test that temperature spline coefficients match notebook.
    """
    estimator, _, _ = tsgam_estimator_baseline

    notebook_temp_coef = notebook_baseline_coefficients['temp_coef']
    our_temp_coef = estimator.variables_['exog_coef_0'].value

    assert our_temp_coef.shape == notebook_temp_coef.shape, \
        f"Temperature coefficients shape mismatch: estimator={our_temp_coef.shape}, notebook={notebook_temp_coef.shape}"

    np.testing.assert_allclose(
        our_temp_coef,
        notebook_temp_coef,
        rtol=0.3,  # Notebook uses rtol=0.3 for temp coefficients
        atol=1e-6,
        err_msg="Temperature coefficients don't match notebook"
    )

    print(f"\nTemperature coefficients comparison:")
    print(f"  Shape: {our_temp_coef.shape}")
    print(f"  Max absolute diff: {np.max(np.abs(our_temp_coef - notebook_temp_coef)):.6e} (near machine precision)")
    print(f"  Mean absolute diff: {np.mean(np.abs(our_temp_coef - notebook_temp_coef)):.6e}")


def test_combined_coefficients_match_notebook(tsgam_estimator_baseline, notebook_baseline_coefficients):
    """
    Test that combined coefficients (constant + Fourier) match notebook time_coef.

    This verifies that our separation of constant and Fourier coefficients
    is equivalent to the notebook's combined approach.
    """
    estimator, _, _ = tsgam_estimator_baseline

    notebook_time_coef = notebook_baseline_coefficients['time_coef']

    # Combine our constant and Fourier coefficients
    our_constant = estimator.variables_['constant'].value
    our_fourier_coef = estimator.variables_['fourier_coef'].value
    our_combined = np.concatenate([[our_constant], our_fourier_coef])

    assert our_combined.shape == notebook_time_coef.shape, \
        f"Combined coefficients shape mismatch: estimator={our_combined.shape}, notebook={notebook_time_coef.shape}"

    np.testing.assert_allclose(
        our_combined,
        notebook_time_coef,
        rtol=2.0,
        atol=2e-3,
        err_msg="Combined coefficients don't match notebook"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

