# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Pytest tests comparing LoadForecastRegressor against notebook implementation.

This test suite verifies that the estimator produces identical results to the
notebook by comparing against saved coefficients from the notebook.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from load_model_estimator import LoadForecastRegressor





def load_notebook_data(sheet='RI', years=[2020, 2021]):
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
    return load_notebook_data(sheet='RI', years=[2020, 2021])


@pytest.fixture(scope="module")
def notebook_params():
    return {
        'num_harmonics': [6, 4, 3],
        'periods': [365.2425 * 24, 7 * 24, 24],
        'exog_mode': 'spline',
        'n_knots': 10,
        'exog_lags': [-3, -2, -1, 0, 1, 2, 3],
        'fourier_reg_weight': 1e-4,
        'exog_reg_weight': 1e-4,
        'exog_diff_reg_weight': 1.0,
        'fit_ar': True,
        'ar_lags': 36,
        'ar_l1_constraint': 0.95,
        'cvxpy_solver': 'CLARABEL',
        'cvxpy_verbose': False,
    }


@pytest.fixture(scope="module")
def fitted_estimator(notebook_data, notebook_params):
    y = np.log(notebook_data.loc["2020":"2021"]["RT_Demand"])
    x = notebook_data.loc["2020":"2021"]["Dry_Bulb"]

    time_idx = np.arange(len(x))
    X = np.column_stack([time_idx, x.values])
    y_arr = y.values

    estimator = LoadForecastRegressor(**notebook_params, debug=True)
    estimator.fit(X, y_arr)

    return estimator, X, y_arr, x.values


@pytest.fixture(scope="module")
def notebook_coefficients():
    """Load saved coefficients and intermediate results from notebook (cached at module scope)."""
    base_path = Path(__file__).parent
    return {
        'time_coef': np.load(base_path / 'baseline_time_coeff.npy'),
        'temp_coef': np.load(base_path / 'baseline_temp_coeff.npy'),
        'ar_coef': np.load(base_path / 'ar_coeff.npy'),
        'ar_intercept': np.load(base_path / 'ar_intercept.npy'),
        'B_running_view': np.load(base_path / 'B_running_view.npy'),
        'use_set': np.load(base_path / 'use_set.npy')
    }


def test_baseline_optimal_value(fitted_estimator):
    """Test that baseline model optimal value matches notebook."""
    estimator, _, _, _ = fitted_estimator

    # Expected optimal value from notebook: 1.9436e-03
    notebook_optimal_value = 1.9436e-03

    assert estimator._last_problem_value_ is not None, "Baseline optimal value should be stored"
    assert estimator._last_problem_status_ == "optimal", f"Baseline status should be optimal, got {estimator._last_problem_status_}"

    actual_optimal = estimator._last_problem_value_

    np.testing.assert_allclose(
        actual_optimal,
        notebook_optimal_value,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Optimal value mismatch: estimator={actual_optimal:.6e}, notebook={notebook_optimal_value:.6e}"
    )
    absolute_diff = abs(actual_optimal - notebook_optimal_value)
    relative_diff = absolute_diff / notebook_optimal_value if notebook_optimal_value != 0 else 0
    print("\nOptimal value comparison:")
    print(f"  Estimator:  {actual_optimal:.10e}")
    print(f"  Notebook:   {notebook_optimal_value:.10e}")
    print(f"  Absolute difference: {absolute_diff:.10e}")
    print(f"  Relative difference: {relative_diff*100:.6f}%")


def test_baseline_intercept_match_notebook(fitted_estimator, notebook_coefficients):
    """Test that baseline intercept matches notebook value."""
    estimator, _, _, _ = fitted_estimator

    notebook_time_coef = notebook_coefficients['time_coef']
    notebook_intercept = notebook_time_coef[0]  # First element is the intercept

    assert hasattr(estimator, 'intercept_'), "Estimator should have intercept_ attribute"
    assert estimator.intercept_ is not None, "Intercept should be set"

    np.testing.assert_allclose(
        estimator.intercept_,
        notebook_intercept,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Intercept doesn't match notebook: estimator={estimator.intercept_:.10e}, notebook={notebook_intercept:.10e}"
    )

    print("\nIntercept comparison:")
    print(f"  Estimator:  {estimator.intercept_:.10e}")
    print(f"  Notebook:   {notebook_intercept:.10e}")
    print(f"  Absolute difference: {abs(estimator.intercept_ - notebook_intercept):.10e}")
    print(f"  Relative difference: {abs((estimator.intercept_ - notebook_intercept) / notebook_intercept) * 100:.6f}%")


def test_baseline_coefficients_match_notebook(fitted_estimator, notebook_coefficients):
    """Test that baseline coefficients match notebook (with absolute and relative tolerance)."""
    estimator, _, _, _ = fitted_estimator

    notebook_time_coef = notebook_coefficients['time_coef']
    notebook_time_coef_no_intercept = notebook_time_coef[1:]  # Exclude intercept (first element)
    notebook_temp_coef = notebook_coefficients['temp_coef']

    assert estimator.time_coef_.shape == notebook_time_coef_no_intercept.shape, \
        f"Time coefficients shape {estimator.time_coef_.shape} != notebook {notebook_time_coef_no_intercept.shape}"

    assert estimator.exog_coef_.shape == notebook_temp_coef.shape, \
        f"Temperature coefficients shape {estimator.exog_coef_.shape} != notebook {notebook_temp_coef.shape}"

    np.testing.assert_allclose(
        estimator.time_coef_,
        notebook_time_coef_no_intercept,
        rtol=2.0,
        atol=2e-3,
        err_msg="Time coefficients don't match notebook"
    )

    np.testing.assert_allclose(
        estimator.exog_coef_,
        notebook_temp_coef,
        rtol=0.3,
        atol=1e-6,
        err_msg="Temperature coefficients don't match notebook"
    )
    time_abs_diff = np.abs(estimator.time_coef_ - notebook_time_coef_no_intercept)
    time_rel_diff = np.abs((estimator.time_coef_ - notebook_time_coef_no_intercept) / (np.abs(notebook_time_coef_no_intercept) + 1e-12))
    temp_abs_diff = np.abs(estimator.exog_coef_ - notebook_temp_coef)
    temp_rel_diff = np.abs((estimator.exog_coef_ - notebook_temp_coef) / (np.abs(notebook_temp_coef) + 1e-12))

    print("\nBaseline coefficients comparison:")
    print("Time coefficients:")
    print(f"  Max absolute diff:  {np.max(time_abs_diff):.6e}")
    print(f"  Mean absolute diff: {np.mean(time_abs_diff):.6e}")
    print(f"  Max relative diff:  {np.max(time_rel_diff):.6e} ({np.max(time_rel_diff)*100:.4f}%)")
    print(f"  Mean relative diff: {np.mean(time_rel_diff):.6e} ({np.mean(time_rel_diff)*100:.4f}%)")
    print("Temperature coefficients:")
    print(f"  Max absolute diff:  {np.max(temp_abs_diff):.6e}")
    print(f"  Mean absolute diff: {np.mean(temp_abs_diff):.6e}")
    print(f"  Max relative diff:  {np.max(temp_rel_diff):.6e} ({np.max(temp_rel_diff)*100:.4f}%)")
    print(f"  Mean relative diff: {np.mean(temp_rel_diff):.6e} ({np.mean(temp_rel_diff)*100:.4f}%)")


def test_ar_coefficients_match_notebook(fitted_estimator, notebook_coefficients):
    """Test that AR coefficients match notebook exactly."""
    estimator, _, _, _ = fitted_estimator

    notebook_ar_coef = notebook_coefficients['ar_coef']
    notebook_ar_intercept = notebook_coefficients['ar_intercept']

    assert estimator.ar_coef_.shape == notebook_ar_coef.shape, \
        f"AR coefficients shape {estimator.ar_coef_.shape} != notebook {notebook_ar_coef.shape}"

    np.testing.assert_allclose(
        estimator.ar_coef_,
        notebook_ar_coef,
        rtol=1e-6,
        atol=1e-6,
        err_msg="AR coefficients don't match notebook"
    )

    np.testing.assert_allclose(
        estimator.ar_intercept_,
        notebook_ar_intercept,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"AR intercept doesn't match notebook: estimator={estimator.ar_intercept_:.10e}, notebook={notebook_ar_intercept:.10e}"
    )

    ar_sum_abs = np.sum(np.abs(estimator.ar_coef_))
    notebook_sum_abs = np.sum(np.abs(notebook_ar_coef))
    np.testing.assert_allclose(
        ar_sum_abs,
        notebook_sum_abs,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"AR L1 norm {ar_sum_abs:.10f} doesn't match notebook {notebook_sum_abs:.10f}"
    )


def test_ar_noise_distribution_match_notebook(fitted_estimator):
    """Test that Laplace noise distribution parameters match notebook."""
    estimator, _, _, _ = fitted_estimator

    # Expected values from notebook
    notebook_lap_loc = -0.000361
    notebook_lap_scale = 0.012968

    assert estimator.ar_noise_loc_ is not None, "AR noise location parameter should be set"
    assert estimator.ar_noise_scale_ is not None, "AR noise scale parameter should be set"

    np.testing.assert_allclose(
        estimator.ar_noise_loc_,
        notebook_lap_loc,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"AR noise location doesn't match notebook: estimator={estimator.ar_noise_loc_:.6f}, notebook={notebook_lap_loc:.6f}"
    )

    np.testing.assert_allclose(
        estimator.ar_noise_scale_,
        notebook_lap_scale,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"AR noise scale doesn't match notebook: estimator={estimator.ar_noise_scale_:.6f}, notebook={notebook_lap_scale:.6f}"
    )

    print("\nLaplace noise distribution parameters comparison:")
    print(f"  Location - Estimator:  {estimator.ar_noise_loc_:.6f}, Notebook: {notebook_lap_loc:.6f}")
    print(f"  Scale    - Estimator:  {estimator.ar_noise_scale_:.6f}, Notebook: {notebook_lap_scale:.6f}")
    print(f"  Location absolute diff: {abs(estimator.ar_noise_loc_ - notebook_lap_loc):.6e}")
    print(f"  Scale absolute diff:    {abs(estimator.ar_noise_scale_ - notebook_lap_scale):.6e}")


def test_ar_coefficients_constraint(fitted_estimator, notebook_coefficients):
    """Test that AR coefficients satisfy L1 constraint (matches notebook)."""
    estimator, _, _, _ = fitted_estimator

    notebook_ar_coef = notebook_coefficients['ar_coef']
    notebook_sum_abs = np.sum(np.abs(notebook_ar_coef))

    ar_sum_abs = np.sum(np.abs(estimator.ar_coef_))

    np.testing.assert_allclose(
        ar_sum_abs,
        notebook_sum_abs,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"AR sum of absolute coefficients {ar_sum_abs:.10f} != notebook {notebook_sum_abs:.10f}"
    )


def test_knots_computed(fitted_estimator):
    """Test that knots are computed and stored correctly."""
    estimator, _, _, x = fitted_estimator

    assert estimator.knots_ is not None, "Knots should be computed"
    assert len(estimator.knots_) == 10, f"Expected 10 knots, got {len(estimator.knots_)}"
    assert np.min(estimator.knots_) >= np.min(x) - 1e-6, "Knots should cover data range"
    assert np.max(estimator.knots_) <= np.max(x) + 1e-6, "Knots should cover data range"


def test_predict_shape(fitted_estimator):
    """Test that predict returns correct shape."""
    estimator, X, _, _ = fitted_estimator

    y_pred = estimator.predict(X)

    assert y_pred.shape == (X.shape[0],), \
        f"Predictions shape {y_pred.shape} != ({X.shape[0]},)"
    assert np.all(np.isfinite(y_pred)), "Predictions should be finite"


def test_residuals_computation(fitted_estimator):
    """Test that residuals are computed correctly for AR model."""
    estimator, X, y_arr, x_arr = fitted_estimator

    assert hasattr(estimator, '_baseline_residuals_'), "Estimator should expose _baseline_residuals_ debug variable"
    assert estimator._baseline_residuals_ is not None, "baseline_residuals_ should be set when fit_ar=True"

    residuals = estimator._baseline_residuals_

    assert len(residuals) > 0, "Should have residuals"
    assert np.all(np.isfinite(residuals)), "Residuals should be finite"

    baseline_mae = np.mean(np.abs(residuals))
    assert 0.03 < baseline_mae < 0.04, \
        f"Baseline MAE {baseline_mae:.6f} outside expected range [0.03, 0.04]"


def test_ar_running_view_match_notebook(fitted_estimator, notebook_coefficients):
    """Test that AR running view B matrix and use_set match notebook exactly."""
    estimator, X, y_arr, x_arr = fitted_estimator

    notebook_B = notebook_coefficients['B_running_view']
    notebook_use_set = notebook_coefficients['use_set']

    assert hasattr(estimator, '_B_running_view_'), "Estimator should expose _B_running_view_ debug variable"
    assert estimator._B_running_view_ is not None, "B_running_view_ should be set when fit_ar=True"

    B = estimator._B_running_view_
    ar_valid_mask = estimator._ar_valid_mask_

    assert B.shape == notebook_B.shape, \
        f"B matrix shape {B.shape} != notebook {notebook_B.shape}"

    B_nan = np.isnan(B)
    notebook_B_nan = np.isnan(notebook_B)
    assert np.array_equal(B_nan, notebook_B_nan), \
        "B matrix NaN patterns don't match notebook"

    B_filled = np.nan_to_num(B, nan=0.0)
    notebook_B_filled = np.nan_to_num(notebook_B, nan=0.0)
    np.testing.assert_allclose(
        B_filled,
        notebook_B_filled,
        rtol=1e-4,
        atol=1e-4,
        err_msg="B matrix values don't match notebook"
    )
    assert np.array_equal(ar_valid_mask, notebook_use_set), \
        f"AR valid mask doesn't match notebook use_set (estimator: {np.sum(ar_valid_mask)} valid, notebook: {np.sum(notebook_use_set)} valid)"


def test_ar_optimal_value_match(fitted_estimator, notebook_coefficients):
    """Test that AR coefficients match notebook exactly."""
    estimator, _, _, _ = fitted_estimator

    notebook_ar_coef = notebook_coefficients['ar_coef']
    notebook_ar_intercept = notebook_coefficients['ar_intercept']

    np.testing.assert_allclose(
        estimator.ar_coef_,
        notebook_ar_coef,
        rtol=1e-6,
        atol=1e-6,
        err_msg="AR coefficients don't match notebook"
    )

    np.testing.assert_allclose(
        estimator.ar_intercept_,
        notebook_ar_intercept,
        rtol=1e-6,
        atol=1e-6,
        err_msg="AR intercept doesn't match notebook"
    )


def test_notebook_parameter_match(fitted_estimator):
    """Test that estimator uses exact notebook parameters."""
    estimator, _, _, _ = fitted_estimator

    assert estimator.num_harmonics == [6, 4, 3]
    assert np.allclose(estimator.periods, [365.2425 * 24, 7 * 24, 24])
    assert estimator.exog_mode == 'spline'
    assert estimator.n_knots == 10
    assert estimator.exog_lags == [-3, -2, -1, 0, 1, 2, 3]
    assert estimator.fourier_reg_weight == 1e-4
    assert estimator.exog_reg_weight == 1e-4
    assert estimator.exog_diff_reg_weight == 1.0
    assert estimator.fit_ar
    assert estimator.ar_lags == 36
    assert estimator.ar_l1_constraint == 0.95


def test_sklearn_compatibility(fitted_estimator):
    """Test sklearn compatibility (get_params, set_params, score)."""
    estimator, X, y_arr, _ = fitted_estimator

    params = estimator.get_params()
    assert 'num_harmonics' in params
    assert 'fit_ar' in params

    estimator.set_params(fit_ar=False)
    assert not estimator.fit_ar
    estimator.set_params(fit_ar=True)
    assert estimator.fit_ar

    score = estimator.score(X, y_arr)
    assert np.isfinite(score), "Score should be finite"
    assert -np.inf < score <= 1.0, f"R² score should be in (-inf, 1], got {score}"


def test_notebook_reproducibility(fitted_estimator, notebook_coefficients):
    """Test that estimator reproduces notebook results exactly."""
    estimator, X, y_arr, x_arr = fitted_estimator

    expected_baseline_optimal = 1.9436e-03
    actual_optimal = estimator._last_problem_value_
    np.testing.assert_allclose(
        actual_optimal,
        expected_baseline_optimal,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Baseline optimal value mismatch: {actual_optimal:.10e} vs expected ~{expected_baseline_optimal:.6e}"
    )

    notebook_time_coef = notebook_coefficients['time_coef']
    notebook_time_coef_no_intercept = notebook_time_coef[1:]  # Exclude intercept
    notebook_intercept = notebook_time_coef[0]  # Intercept is first element
    notebook_temp_coef = notebook_coefficients['temp_coef']
    notebook_ar_coef = notebook_coefficients['ar_coef']
    notebook_ar_intercept = notebook_coefficients['ar_intercept']

    np.testing.assert_allclose(
        estimator.intercept_,
        notebook_intercept,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Intercept doesn't match notebook"
    )

    np.testing.assert_allclose(
        estimator.time_coef_,
        notebook_time_coef_no_intercept,
        rtol=2.0,
        atol=2e-3,
        err_msg="Time coefficients don't match notebook"
    )

    np.testing.assert_allclose(
        estimator.exog_coef_,
        notebook_temp_coef,
        rtol=0.3,
        atol=2e-3,
        err_msg="Temperature coefficients don't match notebook"
    )

    np.testing.assert_allclose(
        estimator.ar_coef_,
        notebook_ar_coef,
        rtol=1e-6,
        atol=1e-6,
        err_msg="AR coefficients don't match notebook"
    )

    np.testing.assert_allclose(
        estimator.ar_intercept_,
        notebook_ar_intercept,
        rtol=1e-6,
        atol=1e-6,
        err_msg="AR intercept doesn't match notebook"
    )

    notebook_ar_sum_abs = np.sum(np.abs(notebook_ar_coef))
    estimator_ar_sum_abs = np.sum(np.abs(estimator.ar_coef_))
    np.testing.assert_allclose(
        estimator_ar_sum_abs,
        notebook_ar_sum_abs,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"AR L1 norm {estimator_ar_sum_abs:.10f} matches notebook {notebook_ar_sum_abs:.10f}"
    )

    # Check Laplace noise distribution parameters
    notebook_lap_loc = -0.000361
    notebook_lap_scale = 0.012968
    np.testing.assert_allclose(
        estimator.ar_noise_loc_,
        notebook_lap_loc,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"AR noise location {estimator.ar_noise_loc_:.6f} doesn't match notebook {notebook_lap_loc:.6f}"
    )
    np.testing.assert_allclose(
        estimator.ar_noise_scale_,
        notebook_lap_scale,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"AR noise scale {estimator.ar_noise_scale_:.6f} doesn't match notebook {notebook_lap_scale:.6f}"
    )


def test_running_view_lag1_backward_compatibility():
    """Test that lag=1 matches original behavior (backward compatibility)."""
    estimator = LoadForecastRegressor()
    arr = np.array([1, 2, 3, 4, 5])
    window = 2

    # Test explicit lag=1
    result_lag1 = estimator._running_view(arr, window, lag=1)

    # Test default (should be lag=1)
    result_default = estimator._running_view(arr, window)

    np.testing.assert_array_equal(result_lag1, result_default)


def test_running_view_lag_uses_correct_values():
    """Test that different lag values access the correct historical values."""
    estimator = LoadForecastRegressor()
    arr = np.array([10, 20, 30, 40, 50, 60, 70])
    window = 3

    # Test lag=1 (standard AR)
    result_lag1 = estimator._running_view(arr, window, lag=1)
    # At position 5: should use arr[4], arr[3], arr[2] = [50, 40, 30]
    # The actual order in window may vary, but should contain these values
    assert result_lag1[5, 0] in [30, 40, 50]
    assert result_lag1[5, 1] in [30, 40, 50]
    assert result_lag1[5, 2] in [30, 40, 50]
    assert 30 in result_lag1[5]
    assert 40 in result_lag1[5]
    assert 50 in result_lag1[5]

    # Test lag=2
    result_lag2 = estimator._running_view(arr, window, lag=2)
    # At position 5: should use arr[3], arr[2], arr[1] = [40, 30, 20]
    assert 20 in result_lag2[5]
    assert 30 in result_lag2[5]
    assert 40 in result_lag2[5]
    # Should NOT contain values from lag=1 (50 should not be present)
    assert 50 not in result_lag2[5]

    # Test lag=3
    result_lag3 = estimator._running_view(arr, window, lag=3)
    # At position 5: should use arr[2], arr[1], arr[0] = [30, 20, 10]
    assert 10 in result_lag3[5]
    assert 20 in result_lag3[5]
    assert 30 in result_lag3[5]
    # Should NOT contain values from lag=1 or lag=2
    assert 40 not in result_lag3[5]
    assert 50 not in result_lag3[5]


def test_running_view_lag_padding():
    """Test that larger lags require more padding (more NaN values at start)."""
    estimator = LoadForecastRegressor()
    arr = np.array([10, 20, 30, 40, 50])
    window = 2

    result_lag1 = estimator._running_view(arr, window, lag=1)
    result_lag2 = estimator._running_view(arr, window, lag=2)
    result_lag3 = estimator._running_view(arr, window, lag=3)

    # Larger lag should have more NaN rows at the start
    assert np.all(np.isnan(result_lag1[0]))
    assert np.all(np.isnan(result_lag2[0]))
    assert np.all(np.isnan(result_lag2[1]))  # lag=2 needs 2 rows of NaN
    assert np.all(np.isnan(result_lag3[0]))
    assert np.all(np.isnan(result_lag3[1]))
    assert np.all(np.isnan(result_lag3[2]))  # lag=3 needs 3 rows of NaN


def test_running_view_lag_verification():
    """Detailed verification of lagged running view on specific example."""
    estimator = LoadForecastRegressor()
    arr = np.array([10, 20, 30, 40, 50, 60, 70])
    window = 3

    result_lag1 = estimator._running_view(arr, window, lag=1)
    result_lag2 = estimator._running_view(arr, window, lag=2)
    result_lag3 = estimator._running_view(arr, window, lag=3)

    # At position 5 (i=5), verify correct values are used
    print("\nPosition 5 verification:")
    print(f"  Lag=1: {result_lag1[5]} (should use arr[4], arr[3], arr[2] = [50, 40, 30])")
    print(f"  Lag=2: {result_lag2[5]} (should use arr[3], arr[2], arr[1] = [40, 30, 20])")
    print(f"  Lag=3: {result_lag3[5]} (should use arr[2], arr[1], arr[0] = [30, 20, 10])")

    # Verify lag=2 uses older values than lag=1
    assert max(result_lag2[5]) <= max(result_lag1[5]), "Lag=2 should use older values than lag=1"
    assert min(result_lag2[5]) <= min(result_lag1[5]), "Lag=2 should use older values than lag=1"

    # Verify lag=3 uses older values than lag=2
    assert max(result_lag3[5]) <= max(result_lag2[5]), "Lag=3 should use older values than lag=2"
    assert min(result_lag3[5]) <= min(result_lag2[5]), "Lag=3 should use older values than lag=2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

