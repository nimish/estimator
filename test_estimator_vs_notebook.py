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


def test_baseline_coefficients_match_notebook(fitted_estimator, notebook_coefficients):
    """Test that baseline coefficients match notebook (with absolute and relative tolerance)."""
    estimator, _, _, _ = fitted_estimator

    notebook_time_coef = notebook_coefficients['time_coef']
    notebook_temp_coef = notebook_coefficients['temp_coef']

    assert estimator.time_coef_.shape == notebook_time_coef.shape, \
        f"Time coefficients shape {estimator.time_coef_.shape} != notebook {notebook_time_coef.shape}"

    assert estimator.exog_coef_.shape == notebook_temp_coef.shape, \
        f"Temperature coefficients shape {estimator.exog_coef_.shape} != notebook {notebook_temp_coef.shape}"

    np.testing.assert_allclose(
        estimator.time_coef_,
        notebook_time_coef,
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
    time_abs_diff = np.abs(estimator.time_coef_ - notebook_time_coef)
    time_rel_diff = np.abs((estimator.time_coef_ - notebook_time_coef) / (np.abs(notebook_time_coef) + 1e-12))
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
    notebook_temp_coef = notebook_coefficients['temp_coef']
    notebook_ar_coef = notebook_coefficients['ar_coef']
    notebook_ar_intercept = notebook_coefficients['ar_intercept']

    np.testing.assert_allclose(
        estimator.time_coef_,
        notebook_time_coef,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

