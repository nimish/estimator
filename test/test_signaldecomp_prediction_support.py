import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats
from signaldecomp.basis import make_basis_matrix
from signaldecomp import components_to_frame

from tsgam_estimator._design import _make_fourier_basis

from tsgam_estimator import (
    TsgamArConfig, TsgamEstimator, TsgamEstimatorConfig,
    TsgamLinearConfig, TsgamSolverConfig,
    TsgamMultiPeriodicConfig,
    TsgamSplineConfig, TsgamForecastEstimator, TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamTrendConfig, TsgamOutlierConfig, TrendType,
)


def test_native_decomposition_reconstructs_weighted_gapped_signal():
    rng = np.random.default_rng(64)
    index = pd.date_range("2024", periods=120, freq="6min")
    x = rng.normal(size=len(index))
    y = 2 + 0.5 * x + np.sin(np.arange(len(index)) / 5) + rng.normal(scale=0.1, size=len(index))
    keep = np.arange(len(index)) != 30
    model = TsgamEstimator(TsgamEstimatorConfig(
        TsgamMultiPeriodicConfig(periods=[30], num_harmonics=[1]),
        [TsgamLinearConfig(lags=[-1, 0])],
        trend_config=TsgamTrendConfig(trend_type=TrendType.LINEAR, grouping=10),
        outlier_config=TsgamOutlierConfig(period_hours=10, reg_weight=0.1),
        solver_config=TsgamSolverConfig(verbose=False),
    )).fit(pd.DataFrame({"x": x[keep]}, index=index[keep]), y[keep],
           sample_weight=rng.uniform(0.1, 2, keep.sum()))
    out = model.decomposition_
    mask = out["fit_mask"]
    assert out["problem"].is_dcp()
    assert not mask[[0, 30, 31]].any()
    frame = components_to_frame(out, index=index, mask=mask)
    np.testing.assert_allclose(
        (frame.reconstruction + frame.residual)[mask], y[mask], atol=1e-6,
    )
    assert frame.loc[~mask].isna().all().all()


@pytest.mark.parametrize("status", ["infeasible_inaccurate", "unbounded_inaccurate", "user_limit"])
def test_ar_nonoptimal_status_does_not_publish_coefficients(monkeypatch, status):
    import cvxpy

    model = TsgamEstimator(TsgamEstimatorConfig(None, None, ar_config=TsgamArConfig([1])))
    model.decomposition_ = {
        "fit_mask": np.ones(10, dtype=bool),
        "values": {"residual": np.linspace(-1, 1, 10)},
    }
    monkeypatch.setattr(cvxpy.Problem, "solve", lambda self, **kwargs: setattr(self, "_status", status))
    model._fit_ar_model()
    assert model.ar_coef_ is None
    assert model.ar_noise_scale_ is None


def test_coupled_prediction_reuses_bases_and_detaches_solver_values(monkeypatch):
    import tsgam_estimator._design as design_module
    import tsgam_estimator._problem as problem_module

    def unused_interaction_basis(*args, **kwargs):
        raise AssertionError("No interaction basis is needed")

    monkeypatch.setattr(problem_module, "make_spline_basis", unused_interaction_basis)
    X = pd.DataFrame(
        {"x": np.sin(np.arange(80) / 3)},
        index=pd.date_range("2024", periods=80, freq="1h"),
    )
    model = TsgamForecastEstimator(TsgamForecastConfig(
        horizon=3, mode="coupled",
        coupling_config=TsgamForecastCouplingConfig(roughness_weight=1),
        base_config=TsgamEstimatorConfig(
            None, [TsgamSplineConfig(n_knots=5)],
            solver_config=TsgamSolverConfig(verbose=False),
        ),
    )).fit(X, X.x.to_numpy() ** 2)
    expected = model.predict(X.iloc[-10:])
    calls = []
    original = design_module._process_exog_config

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(design_module, "_process_exog_config", counted)
    for variable in model.problem_.variables():
        variable.value = None
    np.testing.assert_array_equal(model.predict(X.iloc[-10:]), expected)
    assert len(calls) == 1


@pytest.mark.parametrize("start", [0, 37, 87672, -37])
def test_fourier_window_preserves_absolute_phase_and_cross_terms(start):
    periods, harmonics = [167.5, 24], [2, 3]
    config = TsgamEstimatorConfig(
        TsgamMultiPeriodicConfig(periods=periods, num_harmonics=harmonics), None,
    )
    times = np.arange(start, start + 10)
    actual = _make_fourier_basis(config, times)
    if start >= 0:
        expected = make_basis_matrix(harmonics, start + 10, periods)[times, 1:]
    else:
        # cos(-t)=cos(t), sin(-t)=-sin(t), including pairwise products.
        blocks = {}
        for ix, (period, count) in enumerate(zip(periods, harmonics)):
            block = make_basis_matrix(count, abs(start) + 1, period)[-times, 1:]
            block[:, 1::2] *= -1
            blocks[ix] = block
        expected = make_basis_matrix(harmonics, 10, periods, custom_basis=blocks)[:, 1:]
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


def test_batched_ar_samples_preserve_seed_and_filter_state():
    model = TsgamEstimator(TsgamEstimatorConfig(None, None, ar_config=TsgamArConfig([1, 3])))
    model.ar_coef_ = np.array([0.1, 0.6])
    model.ar_intercept_ = 0.2
    model.ar_noise_loc_, model.ar_noise_scale_ = 0.1, 0.3
    baseline = np.linspace(0, 1, 20)
    baseline[0] = np.nan
    rng = np.random.RandomState(12)
    expected = []
    for _ in range(7):
        noise = stats.laplace.rvs(loc=0.1, scale=0.3, size=26, random_state=rng)
        filtered, _ = signal.lfilter(
            [1], [1, -0.6, 0, -0.1], 0.2 + noise[3:], zi=noise[:3][::-1],
        )
        expected.append(baseline + filtered[-20:])
    actual = model._generate_ar_samples(baseline, 7, np.random.RandomState(12))
    np.testing.assert_array_equal(actual, expected)


def test_interaction_prediction_accepts_single_rows_and_constant_drivers():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=['x', 'z'],
                     index=pd.date_range('2024-01-01', periods=80, freq='h'))
    y = 1 + 2 * X.x + 3 * X.z + X.x * X.z
    est = TsgamEstimator(TsgamEstimatorConfig(
        None, [TsgamLinearConfig(reg_weight=0), TsgamLinearConfig(reg_weight=0)],
        interaction_pairs=[(0, 1)], solver_config=TsgamSolverConfig(verbose=False),
    )).fit(X, y)
    for query in (X.iloc[[20]], X.iloc[:10].assign(x=1.0)):
        np.testing.assert_allclose(est.predict(query),
                                   1 + 2 * query.x + 3 * query.z + query.x * query.z,
                                   atol=1e-6)


def test_offset_prediction_requires_padding_and_ar_does_not_bridge_gaps():
    rng = np.random.default_rng(9)
    X = pd.DataFrame({'x': rng.normal(size=80)},
                     index=pd.date_range('2024-01-01', periods=80, freq='h'))
    y = 1 + np.r_[0, 2 * X.x.to_numpy()[:-1]]
    est = TsgamEstimator(TsgamEstimatorConfig(
        None, [TsgamLinearConfig(lags=[-1], reg_weight=0)],
        solver_config=TsgamSolverConfig(verbose=False),
    )).fit(X, y)
    assert np.isnan(est.predict(X.iloc[20:30])[0])
    np.testing.assert_allclose(est.predict(X.iloc[19:30])[1:], y[20:30], atol=1e-6)
    assert np.isfinite(est.predict(X.iloc[20:30], remove_exogenous=True)).all()
    ar = TsgamEstimator(TsgamEstimatorConfig(
        None, None, ar_config=TsgamArConfig([1, 3]), debug=True,
        solver_config=TsgamSolverConfig(verbose=False),
    )).fit(X.drop(X.index[20:40]), rng.normal(size=60))
    assert len(ar._baseline_residuals_) == 80
    assert not ar._ar_valid_mask_[20:43].any()
    assert ar._ar_valid_mask_[43]
    np.testing.assert_allclose(ar._B_running_view_[43],
                               ar._baseline_residuals_[[40, 42]])
