# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for non-hourly data frequencies with both linear and spline exog.

Exercises the full fit/predict/sample pipeline across:
- Frequencies: 15-minute, daily
- Exog types: spline, linear, mixed (spline+linear)
- Operations: fit, predict, sample (AR)

Periods in TsgamMultiPeriodicConfig are specified in frequency-step units
(not hours): for 15-min data daily=96 steps, for daily data weekly=7 steps.
"""

import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TsgamArConfig,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    get_recommended_periods,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _synth(freq: str, n_samples: int, n_exog: int = 1, seed: int = 42):
    """Synthetic time series at *freq* with *n_exog* exogenous columns."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n_samples, freq=freq)
    cols = {f"x{i}": rng.standard_normal(n_samples) for i in range(n_exog)}
    X = pd.DataFrame(cols, index=timestamps)
    y = 3.0 + sum(0.5 * X.iloc[:, i].values for i in range(n_exog)) + rng.normal(0, 0.3, n_samples)
    return X, y


SOLVER = TsgamSolverConfig(solver="CLARABEL", verbose=False)

# Periods are in frequency-step units:
#   15-min  →  daily = 96 steps,  weekly = 672 steps
#   daily   →  weekly = 7 steps,  yearly = 365.2425 steps
_FREQ_TABLE = {
    "15min": {
        "n_samples": 672,       # 7 days of 15-min data
        "expected_freq": "15min",
        "multi": TsgamMultiPeriodicConfig(num_harmonics=[3, 2], periods=[96, 672]),
    },
    "1D": {
        "n_samples": 400,       # ~13 months of daily data
        "expected_freq": "1d",
        "multi": TsgamMultiPeriodicConfig(num_harmonics=[3, 2], periods=[7, 365.2425]),
    },
}


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(params=["15min", "1D"], ids=["15min", "daily"])
def freq_key(request):
    return request.param


@pytest.fixture
def freq_setup(freq_key):
    """(X, y, multi_periodic_config, expected_freq) per frequency."""
    info = _FREQ_TABLE[freq_key]
    X, y = _synth(freq_key, info["n_samples"])
    return X, y, info["multi"], info["expected_freq"]


@pytest.fixture(params=["spline", "linear"], ids=["spline", "linear"])
def exog_cfg(request):
    """Single-element exog_config list."""
    if request.param == "spline":
        return [TsgamSplineConfig(n_knots=5, lags=[0])]
    return [TsgamLinearConfig(lags=[0])]


@pytest.fixture(params=["spline", "linear"], ids=["spline_lag", "linear_lag"])
def exog_cfg_with_lags(request):
    """Single-element exog_config list with lead/lag terms."""
    if request.param == "spline":
        return [TsgamSplineConfig(n_knots=5, lags=[-1, 0, 1])]
    return [TsgamLinearConfig(lags=[-1, 0, 1])]


# ── frequency inference ──────────────────────────────────────────────────


class TestFrequencyInference:
    """Verify freq_ is inferred correctly for each non-hourly frequency."""

    def test_inferred_freq(self, freq_setup):
        X, y, multi, expected_freq = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=None,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        assert est.freq_ == expected_freq

    def test_time_indices_start_at_zero(self, freq_setup):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=None,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        assert est.time_indices_[0] == 0
        assert est.time_indices_[-1] == len(X) - 1


# ── Fourier-only ─────────────────────────────────────────────────────────


class TestFourierOnly:
    """Fourier-only (no exog) fit/predict across non-hourly frequencies."""

    def test_fit_predict(self, freq_setup):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=None,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


# ── single exog: spline or linear ───────────────────────────────────────


class TestExogFitPredict:
    """Fit/predict with single exog across frequencies × exog types."""

    def test_with_fourier(self, freq_setup, exog_cfg):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=exog_cfg,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_exog_only_no_fourier(self, freq_setup, exog_cfg):
        X, y, _, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=exog_cfg,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


# ── exog with lead/lag terms ────────────────────────────────────────────


class TestExogWithLags:
    """Exog with lead/lag terms across frequencies × exog types."""

    def test_with_lags(self, freq_setup, exog_cfg_with_lags):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=exog_cfg_with_lags,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


# ── mixed spline + linear exog ──────────────────────────────────────────


class TestMixedExog:
    """Two exogenous columns: one spline, one linear."""

    def test_spline_and_linear(self, freq_key):
        info = _FREQ_TABLE[freq_key]
        X, y = _synth(freq_key, info["n_samples"], n_exog=2)
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=info["multi"],
            exog_config=[
                TsgamSplineConfig(n_knots=5, lags=[0]),
                TsgamLinearConfig(lags=[0]),
            ],
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_spline_and_linear_with_lags(self, freq_key):
        info = _FREQ_TABLE[freq_key]
        X, y = _synth(freq_key, info["n_samples"], n_exog=2)
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=info["multi"],
            exog_config=[
                TsgamSplineConfig(n_knots=5, lags=[-1, 0, 1]),
                TsgamLinearConfig(lags=[-1, 0, 1]),
            ],
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


def test_interactions_with_non_hourly_data(freq_key):
    info = _FREQ_TABLE[freq_key]
    X, y = _synth(freq_key, info["n_samples"], n_exog=2)
    y = y + 0.8 * (X["x0"] * X["x1"]).to_numpy()

    cfg = TsgamEstimatorConfig(
        multi_periodic_config=info["multi"],
        exog_config=[
            TsgamLinearConfig(lags=[-1, 0, 1]),
            TsgamLinearConfig(lags=[0, 1]),
        ],
        interaction_pairs=[(0, 1)],
        solver_config=SOLVER,
    )
    est = TsgamEstimator(config=cfg)
    est.fit(X, y)
    preds = est.predict(X)

    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))
    assert est.variables_["exog_coef_0"].value.shape == (1, 3)
    assert est.variables_["exog_coef_1"].value.shape == (1, 2)
    assert est.variables_["interaction_coef_0"].value.shape == (1,)


# ── AR model + sample ───────────────────────────────────────────────────


class TestArModel:
    """AR fit + sample across frequencies × exog types."""

    def test_ar_sample(self, freq_setup, exog_cfg):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=exog_cfg,
            ar_config=TsgamArConfig(lags=[1, 2]),
            solver_config=SOLVER,
            random_state=np.random.RandomState(0),
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))
        samples = est.sample(X, n_samples=3)
        assert samples.shape == (3, len(X))
        assert np.all(np.isfinite(samples))


# ── train/test split ────────────────────────────────────────────────────


class TestTrainTestSplit:
    """Predict on held-out data across frequencies × exog types."""

    def test_split_predict(self, freq_setup, exog_cfg):
        X, y, multi, _ = freq_setup
        split = int(0.8 * len(X))
        X_train, y_train = X.iloc[:split], y[:split]
        X_test = X.iloc[split:]
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=exog_cfg,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert np.all(np.isfinite(preds))


# ── coefficient shapes ──────────────────────────────────────────────────


class TestCoefShapes:
    """Coefficient matrix dimensions for spline vs linear across frequencies."""

    def test_spline_coef_shape(self, freq_key):
        info = _FREQ_TABLE[freq_key]
        X, y = _synth(freq_key, info["n_samples"])
        lags = [-1, 0, 1]
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=info["multi"],
            exog_config=[TsgamSplineConfig(n_knots=5, lags=lags)],
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        coef = est.variables_["exog_coef_0"].value
        assert coef.shape[1] == len(lags)
        assert coef.shape[0] > 1

    def test_linear_coef_shape(self, freq_key):
        info = _FREQ_TABLE[freq_key]
        X, y = _synth(freq_key, info["n_samples"])
        lags = [-1, 0, 1]
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=info["multi"],
            exog_config=[TsgamLinearConfig(lags=lags)],
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        coef = est.variables_["exog_coef_0"].value
        assert coef.shape == (1, len(lags))


# ── get_recommended_periods ─────────────────────────────────────────────


class TestGetRecommendedPeriods:
    """Verify get_recommended_periods for non-hourly frequencies."""

    def test_15min_period_count(self):
        X = pd.DataFrame(
            {"x": np.zeros(672)},
            index=pd.date_range("2020-01-01", periods=672, freq="15min"),
        )
        periods = get_recommended_periods(X)
        assert len(periods) == 4  # [1, 4, 96, 672] multiples × 0.25h

    def test_daily_period_count(self):
        X = pd.DataFrame(
            {"x": np.zeros(400)},
            index=pd.date_range("2020-01-01", periods=400, freq="1D"),
        )
        periods = get_recommended_periods(X)
        assert len(periods) == 2  # [7, 365.2425] multiples × 24h

    def test_15min_includes_harmonics(self):
        X = pd.DataFrame(
            {"x": np.zeros(672)},
            index=pd.date_range("2020-01-01", periods=672, freq="15min"),
        )
        periods, harmonics = get_recommended_periods(X, include_harmonics=True)
        assert len(periods) == len(harmonics)
        assert all(h > 0 for h in harmonics)

    def test_daily_includes_harmonics(self):
        X = pd.DataFrame(
            {"x": np.zeros(400)},
            index=pd.date_range("2020-01-01", periods=400, freq="1D"),
        )
        periods, harmonics = get_recommended_periods(X, include_harmonics=True)
        assert len(periods) == len(harmonics)
        assert all(h > 0 for h in harmonics)

    def test_15min_periods_are_positive(self):
        X = pd.DataFrame(
            {"x": np.zeros(672)},
            index=pd.date_range("2020-01-01", periods=672, freq="15min"),
        )
        periods = get_recommended_periods(X)
        assert all(p > 0 for p in periods)

    def test_daily_periods_are_positive(self):
        X = pd.DataFrame(
            {"x": np.zeros(400)},
            index=pd.date_range("2020-01-01", periods=400, freq="1D"),
        )
        periods = get_recommended_periods(X)
        assert all(p > 0 for p in periods)


# ── repeated fit with different frequencies ──────────────────────────────


class TestRepeatedFit:
    """Repeated fit on same frequency data yields consistent results."""

    def test_same_data_same_coefficients(self, freq_setup, exog_cfg):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=exog_cfg,
            solver_config=SOLVER,
        )
        est = TsgamEstimator(config=cfg)
        est.fit(X, y)
        coefs_first = {
            k: np.copy(v.value) for k, v in est.variables_.items() if v.value is not None
        }
        est.fit(X, y)
        for k, v in est.variables_.items():
            if v.value is None:
                continue
            np.testing.assert_allclose(
                coefs_first[k], v.value, rtol=1e-6, atol=1e-6,
                err_msg=f"Coefficient {k} changed on repeated fit",
            )


# ── sample_weight with non-hourly data ──────────────────────────────────


class TestSampleWeight:
    """sample_weight support across non-hourly frequencies."""

    def test_ones_matches_unweighted(self, freq_setup, exog_cfg):
        X, y, multi, _ = freq_setup
        cfg = TsgamEstimatorConfig(
            multi_periodic_config=multi,
            exog_config=exog_cfg,
            solver_config=SOLVER,
        )
        est_no_w = TsgamEstimator(config=cfg)
        est_no_w.fit(X, y)

        est_ones = TsgamEstimator(config=cfg)
        est_ones.fit(X, y, sample_weight=np.ones(len(y)))

        np.testing.assert_allclose(
            est_no_w.predict(X), est_ones.predict(X), rtol=1e-5,
        )
