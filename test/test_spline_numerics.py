from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from signaldecomp import make_offset_basis, offset_source_mask

from tsgam_estimator import (
    TsgamEstimator, TsgamEstimatorConfig, TsgamForecastConfig,
    TsgamForecastCouplingConfig, TsgamForecastEstimator, TsgamLinearConfig,
    TsgamSolverConfig, TsgamSplineConfig,
)


def numerical_case(solver):
    rng = np.random.default_rng(57)
    x = rng.uniform(-2, 2, 220)
    z = rng.normal(size=len(x))
    X = pd.DataFrame({"x": x, "z": z}, index=pd.date_range("2024", periods=len(x), freq="1h"))
    y = 0.7 + np.sin(x) + 0.3 * np.sin(np.roll(x, 2)) + 0.2 * z
    config = TsgamEstimatorConfig(None, [
        TsgamSplineConfig(knots=[-2, -1, 0, 1, 2], lags=[-2, 0], reg_weight=0.02, diff_reg_weight=0.03),
        TsgamLinearConfig(lags=[-3, 0], reg_weight=0.01),
    ], solver_config=TsgamSolverConfig(
        solver=solver, verbose=False, solver_opts={"eps": 1e-7} if solver == "SCS" else {},
    ))
    return X, y, config


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
def test_whitening_preserves_weighted_gapped_original_problem(solver):
    X, y, config = numerical_case(solver)
    keep = np.arange(len(X)) != 37
    weight = np.linspace(0, 2, len(X))
    models = []
    for whiten in [False, True]:
        cfg = deepcopy(config)
        cfg.exog_config[0].whiten = whiten
        models.append(TsgamEstimator(cfg).fit(X.loc[keep], y[keep], sample_weight=weight[keep]))
    raw, white = models
    mask = white.decomposition_["fit_mask"]
    drivers = X.to_numpy().copy()
    drivers[37] = np.nan
    expected = np.isfinite(y)
    expected[37] = False
    expected &= make_offset_basis(drivers[:, 0], (2, 0)).valid_mask
    expected &= make_offset_basis(drivers[:, 1], (3, 0)).valid_mask
    np.testing.assert_array_equal(mask, expected)
    metadata = white.decomposition_["component_metadata"]["exog_0"]
    np.testing.assert_array_equal(metadata["support_mask"], offset_source_mask(mask, (2, 0)))
    assert metadata["whitening"].training_gram_error < 1e-10
    assert metadata["support_diagnostics"].n_fit == metadata["support_mask"].sum()
    for role in ["intercept", "exog_0", "exog_1", "exog_0_coef", "exog_1_beta"]:
        np.testing.assert_allclose(raw.decomposition_["values"][role], white.decomposition_["values"][role], atol=2e-5)
    np.testing.assert_allclose(raw.problem_.value, white.problem_.value, rtol=1e-6)
    np.testing.assert_allclose(raw.predict(X), white.predict(X), atol=2e-5)
    np.testing.assert_allclose(raw.variables_["exog_coef_0"].value, white.variables_["exog_coef_0"].value, atol=2e-5)


def test_whitening_fails_closed_on_rank_deficient_spline():
    X, y, config = numerical_case("CLARABEL")
    X["x"] = 0.5
    config.exog_config[0].whiten = True
    with pytest.raises(ValueError, match="rank"):
        TsgamEstimator(config).fit(X, y)


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
def test_coupling_uses_original_coefficients_not_numerical_coordinates(solver):
    X, y, config = numerical_case(solver)
    models = []
    for whiten in [False, True]:
        cfg = deepcopy(config)
        cfg.exog_config[0].whiten = whiten
        models.append(TsgamForecastEstimator(TsgamForecastConfig(
            horizon=3, mode="coupled", base_config=cfg,
            coupling_config=TsgamForecastCouplingConfig(roughness_weight=0.3),
        )).fit(X, y))
    raw, white = models
    np.testing.assert_allclose(raw.problem_.value, white.problem_.value, rtol=1e-6)
    np.testing.assert_allclose(raw.predict(X), white.predict(X), atol=3e-5)
    for raw_values, white_values in zip(raw.horizon_values_, white.horizon_values_, strict=True):
        assert not any("numerical" in name for name in white_values)
        for role in raw_values:
            np.testing.assert_allclose(raw_values[role], white_values[role], atol=3e-5)
    assert all(meta["exog_0"]["whitening"].training_gram_error < 1e-10 for meta in white.horizon_component_metadata_)
