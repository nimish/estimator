import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
)


def _data_and_config(lags):
    rng = np.random.default_rng(83)
    X = pd.DataFrame(
        rng.uniform(-1, 1, (96, 2)),
        index=pd.date_range("2024", periods=96, freq="1h"),
        columns=["x", "z"],
    )
    y = 2 + X.x.to_numpy() ** 2 + X.z.to_numpy() + X.prod(axis=1).to_numpy()
    config = TsgamEstimatorConfig(
        exog_config=[
            TsgamSplineConfig(knots=np.linspace(-1, 1, 5), lags=lags),
            TsgamLinearConfig(lags=lags),
        ],
        interaction_pairs=[(0, 1)],
        multi_periodic_config=TsgamMultiPeriodicConfig(periods=[24], num_harmonics=[2]),
        solver_config=TsgamSolverConfig(solver="CLARABEL", verbose=False),
    )
    return X, y, config


@pytest.mark.parametrize("lags", [[0], [-1, 0]])
def test_ordinary_legacy_keys_shapes_and_native_aliases(lags):
    X, y, config = _data_and_config(lags)
    config.trend_config = TsgamTrendConfig(trend_type=TrendType.LINEAR, grouping=12)
    config.outlier_config = TsgamOutlierConfig(period_hours=12, reg_weight=0.1)
    model = TsgamEstimator(config).fit(X, y)
    old = model.variables_
    native = model.decomposition_["values"]

    assert model.output_ is model.decomposition_
    assert model.problem_ is model.decomposition_["problem"]
    assert old["constant"].shape == ()
    assert old["exog_coef_0"].shape == (4, len(lags))
    assert old["exog_coef_1"].shape == (1, len(lags))
    assert old["fourier_coef"].shape == (4,)
    assert old["interaction_coef_0"].shape == (4,)
    mapping = {
        "constant": "intercept_group_values",
        "exog_coef_0": "exog_0_coef",
        "exog_coef_1": "exog_1_beta",
        "fourier_coef": "periodic_theta",
        "interaction_coef_0": "interaction_0_coef",
        "trend": "trend_group_values",
        "trend_slope": "trend_slope",
        "outlier": "outlier_group_values",
    }
    assert set(old) == set(mapping)
    problem_variables = {id(v) for v in model.problem_.variables()}
    for key, role in mapping.items():
        assert isinstance(old[key], cp.Expression)
        expected = np.asarray(native[role]).reshape(old[key].shape, order="F")
        np.testing.assert_array_equal(old[key].value, expected)
        assert {id(v) for v in old[key].variables()} <= problem_variables

    # Compatibility attributes are not a second prediction backend.
    expected = model.predict(X)
    del model.variables_, model.output_, model.problem_
    np.testing.assert_array_equal(model.predict(X), expected)


def test_ordinary_refit_replaces_legacy_views_and_aliases():
    X, y, config = _data_and_config([0])
    model = TsgamEstimator(config).fit(X, y)
    previous = model.output_
    model.config = TsgamEstimatorConfig(exog_config=None, multi_periodic_config=None)
    model.fit(X, y + 1)
    assert model.output_ is model.decomposition_
    assert model.output_ is not previous
    assert set(model.variables_) == {"constant"}
    assert model.variables_["constant"].value == pytest.approx(np.mean(y + 1))


@pytest.mark.parametrize("include_nowcast", [False, True])
def test_coupled_legacy_horizon_layout_and_predictions(include_nowcast):
    X, y, config = _data_and_config([-1, 0])
    model = TsgamForecastEstimator(TsgamForecastConfig(
        horizon=2, base_config=config, mode="coupled", include_nowcast=include_nowcast,
        coupling_config=TsgamForecastCouplingConfig(roughness_weight=0.1),
    )).fit(X, y)
    old = model.variables_
    count = len(model.horizons_)
    assert old["constant"].shape == (count,)
    assert len(old["exog_coef_0"]) == count
    assert old["fourier_coef"].shape[1] == count
    assert old["interaction_coef_0"].shape[1] == count
    problem_variables = {id(v) for v in model.problem_.variables()}
    for ix, values in enumerate(model.horizon_values_):
        np.testing.assert_array_equal(old["constant"].value[ix], values["intercept_group_values"][0])
        for exog, role, shape in [(0, "exog_0_coef", (4, 2)), (1, "exog_1_beta", (1, 2))]:
            expression = old[f"exog_coef_{exog}"][ix]
            assert expression.shape == shape
            np.testing.assert_array_equal(expression.value, values[role].reshape(shape, order="F"))
        for key, role in [("fourier_coef", "periodic_theta"), ("interaction_coef_0", "interaction_0_coef")]:
            np.testing.assert_array_equal(old[key].value[:, ix], values[role].reshape(-1))
    for value in old.values():
        for expression in value if isinstance(value, list) else [value]:
            assert {id(v) for v in expression.variables()} <= problem_variables
    expected = model.predict(X.iloc[-12:])
    del model.variables_
    pd.testing.assert_frame_equal(model.predict(X.iloc[-12:]), expected)


def test_independent_children_support_legacy_access_after_switching_modes():
    X, y, config = _data_and_config([0])
    model = TsgamForecastEstimator(TsgamForecastConfig(
        horizon=1, base_config=config, mode="coupled",
    )).fit(X, y)
    model.config = TsgamForecastConfig(horizon=1, base_config=config)
    model.fit(X, y)
    assert not hasattr(model, "variables_")
    assert not hasattr(model, "horizon_values_")
    assert not hasattr(model, "problem_")
    for child in model.forecast_estimators_.values():
        assert child.variables_["exog_coef_0"].value.shape == (4, 1)
        assert child.problem_ is child.output_["problem"]
