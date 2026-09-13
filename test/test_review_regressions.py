import numpy as np
import pandas as pd
import pytest

from tsgam_estimator import (
    TrendType, TsgamEstimator, TsgamEstimatorConfig, TsgamForecastConfig,
    TsgamForecastEstimator, TsgamSplineConfig, TsgamTrendConfig,
)


@pytest.mark.parametrize("mode", ["ordinary", "independent", "coupled"])
@pytest.mark.parametrize("sort_index", [False, True])
def test_duplicate_timestamps_fail_before_grid_expansion(mode, sort_index):
    X = pd.DataFrame(index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]))
    cfg = TsgamEstimatorConfig(None, None, sort_index=sort_index)
    model = TsgamEstimator(cfg) if mode == "ordinary" else TsgamForecastEstimator(
        TsgamForecastConfig(horizon=1, base_config=cfg, mode=mode)
    )
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        model.fit(X, np.array([0., 10., 0., 0.]), sample_weight=np.array([1., 2., 3., 3.]))


@pytest.mark.parametrize("kind", list(TrendType))
def test_string_and_enum_trends_match(kind):
    X = pd.DataFrame(index=pd.date_range("2024", periods=48, freq="1h"))
    y = np.linspace(2, 3, len(X))
    fitted = []
    for value in (str(kind), kind):
        trend = TsgamTrendConfig(trend_type=value, grouping=12)
        assert trend.trend_type is kind
        model = TsgamEstimator(TsgamEstimatorConfig(None, None, trend_config=trend)).fit(X, y)
        fitted.append(model.predict(X))
    np.testing.assert_array_equal(*fitted)


@pytest.mark.parametrize("kwargs", [{"n_knots": 2}, {"knots": [0, 1]}, {"knots": np.array([0, 1])}])
def test_two_knot_splines_explain_linear_migration(kwargs):
    with pytest.raises(ValueError, match="TsgamLinearConfig"):
        TsgamSplineConfig(**kwargs)


def test_invalid_trend_string_fails_at_config_boundary():
    with pytest.raises(ValueError, match="TrendType"):
        TsgamTrendConfig(trend_type="not-a-trend")
