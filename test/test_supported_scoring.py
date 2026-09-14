import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression

from tsgam_estimator import (
    TsgamEstimator, TsgamEstimatorConfig, TsgamLinearConfig,
    TsgamForecastConfig, TsgamForecastEstimator,
    make_supported_scorer,
)


def lagged_data():
    x = np.random.default_rng(42).normal(size=100)
    X = pd.DataFrame({"x": x}, index=pd.date_range("2024", periods=100, freq="1h"))
    return X, 1 + 2 * np.roll(x, 2)


def lagged_model():
    return TsgamEstimator(TsgamEstimatorConfig(None, [TsgamLinearConfig(lags=[-2])]))


def test_score_masks_predictions_without_recomputing_lags():
    X, y = lagged_data()
    model = lagged_model().fit(X, y)
    prediction = model.predict(X)
    assert np.isnan(prediction[:2]).all()
    weights = np.linspace(1, 3, len(y))
    np.testing.assert_allclose(model.score(X, y, weights), r2_score(
        y[2:], prediction[2:], sample_weight=weights[2:],
    ))
    scorer = make_supported_scorer("neg_mean_squared_error")
    np.testing.assert_allclose(scorer(model, X, y, sample_weight=weights), -mean_squared_error(
        y[2:], prediction[2:], sample_weight=weights[2:],
    ))


@pytest.mark.parametrize("scoring", [None, make_supported_scorer("neg_root_mean_squared_error")])
def test_grid_search_scores_lagged_validation_windows(scoring):
    X, y = lagged_data()
    search = GridSearchCV(
        lagged_model(), {"config__exog_config__0__reg_weight": [0.01, 0.1]},
        scoring=scoring, cv=TimeSeriesSplit(3), error_score="raise",
    ).fit(X, y)
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert np.isnan(search.best_estimator_.predict(X)[:2]).all()


def test_score_rejects_unsupported_window_and_invalid_targets():
    X, y = lagged_data()
    model = TsgamEstimator(TsgamEstimatorConfig(None, [TsgamLinearConfig(lags=[-4])])).fit(X, y)
    with pytest.raises(ValueError, match="No supported"):
        model.score(X.iloc[:4], y[:4])
    y[3] = np.nan
    with pytest.raises(ValueError, match="Targets must be finite"):
        model.score(X, y)


def test_score_rejects_zero_supported_weight():
    X, y = lagged_data()
    model = lagged_model().fit(X, y)
    with pytest.raises(ValueError, match="positive total"):
        model.score(X, y, np.r_[1., 1., np.zeros(98)])


@pytest.mark.parametrize("column_target", [False, True])
@pytest.mark.parametrize("timestamp_column", [False, True])
def test_scoring_aligns_shuffled_targets_and_weights(column_target, timestamp_column):
    X, y = lagged_data()
    model = lagged_model().fit(X, y)
    weights = np.linspace(0.1, 3, len(y))
    prediction = model.predict(X)
    expected = r2_score(y[2:], prediction[2:], sample_weight=weights[2:])
    order = np.random.default_rng(9).permutation(len(y))
    if timestamp_column:
        X = X.reset_index(names="timestamp")
    target = y[:, None] if column_target else y
    for scorer in (lambda est, x, y, sample_weight: est.score(x, y, sample_weight), make_supported_scorer("r2")):
        actual = scorer(model, X.iloc[order], target[order], sample_weight=weights[order])
        np.testing.assert_allclose(actual, expected)


def test_scorer_preserves_unsorted_input_rejection():
    X, y = lagged_data()
    model = lagged_model()
    model.config.sort_index = False
    model.fit(X, y)
    with pytest.raises(ValueError, match="not sorted"):
        make_supported_scorer("r2")(model, X.iloc[::-1], y[::-1])


def test_generic_scorer_does_not_reorder_other_regressors():
    X, y = lagged_data()
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(
        make_supported_scorer("r2")(model, X.iloc[::-1], y[::-1, None]),
        model.score(X, y),
    )


def test_scorer_still_rejects_incompatible_multioutput_targets():
    X, y = lagged_data()
    model = lagged_model().fit(X, y)
    with pytest.raises(ValueError, match="matching"):
        model.score(X, np.column_stack([y, y]))


@pytest.mark.parametrize("mode", ["independent", "coupled"])
def test_forecast_scorer_aligns_multioutput_targets(mode):
    X, y = lagged_data()
    model = TsgamForecastEstimator(TsgamForecastConfig(
        horizon=2, mode=mode, base_config=lagged_model().config,
    )).fit(X, y)
    targets = np.column_stack([np.roll(y, -int(h)) for h in model.horizons_])
    weights = np.linspace(0.2, 2, len(y))
    predicted = model.predict(X).to_numpy()
    supported = np.isfinite(predicted).all(axis=1)
    expected = r2_score(targets[supported], predicted[supported], sample_weight=weights[supported])
    actual = make_supported_scorer("r2")(
        model, X.iloc[::-1], targets[::-1], sample_weight=weights[::-1],
    )
    np.testing.assert_allclose(actual, expected)
