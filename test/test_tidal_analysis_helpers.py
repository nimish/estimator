from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from spcqe import make_basis_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from tidal_analysis_helpers import (  # noqa: E402
    build_day_hour_matrix,
    compute_lagged_correlation,
    compute_periodogram,
    extract_fourier_components,
    infer_samples_per_hour,
)


def test_infer_samples_per_hour_matches_index_spacing():
    hourly_index = pd.date_range("2024-01-01", periods=5, freq="1h")
    six_min_index = pd.date_range("2024-01-01", periods=10, freq="6min")

    assert infer_samples_per_hour(hourly_index) == 1
    assert infer_samples_per_hour(six_min_index) == 10


def test_build_day_hour_matrix_averages_subhourly_values():
    day_1 = pd.date_range("2024-01-01", periods=20, freq="6min")
    day_2 = pd.date_range("2024-01-02", periods=20, freq="6min")
    index = day_1.append(day_2)
    values = np.concatenate([
        np.repeat(1.0, 10),
        np.repeat(3.0, 10),
        np.repeat(2.0, 10),
        np.repeat(4.0, 10),
    ])

    matrix = build_day_hour_matrix(index, values)

    assert matrix.shape == (2, 24)
    assert matrix.loc[pd.Timestamp("2024-01-01"), 0] == 1.0
    assert matrix.loc[pd.Timestamp("2024-01-01"), 1] == 3.0
    assert matrix.loc[pd.Timestamp("2024-01-02"), 0] == 2.0
    assert matrix.loc[pd.Timestamp("2024-01-02"), 1] == 4.0


def test_compute_periodogram_identifies_dominant_period():
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="1h")
    t = np.arange(len(index))
    values = np.sin(2 * np.pi * t / 12.0)

    spectrum = compute_periodogram(index, values, min_period_hours=4.0, max_period_hours=24.0)

    top_period = spectrum.sort_values("power", ascending=False).iloc[0]["period_hours"]
    assert abs(top_period - 12.0) < 0.5


def test_compute_lagged_correlation_finds_feature_lead():
    rng = np.random.default_rng(42)
    base = rng.standard_normal(100)
    target = np.roll(base, 3)

    correlations = compute_lagged_correlation(target, base, max_lag=6)
    best_lag = correlations.loc[correlations["correlation"].abs().idxmax(), "lag"]

    assert best_lag == 3


def test_extract_fourier_components_reconstructs_combined_signal():
    periods = [12.4206, 24.0]
    num_harmonics = [1, 1]
    time_indices = np.arange(0, 48)
    coefs = np.array([1.5, -0.2, 0.7, 0.1, 0.3, -0.4, 0.2, 0.6])

    estimator = SimpleNamespace(
        config=SimpleNamespace(
            multi_periodic_config=SimpleNamespace(
                num_harmonics=num_harmonics,
                periods=periods,
            ),
        ),
        time_indices_=time_indices,
        variables_={"fourier_coef": SimpleNamespace(value=coefs)},
    )

    components = extract_fourier_components(
        estimator,
        labels=["semi", "diurnal"],
    )

    expected = make_basis_matrix(
        num_harmonics=num_harmonics,
        length=time_indices.max() + 1,
        periods=periods,
    )[time_indices, 1:] @ coefs

    np.testing.assert_allclose(components["combined"], expected)
    np.testing.assert_allclose(
        components["combined"],
        components["diurnal"] + components["semi"] + components["diurnal_x_semi"],
    )
