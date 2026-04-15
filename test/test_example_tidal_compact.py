from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from example_tidal import TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS  # noqa: E402
from example_tidal_compact import (  # noqa: E402
    build_diagnostic_figures,
    build_periodogram_selector_options,
    build_periodogram_figure,
    build_shapley_figure,
)


def test_build_periodogram_figure_marks_named_constituents():
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="1h")
    time_steps = np.arange(len(index))
    values = np.sin(2 * np.pi * time_steps / 12.42)

    figure = build_periodogram_figure(index, values, title="Exploration spectrum")

    assert len(figure.data) == 1
    assert figure.data[0].mode == "lines"
    assert any(abs(shape.x0 - PERIODS["M2"]) < 1.0e-6 for shape in figure.layout.shapes)


def test_build_diagnostic_figures_includes_residual_spectrum():
    index = pd.date_range("2024-01-01", periods=24 * 14, freq="1h")
    time_steps = np.arange(len(index))
    observed = np.sin(2 * np.pi * time_steps / 12.42)
    predicted = observed - 0.1 * np.sin(2 * np.pi * time_steps / 24.0)
    residuals = observed - predicted

    fit_result = {
        "metrics_train": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "metrics_test": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "te_index": index,
        "te_obs": observed,
        "te_pred": predicted,
        "te_obs_clean": observed,
        "te_pred_clean": predicted,
        "residuals": residuals,
        "picked": {"M2": (PERIODS["M2"], 1)},
        "active_regs": [],
        "n_train": len(index),
        "n_test": len(index),
        "sph": 1,
    }
    df = pd.DataFrame({"water_level": observed}, index=index)

    figures = build_diagnostic_figures(df, fit_result)

    assert "Residual spectrum" in figures
    assert isinstance(figures["Residual spectrum"], go.Figure)


def test_build_periodogram_selector_options_returns_valid_default_name():
    df = pd.DataFrame(
        {
            "water_level": [1.0, 2.0, 3.0],
            "pressure": [1010.0, 1011.0, 1012.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    options, default_name = build_periodogram_selector_options(df)

    assert default_name in options
    assert options[default_name] == "water_level"


def test_build_shapley_figure_uses_explicit_r2_baseline():
    shapley_result = {
        "components": ["M2", "pressure"],
        "coalitions": 4,
        "failed": 0,
        "baseline_r2": -0.25,
        "baseline_rmse": 1.2,
        "full_r2": 0.4,
        "full_rmse": 0.6,
        "shap_r2": {"M2": 0.45, "pressure": 0.2},
        "shap_rmse": {"M2": -0.4, "pressure": -0.2},
    }

    figure = build_shapley_figure(shapley_result)

    assert figure.data[0]["y"][0] == -0.25
