from dataclasses import dataclass
from datetime import date
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import example_tidal_compact as tidal_compact  # noqa: E402
from example_tidal import (  # noqa: E402
    TIDAL_COMPONENT_LABELS,
    TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS,
    make_constituent_multi_periodic,
)
from example_tidal_compact import (  # noqa: E402
    build_fit_label,
    build_interaction_index_pairs,
    build_interaction_pair_options,
    build_regressor_basis_figure,
    build_regressor_basis_inputs,
    build_model_date_defaults,
    build_model_window_validation_message,
    build_diagnostic_figures,
    build_exog_design_matrices,
    build_knot_count,
    build_shapley_coalition_plan,
    build_shapley_component_list,
    build_shapley_model_inputs,
    build_shapley_result,
    collect_model_params,
    option_name_for_value,
    build_periodogram_figure,
    build_periodogram_selector_options,
    build_shapley_figure,
    load_station_frame,
)


@dataclass
class _Widget:
    value: Any


def test_build_model_date_defaults_keeps_example_split_when_available():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2022, 1, 1),
        date(2024, 3, 31),
    )

    assert train_start == date(2022, 1, 1)
    assert train_end == date(2024, 1, 1)
    assert test_end == date(2024, 3, 31)


def test_build_model_date_defaults_moves_example_split_off_loaded_start_boundary():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2024, 1, 1),
        date(2024, 3, 31),
    )

    assert train_start == date(2024, 1, 1)
    assert train_end == date(2024, 1, 2)
    assert test_end == date(2024, 3, 31)


def test_build_model_date_defaults_uses_midpoint_when_example_split_is_out_of_range():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2024, 2, 1),
        date(2024, 2, 11),
    )

    assert train_start == date(2024, 2, 1)
    assert train_end == date(2024, 2, 6)
    assert test_end == date(2024, 2, 11)


def test_build_model_date_defaults_uses_second_day_for_two_day_window():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2024, 2, 1),
        date(2024, 2, 2),
    )

    assert train_start == date(2024, 2, 1)
    assert train_end == date(2024, 2, 2)
    assert test_end == date(2024, 2, 2)


def test_build_model_date_defaults_leaves_single_day_window_unsplit():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2024, 2, 1),
        date(2024, 2, 1),
    )

    assert train_start == date(2024, 2, 1)
    assert train_end == date(2024, 2, 1)
    assert test_end == date(2024, 2, 1)


def test_build_model_window_validation_message_accepts_non_empty_train_test_split():
    message = build_model_window_validation_message(
        date(2024, 2, 1),
        date(2024, 2, 3),
        date(2024, 2, 1),
        date(2024, 2, 2),
        date(2024, 2, 3),
    )

    assert message is None


def test_build_model_window_validation_message_rejects_single_day_window():
    message = build_model_window_validation_message(
        date(2024, 2, 1),
        date(2024, 2, 1),
        date(2024, 2, 1),
        date(2024, 2, 1),
        date(2024, 2, 1),
    )

    assert message is not None
    assert "single-day" in message.lower()


def test_build_model_window_validation_message_rejects_empty_training_split():
    message = build_model_window_validation_message(
        date(2024, 2, 1),
        date(2024, 2, 3),
        date(2024, 2, 1),
        date(2024, 2, 1),
        date(2024, 2, 3),
    )

    assert message is not None
    assert "train end" in message.lower()


def test_build_model_window_validation_message_rejects_empty_test_split():
    message = build_model_window_validation_message(
        date(2024, 2, 1),
        date(2024, 2, 3),
        date(2024, 2, 1),
        date(2024, 2, 4),
        date(2024, 2, 4),
    )

    assert message is not None
    assert "test end" in message.lower()


def test_build_periodogram_figure_marks_named_constituents():
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="1h")
    time_steps = np.arange(len(index))
    values = np.sin(2 * np.pi * time_steps / 12.42)

    figure = build_periodogram_figure(index, values, title="Exploration spectrum")

    assert len(figure.data) == 1
    assert figure.data[0].mode == "lines"
    assert any(abs(shape.x0 - PERIODS["M2"]) < 1.0e-6 for shape in figure.layout.shapes)


def test_make_constituent_multi_periodic_includes_p1_q1_and_msf_candidates():
    config = make_constituent_multi_periodic(1.0)

    assert {"P1", "Q1", "Msf"} <= set(PERIODS)

    periods_by_label = dict(zip(TIDAL_COMPONENT_LABELS, config.periods, strict=True))
    harmonics_by_label = dict(zip(TIDAL_COMPONENT_LABELS, config.num_harmonics, strict=True))

    assert periods_by_label["P1"] == pytest.approx(PERIODS["P1"])
    assert periods_by_label["Q1"] == pytest.approx(PERIODS["Q1"])
    assert periods_by_label["Msf"] == pytest.approx(PERIODS["Msf"])
    assert harmonics_by_label["P1"] == 1
    assert harmonics_by_label["Q1"] == 1
    assert harmonics_by_label["Msf"] == 1


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


def test_build_diagnostic_figures_adds_train_pred_vs_obs_and_typical_cycles():
    index = pd.date_range("2024-01-01", periods=24 * 14, freq="1h")
    time_steps = np.arange(len(index))
    observed = np.sin(2 * np.pi * time_steps / 12.42)
    predicted = observed - 0.1 * np.sin(2 * np.pi * time_steps / 24.0)
    residuals = observed - predicted

    fit_result = {
        "metrics_train": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "metrics_test": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "tr_index": index,
        "tr_obs": observed,
        "tr_pred": predicted,
        "tr_obs_clean": observed,
        "tr_pred_clean": predicted,
        "tr_mape_n": len(index),
        "te_index": index,
        "te_obs": observed,
        "te_pred": predicted,
        "te_obs_clean": observed,
        "te_pred_clean": predicted,
        "te_mape_n": len(index),
        "residuals": residuals,
        "picked": {"M2": (PERIODS["M2"], 1)},
        "active_regs": [],
        "n_train": len(index),
        "n_test": len(index),
        "sph": 1,
    }
    df = pd.DataFrame({"water_level": observed}, index=index)

    figures = build_diagnostic_figures(df, fit_result)

    assert "Pred vs Obs (Train)" in figures
    assert "Typical day" in figures
    assert "Typical week" in figures
    assert "Typical month" in figures


def test_build_metrics_table_html_mentions_mape_reference_and_counts():
    fit_result = {
        "metrics_train": {"rmse": 0.1, "mae": 0.1, "mape": 1.5, "r2": 0.9},
        "metrics_test": {"rmse": 0.2, "mae": 0.2, "mape": 2.5, "r2": 0.8},
        "tr_index": pd.date_range("2024-01-01", periods=3, freq="1h"),
        "tr_obs": np.array([0.0, 0.5, 1.0]),
        "tr_pred": np.array([0.0, 0.4, 0.9]),
        "tr_obs_clean": np.array([0.0, 0.5, 1.0]),
        "tr_pred_clean": np.array([0.0, 0.4, 0.9]),
        "tr_mape_n": 2,
        "te_index": pd.date_range("2024-01-02", periods=3, freq="1h"),
        "te_obs": np.array([0.0, 0.5, 1.0]),
        "te_pred": np.array([0.0, 0.4, 0.9]),
        "te_obs_clean": np.array([0.0, 0.5, 1.0]),
        "te_pred_clean": np.array([0.0, 0.4, 0.9]),
        "te_mape_n": 2,
        "residuals": np.array([0.0, 0.1, 0.1]),
        "picked": {},
        "active_regs": [],
        "n_train": 3,
        "n_test": 3,
        "sph": 1,
    }

    html = tidal_compact.build_metrics_table_html(fit_result)

    assert "|obs| &gt; 0.01 m" in html
    assert "Train MAPE n" in html
    assert "Test MAPE n" in html


def test_build_train_fit_timeseries_figure_uses_training_window():
    builder = getattr(tidal_compact, "build_train_fit_timeseries_figure", None)

    assert callable(builder)

    fit_result = {
        "metrics_train": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "metrics_test": {"rmse": 0.2, "mae": 0.2, "mape": 2.0, "r2": 0.8},
        "tr_index": pd.date_range("2024-01-01", periods=3, freq="1h"),
        "tr_obs": np.array([0.0, 0.5, 1.0]),
        "tr_pred": np.array([0.1, 0.4, 0.9]),
        "tr_obs_clean": np.array([0.0, 0.5, 1.0]),
        "tr_pred_clean": np.array([0.1, 0.4, 0.9]),
        "tr_mape_n": 2,
        "te_index": pd.date_range("2024-01-02", periods=3, freq="1h"),
        "te_obs": np.array([1.0, 1.5, 2.0]),
        "te_pred": np.array([1.1, 1.4, 1.9]),
        "te_obs_clean": np.array([1.0, 1.5, 2.0]),
        "te_pred_clean": np.array([1.1, 1.4, 1.9]),
        "te_mape_n": 3,
        "residuals": np.array([-0.1, 0.1, 0.1]),
        "picked": {},
        "active_regs": [],
        "active_interactions": [],
        "x_train_fit": pd.DataFrame(index=pd.date_range("2024-01-01", periods=3, freq="1h")),
        "model": None,
        "exog_config": None,
        "n_train": 3,
        "n_test": 3,
        "sph": 1,
    }

    figure = builder("demo", fit_result)

    np.testing.assert_allclose(figure.data[0]["y"], fit_result["tr_obs"])
    np.testing.assert_allclose(figure.data[1]["y"], fit_result["tr_pred"])


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


def test_harmonic_slider_max_is_32():
    assert tidal_compact.HARMONIC_SLIDER_MAX == 32


def test_regressor_response_uses_separate_selector_creation_and_render_cells():
    source = inspect.getsource(tidal_compact)

    assert "return (regressor_response_lag_selectors,)" in source
    assert "def _(fit_result, regressor_response_lag_selectors):" in source


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


def test_build_shapley_component_list_appends_surviving_interactions():
    component_mask = {"M2": True, "S2": False, "pressure": True, "wind_u": True, "air_temp": True}

    components, interaction_lookup = build_shapley_component_list(
        component_mask,
        ["pressure", "wind_u"],
        [("pressure", "wind_u"), ("pressure", "air_temp")],
        {"pressure": (-2, 0), "wind_u": (-1, 0), "air_temp": (0, 0)},
    )

    assert components == ["M2", "pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    assert interaction_lookup == {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}


def test_build_shapley_component_list_drops_interactions_without_zero_lag():
    component_mask = {"pressure": True, "wind_u": True}

    components, interaction_lookup = build_shapley_component_list(
        component_mask,
        ["pressure", "wind_u"],
        [("pressure", "wind_u")],
        {"pressure": (-2, -1), "wind_u": (-1, 0)},
    )

    assert components == ["pressure", "wind_u"]
    assert interaction_lookup == {}


def test_build_shapley_coalition_plan_drops_invalid_interaction_bits():
    components = ["pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    interaction_lookup = {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}

    raw_to_canonical = build_shapley_coalition_plan(components, interaction_lookup)

    assert raw_to_canonical[0b100] == 0b000
    assert raw_to_canonical[0b101] == 0b001
    assert raw_to_canonical[0b110] == 0b010
    assert raw_to_canonical[0b111] == 0b111
    assert len(set(raw_to_canonical.values())) == 5


def test_build_shapley_model_inputs_activates_only_selected_interactions():
    components = ["M2", "pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    interaction_lookup = {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}

    component_mask, interaction_pairs = build_shapley_model_inputs(
        0b1111,
        components,
        {"M2": True, "S2": False, "pressure": True, "wind_u": True, "air_temp": False},
        interaction_lookup,
    )

    assert component_mask == {"M2": True, "S2": False, "pressure": True, "wind_u": True, "air_temp": False}
    assert interaction_pairs == [("pressure", "wind_u")]


def test_build_shapley_result_deduplicates_invalid_interaction_runs(monkeypatch):
    calls: list[tuple[dict[str, bool], tuple[tuple[str, str], ...]]] = []
    progress_ticks: list[str] = []

    def fake_run_tidal_model(component_mask, **model_kwargs):
        interaction_pairs = tuple(model_kwargs["interaction_pairs"])
        calls.append((component_mask.copy(), interaction_pairs))
        reg_score = int(component_mask.get("pressure", False)) + int(component_mask.get("wind_u", False))
        interaction_score = len(interaction_pairs)
        return {
            "metrics_test": {"r2": float(reg_score + interaction_score), "rmse": float(10 - reg_score - interaction_score)},
            "picked": {},
            "active_regs": [name for name in ["pressure", "wind_u"] if component_mask.get(name, False)],
            "active_interactions": ["Pressure (hPa) × Wind U (m/s)"] if interaction_pairs else [],
        }

    monkeypatch.setattr(tidal_compact, "run_tidal_model", fake_run_tidal_model)

    components = ["pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    interaction_lookup = {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}
    raw_to_canonical = tidal_compact.build_shapley_coalition_plan(components, interaction_lookup)

    shapley_result = build_shapley_result(
        {"pressure": True, "wind_u": True},
        components=components,
        interaction_lookup=interaction_lookup,
        raw_to_canonical=raw_to_canonical,
        df=pd.DataFrame({"water_level": [0.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h")),
        sph=1,
        harmonic_orders={},
        lag_ranges={},
        knot_presets={},
        interaction_pairs=[("pressure", "wind_u")],
        train_start="2024-01-01",
        train_end="2024-01-01",
        test_end="2024-01-01",
        progress_callback=lambda: progress_ticks.append("tick"),
    )

    assert shapley_result["components"] == ["pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    assert shapley_result["coalitions"] == 5
    assert len(calls) == 5
    assert len(progress_ticks) == 5
    assert shapley_result["shap_r2"] == pytest.approx(
        {
            "pressure": 4.0 / 3.0,
            "wind_u": 4.0 / 3.0,
            "Pressure (hPa) × Wind U (m/s)": 1.0 / 3.0,
        }
    )
    assert shapley_result["shap_rmse"] == pytest.approx(
        {
            "pressure": -4.0 / 3.0,
            "wind_u": -4.0 / 3.0,
            "Pressure (hPa) × Wind U (m/s)": -1.0 / 3.0,
        }
    )
    assert all(
        not interaction_pairs or (component_mask["pressure"] and component_mask["wind_u"])
        for component_mask, interaction_pairs in calls
    )


def test_build_shapley_result_accepts_model_solver_keywords(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_run_tidal_model(component_mask, **model_kwargs):
        calls.append(
            {
                "component_mask": component_mask.copy(),
                "solver_verbose": model_kwargs["solver_verbose"],
                "debug": model_kwargs["debug"],
            }
        )
        score = int(component_mask.get("M2", False)) + int(component_mask.get("pressure", False))
        return {
            "metrics_test": {"r2": float(score), "rmse": float(10 - score)},
            "picked": {},
            "active_regs": ["pressure"] if component_mask.get("pressure", False) else [],
            "active_interactions": [],
        }

    monkeypatch.setattr(tidal_compact, "run_tidal_model", fake_run_tidal_model)

    components = ["M2", "pressure"]
    shapley_result = build_shapley_result(
        {"M2": True, "pressure": True},
        components=components,
        interaction_lookup={},
        raw_to_canonical={bits: bits for bits in range(2 ** len(components))},
        df=pd.DataFrame({"water_level": [0.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h")),
        sph=1,
        harmonic_orders={"M2": 1},
        lag_ranges={"pressure": (-2, 0)},
        knot_presets={"pressure": "med"},
        interaction_pairs=[],
        train_start="2024-01-01",
        train_end="2024-01-01",
        test_end="2024-01-01",
        solver_verbose=True,
        debug=True,
    )

    assert shapley_result["components"] == components
    assert calls
    assert all(call["solver_verbose"] is True for call in calls)
    assert all(call["debug"] is True for call in calls)


@pytest.mark.parametrize(
    ("preset", "expected"),
    [("low", 4), ("med", 8), ("high", 12)],
)
def test_build_knot_count_maps_named_presets(preset, expected):
    assert build_knot_count(preset) == expected


def test_option_name_for_value_returns_dropdown_label():
    assert option_name_for_value({"Low": "low", "Med": "med", "High": "high"}, "med") == "Med"
    assert option_name_for_value({"Pressure (hPa)": "pressure"}, "pressure") == "Pressure (hPa)"


def test_build_interaction_pair_options_returns_human_readable_labels():
    options = build_interaction_pair_options(["pressure", "wind_u", "air_temp"])

    assert options == {
        "Pressure (hPa) × Wind U (m/s)": "pressure|wind_u",
        "Pressure (hPa) × Air temp (degC)": "pressure|air_temp",
        "Wind U (m/s) × Air temp (degC)": "wind_u|air_temp",
    }


def test_build_interaction_index_pairs_filters_to_active_regressors():
    interaction_pairs = [("pressure", "wind_u"), ("pressure", "air_temp")]

    assert build_interaction_index_pairs(["pressure", "wind_u"], interaction_pairs) == [(0, 1)]


def test_collect_model_params_includes_selected_interaction_pairs_and_fourier_reg_weight():
    harmonic_inputs = {"M2": _Widget(2), "S2": _Widget(0)}
    regressor_toggles = {"pressure": _Widget(True), "wind_u": _Widget(True)}
    regressor_lags = {"pressure": _Widget((-2, 0)), "wind_u": _Widget((-1, 0))}
    regressor_knots = {"pressure": _Widget("high"), "wind_u": _Widget("med")}
    interaction_pairs_select = _Widget(["pressure|wind_u"])
    fourier_reg_weight = _Widget(0.02)
    df = pd.DataFrame(
        {
            "water_level": np.linspace(0.0, 1.0, 6),
            "pressure": np.linspace(1010.0, 1015.0, 6),
            "wind_u": np.linspace(-1.0, 1.0, 6),
        },
        index=pd.date_range("2024-01-01", periods=6, freq="1h"),
    )

    component_mask, model_kwargs = collect_model_params(
        harmonic_inputs,
        regressor_lags,
        regressor_toggles,
        regressor_knots,
        interaction_pairs_select,
        df=df,
        sph=1,
        train_start=_Widget(date(2024, 1, 1)),
        train_end=_Widget(date(2024, 1, 2)),
        test_end=_Widget(date(2024, 1, 3)),
        fourier_reg_weight=fourier_reg_weight,
    )

    assert component_mask == {"M2": True, "S2": False, "pressure": True, "wind_u": True}
    assert model_kwargs["interaction_pairs"] == [("pressure", "wind_u")]
    assert model_kwargs["fourier_reg_weight"] == pytest.approx(0.02)


def test_build_fit_label_includes_active_interactions():
    fit_result = {
        "metrics_train": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "metrics_test": {"rmse": 0.1, "mae": 0.1, "mape": 1.0, "r2": 0.9},
        "te_index": pd.date_range("2024-01-01", periods=2, freq="1h"),
        "te_obs": np.array([0.0, 1.0]),
        "te_pred": np.array([0.0, 1.0]),
        "te_obs_clean": np.array([0.0, 1.0]),
        "te_pred_clean": np.array([0.0, 1.0]),
        "residuals": np.array([0.0, 0.0]),
        "picked": {},
        "active_regs": ["pressure", "wind_u"],
        "active_interactions": ["Pressure (hPa) × Wind U (m/s)"],
        "n_train": 10,
        "n_test": 2,
        "sph": 1,
    }

    label = build_fit_label(fit_result)

    assert "pressure, wind_u" in label
    assert "Pressure (hPa) × Wind U (m/s)" in label


def test_select_default_plot_lag_prefers_zero_then_nearest():
    selector = getattr(tidal_compact, "select_default_plot_lag", None)

    assert callable(selector)
    assert selector([-3, 0, 2]) == 0
    assert selector([-4, -1, 3]) == -1


def test_build_typical_cycle_frame_collapses_to_phase_average():
    builder = getattr(tidal_compact, "build_typical_cycle_frame", None)

    assert callable(builder)

    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    frame = builder(
        index=index,
        observed=np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]),
        predicted=np.array([1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0]),
        cycle_length_samples=4,
    )

    assert list(frame["phase"]) == [0, 1, 2, 3]
    np.testing.assert_allclose(frame["observed"], [5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(frame["predicted"], [6.0, 7.0, 8.0, 9.0])


def test_build_regressor_response_inputs_aligns_selected_lag():
    builder = getattr(tidal_compact, "build_regressor_response_inputs", None)

    assert callable(builder)

    estimator = tidal_compact.TsgamEstimator(
        tidal_compact.TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=None,
        )
    )
    estimator.variables_ = {
        "exog_coef_0": SimpleNamespace(value=np.array([[1.5, 2.5]])),
    }

    fit_result = {
        "tr_obs_clean": np.array([1.0, 2.0, 3.0, 4.0]),
        "x_train_fit": pd.DataFrame(
            {"pressure": [10.0, 20.0, 30.0, 40.0]},
            index=pd.date_range("2024-01-01", periods=4, freq="1h"),
        ),
        "active_regs": ["pressure"],
        "model": estimator,
        "exog_config": [tidal_compact.TsgamLinearConfig(lags=[-1, 0], reg_weight=1e-5)],
    }

    response_inputs = builder(fit_result, "pressure", lag=0)

    assert response_inputs["selected_lag"] == 0
    np.testing.assert_allclose(response_inputs["x_scatter"], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(response_inputs["y_scatter"], [1.0, 2.0, 3.0, 4.0])
    assert response_inputs["grid"].shape == (200,)
    assert response_inputs["curve"].shape == (200,)


def test_run_tidal_model_passes_active_interaction_pairs_to_estimator(monkeypatch):
    index = pd.date_range("2024-01-01", periods=48, freq="1h")
    df = pd.DataFrame(
        {
            "water_level": np.sin(np.linspace(0.0, 4.0 * np.pi, len(index))),
            "pressure": np.linspace(1010.0, 1018.0, len(index)),
            "wind_u": np.linspace(-2.0, 2.0, len(index)),
            "air_temp": np.linspace(5.0, 9.0, len(index)),
        },
        index=index,
    )
    captured: dict[str, object] = {}

    def fake_build_periodic_config(component_mask, harmonic_orders, sph, fourier_reg_weight=None):
        captured["fourier_reg_weight"] = fourier_reg_weight
        return {}, None

    def fake_build_exog_design_matrices(df_train, df_test, ok_train, reg_names, lag_ranges, knot_presets, sph):
        assert reg_names == ["pressure", "wind_u"]
        x_train_fit = df_train.loc[ok_train, ["pressure", "wind_u"]].copy()
        x_train_pred = df_train[["pressure", "wind_u"]].copy()
        x_test_pred = df_test[["pressure", "wind_u"]].copy()
        exog_config = [
            tidal_compact.TsgamLinearConfig(lags=[0], reg_weight=1e-5),
            tidal_compact.TsgamLinearConfig(lags=[0], reg_weight=1e-5),
        ]
        return x_train_fit, x_train_pred, x_test_pred, ["pressure", "wind_u"], exog_config

    class FakeEstimator:
        def __init__(self, config):
            captured["config"] = config

        def fit(self, x, y):
            captured["fit_shape"] = x.shape
            captured["y_len"] = len(y)

        def predict(self, x):
            return np.zeros(len(x), dtype=float)

    monkeypatch.setattr(tidal_compact, "build_periodic_config", fake_build_periodic_config)
    monkeypatch.setattr(tidal_compact, "build_exog_design_matrices", fake_build_exog_design_matrices)
    monkeypatch.setattr(tidal_compact, "TsgamEstimator", FakeEstimator)

    parameters = inspect.signature(tidal_compact.run_tidal_model).parameters

    assert "solver_verbose" in parameters
    assert "debug" in parameters

    result = tidal_compact.run_tidal_model(
        component_mask={"pressure": True, "wind_u": True},
        df=df,
        sph=1,
        harmonic_orders={},
        lag_ranges={"pressure": (-2, 0), "wind_u": (-1, 0)},
        knot_presets={"pressure": "med", "wind_u": "med"},
        interaction_pairs=[("pressure", "wind_u"), ("pressure", "air_temp")],
        train_start="2024-01-01",
        train_end="2024-01-02",
        test_end="2024-01-02",
        fourier_reg_weight=0.02,
        solver_verbose=True,
        debug=True,
    )

    config = captured["config"]

    assert config.interaction_pairs == [(0, 1)]
    assert config.solver_config.verbose is True
    assert config.debug is True
    assert captured["fourier_reg_weight"] == pytest.approx(0.02)
    assert result["active_regs"] == ["pressure", "wind_u"]
    assert result["active_interactions"] == ["Pressure (hPa) × Wind U (m/s)"]


def test_build_periodic_config_increases_weight_for_high_order_long_periods():
    picked, config = tidal_compact.build_periodic_config(
        component_mask={"Mf": True, "M2": True},
        harmonic_orders={"Mf": 16, "M2": 4},
        sph=1,
    )

    assert picked == {"Mf": (PERIODS["Mf"], 16), "M2": (PERIODS["M2"], 4)}
    assert config is not None
    assert config.reg_weight > 1e-4


def test_build_periodic_config_applies_manual_reg_weight():
    picked, config = tidal_compact.build_periodic_config(
        component_mask={"annual": True},
        harmonic_orders={"annual": 8},
        sph=1,
        fourier_reg_weight=0.02,
    )

    assert picked == {"annual": (PERIODS["annual"], 8)}
    assert config is not None
    assert config.reg_weight == pytest.approx(0.02)


def test_build_exog_design_matrices_uses_selected_knot_preset():
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    df_train = pd.DataFrame({"pressure": np.linspace(1010.0, 1015.0, len(index))}, index=index)
    df_test = pd.DataFrame(
        {"pressure": np.linspace(1016.0, 1017.0, 2)},
        index=pd.date_range("2024-01-01 06:00:00", periods=2, freq="1h"),
    )
    ok_train = pd.Series([True] * len(df_train), index=df_train.index)

    _x_train_fit, _x_train_pred, _x_test_pred, active_regs, exog_config = build_exog_design_matrices(
        df_train,
        df_test,
        ok_train,
        ["pressure"],
        {"pressure": (-2, 0)},
        {"pressure": "high"},
        1,
    )

    assert active_regs == ["pressure"]
    assert exog_config is not None
    assert len(exog_config) == 1
    assert exog_config[0].n_knots == 12


def test_build_exog_design_matrices_raises_for_missing_active_knot_preset():
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    df_train = pd.DataFrame({"pressure": np.linspace(1010.0, 1015.0, len(index))}, index=index)
    df_test = pd.DataFrame(
        {"pressure": np.linspace(1016.0, 1017.0, 2)},
        index=pd.date_range("2024-01-01 06:00:00", periods=2, freq="1h"),
    )
    ok_train = pd.Series([True] * len(df_train), index=df_train.index)

    with pytest.raises(ValueError, match="Missing knot preset for active regressor: pressure"):
        build_exog_design_matrices(
            df_train,
            df_test,
            ok_train,
            ["pressure"],
            {"pressure": (-2, 0)},
            {},
            1,
        )


def test_build_regressor_basis_inputs_uses_preset_knot_count_and_spans_regressor_range():
    regressor = pd.Series(
        [np.nan, -2.0, -0.5, 1.5, 3.0],
        index=pd.date_range("2024-01-01", periods=5, freq="1h"),
        name="pressure",
    )

    basis_inputs = build_regressor_basis_inputs(regressor, "high")

    assert basis_inputs["regressor_name"] == "pressure"
    assert len(basis_inputs["knots"]) == 12
    assert basis_inputs["knots"][0] == pytest.approx(-2.0)
    assert basis_inputs["knots"][-1] == pytest.approx(3.0)
    assert basis_inputs["grid"][0] == pytest.approx(-2.0)
    assert basis_inputs["grid"][-1] == pytest.approx(3.0)
    assert basis_inputs["basis"].shape == (len(basis_inputs["grid"]), 11)


def test_build_regressor_basis_inputs_rejects_constant_valued_regressors():
    regressor = pd.Series(
        [1012.5, 1012.5, 1012.5, 1012.5],
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
        name="pressure",
    )

    with pytest.raises(ValueError, match="constant"):
        build_regressor_basis_inputs(regressor, "med")


def test_build_regressor_basis_inputs_matches_tsgam_spline_basis():
    regressor = pd.Series(
        [-1.5, -0.25, 0.5, 1.75, 3.25],
        index=pd.date_range("2024-01-01", periods=5, freq="1h"),
        name="wind_u",
    )

    basis_inputs = build_regressor_basis_inputs(regressor, "med")
    estimator = tidal_compact.TsgamEstimator(
        tidal_compact.TsgamEstimatorConfig(
            multi_periodic_config=None,
            exog_config=None,
        )
    )

    expected_basis = estimator._make_H(
        basis_inputs["grid"],
        basis_inputs["knots"],
        include_offset=False,
    )

    np.testing.assert_allclose(basis_inputs["basis"], expected_basis)


def test_build_model_regressor_basis_inputs_uses_processed_training_regressor_range():
    index = pd.date_range("2024-01-01", periods=24 * 4, freq="1h")
    water_level = np.linspace(0.0, 1.0, len(index))
    water_level[10] = np.nan
    pressure = np.concatenate(
        [
            np.linspace(1000.0, 1023.0, 24),
            np.linspace(1100.0, 1123.0, 24),
            np.linspace(1200.0, 1223.0, 24),
            np.linspace(3000.0, 3023.0, 24),
        ]
    )
    pressure[10] = 1500.0
    df = pd.DataFrame(
        {
            "water_level": water_level,
            "pressure": pressure,
        },
        index=index,
    )

    basis_inputs = tidal_compact.build_model_regressor_basis_inputs(
        df,
        "pressure",
        "low",
        date(2024, 1, 1),
        date(2024, 1, 4),
        date(2024, 1, 4),
    )

    split_time = pd.Timestamp("2024-01-04")
    df_train = df[df.index < split_time]
    df_test = df[df.index >= split_time]
    ok_train = df_train["water_level"].notna()
    x_train_raw, _x_test_raw, active_regs, _dropped = tidal_compact.prepare_split_regressors(
        df_train,
        df_test,
        ["pressure"],
    )
    expected_train_regressor = x_train_raw.loc[ok_train, "pressure"]
    expected_knots = np.linspace(
        float(expected_train_regressor.min()),
        float(expected_train_regressor.max()),
        tidal_compact.build_knot_count("low"),
    )

    assert active_regs == ["pressure"]
    np.testing.assert_allclose(basis_inputs["knots"], expected_knots)
    assert basis_inputs["grid"][0] == pytest.approx(float(expected_train_regressor.min()))
    assert basis_inputs["grid"][-1] == pytest.approx(float(expected_train_regressor.max()))
    assert basis_inputs["grid"][-1] < float(df["pressure"].max())


def test_build_regressor_basis_figure_marks_knot_locations():
    regressor = pd.Series(
        [-1.0, 0.0, 1.0, 2.0],
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
        name="wind_u",
    )
    basis_inputs = build_regressor_basis_inputs(regressor, "low")

    figure = build_regressor_basis_figure(basis_inputs, knot_preset="low")

    knot_positions = {float(knot) for knot in basis_inputs["knots"]}
    figure_knot_positions = {
        float(shape.x0)
        for shape in figure.layout.shapes
        if getattr(shape, "type", None) == "line"
    }

    assert knot_positions <= figure_knot_positions


def test_load_station_frame_reuses_covering_cache_and_trims_requested_window(
    monkeypatch, tmp_path
):
    station_id = "8518750"
    cached_file = tmp_path / f"{station_id}_2024-01-01_2024-01-05_combined.csv"
    cached_file.touch()
    cached_index = pd.date_range("2024-01-01 00:00:00", "2024-01-05 23:00:00", freq="1h")
    cached_frame = pd.DataFrame(
        {
            "water_level": np.linspace(0.0, 1.0, len(cached_index)),
            "pressure": np.linspace(1010.0, 1012.0, len(cached_index)),
            "wind_u": np.linspace(-2.0, 2.0, len(cached_index)),
            "wind_v": np.linspace(1.0, -1.0, len(cached_index)),
        },
        index=cached_index,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(tidal_compact, "find_station", lambda _: station_id)

    def fake_resolve(data_dir, station, begin_date, end_date):
        calls["resolve"] = (data_dir, station, begin_date, end_date)
        return cached_file

    def fake_load_tidal_data(data_file):
        calls["load_tidal_data"] = data_file
        return cached_frame

    monkeypatch.setattr(
        tidal_compact, "resolve_tidal_cache_path", fake_resolve, raising=False
    )
    monkeypatch.setattr(tidal_compact, "load_tidal_data", fake_load_tidal_data)
    monkeypatch.setattr(
        tidal_compact,
        "download_tidal_data",
        lambda *args, **kwargs: pytest.fail("expected cached tidal data reuse"),
    )

    result = load_station_frame(
        "The Battery, NY",
        use_weather=False,
        download_tidal=False,
        download_weather=False,
        data_start=date(2024, 1, 2),
        data_end=date(2024, 1, 3),
    )

    assert calls["resolve"] == (
        tidal_compact.DEFAULT_DATA_DIR,
        station_id,
        "20240102",
        "20240103",
    )
    assert calls["load_tidal_data"] == cached_file
    assert result["df"].index.min() == pd.Timestamp("2024-01-02 00:00:00")
    assert result["df"].index.max() == pd.Timestamp("2024-01-03 23:00:00")
    assert result["date_min"] == date(2024, 1, 2)
    assert result["date_max"] == date(2024, 1, 3)
    assert "cache" in result["status_message"].lower()
    assert "2024-01-02" in result["status_message"]
    assert "2024-01-03" in result["status_message"]
    assert "dp_dt" in result["df"].columns
    expected_dp_dt = cached_frame["pressure"].diff().loc[pd.Timestamp("2024-01-02 00:00:00")]
    assert result["df"].loc[pd.Timestamp("2024-01-02 00:00:00"), "dp_dt"] == pytest.approx(
        expected_dp_dt
    )
    assert "wind_stress" in result["df"].columns


def test_load_station_frame_raises_clear_error_when_requested_tidal_window_is_missing(
    monkeypatch, tmp_path
):
    station_id = "8518750"
    missing_path = tmp_path / f"{station_id}_2024-02-01_2024-02-07_combined.csv"

    monkeypatch.setattr(tidal_compact, "find_station", lambda _: station_id)
    monkeypatch.setattr(
        tidal_compact,
        "resolve_tidal_cache_path",
        lambda data_dir, station, begin_date, end_date: missing_path,
        raising=False,
    )
    monkeypatch.setattr(
        tidal_compact,
        "download_tidal_data",
        lambda *args, **kwargs: pytest.fail("unexpected tidal download"),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        load_station_frame(
            "The Battery, NY",
            use_weather=False,
            download_tidal=False,
            download_weather=False,
            data_start=date(2024, 2, 1),
            data_end=date(2024, 2, 7),
        )

    message = str(exc_info.value)
    assert "tidal" in message.lower()
    assert "2024-02-01" in message
    assert "2024-02-07" in message


def test_load_station_frame_raises_for_partial_lcd_year_coverage_when_download_disabled(
    monkeypatch, tmp_path
):
    station_id = "8518750"
    weather_station_id = "TESTWX"
    cached_file = tmp_path / f"{station_id}_20241231_20250101_combined.csv"
    cached_file.touch()
    cached_index = pd.date_range("2024-12-30 00:00:00", "2025-01-01 23:00:00", freq="1h")
    cached_frame = pd.DataFrame(
        {"water_level": np.linspace(0.0, 1.0, len(cached_index))},
        index=cached_index,
    )
    partial_weather = pd.DataFrame(
        {"air_temp": [5.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-12-31 00:00:00")], name="datetime"),
    )
    (tmp_path / f"lcd_{weather_station_id}_2024.csv").touch()

    monkeypatch.setattr(tidal_compact, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(tidal_compact, "find_station", lambda _: station_id)
    monkeypatch.setattr(
        tidal_compact,
        "TIDE_TO_WEATHER",
        {station_id: (weather_station_id, "Mock Weather")},
    )
    monkeypatch.setattr(
        tidal_compact,
        "resolve_tidal_cache_path",
        lambda data_dir, station, begin_date, end_date: cached_file,
        raising=False,
    )
    monkeypatch.setattr(tidal_compact, "load_tidal_data", lambda _: cached_frame.copy())
    monkeypatch.setattr(
        tidal_compact,
        "load_lcd_weather",
        lambda *args, **kwargs: partial_weather.copy(),
    )
    monkeypatch.setattr(
        tidal_compact,
        "download_lcd_weather",
        lambda *args, **kwargs: pytest.fail("unexpected weather download"),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        load_station_frame(
            "The Battery, NY",
            use_weather=True,
            download_tidal=False,
            download_weather=False,
            data_start=date(2024, 12, 31),
            data_end=date(2025, 1, 1),
        )

    message = str(exc_info.value)
    assert "lcd weather" in message.lower()
    assert "2024-12-31" in message
    assert "2025-01-01" in message


def test_load_station_frame_downloads_missing_lcd_years_before_loading_weather(
    monkeypatch, tmp_path
):
    station_id = "8518750"
    weather_station_id = "TESTWX"
    cached_file = tmp_path / f"{station_id}_20241231_20250101_combined.csv"
    cached_file.touch()
    cached_index = pd.date_range("2024-12-30 00:00:00", "2025-01-01 23:00:00", freq="1h")
    cached_frame = pd.DataFrame(
        {"water_level": np.linspace(0.0, 1.0, len(cached_index))},
        index=cached_index,
    )
    weather_frame = pd.DataFrame(
        {"air_temp": [5.0, 6.0]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-12-31 00:00:00"),
                pd.Timestamp("2025-01-01 00:00:00"),
            ],
            name="datetime",
        ),
    )
    (tmp_path / f"lcd_{weather_station_id}_2024.csv").touch()
    calls: dict[str, object] = {}

    monkeypatch.setattr(tidal_compact, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(tidal_compact, "find_station", lambda _: station_id)
    monkeypatch.setattr(
        tidal_compact,
        "TIDE_TO_WEATHER",
        {station_id: (weather_station_id, "Mock Weather")},
    )
    monkeypatch.setattr(
        tidal_compact,
        "resolve_tidal_cache_path",
        lambda data_dir, station, begin_date, end_date: cached_file,
        raising=False,
    )
    monkeypatch.setattr(tidal_compact, "load_tidal_data", lambda _: cached_frame.copy())

    def fake_download_lcd_weather(data_dir, station_id, begin_year, end_year):
        calls["download"] = (data_dir, station_id, begin_year, end_year)
        (tmp_path / f"lcd_{weather_station_id}_2025.csv").touch()

    def fake_load_lcd_weather(data_dir, station_id, begin_date, end_date):
        calls["load_weather"] = (data_dir, station_id, begin_date, end_date)
        return weather_frame.copy()

    monkeypatch.setattr(
        tidal_compact,
        "download_lcd_weather",
        fake_download_lcd_weather,
    )
    monkeypatch.setattr(tidal_compact, "load_lcd_weather", fake_load_lcd_weather)

    result = load_station_frame(
        "The Battery, NY",
        use_weather=True,
        download_tidal=False,
        download_weather=True,
        data_start=date(2024, 12, 31),
        data_end=date(2025, 1, 1),
    )

    assert calls["download"] == (tmp_path, weather_station_id, 2024, 2025)
    assert calls["load_weather"] == (
        tmp_path,
        weather_station_id,
        "2024-12-31",
        "2025-01-01",
    )
    assert "download" in result["status_message"].lower()


def test_load_station_frame_raises_for_existing_partial_weather_cache_when_download_disabled(
    monkeypatch, tmp_path
):
    station_id = "8518750"
    weather_station_id = "TESTWX"
    cached_file = tmp_path / f"{station_id}_20241231_20250101_combined.csv"
    cached_file.touch()
    cached_index = pd.date_range("2024-12-30 00:00:00", "2025-01-01 23:00:00", freq="1h")
    cached_frame = pd.DataFrame(
        {"water_level": np.linspace(0.0, 1.0, len(cached_index))},
        index=cached_index,
    )
    partial_weather = pd.DataFrame(
        {"air_temp": [5.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-12-31 00:00:00")], name="datetime"),
    )
    (tmp_path / f"lcd_{weather_station_id}_2024.csv").touch()
    (tmp_path / f"lcd_{weather_station_id}_2025.csv").touch()

    monkeypatch.setattr(tidal_compact, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(tidal_compact, "find_station", lambda _: station_id)
    monkeypatch.setattr(
        tidal_compact,
        "TIDE_TO_WEATHER",
        {station_id: (weather_station_id, "Mock Weather")},
    )
    monkeypatch.setattr(
        tidal_compact,
        "resolve_tidal_cache_path",
        lambda data_dir, station, begin_date, end_date: cached_file,
        raising=False,
    )
    monkeypatch.setattr(tidal_compact, "load_tidal_data", lambda _: cached_frame.copy())
    monkeypatch.setattr(tidal_compact, "load_lcd_weather", lambda *args, **kwargs: partial_weather.copy())
    monkeypatch.setattr(
        tidal_compact,
        "download_lcd_weather",
        lambda *args, **kwargs: pytest.fail("unexpected weather download"),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        load_station_frame(
            "The Battery, NY",
            use_weather=True,
            download_tidal=False,
            download_weather=False,
            data_start=date(2024, 12, 31),
            data_end=date(2025, 1, 1),
        )

    message = str(exc_info.value)
    assert "lcd weather" in message.lower()
    assert "coverage" in message.lower()
    assert "2024-12-31" in message
    assert "2025-01-01" in message


def test_load_station_frame_redownloads_when_existing_weather_cache_is_partial(
    monkeypatch, tmp_path
):
    station_id = "8518750"
    weather_station_id = "TESTWX"
    cached_file = tmp_path / f"{station_id}_20241231_20250101_combined.csv"
    cached_file.touch()
    cached_index = pd.date_range("2024-12-30 00:00:00", "2025-01-01 23:00:00", freq="1h")
    cached_frame = pd.DataFrame(
        {"water_level": np.linspace(0.0, 1.0, len(cached_index))},
        index=cached_index,
    )
    partial_weather = pd.DataFrame(
        {"air_temp": [5.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-12-31 00:00:00")], name="datetime"),
    )
    full_weather = pd.DataFrame(
        {"air_temp": [5.0, 6.0]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-12-31 00:00:00"),
                pd.Timestamp("2025-01-01 00:00:00"),
            ],
            name="datetime",
        ),
    )
    (tmp_path / f"lcd_{weather_station_id}_2024.csv").touch()
    (tmp_path / f"lcd_{weather_station_id}_2025.csv").touch()
    calls: dict[str, object] = {"load_count": 0}

    monkeypatch.setattr(tidal_compact, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(tidal_compact, "find_station", lambda _: station_id)
    monkeypatch.setattr(
        tidal_compact,
        "TIDE_TO_WEATHER",
        {station_id: (weather_station_id, "Mock Weather")},
    )
    monkeypatch.setattr(
        tidal_compact,
        "resolve_tidal_cache_path",
        lambda data_dir, station, begin_date, end_date: cached_file,
        raising=False,
    )
    monkeypatch.setattr(tidal_compact, "load_tidal_data", lambda _: cached_frame.copy())

    def fake_load_lcd_weather(data_dir, station_id, begin_date, end_date):
        calls["load_count"] = int(calls["load_count"]) + 1
        calls["last_load"] = (data_dir, station_id, begin_date, end_date)
        return partial_weather.copy() if calls["load_count"] == 1 else full_weather.copy()

    def fake_download_lcd_weather(data_dir, station_id, begin_year, end_year):
        calls["download"] = (data_dir, station_id, begin_year, end_year)

    monkeypatch.setattr(tidal_compact, "load_lcd_weather", fake_load_lcd_weather)
    monkeypatch.setattr(tidal_compact, "download_lcd_weather", fake_download_lcd_weather)

    result = load_station_frame(
        "The Battery, NY",
        use_weather=True,
        download_tidal=False,
        download_weather=True,
        data_start=date(2024, 12, 31),
        data_end=date(2025, 1, 1),
    )

    assert calls["download"] == (tmp_path, weather_station_id, 2024, 2025)
    assert calls["load_count"] == 2
    assert calls["last_load"] == (
        tmp_path,
        weather_station_id,
        "2024-12-31",
        "2025-01-01",
    )
    assert "download" in result["status_message"].lower()
