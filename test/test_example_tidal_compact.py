from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import example_tidal_compact as tidal_compact  # noqa: E402
from example_tidal import TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS  # noqa: E402
from example_tidal_compact import (  # noqa: E402
    build_model_date_defaults,
    build_model_window_validation_message,
    build_diagnostic_figures,
    build_periodogram_figure,
    build_periodogram_selector_options,
    build_shapley_figure,
    load_station_frame,
)


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
