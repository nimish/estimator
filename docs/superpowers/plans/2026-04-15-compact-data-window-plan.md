# Compact Data Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `examples/example_tidal_compact.py` load an explicit top-level station/date window on demand, reuse/download the needed data range, and derive later exploration/model datepickers from the loaded dataset.

**Architecture:** Keep the existing data backends in `examples/example_tidal.py` as the source of truth. Extend `load_station_frame(...)` in the compact notebook so it accepts the requested date window, resolves a covering tidal cache or downloads the requested range, trims the loaded frame to that window, and then merges LCD weather for the same span. Rewire the marimo UI so data loading is gated behind a top-level `Load data` button, while downstream datepickers use a small pure helper to derive valid defaults from the loaded frame.

**Tech Stack:** Python 3.12, marimo, pandas, NumPy, pytest, `unittest.mock`

---

## File Map

- Modify: `examples/example_tidal_compact.py`
  - Import `resolve_tidal_cache_path(...)` and date utilities.
  - Extend `load_station_frame(...)` with requested window, cache/download handling, trimming, and clearer status/error messages.
  - Add `DATA_WINDOW_DEFAULT_START`, `DATA_WINDOW_DEFAULT_END`, and `build_model_date_defaults(...)`.
  - Rewire the top-level notebook controls around an explicit `Load data` button.
  - Rebuild model datepickers from the loaded frame bounds instead of assuming the full cached dataset.
- Modify: `test/test_example_tidal_compact.py`
  - Add focused regressions for range-aware loading, cache-miss errors, and downstream model date defaults.

### Task 1: Range-Aware Data Loading

**Files:**
- Modify: `examples/example_tidal_compact.py:20-32`
- Modify: `examples/example_tidal_compact.py:103-109`
- Modify: `examples/example_tidal_compact.py:470-530`
- Modify: `test/test_example_tidal_compact.py:1-92`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from example_tidal import TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS  # noqa: E402
from example_tidal_compact import (  # noqa: E402
    build_diagnostic_figures,
    build_periodogram_selector_options,
    build_periodogram_figure,
    build_shapley_figure,
    load_station_frame,
)


def test_load_station_frame_uses_covering_cache_and_trims_requested_window(tmp_path):
    cache_file = tmp_path / "8518750_20231201_20240131_combined.csv"
    cache_file.write_text("datetime,water_level\n")
    full_index = pd.date_range("2023-12-31 00:00", "2024-01-03 23:00", freq="1h")
    full_df = pd.DataFrame(
        {"water_level": np.arange(len(full_index), dtype=float)},
        index=full_index,
    )

    with (
        patch("example_tidal_compact.find_station", return_value="8518750"),
        patch("example_tidal_compact.resolve_tidal_cache_path", return_value=cache_file),
        patch("example_tidal_compact.load_tidal_data", return_value=full_df),
    ):
        station_data = load_station_frame(
            "The Battery, NY",
            data_start=date(2024, 1, 1),
            data_end=date(2024, 1, 2),
            use_weather=False,
            download_tidal=False,
            download_weather=False,
        )

    assert station_data["date_min"] == date(2024, 1, 1)
    assert station_data["date_max"] == date(2024, 1, 2)
    assert station_data["df"].index.min() == pd.Timestamp("2024-01-01 00:00:00")
    assert station_data["df"].index.max() == pd.Timestamp("2024-01-02 23:00:00")
    assert "cache" in station_data["status_message"].lower()
    assert "loaded window: 2024-01-01 to 2024-01-02" in station_data["status_message"].lower()


def test_load_station_frame_requires_cached_window_when_download_disabled(tmp_path):
    missing_cache = tmp_path / "8518750_20240101_20250101_combined.csv"

    with (
        patch("example_tidal_compact.find_station", return_value="8518750"),
        patch("example_tidal_compact.resolve_tidal_cache_path", return_value=missing_cache),
    ):
        with pytest.raises(
            FileNotFoundError,
            match="Requested tidal window 2024-01-01 to 2025-01-01 is not cached",
        ):
            load_station_frame(
                "The Battery, NY",
                data_start=date(2024, 1, 1),
                data_end=date(2025, 1, 1),
                use_weather=False,
                download_tidal=False,
                download_weather=False,
            )
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest test/test_example_tidal_compact.py -k "load_station_frame" -v`

Expected: FAIL with `TypeError: load_station_frame() got an unexpected keyword argument 'data_start'`.

- [ ] **Step 3: Implement range-aware loading in the compact notebook**

```python
from datetime import date, timedelta

from example_tidal import (
    DEFAULT_DATA_DIR,
    STATION_CATALOG,
    TIDE_TO_WEATHER,
    download_lcd_weather,
    download_tidal_data,
    find_station,
    load_lcd_weather,
    load_tidal_data,
    merge_tidal_weather,
    resolve_tidal_cache_path,
    TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS,
)


class StationData(TypedDict):
    df: pd.DataFrame
    sph: int
    date_min: date
    date_max: date
    status_message: str


@app.function
def load_station_frame(
    station_name: str,
    data_start: date,
    data_end: date,
    use_weather: bool,
    download_tidal: bool,
    download_weather: bool,
) -> StationData:
    if data_start > data_end:
        raise ValueError("Data start must be on or before data end.")

    station_id = find_station(station_name)
    request_begin = data_start.strftime("%Y%m%d")
    request_end = data_end.strftime("%Y%m%d")
    status_messages: list[str] = []

    tidal_path = resolve_tidal_cache_path(DEFAULT_DATA_DIR, station_id, request_begin, request_end)
    if tidal_path.exists():
        df = load_tidal_data(tidal_path).copy()
        status_messages.append(f"Tidal data: cache {tidal_path.name}")
    elif not download_tidal:
        raise FileNotFoundError(
            f"Requested tidal window {data_start} to {data_end} is not cached for {STATION_CATALOG[station_id]['name']}."
        )
    else:
        tidal_path = download_tidal_data(
            DEFAULT_DATA_DIR,
            station=station_id,
            begin_date=request_begin,
            end_date=request_end,
        )
        df = load_tidal_data(tidal_path).copy()
        status_messages.append(f"Tidal data: downloaded {tidal_path.name}")

    time_index = pd.DatetimeIndex(df.index)
    if time_index.tz is not None:
        time_index = time_index.tz_convert(None)
    df.index = time_index
    df = df[str(data_start) : str(data_end)]
    if df.empty:
        raise FileNotFoundError(
            f"No tidal samples available between {data_start} and {data_end} for {STATION_CATALOG[station_id]['name']}."
        )

    weather_station = TIDE_TO_WEATHER.get(station_id)
    if use_weather and weather_station:
        weather_station_id, station_label = weather_station
        try:
            weather_df = load_lcd_weather(
                DEFAULT_DATA_DIR,
                station_id=weather_station_id,
                begin_date=str(data_start),
                end_date=str(data_end),
            )
        except FileNotFoundError:
            if not download_weather:
                raise FileNotFoundError(
                    f"Requested LCD weather window {data_start} to {data_end} is not cached for {station_label}."
                ) from None
            download_lcd_weather(
                DEFAULT_DATA_DIR,
                station_id=weather_station_id,
                begin_year=data_start.year,
                end_year=data_end.year,
            )
            weather_df = load_lcd_weather(
                DEFAULT_DATA_DIR,
                station_id=weather_station_id,
                begin_date=str(data_start),
                end_date=str(data_end),
            )
        df = merge_tidal_weather(df, weather_df)
        status_messages.append(f"Weather: merged {station_label}")

    if "pressure" in df.columns:
        df["dp_dt"] = df["pressure"].diff()
    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_stress"] = df["wind_u"] ** 2 + df["wind_v"] ** 2

    loaded_index = pd.DatetimeIndex(df.index)
    status_messages.append(
        f"Loaded window: {loaded_index[0].date()} to {loaded_index[-1].date()}"
    )
    return {
        "df": df,
        "sph": infer_samples_per_hour(loaded_index),
        "date_min": loaded_index[0].date(),
        "date_max": loaded_index[-1].date(),
        "status_message": "; ".join(status_messages),
    }
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run pytest test/test_example_tidal_compact.py -k "load_station_frame" -v`

Expected: PASS for both new range-aware loading regressions.

- [ ] **Step 5: Commit the loading changes**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add explicit compact data-window loading"
```

### Task 2: Explicit Load Controls And Downstream Date Defaults

**Files:**
- Modify: `examples/example_tidal_compact.py:152-190`
- Modify: `examples/example_tidal_compact.py:261-321`
- Modify: `test/test_example_tidal_compact.py:1-92`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing tests for model date defaults**

```python
from datetime import date

from example_tidal_compact import build_model_date_defaults  # noqa: E402


def test_build_model_date_defaults_keeps_example_split_when_available():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2023, 12, 15),
        date(2024, 2, 15),
    )

    assert train_start == date(2023, 12, 15)
    assert train_end == date(2024, 1, 1)
    assert test_end == date(2024, 2, 15)


def test_build_model_date_defaults_uses_midpoint_when_example_split_is_out_of_range():
    train_start, train_end, test_end = build_model_date_defaults(
        date(2024, 6, 1),
        date(2024, 6, 10),
    )

    assert train_start == date(2024, 6, 1)
    assert train_end == date(2024, 6, 5)
    assert test_end == date(2024, 6, 10)
```

- [ ] **Step 2: Run the new regression to verify it fails**

Run: `uv run pytest test/test_example_tidal_compact.py -k "model_date_defaults" -v`

Expected: FAIL with `ImportError: cannot import name 'build_model_date_defaults'`.

- [ ] **Step 3: Add explicit load controls and derive downstream defaults from the loaded frame**

```python
DATA_WINDOW_DEFAULT_START = pd.Timestamp("2022-01-01").date()
DATA_WINDOW_DEFAULT_END = pd.Timestamp("2024-03-31").date()


@app.function
def build_model_date_defaults(date_min: date, date_max: date) -> tuple[date, date, date]:
    example_split = pd.Timestamp("2024-01-01").date()
    if example_split < date_min or example_split > date_max:
        example_split = date_min + timedelta(days=(date_max - date_min).days // 2)
    return date_min, example_split, date_max
```

```python
@app.cell
def _():
    _station_options = {meta["name"]: station_name for station_name, meta in STATION_CATALOG.items()}
    station_picker = mo.ui.dropdown(options=_station_options, value="The Battery, NY", label="Station")
    return (station_picker,)


@app.cell
def _(station_picker):
    mo.stop(not station_picker.value, mo.md("*Select a station to enable data and weather options.*"))
    _station_id = find_station(station_picker.value)
    _weather_station = TIDE_TO_WEATHER.get(_station_id)
    data_start = mo.ui.date(value=DATA_WINDOW_DEFAULT_START, label="Data start")
    data_end = mo.ui.date(value=DATA_WINDOW_DEFAULT_END, label="Data end")
    download_tidal = mo.ui.switch(label="Download tidal data if not found", value=True)
    if _weather_station:
        _, _station_label = _weather_station
        use_weather = mo.ui.switch(value=True, label=f"Merge LCD weather from {_station_label}")
        download_weather = mo.ui.switch(label="Download weather data if not found", value=True)
    else:
        use_weather = mo.ui.switch(value=False, label="No mapped weather station", disabled=True)
        download_weather = mo.ui.switch(
            value=False,
            label="Download weather data if not found",
            disabled=True,
        )

    load_data = mo.ui.run_button(label="Load data")
    mo.vstack(
        [
            mo.hstack([station_picker, data_start, data_end], justify="start"),
            mo.hstack([download_tidal, use_weather, download_weather], justify="start"),
            load_data,
        ]
    )
    return data_end, data_start, download_tidal, download_weather, load_data, use_weather


@app.cell
def _(data_end, data_start, download_tidal, download_weather, load_data, station_picker, use_weather):
    mo.stop(not load_data.value, mo.md("*Choose a station and data window above, then click **Load data**.*"))
    mo.stop(data_start.value > data_end.value, mo.md("*`Data start` must be on or before `Data end`.*"))
    station_data = load_station_frame(
        station_picker.value,
        data_start=data_start.value,
        data_end=data_end.value,
        use_weather=bool(use_weather.value),
        download_tidal=bool(download_tidal.value),
        download_weather=bool(download_weather.value),
    )
    return (station_data,)
```

```python
@app.cell
def _(station_data):
    _df = station_data["df"]
    harmonic_inputs = {
        name: mo.ui.slider(
            start=0,
            stop=8,
            value=order,
            label=f"{name} ({PERIODS.get(name, 8766.0):.1f} h)",
            show_value=True,
            full_width=True,
        )
        for name, order in DEFAULT_HARMONICS.items()
    }

    _available_regressors = available_columns(_df, MODEL_REGRESSOR_CANDIDATES)
    regressor_toggles = {}
    regressor_lags = {}
    _regressor_rows = []
    for name in _available_regressors:
        regressor_toggles[name] = mo.ui.switch(value=False, label=name)
        _lag_start, _lag_end = LAG_DEFAULTS.get(name, (-2, 0))
        regressor_lags[name] = mo.ui.range_slider(
            start=-6,
            stop=6,
            value=(_lag_start, _lag_end),
            label="lag (h)",
            show_value=True,
        )
        _regressor_rows.append(mo.hstack([regressor_toggles[name], regressor_lags[name]], justify="start"))

    _train_start_default, _train_end_default, _test_end_default = build_model_date_defaults(
        station_data["date_min"],
        station_data["date_max"],
    )
    train_start = mo.ui.date(value=_train_start_default, label="Train start")
    train_end = mo.ui.date(value=_train_end_default, label="Train end")
    test_end = mo.ui.date(value=_test_end_default, label="Test end")
    run_fit = mo.ui.run_button(label="Fit model")

    mo.vstack(
        [
            mo.md("## Configure model"),
            mo.hstack(
                [
                    mo.vstack([mo.md("**Harmonics** (0 = exclude)")] + list(harmonic_inputs.values())),
                    mo.vstack([mo.md("**Regressors** (toggle + lag range in hours)")] + _regressor_rows)
                    if _available_regressors
                    else mo.md(""),
                    mo.vstack([mo.md("**Date ranges**"), train_start, train_end, test_end]),
                ],
                justify="start",
                gap=2,
            ),
            run_fit,
        ]
    )
    return (
        harmonic_inputs,
        regressor_lags,
        regressor_toggles,
        run_fit,
        test_end,
        train_end,
        train_start,
    )
```

- [ ] **Step 4: Run the full notebook-adjacent test file**

Run: `uv run pytest test/test_example_tidal_compact.py -v`

Expected: PASS for the existing periodogram/Shapley regressions plus the new data-window tests.

- [ ] **Step 5: Smoke-test the notebook manually**

Run: `uv run marimo edit examples/example_tidal_compact.py`

Expected:
- Before clicking `Load data`, the notebook shows the top-level station/date window controls and a prompt to load data.
- Loading the default window shows a status message with cache/download origin and the actual loaded date span.
- Extending `Data end` and clicking `Load data` updates the available exploration/model end dates if the data can be loaded.
- Disabling `Download tidal data if not found` and requesting an uncached future window shows a clear notebook-visible missing-cache message.
- Loading a short window that does not include `2024-01-01` causes the model `Train end` default to fall back to the midpoint of the loaded window.

- [ ] **Step 6: Commit the UI/default changes**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add explicit compact notebook data windows"
```
