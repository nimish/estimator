import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import os
    from collections.abc import Mapping
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import date, timedelta
    from functools import partial
    from math import factorial
    from pathlib import Path
    from typing import TypedDict, cast

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats

    from example_tidal import (
        DEFAULT_DATA_DIR,
        STATION_CATALOG,
        TIDE_TO_WEATHER,
        download_lcd_weather,
        download_tidal_data,
        find_station,
        load_lcd_weather,
        load_station,
        load_tidal_data,
        merge_tidal_weather,
        resolve_tidal_cache_path,
        TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS,
    )
    from tidal_analysis_helpers import compute_lagged_correlation, compute_periodogram, infer_samples_per_hour
    from tidal_model_shared import prepare_split_regressors, tidal_metrics
    from tsgam_estimator import (
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamLinearConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
    )

    COLUMN_LABELS = {
        "water_level": "Water level (m)",
        "pressure": "Pressure (hPa)",
        "dp_dt": "dP/dt (hPa/step)",
        "water_temp": "Water temp (degC)",
        "air_temp": "Air temp (degC)",
        "wind_u": "Wind U (m/s)",
        "wind_v": "Wind V (m/s)",
        "wind_speed": "Wind speed (m/s)",
        "wind_stress": "Wind stress (m^2/s^2)",
        "lcd_slp": "LCD sea level pressure (hPa)",
    }
    REGRESSOR_ANALYSIS_CANDIDATES = [
        "pressure",
        "dp_dt",
        "water_temp",
        "wind_u",
        "wind_v",
        "air_temp",
        "wind_speed",
        "wind_stress",
        "lcd_slp",
    ]
    CORRELATION_CANDIDATES = ["water_level", *REGRESSOR_ANALYSIS_CANDIDATES]
    MODEL_REGRESSOR_CANDIDATES = [
        "pressure",
        "dp_dt",
        "water_temp",
        "wind_u",
        "wind_v",
        "air_temp",
        "wind_stress",
    ]
    LAG_DEFAULTS = {
        "pressure": (-2, 0),
        "dp_dt": (-2, 0),
        "water_temp": (0, 0),
        "wind_u": (-1, 0),
        "wind_v": (-1, 0),
        "air_temp": (0, 0),
        "wind_stress": (-1, 0),
    }
    KNOT_PRESET_TO_COUNT = {"low": 4, "med": 8, "high": 12}
    KNOT_PRESET_OPTIONS = {"Low": "low", "Med": "med", "High": "high"}
    DEFAULT_HARMONICS = {
        "M2": 4,
        "S2": 1,
        "N2": 1,
        "K1": 2,
        "O1": 1,
        "Mf": 1,
        "Mm": 1,
        "annual": 2,
    }
    METRIC_SPECS = {
        "rmse": ("RMSE", "m", 3),
        "mae": ("MAE", "m", 3),
        "mape": ("MAPE", "%", 2),
        "r2": ("R^2", "", 3),
    }
    DATA_WINDOW_DEFAULT_START = pd.Timestamp("2022-01-01").date()
    DATA_WINDOW_DEFAULT_END = pd.Timestamp("2024-03-31").date()

    class StationData(TypedDict):
        df: pd.DataFrame
        sph: int
        date_min: date
        date_max: date
        status_message: str

    class ModelKwargs(TypedDict):
        df: pd.DataFrame
        sph: int
        harmonic_orders: dict[str, int]
        lag_ranges: dict[str, tuple[int, int]]
        knot_presets: dict[str, str]
        train_start: str
        train_end: str
        test_end: str

    class MetricDict(TypedDict):
        rmse: float
        mae: float
        mape: float
        r2: float

    class RegressorBasisInputs(TypedDict):
        regressor_name: str
        index: pd.DatetimeIndex
        values: np.ndarray
        knots: np.ndarray
        grid: np.ndarray
        basis: np.ndarray

    class FitResult(TypedDict):
        metrics_train: MetricDict
        metrics_test: MetricDict
        te_index: pd.DatetimeIndex
        te_obs: np.ndarray
        te_pred: np.ndarray
        te_obs_clean: np.ndarray
        te_pred_clean: np.ndarray
        residuals: np.ndarray
        picked: dict[str, tuple[float, int]]
        active_regs: list[str]
        n_train: int
        n_test: int
        sph: int

    class ShapleyResult(TypedDict):
        components: list[str]
        coalitions: int
        failed: int
        baseline_r2: float
        baseline_rmse: float
        full_r2: float
        full_rmse: float
        shap_r2: dict[str, float]
        shap_rmse: dict[str, float]


@app.cell
def _():
    _station_options = {meta["name"]: station_name for station_name, meta in STATION_CATALOG.items()}
    station_picker = mo.ui.dropdown(options=_station_options, value="The Battery, NY", label="Station")
    station_picker
    return (station_picker,)


@app.cell
def _(station_picker):
    mo.stop(not station_picker.value, mo.md("*Select a station to configure data loading.*"))
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
            mo.hstack([data_start, data_end], justify="start"),
            mo.hstack([download_tidal, use_weather, download_weather], justify="start"),
            load_data,
        ]
    )
    return (
        data_end,
        data_start,
        download_tidal,
        download_weather,
        load_data,
        use_weather,
    )


@app.cell
def _(
    data_end,
    data_start,
    download_tidal,
    download_weather,
    load_data,
    station_picker,
    use_weather,
):
    mo.stop(
        not load_data.value,
        mo.md("*Choose the window above and click **Load data** to fetch station data.*"),
    )
    mo.stop(
        data_start.value > data_end.value,
        mo.md("*`Data start` must be on or before `Data end`.*"),
    )
    station_data = load_station_frame(
        station_picker.value,
        use_weather=bool(use_weather.value),
        download_tidal=bool(download_tidal.value),
        download_weather=bool(download_weather.value),
        data_start=data_start.value,
        data_end=data_end.value,
    )
    return (station_data,)


@app.cell
def _(station_data):
    _status_message = station_data["status_message"]
    mo.md(f"*{_status_message}*") if _status_message else mo.md("")
    return


@app.cell
def _(station_data):
    explore_start = mo.ui.date(value=station_data["date_min"], label="From")
    explore_end = mo.ui.date(value=station_data["date_max"], label="To")
    explore_resample = mo.ui.dropdown(
        options={"---": None, "Hourly": "1h", "6-hourly": "6h", "Daily": "1D"},
        value="---",
        label="Resample",
    )
    mo.hstack([explore_start, explore_end, explore_resample], justify="start")
    return explore_end, explore_resample, explore_start


@app.cell
def _(explore_end, explore_resample, explore_start, station_data):
    build_overview_figure(
        station_data["df"],
        explore_start.value,
        explore_end.value,
        explore_resample.value,
    )
    return


@app.cell
def _(station_data):
    options, default_name = build_periodogram_selector_options(station_data["df"])
    explore_periodogram_series = mo.ui.dropdown(
        options=options,
        value=default_name,
        label="Spectrum series",
    )
    explore_periodogram_series
    return (explore_periodogram_series,)


@app.cell
def _(explore_end, explore_periodogram_series, explore_start, station_data):
    window = station_data["df"][str(explore_start.value) : str(explore_end.value)]
    series = window[explore_periodogram_series.value]
    mo.stop(
        series.notna().sum() < 4,
        mo.md("*Need at least a few non-null samples in the selected window to compute a periodogram.*"),
    )
    build_periodogram_figure(
        window.index,
        series.to_numpy(dtype=float),
        title=f"{COLUMN_LABELS.get(explore_periodogram_series.value, explore_periodogram_series.value)} spectrum",
    )
    return


@app.cell
def _(station_data):
    _columns = available_columns(station_data["df"], CORRELATION_CANDIDATES)
    mo.stop(len(_columns) < 2, mo.md("*Merge weather data above to see regressor correlations.*"))
    build_correlation_heatmap(station_data["df"])
    return


@app.cell
def _(station_data):
    _df = station_data["df"]
    _train_start_default, _train_end_default, _test_end_default = build_model_date_defaults(
        station_data["date_min"],
        station_data["date_max"],
    )
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
    regressor_knots = {}
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
        regressor_knots[name] = mo.ui.dropdown(
            options=KNOT_PRESET_OPTIONS,
            value=option_name_for_value(KNOT_PRESET_OPTIONS, "med"),
            label="Knots",
        )
        _regressor_rows.append(
            mo.hstack([regressor_toggles[name], regressor_lags[name], regressor_knots[name]], justify="start")
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
                    mo.vstack([mo.md("**Regressors** (toggle + lag range in hours + knots)")] + _regressor_rows)
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
        regressor_knots,
        regressor_lags,
        regressor_toggles,
        run_fit,
        test_end,
        train_end,
        train_start,
    )


@app.cell
def _(station_data):
    _available_regressors = available_columns(station_data["df"], MODEL_REGRESSOR_CANDIDATES)
    mo.stop(
        not _available_regressors,
        mo.md("*No model regressors are available in the loaded dataset for inspection.*"),
    )
    _options = {
        COLUMN_LABELS.get(name, name): name for name in _available_regressors
    }
    inspect_regressor = mo.ui.dropdown(
        options=_options,
        value=option_name_for_value(_options, _available_regressors[0]),
        label="Inspect regressor",
    )
    mo.vstack(
        [
            mo.md("## Regressor inspection"),
            inspect_regressor,
        ]
    )
    return (inspect_regressor,)


@app.cell
def _(
    inspect_regressor,
    regressor_knots,
    station_data,
    test_end,
    train_end,
    train_start,
):
    _regressor_name = inspect_regressor.value
    _knot_preset = regressor_knots[_regressor_name].value
    _window_message = build_model_window_validation_message(
        station_data["date_min"],
        station_data["date_max"],
        train_start.value,
        train_end.value,
        test_end.value,
    )
    mo.stop(_window_message is not None, mo.md(f"*{_window_message}*"))
    try:
        _basis_inputs = build_model_regressor_basis_inputs(
            station_data["df"],
            _regressor_name,
            _knot_preset,
            train_start.value,
            train_end.value,
            test_end.value,
        )
    except ValueError as exc:
        mo.stop(True, mo.md(f"*{exc}*"))
    mo.vstack(
        [
            mo.md(f"*Current knot preset for `{_regressor_name}`: `{_knot_preset}`*"),
            build_regressor_basis_figure(_basis_inputs, knot_preset=_knot_preset),
        ]
    )
    return


@app.cell
def _(
    harmonic_inputs,
    regressor_knots,
    regressor_lags,
    regressor_toggles,
    run_fit,
    station_data,
    test_end,
    train_end,
    train_start,
):
    mo.stop(not run_fit.value, mo.md("*Configure parameters above and click **Fit model**.*"))
    _window_message = build_model_window_validation_message(
        station_data["date_min"],
        station_data["date_max"],
        train_start.value,
        train_end.value,
        test_end.value,
    )
    mo.stop(_window_message is not None, mo.md(f"*{_window_message}*"))
    _component_mask, _model_kwargs = collect_model_params(
        harmonic_inputs,
        regressor_lags,
        regressor_toggles,
        regressor_knots,
        df=station_data["df"],
        sph=station_data["sph"],
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
    )
    mo.stop(not any(_component_mask.values()), mo.md("*Set at least one constituent to harmonics > 0.*"))

    fit_result = run_tidal_model(_component_mask, **_model_kwargs)
    fit_label = build_fit_label(fit_result)
    return fit_label, fit_result


@app.cell
def _(fit_result):
    mo.Html(build_metrics_table_html(fit_result))
    return


@app.cell
def _(fit_label, fit_result):
    build_fit_timeseries_figure(fit_label, fit_result)
    return


@app.cell
def _(fit_result, station_data):
    mo.ui.tabs(build_diagnostic_figures(station_data["df"], fit_result), lazy=False)
    return


@app.cell
def _():
    run_shapley = mo.ui.run_button(label="Run Shapley analysis")
    mo.vstack([mo.md("## Shapley value analysis"), run_shapley])
    return (run_shapley,)


@app.cell
def _(
    harmonic_inputs,
    regressor_knots,
    regressor_lags,
    regressor_toggles,
    run_shapley,
    station_data,
    test_end,
    train_end,
    train_start,
):
    mo.stop(not run_shapley.value, mo.md("*Click above to compute Shapley values for active components.*"))
    _window_message = build_model_window_validation_message(
        station_data["date_min"],
        station_data["date_max"],
        train_start.value,
        train_end.value,
        test_end.value,
    )
    mo.stop(_window_message is not None, mo.md(f"*{_window_message}*"))
    _component_mask, _model_kwargs = collect_model_params(
        harmonic_inputs,
        regressor_lags,
        regressor_toggles,
        regressor_knots,
        df=station_data["df"],
        sph=station_data["sph"],
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
    )
    _components = [name for name, active in _component_mask.items() if active]
    _num_components = len(_components)

    mo.stop(_num_components < 2, mo.md("*Need at least 2 active components for Shapley analysis.*"))
    mo.stop(
        _num_components > 12,
        mo.md(f"*{_num_components} components -> {2**_num_components:,} model runs - cap at 12.*"),
    )

    _run_model = partial(run_tidal_model, **_model_kwargs)
    _total_coalitions = 2**_num_components
    _failed_runs = 0
    _baseline_metrics = _run_model({component: False for component in _components})["metrics_test"]
    _coalition_metrics: dict[int, dict[str, float]] = {0: _baseline_metrics}
    _max_workers = min(os.cpu_count() or 4, _total_coalitions, 8)

    with mo.status.progress_bar(total=_total_coalitions) as _progress:
        _progress.update()
        _coalition_masks = {
            bits: {component: bool(bits & (1 << idx)) for idx, component in enumerate(_components)}
            for bits in range(1, 2**_num_components)
        }
        with ThreadPoolExecutor(max_workers=_max_workers) as _pool:
            _futures = {
                _pool.submit(_run_model, coalition_mask): bits
                for bits, coalition_mask in _coalition_masks.items()
            }
            for _future in as_completed(_futures):
                _bits = _futures[_future]
                _metrics = _future.result()["metrics_test"]
                if np.isnan(_metrics.get("r2", 0.0)):
                    _failed_runs += 1
                _coalition_metrics[_bits] = _metrics
                _progress.update()

    _baseline_r2 = _baseline_metrics.get("r2", 0.0)
    if not np.isfinite(_baseline_r2):
        _baseline_r2 = 0.0
    _baseline_rmse = _baseline_metrics.get("rmse", np.nan)
    if not np.isfinite(_baseline_rmse):
        _baseline_rmse = np.nan
    _full_mask = 2**_num_components - 1
    shapley_result: ShapleyResult = {
        "components": _components,
        "coalitions": _total_coalitions,
        "failed": _failed_runs,
        "baseline_r2": _baseline_r2,
        "baseline_rmse": _baseline_rmse,
        "full_r2": _coalition_metrics[_full_mask].get("r2", 0.0)
        if np.isfinite(_coalition_metrics[_full_mask].get("r2", 0.0))
        else 0.0,
        "full_rmse": _coalition_metrics[_full_mask].get("rmse", _baseline_rmse),
        "shap_r2": compute_shapley(_coalition_metrics, _components, "r2", _baseline_r2),
        "shap_rmse": compute_shapley(_coalition_metrics, _components, "rmse", _baseline_rmse),
    }
    return (shapley_result,)


@app.cell
def _(shapley_result: ShapleyResult):
    build_shapley_figure(shapley_result)
    return


@app.function
def available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns and df[col].notna().any()]


@app.function
def missing_lcd_weather_years(
    data_dir: Path | str,
    station_id: str,
    begin_year: int,
    end_year: int,
) -> list[int]:
    data_dir = Path(data_dir)
    return [
        year
        for year in range(begin_year, end_year + 1)
        if not (data_dir / f"lcd_{station_id}_{year}.csv").exists()
    ]


@app.function
def weather_frame_covers_window(
    weather_df: pd.DataFrame,
    window_start: date,
    window_end: date,
) -> bool:
    if weather_df.empty:
        return False
    weather_index = pd.DatetimeIndex(weather_df.index)
    if weather_index.tz is not None:
        weather_index = weather_index.tz_convert(None)
    return weather_index.min().date() <= window_start and weather_index.max().date() >= window_end


@app.function
def build_model_date_defaults(date_min: date, date_max: date) -> tuple[date, date, date]:
    if date_min > date_max:
        raise ValueError("date_min must be on or before date_max")

    window_days = (date_max - date_min).days + 1
    if window_days == 1:
        return date_min, date_min, date_max

    preferred_train_end = pd.Timestamp("2024-01-01").date()
    if date_min <= preferred_train_end <= date_max:
        train_end = preferred_train_end
    else:
        midpoint_days = (date_max - date_min).days // 2
        train_end = date_min + timedelta(days=midpoint_days)

    if train_end <= date_min:
        train_end = date_min + timedelta(days=1)
    return date_min, train_end, date_max


@app.function
def build_model_window_validation_message(
    date_min: date,
    date_max: date,
    train_start: date,
    train_end: date,
    test_end: date,
) -> str | None:
    if date_min == date_max:
        return (
            "The loaded window is single-day. "
            "Load at least two days of data to create non-empty train and test splits."
        )
    if train_end <= train_start:
        return "`Train end` must be after `Train start` so the model has at least one training day."
    if test_end < train_end:
        return "`Test end` must be on or after `Train end` so the model has at least one test day."

    train_window_start = max(date_min, train_start)
    train_window_end = min(date_max, train_end - timedelta(days=1))
    if train_window_start > train_window_end:
        return "`Train end` must leave at least one loaded day before the split."

    test_window_start = max(date_min, train_end)
    test_window_end = min(date_max, test_end)
    if test_window_start > test_window_end:
        return "`Test end` must include at least one loaded day on or after `Train end`."

    return None


@app.function
def load_station_frame(
    station_name: str,
    use_weather: bool,
    download_tidal: bool,
    download_weather: bool,
    data_start: date | None = None,
    data_end: date | None = None,
) -> StationData:
    if (data_start is None) != (data_end is None):
        raise ValueError("data_start and data_end must be provided together")
    if data_start is not None and data_end is not None and data_start > data_end:
        raise ValueError("data_start must be on or before data_end")

    station_id = find_station(station_name)
    station_label = STATION_CATALOG[station_id]["name"]
    status_messages: list[str] = []

    if data_start is None or data_end is None:
        try:
            df = load_station(station_id).copy()
            tidal_source = "cache"
        except FileNotFoundError:
            if not download_tidal:
                raise
            data_file = download_tidal_data(DEFAULT_DATA_DIR, station=station_id)
            df = load_tidal_data(data_file).copy()
            tidal_source = "download"
    else:
        request_begin = data_start.strftime("%Y%m%d")
        request_end = data_end.strftime("%Y%m%d")
        request_window = f"{data_start} to {data_end}"
        data_file = resolve_tidal_cache_path(
            DEFAULT_DATA_DIR,
            station_id,
            request_begin,
            request_end,
        )
        if data_file.exists():
            tidal_source = "cache"
        elif download_tidal:
            data_file = download_tidal_data(
                DEFAULT_DATA_DIR,
                station=station_id,
                begin_date=request_begin,
                end_date=request_end,
            )
            tidal_source = "download"
        else:
            raise FileNotFoundError(
                f"No tidal data found for requested tidal window {request_window}"
            )
        df = load_tidal_data(data_file).copy()

    time_index = pd.DatetimeIndex(df.index)
    if time_index.tz is not None:
        time_index = time_index.tz_convert(None)
    df.index = time_index
    if "pressure" in df.columns:
        df["dp_dt"] = df["pressure"].diff()
    if data_start is not None and data_end is not None:
        df = df.loc[str(data_start) : str(data_end)].copy()
        if df.empty:
            raise FileNotFoundError(
                f"No tidal data found for requested tidal window {data_start} to {data_end}"
            )
        time_index = pd.DatetimeIndex(df.index)
    status_messages.append(f"Loaded tidal data from {tidal_source} for {station_label}")

    weather_station = TIDE_TO_WEATHER.get(station_id)
    weather_start = data_start if data_start is not None else time_index[0].date()
    weather_end = data_end if data_end is not None else time_index[-1].date()
    if use_weather and weather_station:
        weather_station_id, weather_label = weather_station

        def weather_window_error(weather_df: pd.DataFrame | None = None) -> FileNotFoundError:
            message = (
                "No LCD weather data found for requested LCD weather window "
                f"{weather_start} to {weather_end}"
            )
            if weather_df is None or weather_df.empty:
                return FileNotFoundError(message)
            weather_index = pd.DatetimeIndex(weather_df.index)
            if weather_index.tz is not None:
                weather_index = weather_index.tz_convert(None)
            return FileNotFoundError(
                f"{message} (loaded coverage: {weather_index.min().date()} to {weather_index.max().date()})"
            )

        missing_weather_years = missing_lcd_weather_years(
            DEFAULT_DATA_DIR,
            weather_station_id,
            weather_start.year,
            weather_end.year,
        )
        if missing_weather_years:
            if not download_weather:
                missing_year_text = ", ".join(str(year) for year in missing_weather_years)
                raise FileNotFoundError(
                    "No LCD weather data found for requested LCD weather window "
                    f"{weather_start} to {weather_end} "
                    f"(missing yearly files: {missing_year_text})"
                )
            download_lcd_weather(
                DEFAULT_DATA_DIR,
                station_id=weather_station_id,
                begin_year=weather_start.year,
                end_year=weather_end.year,
            )
            missing_weather_years = missing_lcd_weather_years(
                DEFAULT_DATA_DIR,
                weather_station_id,
                weather_start.year,
                weather_end.year,
            )
            if missing_weather_years:
                missing_year_text = ", ".join(str(year) for year in missing_weather_years)
                raise FileNotFoundError(
                    "No LCD weather data found for requested LCD weather window "
                    f"{weather_start} to {weather_end} "
                    f"(missing yearly files: {missing_year_text})"
                )
            weather_source = "download"
        else:
            weather_source = "cache"

        try:
            weather_df = load_lcd_weather(
                DEFAULT_DATA_DIR,
                station_id=weather_station_id,
                begin_date=str(weather_start),
                end_date=str(weather_end),
            )
        except FileNotFoundError:
            raise weather_window_error() from None

        if not weather_frame_covers_window(weather_df, weather_start, weather_end):
            if not download_weather:
                raise weather_window_error(weather_df)
            download_lcd_weather(
                DEFAULT_DATA_DIR,
                station_id=weather_station_id,
                begin_year=weather_start.year,
                end_year=weather_end.year,
            )
            weather_source = "download"
            try:
                weather_df = load_lcd_weather(
                    DEFAULT_DATA_DIR,
                    station_id=weather_station_id,
                    begin_date=str(weather_start),
                    end_date=str(weather_end),
                )
            except FileNotFoundError:
                raise weather_window_error() from None
            if not weather_frame_covers_window(weather_df, weather_start, weather_end):
                raise weather_window_error(weather_df)

        df = merge_tidal_weather(df, weather_df)
        status_messages.append(
            f"Merged {len(weather_df.columns)} weather columns from {weather_label} "
            f"via {weather_source}"
        )

    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_stress"] = df["wind_u"] ** 2 + df["wind_v"] ** 2

    time_index = pd.DatetimeIndex(df.index)
    loaded_start = time_index[0].date()
    loaded_end = time_index[-1].date()
    status_messages.append(f"Loaded window {loaded_start} to {loaded_end}")

    return {
        "df": df,
        "sph": infer_samples_per_hour(time_index),
        "date_min": loaded_start,
        "date_max": loaded_end,
        "status_message": "; ".join(status_messages),
    }


@app.function
def build_overview_figure(
    df: pd.DataFrame,
    start_date,
    end_date,
    resample_rule: str | None,
) -> go.Figure:
    window = df[str(start_date) : str(end_date)]
    if resample_rule is not None:
        window = window.resample(resample_rule).mean()

    columns = available_columns(window, list(COLUMN_LABELS))
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for idx, column in enumerate(columns, 1):
        fig.add_trace(
            go.Scattergl(
                x=window.index,
                y=window[column],
                mode="lines",
                line=dict(width=0.8),
                name=column,
            ),
            row=idx,
            col=1,
        )
        fig.update_yaxes(title_text=COLUMN_LABELS.get(column, column), row=idx, col=1)

    points = len(window.dropna(how="all"))
    suffix = f", {resample_rule}" if resample_rule else ""
    fig.update_layout(
        height=180 * len(columns),
        title=f"{points:,} points ({start_date} - {end_date}{suffix})",
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


@app.function
def build_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    columns = available_columns(df, CORRELATION_CANDIDATES)
    corr = df[columns].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Pairwise Pearson correlation",
        width=520,
        height=520,
        margin=dict(l=100, r=20, t=40, b=100),
    )
    return fig


@app.function
def build_periodogram_selector_options(df: pd.DataFrame) -> tuple[dict[str, str], str]:
    options = {
        COLUMN_LABELS.get(column, column): column
        for column in available_columns(df, list(COLUMN_LABELS))
    }
    default_name = COLUMN_LABELS.get("water_level", "water_level")
    if default_name not in options:
        default_name = next(iter(options))
    return options, default_name


@app.function
def build_periodogram_figure(
    index: pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
    title: str,
    *,
    min_period_hours: float = 2.0,
    max_period_hours: float | None = None,
) -> go.Figure:
    spectrum = compute_periodogram(
        pd.DatetimeIndex(index),
        values,
        min_period_hours=min_period_hours,
        max_period_hours=max_period_hours,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spectrum["period_hours"],
            y=spectrum["power"],
            mode="lines",
            line=dict(width=1.5, color="steelblue"),
            name="power",
        )
    )

    axis_max = float(spectrum["period_hours"].max()) if not spectrum.empty else max_period_hours or 48.0
    for label, period_hours in PERIODS.items():
        if period_hours < min_period_hours or period_hours > axis_max:
            continue
        fig.add_vline(x=period_hours, line_width=0.8, line_color="gray", line_dash="dot")
        fig.add_annotation(
            x=period_hours,
            y=1.0,
            yref="paper",
            text=label,
            textangle=-90,
            showarrow=False,
            yanchor="top",
            xanchor="right",
            font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Period (hours)",
        yaxis_title="Power",
        margin=dict(l=60, r=20, t=50, b=50),
        height=360,
        showlegend=False,
    )
    return fig


@app.function
def collect_model_params(
    harmonic_inputs,
    regressor_lags,
    regressor_toggles,
    regressor_knots,
    *,
    df: pd.DataFrame,
    sph: int,
    train_start,
    train_end,
    test_end,
) -> tuple[dict[str, bool], ModelKwargs]:
    mask = {name: int(widget.value) > 0 for name, widget in harmonic_inputs.items()}
    mask.update({name: widget.value for name, widget in regressor_toggles.items()})
    return mask, {
        "df": df,
        "sph": sph,
        "harmonic_orders": {name: int(widget.value) for name, widget in harmonic_inputs.items()},
        "lag_ranges": {name: widget.value for name, widget in regressor_lags.items()},
        "knot_presets": {name: widget.value for name, widget in regressor_knots.items()},
        "train_start": str(train_start.value),
        "train_end": str(train_end.value),
        "test_end": str(test_end.value),
    }


@app.function
def option_name_for_value(options: Mapping[str, str], value: str) -> str:
    for option_name, option_value in options.items():
        if option_value == value:
            return option_name
    raise ValueError(f"No option name found for value: {value!r}")


@app.function
def build_knot_count(preset: str) -> int:
    try:
        return KNOT_PRESET_TO_COUNT[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown knot preset: {preset}") from exc


@app.function
def build_spline_basis_matrix(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)

    def truncated_cubic_distance(values: np.ndarray, knot: float, knot_max: float) -> np.ndarray:
        numerator = np.clip(np.power(values - knot, 3), 0.0, np.inf)
        numerator -= np.clip(np.power(values - knot_max, 3), 0.0, np.inf)
        return numerator / (knot_max - knot)

    basis = np.ones((len(x), len(knots)), dtype=float)
    basis[:, 1] = x
    for knot_idx in range(len(knots) - 2):
        basis_col = knot_idx + 2
        basis[:, basis_col] = truncated_cubic_distance(x, knots[knot_idx], knots[-1]) - truncated_cubic_distance(
            x,
            knots[-2],
            knots[-1],
        )
    return basis[:, 1:]


@app.function
def build_regressor_basis_inputs(
    regressor: pd.Series,
    knot_preset: str,
    *,
    basis_regressor: pd.Series | None = None,
    grid_size: int = 200,
) -> RegressorBasisInputs:
    basis_regressor = regressor if basis_regressor is None else basis_regressor
    clean_basis_regressor = basis_regressor.dropna()
    if clean_basis_regressor.empty:
        raise ValueError("regressor must contain at least one finite value")

    knot_count = build_knot_count(knot_preset)
    regressor_min = float(clean_basis_regressor.min())
    regressor_max = float(clean_basis_regressor.max())
    if not np.isfinite(regressor_min) or not np.isfinite(regressor_max):
        raise ValueError("Selected regressor must have finite values for basis inspection.")
    if np.isclose(regressor_min, regressor_max):
        regressor_name = regressor.name or basis_regressor.name or "Selected regressor"
        basis_window_label = "loaded window" if basis_regressor is regressor else "processed training window"
        raise ValueError(
            f"{regressor_name} is constant over the {basis_window_label}; basis inspection requires variation."
        )
    knots = np.linspace(regressor_min, regressor_max, knot_count)
    grid = np.linspace(regressor_min, regressor_max, grid_size)
    basis = build_spline_basis_matrix(grid, knots)
    return {
        "regressor_name": regressor.name or basis_regressor.name or "regressor",
        "index": pd.DatetimeIndex(regressor.index),
        "values": regressor.to_numpy(dtype=float),
        "knots": knots,
        "grid": grid,
        "basis": basis,
    }


@app.function
def split_model_window(
    df: pd.DataFrame,
    train_start: str | date,
    train_end: str | date,
    test_end: str | date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_time = pd.Timestamp(train_end)
    window = df[str(train_start) : str(test_end)]
    df_train = window[window.index < split_time]
    df_test = window[window.index >= split_time]
    ok_train = df_train["water_level"].notna()
    ok_test = df_test["water_level"].notna()
    return df_train, df_test, ok_train, ok_test


@app.function
def build_model_regressor_basis_inputs(
    df: pd.DataFrame,
    regressor_name: str,
    knot_preset: str,
    train_start: str | date,
    train_end: str | date,
    test_end: str | date,
    *,
    grid_size: int = 200,
) -> RegressorBasisInputs:
    df_train, df_test, ok_train, _ok_test = split_model_window(
        df,
        train_start,
        train_end,
        test_end,
    )
    x_train_fit, _x_train_pred, _x_test_pred, active_regs, _exog_config = build_exog_design_matrices(
        df_train,
        df_test,
        ok_train,
        [regressor_name],
        {regressor_name: (0, 0)},
        {regressor_name: knot_preset},
        1,
    )
    if regressor_name not in active_regs or regressor_name not in x_train_fit:
        raise ValueError(
            f"{regressor_name} is unavailable after model preprocessing for the current train/test window."
        )
    if x_train_fit.empty:
        raise ValueError(
            f"{regressor_name} has no usable training samples after model preprocessing for the current train/test window."
        )
    return build_regressor_basis_inputs(
        df[regressor_name],
        knot_preset,
        basis_regressor=x_train_fit[regressor_name],
        grid_size=grid_size,
    )


@app.function
def build_regressor_basis_figure(
    basis_inputs: RegressorBasisInputs,
    *,
    knot_preset: str,
) -> go.Figure:
    regressor_name = basis_inputs["regressor_name"]
    regressor_label = COLUMN_LABELS.get(regressor_name, regressor_name)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=(
            f"{regressor_label} over loaded window",
            f"Spline basis implied by `{knot_preset}` knots on processed training regressor",
        ),
    )
    figure.add_trace(
        go.Scattergl(
            x=basis_inputs["index"],
            y=basis_inputs["values"],
            mode="lines",
            line=dict(width=0.9, color="steelblue"),
            name="loaded values",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for basis_idx in range(basis_inputs["basis"].shape[1]):
        figure.add_trace(
            go.Scatter(
                x=basis_inputs["grid"],
                y=basis_inputs["basis"][:, basis_idx],
                mode="lines",
                line=dict(width=1.0),
                name=f"basis {basis_idx + 1}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    for knot in basis_inputs["knots"]:
        figure.add_vline(
            x=float(knot),
            line_width=0.8,
            line_color="gray",
            line_dash="dot",
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text=regressor_label, row=1, col=1)
    figure.update_yaxes(title_text="Basis weight", row=2, col=1)
    figure.update_xaxes(title_text="Time", row=1, col=1)
    figure.update_xaxes(title_text=regressor_label, row=2, col=1)
    figure.update_layout(
        height=650,
        title=f"Regressor inspection: {regressor_label}",
        margin=dict(l=60, r=20, t=70, b=40),
    )
    return figure


@app.function
def build_periodic_config(
    component_mask: dict[str, bool],
    harmonic_orders: dict[str, int],
    sph: int,
) -> tuple[dict[str, tuple[float, int]], TsgamMultiPeriodicConfig | None]:
    picked: dict[str, tuple[float, int]] = {}
    for name, active in component_mask.items():
        if not active or name not in PERIODS:
            continue
        order = harmonic_orders.get(name, 0)
        if order > 0:
            picked[name] = (PERIODS[name], order)

    config = None
    if picked:
        config = TsgamMultiPeriodicConfig(
            periods=[period * sph for period, _ in picked.values()],
            num_harmonics=[order for _, order in picked.values()],
            reg_weight=1e-4,
        )
    return picked, config


@app.function
def build_exog_design_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ok_train: pd.Series,
    reg_names: list[str],
    lag_ranges: dict[str, tuple[int, int]],
    knot_presets: dict[str, str],
    sph: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[TsgamSplineConfig | TsgamLinearConfig] | None]:
    x_train_fit = pd.DataFrame(index=df_train.index[ok_train])
    x_train_pred = pd.DataFrame(index=df_train.index)
    x_test_pred = pd.DataFrame(index=df_test.index)
    if not reg_names:
        return x_train_fit, x_train_pred, x_test_pred, [], None

    x_train_raw, x_test_raw, active_regs, _ = prepare_split_regressors(df_train, df_test, reg_names)
    if not active_regs:
        return x_train_fit, x_train_pred, x_test_pred, [], None

    exog_config = []
    for column in active_regs:
        lag_start, lag_end = lag_ranges.get(column, (-2, 0))
        try:
            knot_preset = knot_presets[column]
        except KeyError as exc:
            raise ValueError(f"Missing knot preset for active regressor: {column}") from exc
        knot_count = build_knot_count(knot_preset)
        exog_config.append(
            TsgamSplineConfig(
                n_knots=knot_count,
                lags=[hour * sph for hour in range(lag_start, lag_end + 1)],
                reg_weight=1e-5,
                diff_reg_weight=0.3,
            )
        )

    return (
        x_train_raw.loc[ok_train],
        x_train_raw,
        x_test_raw,
        active_regs,
        exog_config,
    )


@app.function
def pack_model_result(
    df_test: pd.DataFrame,
    y_train: np.ndarray,
    ok_train: pd.Series,
    ok_test: pd.Series,
    yhat_train: np.ndarray,
    yhat_test: np.ndarray,
    picked: dict[str, tuple[float, int]],
    active_regs: list[str],
    sph: int,
) -> FitResult:
    ok_train_np = ok_train.to_numpy(dtype=bool)
    ok_test_np = ok_test.to_numpy(dtype=bool)
    te_index = pd.DatetimeIndex(df_test.index)
    test_obs = df_test["water_level"].to_numpy(dtype=float)
    test_obs_clean = df_test.loc[ok_test, "water_level"].to_numpy(dtype=float)
    train_metrics = cast(MetricDict, tidal_metrics(y_train, yhat_train[ok_train_np]))
    test_metrics = cast(MetricDict, tidal_metrics(test_obs_clean, yhat_test[ok_test_np]))
    return cast(FitResult, {
        "metrics_train": train_metrics,
        "metrics_test": test_metrics,
        "te_index": te_index,
        "te_obs": test_obs,
        "te_pred": yhat_test,
        "te_obs_clean": test_obs_clean,
        "te_pred_clean": yhat_test[ok_test_np],
        "residuals": test_obs - yhat_test,
        "picked": picked,
        "active_regs": active_regs,
        "n_train": len(y_train),
        "n_test": int(ok_test.sum()),
        "sph": sph,
    })


@app.function
def run_tidal_model(
    component_mask: dict[str, bool],
    *,
    df: pd.DataFrame,
    sph: int,
    harmonic_orders: dict[str, int],
    lag_ranges: dict[str, tuple[int, int]],
    knot_presets: dict[str, str],
    train_start: str,
    train_end: str,
    test_end: str,
) -> FitResult:
    df_train, df_test, ok_train, ok_test = split_model_window(
        df,
        train_start,
        train_end,
        test_end,
    )
    y_train = df_train.loc[ok_train, "water_level"].to_numpy(dtype=float)

    picked, periodic_config = build_periodic_config(component_mask, harmonic_orders, sph)
    reg_names = [name for name, active in component_mask.items() if active and name not in PERIODS]
    x_train_fit, x_train_pred, x_test_pred, active_regs, exog_config = build_exog_design_matrices(
        df_train,
        df_test,
        ok_train,
        reg_names,
        lag_ranges,
        knot_presets,
        sph,
    )

    if periodic_config is None and exog_config is None:
        baseline = float(np.nanmean(y_train))
        return pack_model_result(
            df_test,
            y_train,
            ok_train,
            ok_test,
            np.full(len(df_train), baseline),
            np.full(len(df_test), baseline),
            picked,
            active_regs,
            sph,
        )

    try:
        model = TsgamEstimator(
            TsgamEstimatorConfig(
                multi_periodic_config=periodic_config,
                exog_config=exog_config,
                solver_config=TsgamSolverConfig(solver="SCS", verbose=False),
            )
        )
        model.fit(x_train_fit, y_train)
        return pack_model_result(
            df_test,
            y_train,
            ok_train,
            ok_test,
            model.predict(x_train_pred),
            model.predict(x_test_pred),
            picked,
            active_regs,
            sph,
        )
    except Exception:
        return pack_model_result(
            df_test,
            y_train,
            ok_train,
            ok_test,
            np.full(len(df_train), np.nan),
            np.full(len(df_test), np.nan),
            picked,
            active_regs,
            sph,
        )


@app.function
def build_fit_label(fit_result: FitResult) -> str:
    parts = []
    picked = list(fit_result["picked"])
    active_regs = fit_result["active_regs"]
    if picked:
        parts.append(", ".join(picked))
    if active_regs:
        parts.append(f"+ {', '.join(active_regs)}")
    summary = " ".join(parts) if parts else "Mean baseline"
    return f"{summary} - {fit_result['n_train']:,} train / {fit_result['n_test']:,} test"


@app.function
def format_metric_value(metric_name: str, value: float) -> str:
    _, unit, digits = METRIC_SPECS[metric_name]
    if not np.isfinite(value):
        return "nan"
    value_text = f"{value:.{digits}f}"
    return f"{value_text} {unit}".strip()


@app.function
def build_metrics_table_html(fit_result: FitResult) -> str:
    rows = []
    for metric_name, (label, _, _) in METRIC_SPECS.items():
        rows.append(
            {
                "Metric": label,
                "Train": format_metric_value(metric_name, fit_result["metrics_train"].get(metric_name, np.nan)),
                "Test": format_metric_value(metric_name, fit_result["metrics_test"].get(metric_name, np.nan)),
            }
        )
    return pd.DataFrame(rows).to_html(index=False)


@app.function
def build_fit_timeseries_figure(fit_label: str, fit_result: FitResult) -> go.Figure:
    time_index = fit_result["te_index"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(
        go.Scattergl(
            x=time_index,
            y=fit_result["te_obs"],
            mode="lines",
            line=dict(width=0.8, color="steelblue"),
            name="observed",
            opacity=0.8,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=time_index,
            y=fit_result["te_pred"],
            mode="lines",
            line=dict(width=0.8, color="coral"),
            name="predicted",
            opacity=0.8,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=time_index,
            y=fit_result["residuals"],
            mode="lines",
            line=dict(width=0.6, color="seagreen"),
            name="residual",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_width=0.5, line_color="black", row=2, col=1)
    fig.update_yaxes(title_text="Water level (m)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (m)", row=2, col=1)
    fig.update_layout(
        height=500,
        title=fit_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=30),
    )
    return fig


@app.function
def build_pred_vs_obs_figure(fit_result: FitResult) -> go.Figure:
    observed = fit_result["te_obs_clean"]
    predicted = fit_result["te_pred_clean"]
    lo = min(predicted.min(), observed.min())
    hi = max(predicted.max(), observed.max())
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=predicted,
            y=observed,
            mode="markers",
            marker=dict(size=2, opacity=0.15, color="black"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", width=0.8, color="black"),
            showlegend=False,
        )
    )
    fig.update_layout(
        width=450,
        height=450,
        title="Predicted vs Observed",
        xaxis=dict(title="Predicted (m)", scaleanchor="y", range=[lo, hi]),
        yaxis=dict(title="Observed (m)", range=[lo, hi]),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


@app.function
def build_residual_qq_figure(fit_result: FitResult) -> go.Figure:
    residuals = fit_result["residuals"]
    clean = residuals[np.isfinite(residuals)]
    (theoretical, sample), (slope, intercept, _) = stats.probplot(clean, dist="norm")
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=theoretical,
            y=sample,
            mode="markers",
            marker=dict(size=3, opacity=0.3, color="steelblue"),
            name="residuals",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=slope * theoretical + intercept,
            mode="lines",
            line=dict(color="red", width=1, dash="dash"),
            name="normal ref",
        )
    )
    fig.update_layout(
        width=450,
        height=450,
        title="Q-Q plot (residuals)",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Sample quantiles (m)",
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


@app.function
def build_residual_hist_figure(fit_result: FitResult) -> go.Figure:
    residuals = fit_result["residuals"]
    clean = residuals[np.isfinite(residuals)]
    mean = clean.mean()
    sigma = clean.std()
    x_grid = np.linspace(clean.min(), clean.max(), 200)
    bin_width = (clean.max() - clean.min()) / 80

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clean,
            nbinsx=80,
            name="residuals",
            marker_color="steelblue",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=stats.norm.pdf(x_grid, mean, sigma) * len(clean) * bin_width,
            mode="lines",
            line=dict(color="red", width=1.5),
            name=f"N({mean:.3f}, {sigma:.3f}^2)",
        )
    )
    fig.update_layout(
        title="Residual distribution",
        xaxis_title="Residual (m)",
        yaxis_title="Count",
        margin=dict(l=60, r=20, t=40, b=50),
        height=350,
        barmode="overlay",
    )
    return fig


@app.function
def build_rolling_rmse_figure(fit_result: FitResult) -> go.Figure:
    residuals = fit_result["residuals"]
    time_index = fit_result["te_index"]
    samples_per_hour = fit_result["sph"]
    window = int(168 * samples_per_hour)
    rolling_rmse = (
        pd.Series(residuals ** 2, index=time_index)
        .rolling(window, min_periods=window // 2)
        .mean()
        .pipe(np.sqrt)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=rolling_rmse.index,
            y=rolling_rmse.values,
            mode="lines",
            line=dict(width=0.8, color="steelblue"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Rolling RMSE (7-day window)",
        yaxis_title="RMSE (m)",
        height=300,
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


@app.function
def build_residual_regressor_xcorr_figure(
    df: pd.DataFrame,
    fit_result: FitResult,
) -> go.Figure | None:
    regressors = available_columns(df, REGRESSOR_ANALYSIS_CANDIDATES)
    if not regressors:
        return None

    residuals = fit_result["residuals"]
    time_index = fit_result["te_index"]
    samples_per_hour = fit_result["sph"]
    ok = np.isfinite(residuals)
    reg_df = df.loc[time_index[ok]]
    max_lag = int(12 * samples_per_hour)

    fig = go.Figure()
    for column in regressors:
        lagged_corr = compute_lagged_correlation(
            np.asarray(residuals[ok], dtype=float),
            np.asarray(reg_df[column].values, dtype=float),
            max_lag,
        )
        lagged_corr["lag_hours"] = lagged_corr["lag"] / samples_per_hour
        fig.add_trace(
            go.Scatter(
                x=lagged_corr["lag_hours"],
                y=lagged_corr["correlation"],
                mode="lines",
                name=column,
                line=dict(width=1.5),
            )
        )

    fig.add_hline(y=0, line_width=0.5, line_color="black")
    fig.add_vline(x=0, line_width=2.0, line_color="red")
    fig.update_layout(
        title="Residual-regressor cross-correlation (+/-12 h)",
        xaxis_title="Lag (hours)",
        yaxis_title="Pearson r",
        height=400,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


@app.function
def build_diagnostic_figures(
    df: pd.DataFrame,
    fit_result: FitResult,
) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {
        "Pred vs Obs": build_pred_vs_obs_figure(fit_result),
        "Q-Q": build_residual_qq_figure(fit_result),
        "Residual dist": build_residual_hist_figure(fit_result),
        "Rolling RMSE": build_rolling_rmse_figure(fit_result),
        "Residual spectrum": build_periodogram_figure(
            fit_result["te_index"],
            fit_result["residuals"],
            title="Residual spectrum",
            min_period_hours=2.0,
        ),
    }
    xcorr_figure = build_residual_regressor_xcorr_figure(df, fit_result)
    if xcorr_figure is not None:
        figures["Resid x regressor"] = xcorr_figure
    return figures


@app.function
def compute_shapley(
    results: dict[int, dict[str, float]],
    components: list[str],
    metric: str,
    baseline: float,
) -> dict[str, float]:
    values = {0: baseline}
    for bits, metrics in results.items():
        value = metrics.get(metric, baseline)
        values[bits] = value if np.isfinite(value) else baseline

    shapley: dict[str, float] = {}
    num_components = len(components)
    for idx, component in enumerate(components):
        contribution = 0.0
        for subset in range(2**num_components):
            if subset & (1 << idx):
                continue
            subset_size = bin(subset).count("1")
            weight = factorial(subset_size) * factorial(num_components - subset_size - 1) / factorial(num_components)
            contribution += weight * (values[subset | (1 << idx)] - values[subset])
        shapley[component] = contribution
    return shapley


@app.function
def build_shapley_figure(shapley_result: ShapleyResult) -> go.Figure:
    components = shapley_result["components"]
    shap_r2 = shapley_result["shap_r2"]
    shap_rmse = shapley_result["shap_rmse"]
    baseline_r2 = shapley_result["baseline_r2"]
    baseline_rmse = shapley_result["baseline_rmse"]
    full_r2 = shapley_result["full_r2"]
    full_rmse = shapley_result["full_rmse"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["R^2 attribution", "RMSE attribution"])
    for col_idx, (shapley_values, baseline, full_value, ylabel) in enumerate(
        [
            (shap_r2, baseline_r2, full_r2, "R^2"),
            (shap_rmse, baseline_rmse, full_rmse, "RMSE (m)"),
        ],
        1,
    ):
        ordered = sorted(shapley_values.items(), key=lambda item: abs(item[1]), reverse=True)
        names = [item[0] for item in ordered]
        values = [item[1] for item in ordered]
        fig.add_trace(
            go.Waterfall(
                x=["baseline"] + names + ["full model"],
                y=[baseline] + values + [0.0],
                measure=["absolute"] + ["relative"] * len(components) + ["total"],
                text=[f"{value:.4f}" for value in [baseline] + values + [full_value]],
                textposition="outside",
                increasing=dict(marker_color="steelblue" if col_idx == 1 else "coral"),
                decreasing=dict(marker_color="coral" if col_idx == 1 else "steelblue"),
                totals=dict(marker_color="midnightblue"),
                connector=dict(line=dict(color="gray", width=0.5, dash="dot")),
            ),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(title_text=ylabel, row=1, col=col_idx)

    title = f"Shapley component attribution ({shapley_result['coalitions']} coalitions)"
    if shapley_result["failed"]:
        title += f", {shapley_result['failed']} solver failures"
    fig.update_layout(
        title=title,
        showlegend=False,
        height=max(400, 40 * len(components)),
        margin=dict(l=60, r=40, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


if __name__ == "__main__":
    app.run()
