#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Tidal Water Level Prediction with TSGAM

Demonstrates predicting tidal water levels at NOAA tide gauge stations using:
- Constituent-aware multi-periodic Fourier basis for astronomical tides
  (M2, S2, N2, K1, O1, Mf, Mm, annual)
- Meteorological regressors (barometric pressure, water temperature, wind)
  to capture the non-tidal residual (storm surge, inverse barometer effect)
- Optional linear trend for long-term sea level change

Astronomical tides are dominated by periodic gravitational forcing from the
Moon and Sun, mapping naturally to TSGAM's multi-periodic Fourier basis.
The non-tidal residual—storm surge, inverse barometer effect, wind setup—is
driven by meteorological forcing captured by exogenous regressor splines.

Data sources:
  Tide station data (water level, pressure, water temp):
    NOAA CO-OPS API — https://tidesandcurrents.noaa.gov/api/
  Historical weather station data (wind, air temp, dew point):
    NOAA NCEI Local Climatological Data (LCD) — https://www.ncei.noaa.gov/
    products/land-based-station/local-climatological-data

Station map (tide gauge data availability):
  https://tidesandcurrents.noaa.gov/map/index.html
"""

import json
import sys
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tsgam_estimator import (
    PERIOD_HOURLY_YEARLY,
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
)


# ---------------------------------------------------------------------------
# Tidal period constants (hours, for use with hourly-sampled data)
# ---------------------------------------------------------------------------

PERIOD_HOURLY_SEMIDIURNAL = 12.4206
"""M2 principal lunar semidiurnal constituent (~12 h 25 min)."""

PERIOD_HOURLY_N2 = 12.65835
"""N2 larger lunar elliptic semidiurnal constituent (~12 h 40 min)."""

PERIOD_HOURLY_DIURNAL = 24.0
"""Solar day.  With harmonics: n=1 captures K1/P1 diurnal band,
n=2 captures S2 solar semidiurnal (12.0 h)."""

PERIOD_HOURLY_K1 = 23.93447
"""K1 lunisolar diurnal constituent (~23 h 56 min)."""

PERIOD_HOURLY_O1 = 25.81934
"""O1 principal lunar diurnal constituent (~25 h 49 min)."""

PERIOD_HOURLY_P1 = 24.06589
"""P1 principal solar diurnal constituent (~24 h 4 min)."""

PERIOD_HOURLY_MF = 327.85994
"""Mf lunar fortnightly constituent (~13.66 days)."""

PERIOD_HOURLY_FORTNIGHTLY = 354.37
"""MSf spring-neap cycle (~14.77 days).  Modulation of tidal range
due to lunar-solar alignment."""

PERIOD_HOURLY_MM = 661.31
"""Mm lunar monthly constituent (~27.55 days)."""

TIDAL_CONSTITUENT_PERIODS_HOURS = OrderedDict(
    [
        ('M2', PERIOD_HOURLY_SEMIDIURNAL),
        ('S2', PERIOD_HOURLY_DIURNAL / 2),
        ('N2', PERIOD_HOURLY_N2),
        ('K1', PERIOD_HOURLY_K1),
        ('O1', PERIOD_HOURLY_O1),
        ('Mf', PERIOD_HOURLY_MF),
        ('Mm', PERIOD_HOURLY_MM),
        ('annual', PERIOD_HOURLY_YEARLY),
    ],
)
"""Named constituent periods used by the shared tidal Fourier basis."""

TIDAL_CONSTITUENT_HARMONICS = OrderedDict(
    [
        ('M2', 4),      # M4/M6/M8 overtides
        ('S2', 1),
        ('N2', 1),
        ('K1', 2),      # second harmonic reaches the K2 band
        ('O1', 1),
        ('Mf', 1),
        ('Mm', 1),
        ('annual', 2),  # annual + semiannual structure
    ],
)
"""Per-constituent harmonic counts for the shared tidal Fourier basis."""

TIDAL_COMPONENT_LABELS = list(TIDAL_CONSTITUENT_PERIODS_HOURS.keys())


def make_constituent_multi_periodic(samples_per_hour: float) -> TsgamMultiPeriodicConfig:
    """Build the shared constituent-aware Fourier config on the fit-time sample grid."""
    return TsgamMultiPeriodicConfig(
        num_harmonics=list(TIDAL_CONSTITUENT_HARMONICS.values()),
        periods=[p * samples_per_hour for p in TIDAL_CONSTITUENT_PERIODS_HOURS.values()],
        reg_weight=1e-5,
    )


# ---------------------------------------------------------------------------
# Station defaults
# ---------------------------------------------------------------------------

DEFAULT_STATION = '8518750'
DEFAULT_STATION_NAME = 'The Battery, NY'

NOAA_API_BASE = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'


# ---------------------------------------------------------------------------
# NCEI LCD weather station defaults
# ---------------------------------------------------------------------------

LCD_BASE_URL = 'https://www.ncei.noaa.gov/data/local-climatological-data/access'

DEFAULT_WEATHER_STATION = '72505394728'
DEFAULT_WEATHER_STATION_NAME = 'Central Park, NY'

TIDE_TO_WEATHER: dict[str, tuple[str, str]] = {
    '8518750': ('72505394728', 'Central Park, NY'),
    '8443970': ('72509014739', 'Boston Logan, MA'),
    '8665530': ('72208013880', 'Charleston Intl, SC'),
    '8771450': ('72251012924', 'Galveston Scholes, TX'),
    '9414290': ('72494023234', 'San Francisco Intl, CA'),
    '1612340': ('91182022521', 'Honolulu Intl, HI'),
}

DEFAULT_DATA_DIR = Path(__file__).parent / 'data' / 'tidal'


def find_station(query: str) -> str:
    """Look up a station ID by name substring (case-insensitive)."""
    if query in STATION_CATALOG:
        return query
    matches = [
        sid for sid, info in STATION_CATALOG.items()
        if query.lower() in info['name'].lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f'No station matching {query!r}')
    names = [STATION_CATALOG[s]['name'] for s in matches]
    raise KeyError(f'Ambiguous query {query!r}: {names}')


def load_station(
    query: str,
    data_dir: Path | str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load tidal data by station name or ID.

    >>> df = load_station("San Francisco")
    """
    sid = find_station(query)
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return load_tidal_data(data_dir / f'{sid}_combined.csv', **kwargs)


STATION_CATALOG: dict[str, dict] = {
    '8518750': {
        'name': 'The Battery, NY',
        'region': 'Northeast Atlantic',
        'tidal_regime': 'Semi-diurnal',
        'notes': 'Storm surge from nor\'easters; ~1.5 m tidal range',
    },
    '8443970': {
        'name': 'Boston, MA',
        'region': 'Northeast Atlantic',
        'tidal_regime': 'Semi-diurnal',
        'notes': 'Large tidal range (~3 m); strong spring-neap modulation',
    },
    '8665530': {
        'name': 'Charleston, SC',
        'region': 'Southeast Atlantic',
        'tidal_regime': 'Semi-diurnal',
        'notes': 'Hurricane-prone coast; ~1.5 m tidal range',
    },
    '8771450': {
        'name': 'Galveston, TX',
        'region': 'Gulf of Mexico',
        'tidal_regime': 'Diurnal/mixed',
        'notes': 'Small tidal range (~0.4 m); wind-driven surge dominant',
    },
    '9414290': {
        'name': 'San Francisco, CA',
        'region': 'Pacific',
        'tidal_regime': 'Mixed semi-diurnal',
        'notes': 'Strong diurnal inequality; ~1.5 m tidal range',
    },
    '1612340': {
        'name': 'Honolulu, HI',
        'region': 'Tropical Pacific',
        'tidal_regime': 'Mixed',
        'notes': 'Small tidal range (~0.5 m); minimal storm surge',
    },
}


# ---------------------------------------------------------------------------
# NOAA CO-OPS API helpers
# ---------------------------------------------------------------------------

_WATER_LEVEL_PRODUCTS = frozenset({
    'hourly_height', 'water_level', 'high_low', 'daily_mean',
    'monthly_mean', 'predictions',
})
_MET_PRODUCTS = frozenset({
    'air_pressure', 'water_temperature', 'wind',
    'air_temperature', 'humidity',
})


def tidal_cache_path(data_dir, station, begin_date, end_date):
    """Build the cache path for a station/date-range tidal CSV."""
    return Path(data_dir) / f'{station}_{begin_date}_{end_date}_combined.csv'


def _is_date_only_text(text: str) -> bool:
    stripped = str(text).strip()
    return ' ' not in stripped and 'T' not in stripped and ':' not in stripped


def _window_boundary(text, is_end: bool) -> pd.Timestamp:
    timestamp = pd.Timestamp(text)
    if is_end and _is_date_only_text(str(text)):
        return timestamp.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def resolve_tidal_cache_path(data_dir, station, begin_date, end_date):
    """Resolve the best cached tidal CSV for a requested date range."""
    preferred = tidal_cache_path(data_dir, station, begin_date, end_date)
    if preferred.exists():
        return preferred

    request_begin = pd.Timestamp(begin_date)
    request_end = pd.Timestamp(end_date)
    candidate_paths = [Path(data_dir) / f'{station}_combined.csv']
    candidate_paths.extend(sorted(Path(data_dir).glob(f'{station}_*_combined.csv')))

    covering: list[tuple[pd.Timestamp, pd.Timestamp, Path]] = []
    for candidate in candidate_paths:
        if not candidate.exists():
            continue

        parts = candidate.stem.split('_')
        if len(parts) == 4 and parts[0] == station and parts[-1] == 'combined':
            try:
                candidate_begin = pd.Timestamp(parts[1])
                candidate_end = pd.Timestamp(parts[2])
            except Exception:
                continue
            if (
                candidate_begin <= request_begin
                and candidate_end.normalize() >= request_end.normalize()
            ):
                covering.append((candidate_begin, candidate_end, candidate))
            continue

        try:
            index = pd.read_csv(
                candidate,
                usecols=['datetime'],
                parse_dates=['datetime'],
            )['datetime']
            if (
                not index.empty
                and index.min() <= request_begin
                and (
                    index.max().normalize() >= request_end.normalize()
                    if _is_date_only_text(str(end_date))
                    else index.max() >= request_end
                )
            ):
                covering.append((index.min(), index.max(), candidate))
        except Exception:
            continue

    if covering:
        covering.sort(key=lambda item: (item[1] - item[0], item[2].name))
        return covering[0][2]

    return preferred


def _noaa_url(station, product, begin_date, end_date):
    """Build a NOAA CO-OPS Data Retrieval API URL."""
    params = {
        'station': station,
        'product': product,
        'begin_date': begin_date,
        'end_date': end_date,
        'units': 'metric',
        'time_zone': 'gmt',
        'format': 'json',
        'application': 'tsgam_estimator',
    }
    if product in _WATER_LEVEL_PRODUCTS:
        params['datum'] = 'MLLW'
    if product in _MET_PRODUCTS:
        params['interval'] = 'h'
    qs = '&'.join(f'{k}={v}' for k, v in params.items())
    return f'{NOAA_API_BASE}?{qs}'


def _fetch_json(url):
    """Fetch and parse JSON from a URL."""
    req = urllib.request.Request(
        url, headers={'User-Agent': 'tsgam_estimator/0.1'},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _parse_water_level(records):
    rows = []
    for r in records:
        try:
            rows.append({
                'datetime': pd.Timestamp(r['t']),
                'water_level': float(r['v']),
            })
        except (ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(rows).set_index('datetime')


def _parse_scalar(records, col_name):
    rows = []
    for r in records:
        try:
            rows.append({
                'datetime': pd.Timestamp(r['t']),
                col_name: float(r['v']),
            })
        except (ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(rows).set_index('datetime')


def _parse_wind(records):
    rows = []
    for r in records:
        try:
            speed = float(r['s'])
            direction = float(r['d'])
        except (ValueError, KeyError, TypeError):
            continue
        # Met convention: direction is where wind blows FROM.
        # Convert to u (eastward) / v (northward) components.
        dir_rad = np.radians(direction)
        rows.append({
            'datetime': pd.Timestamp(r['t']),
            'wind_speed': speed,
            'wind_u': -speed * np.sin(dir_rad),
            'wind_v': -speed * np.cos(dir_rad),
        })
    return pd.DataFrame(rows).set_index('datetime')


def _year_chunks(begin_date, end_date):
    """Split a date range into <=1-year chunks (NOAA hourly data limit)."""
    start = pd.Timestamp(begin_date)
    end = pd.Timestamp(end_date)
    chunks = []
    while start <= end:
        chunk_end = min(
            start + pd.DateOffset(years=1) - pd.Timedelta('1d'), end,
        )
        chunks.append((start.strftime('%Y%m%d'), chunk_end.strftime('%Y%m%d')))
        start = chunk_end + pd.Timedelta('1d')
    return chunks


def _month_chunks(begin_date, end_date):
    """Split a date range into <=31-day chunks (NOAA 6-min data limit)."""
    start = pd.Timestamp(begin_date)
    end = pd.Timestamp(end_date)
    chunks = []
    while start <= end:
        chunk_end = min(start + pd.Timedelta(days=30), end)
        chunks.append((start.strftime('%Y%m%d'), chunk_end.strftime('%Y%m%d')))
        start = chunk_end + pd.Timedelta('1d')
    return chunks


def _fetch_product_chunks(station, product, parser, chunks):
    """Download one CO-OPS product across date chunks, return DataFrame or None."""
    print(f'  Fetching {product} for station {station} ...')
    parts: list[pd.DataFrame] = []
    for i, (b, e) in enumerate(chunks):
        url = _noaa_url(station, product, b, e)
        try:
            data = _fetch_json(url)
        except Exception as exc:
            print(f'    Warning: chunk {b}\u2013{e} failed ({exc})')
            continue
        if 'error' in data:
            msg = data['error'].get('message', str(data['error']))
            if 'No data' not in msg:
                print(f'    {product} not available: {msg}')
                return None
            continue
        records = data.get('data', [])
        if records:
            parts.append(parser(records))
        if i < len(chunks) - 1:
            time.sleep(1.0)
    if not parts:
        return None
    combined = pd.concat(parts).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------

def download_tidal_data(
    data_dir,
    station=DEFAULT_STATION,
    begin_date='20220101',
    end_date='20240331',
):
    """
    Download verified hourly water level and meteorological data from NOAA.

    Fetches ``hourly_height``, ``air_pressure``, ``water_temperature``, and
    ``wind`` products.  Results are cached as a single CSV in *data_dir*;
    subsequent calls with the same station skip the download.

    Parameters
    ----------
    data_dir : Path or str
        Directory for cached CSV files.
    station : str
        7-character NOAA station ID (default: 8518750 = The Battery, NY).
    begin_date, end_date : str
        YYYYMMDD date strings for the retrieval window.

    Returns
    -------
    Path
        Path to the cached CSV.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cached = resolve_tidal_cache_path(data_dir, station, begin_date, end_date)
    if cached.exists():
        print(f'Cached data found: {cached}')
        return cached

    cached = tidal_cache_path(data_dir, station, begin_date, end_date)

    met_product_defs = [
        ('air_pressure', 'pressure',
         lambda r: _parse_scalar(r, 'pressure')),
        ('water_temperature', 'water_temp',
         lambda r: _parse_scalar(r, 'water_temp')),
        ('wind', 'wind', _parse_wind),
    ]

    frames: dict[str, pd.DataFrame] = {}
    year_cks = _year_chunks(begin_date, end_date)

    # Water level: try hourly_height (verified, yearly chunks) first,
    # then fall back to water_level (preliminary, 31-day chunks).
    end_ts = pd.Timestamp(end_date)
    wl_frame = _fetch_product_chunks(
        station, 'hourly_height', _parse_water_level, year_cks,
    )
    covers_end = (
        wl_frame is not None
        and len(wl_frame) > 0
        and wl_frame.index.max() >= end_ts - pd.Timedelta(days=7)
    )
    if not covers_end:
        print('    Falling back to water_level (preliminary) ...')
        wl_frame = _fetch_product_chunks(
            station, 'water_level', _parse_water_level,
            _month_chunks(begin_date, end_date),
        )
    if wl_frame is None or len(wl_frame) == 0:
        raise RuntimeError(
            f'Failed to download water level data for station {station}',
        )
    frames['water_level'] = wl_frame
    print(f'    water_level: {len(wl_frame)} records')

    for product, label, parser in met_product_defs:
        result = _fetch_product_chunks(station, product, parser, year_cks)
        if result is not None and len(result) > 0:
            frames[label] = result
            print(f'    {label}: {len(result)} records')
        else:
            print(f'    {label}: skipped (not available at this station)')

    df = frames.pop('water_level')
    for _label, frame in frames.items():
        df = df.join(frame, how='left')

    df.to_csv(cached)
    print(f'Saved: {cached} ({len(df)} rows)')
    return cached


def load_tidal_data(data_file, interpolate_missing: bool = True):
    """
    Load preprocessed NOAA tidal data from a cached CSV.

    The data is kept at its native sampling frequency (e.g. 6-minute for
    ``water_level`` product, hourly for ``hourly_height``).  A regular
    ``DatetimeIndex`` at the detected frequency is created; small gaps
    (<=6 hours) are linearly interpolated and rows with missing
    ``water_level`` are dropped.

    Parameters
    ----------
    data_file : Path or str
        Path to the CSV written by :func:`download_tidal_data`.

    Returns
    -------
    DataFrame
        Columns include ``water_level`` (always) and zero or more of
        ``pressure``, ``water_temp``, ``wind_speed``, ``wind_u``, ``wind_v``.
    """
    df = pd.read_csv(data_file, index_col='datetime', parse_dates=True)

    freq_td = df.index.to_series().diff().median()
    freq = pd.infer_freq(df.index[:500])
    if freq is None:
        freq = pd.tseries.frequencies.to_offset(freq_td)
    samples_per_hour = max(1, round(pd.Timedelta('1h') / freq_td))

    idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(idx)

    if interpolate_missing:
        gap_limit = 6 * samples_per_hour
        for col in df.columns:
            df[col] = df[col].interpolate(
                method='linear', limit=gap_limit, limit_direction='both',
            )

        df = df.ffill(limit=3 * samples_per_hour).bfill(limit=3 * samples_per_hour)

    freq_label = f'{int(freq_td.total_seconds() // 60)}min' if freq_td < pd.Timedelta('1h') else '1h'
    print(f'Loaded {len(df)} samples at {freq_label} resolution')
    print(f'Date range: {df.index.min()} to {df.index.max()}')
    print(
        f'Water level range: {df["water_level"].min():.3f} to '
        f'{df["water_level"].max():.3f} m (MLLW)',
    )
    avail = [
        c for c in ('pressure', 'water_temp', 'wind_speed', 'wind_u', 'wind_v')
        if c in df.columns and df[c].notna().sum() > 100
    ]
    print(f'Available met variables: {", ".join(avail) if avail else "none"}')
    return df


# ---------------------------------------------------------------------------
# NCEI LCD weather station data (wind, air temp, dew point)
# ---------------------------------------------------------------------------

_LCD_USECOLS = [
    'DATE', 'REPORT_TYPE',
    'HourlyDryBulbTemperature', 'HourlyDewPointTemperature',
    'HourlyWindSpeed', 'HourlyWindDirection',
    'HourlySeaLevelPressure',
]


def _clean_lcd_numeric(series):
    """Convert LCD string column to float, handling 'M', 'VRB', trailing 's'."""
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.rstrip('s')
        .replace({'VRB': np.nan, 'M': np.nan, 'T': np.nan, '*': np.nan}),
        errors='coerce',
    )


def download_lcd_weather(
    data_dir,
    station_id=DEFAULT_WEATHER_STATION,
    begin_year=2022,
    end_year=2024,
):
    """
    Download LCD CSV files from NCEI for each year.

    LCD (Local Climatological Data) files contain hourly ASOS/AWOS weather
    observations including wind speed, wind direction, air temperature, and
    dew point—variables not typically available at tide gauge stations.

    Parameters
    ----------
    data_dir : Path or str
        Directory for cached CSV files.
    station_id : str
        NCEI LCD station ID (USAF+WBAN, e.g. '72505394728' = Central Park).
    begin_year, end_year : int
        Year range (inclusive) to download.

    Returns
    -------
    list of Path
        Paths to downloaded per-year CSV files.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for year in range(begin_year, end_year + 1):
        cached = data_dir / f'lcd_{station_id}_{year}.csv'
        if cached.exists():
            print(f'    LCD {year}: cached ({cached.stat().st_size / 1024:.0f} KB)')
            paths.append(cached)
            continue
        url = f'{LCD_BASE_URL}/{year}/{station_id}.csv'
        print(f'    Downloading LCD {year} ...')
        req = urllib.request.Request(
            url, headers={'User-Agent': 'tsgam_estimator/0.1'},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                with open(cached, 'wb') as f:
                    f.write(resp.read())
            print(f'    LCD {year}: {cached.stat().st_size / 1024:.0f} KB')
            paths.append(cached)
        except Exception as exc:
            print(f'    LCD {year}: download failed ({exc})')
        time.sleep(1)

    return paths


def load_lcd_weather(
    data_dir,
    station_id=DEFAULT_WEATHER_STATION,
    begin_date='2022-01-01',
    end_date='2024-03-31',
    interpolate_missing: bool = True,
):
    """
    Parse LCD CSV files and return an hourly weather DataFrame.

    Extracts wind speed, wind direction (decomposed to u/v), air temperature,
    and dew point.  Observations are resampled to the nearest hour.  LCD v1
    reports in US customary units (°F, mph, inHg); this function converts to
    metric (°C, m/s, hPa).

    Parameters
    ----------
    data_dir : Path or str
        Directory containing ``lcd_{station_id}_{year}.csv`` files.
    station_id : str
        NCEI LCD station ID.
    begin_date, end_date : str
        ISO date strings for the desired range.

    Returns
    -------
    DataFrame
        Hourly index with columns: ``air_temp``, ``dewpoint``,
        ``wind_speed``, ``wind_dir``, ``wind_u``, ``wind_v``, ``lcd_slp``.
    """
    data_dir = Path(data_dir)
    begin = _window_boundary(begin_date, is_end=False)
    end = _window_boundary(end_date, is_end=True)

    parts: list[pd.DataFrame] = []
    for year in range(begin.year, end.year + 1):
        f = data_dir / f'lcd_{station_id}_{year}.csv'
        if not f.exists():
            continue
        chunk = pd.read_csv(
            f, usecols=_LCD_USECOLS, dtype=str, low_memory=False,
        )
        parts.append(chunk)

    if not parts:
        raise FileNotFoundError(
            f'No LCD files found for station {station_id} in {data_dir}',
        )

    df = pd.concat(parts, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    # Convert numeric columns
    for col in _LCD_USECOLS[2:]:
        df[col] = _clean_lcd_numeric(df[col])

    # Unit conversions — LCD v1 uses °F, mph, inHg
    df['air_temp'] = (df['HourlyDryBulbTemperature'] - 32) * 5 / 9
    df['dewpoint'] = (df['HourlyDewPointTemperature'] - 32) * 5 / 9
    df['wind_speed'] = df['HourlyWindSpeed'] * 0.44704        # mph -> m/s
    df['wind_speed'] = df['wind_speed'].where(df['wind_speed'] < 80, np.nan)
    df['wind_dir'] = df['HourlyWindDirection']
    df['lcd_slp'] = df['HourlySeaLevelPressure'] * 33.8639    # inHg -> hPa

    # Wind u/v components (met convention: dir = where wind blows FROM)
    dir_rad = np.radians(df['wind_dir'])
    df['wind_u'] = -df['wind_speed'] * np.sin(dir_rad)
    df['wind_v'] = -df['wind_speed'] * np.cos(dir_rad)

    # Resample to nearest hour
    df['hour'] = df['datetime'].dt.round('h')
    result_cols = [
        'air_temp', 'dewpoint', 'wind_speed', 'wind_dir',
        'wind_u', 'wind_v', 'lcd_slp',
    ]
    hourly = df.groupby('hour')[result_cols].mean()
    hourly.index.name = 'datetime'

    # Trim to requested range
    hourly = hourly[begin:end]

    if interpolate_missing:
        # Interpolate small gaps (up to 3 hours)
        for col in hourly.columns:
            hourly[col] = hourly[col].interpolate(
                method='linear', limit=3, limit_direction='both',
            )

    print(f'  LCD weather: {len(hourly)} hourly records')
    for col in ('air_temp', 'wind_speed'):
        if col in hourly.columns and hourly[col].notna().any():
            lo, hi = hourly[col].min(), hourly[col].max()
            unit = '\u00b0C' if 'temp' in col else 'm/s'
            print(f'    {col}: {lo:.1f} to {hi:.1f} {unit}')

    return hourly


def merge_tidal_weather(tidal_df, weather_df, interpolate_missing: bool = True):
    """
    Merge tidal (CO-OPS) and weather (LCD) DataFrames.

    The tidal DataFrame keeps its native frequency.  Weather columns
    (typically hourly) are linearly interpolated to fill sub-hourly gaps
    after joining.  LCD columns overwrite any same-named CO-OPS columns
    (e.g. ``wind_speed``, ``wind_u``, ``wind_v``) since LCD ASOS data is
    generally more reliable for wind than the limited CO-OPS met sensors.

    Parameters
    ----------
    tidal_df : DataFrame
        Output of :func:`load_tidal_data`.
    weather_df : DataFrame
        Output of :func:`load_lcd_weather`.

    Returns
    -------
    DataFrame
        Combined DataFrame on the tidal index.
    """
    overlap_cols = [c for c in weather_df.columns if c in tidal_df.columns]
    if overlap_cols:
        tidal_df = tidal_df.drop(columns=overlap_cols)
    merged = tidal_df.join(weather_df, how='left')

    if interpolate_missing:
        freq_td = merged.index.to_series().diff().median()
        samples_per_hour = max(1, round(pd.Timedelta('1h') / freq_td))
        fill_limit = 3 * samples_per_hour

        wx_cols = weather_df.columns.difference(tidal_df.columns).tolist()
        wx_cols += overlap_cols
        for col in wx_cols:
            if col in merged.columns:
                merged[col] = merged[col].interpolate(
                    method='linear', limit=fill_limit, limit_direction='both',
                )

        merged = merged.ffill(limit=fill_limit).bfill(limit=fill_limit)
    return merged


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main():
    """Quick standalone demo: download, fit, predict, plot."""
    import matplotlib.pyplot as plt

    examples_dir = Path(__file__).parent
    data_dir = examples_dir / 'data' / 'tidal'
    plots_dir = examples_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    data_file = download_tidal_data(data_dir)
    df = load_tidal_data(data_file)

    # Supplement with LCD weather station data (wind, air temp)
    weather_id = TIDE_TO_WEATHER.get(DEFAULT_STATION, (None,))[0]
    if weather_id:
        print(f'\nFetching LCD weather data ({weather_id}) ...')
        download_lcd_weather(
            data_dir, station_id=weather_id,
            begin_year=2022, end_year=2024,
        )
        weather_df = load_lcd_weather(
            data_dir, station_id=weather_id,
            begin_date='2022-01-01', end_date='2024-03-31',
        )
        df = merge_tidal_weather(df, weather_df)

    train_start, train_end = '2022-01-01', '2023-12-31'
    test_start, test_end = '2024-01-01', '2024-03-31'
    df_train = df[train_start:train_end]
    df_test = df[test_start:test_end]
    y_train = df_train['water_level'].values
    y_test = df_test['water_level'].values

    met_cols = [
        c for c in ('pressure', 'water_temp', 'wind_u', 'wind_v', 'air_temp')
        if c in df.columns
        and df_train[c].notna().all()
        and df_test[c].notna().all()
    ]

    if met_cols:
        reg_means = df_train[met_cols].mean()
        reg_stds = df_train[met_cols].std().replace(0, 1)
        X_train = (df_train[met_cols] - reg_means) / reg_stds
        X_test = (df_test[met_cols] - reg_means) / reg_stds
    else:
        X_train = pd.DataFrame(index=df_train.index)
        X_test = pd.DataFrame(index=df_test.index)

    spline_map = {
        'pressure': TsgamSplineConfig(
            n_knots=10, lags=[-2, -1, 0, 1],
            reg_weight=1e-5, diff_reg_weight=0.3,
        ),
        'wind_u': TsgamSplineConfig(
            n_knots=8, lags=[-1, 0, 1],
            reg_weight=1e-5, diff_reg_weight=0.3,
        ),
        'wind_v': TsgamSplineConfig(
            n_knots=8, lags=[-1, 0, 1],
            reg_weight=1e-5, diff_reg_weight=0.3,
        ),
    }
    default_spline = TsgamSplineConfig(
        n_knots=8, lags=[0], reg_weight=1e-5, diff_reg_weight=0.3,
    )
    exog_config = [spline_map.get(c, default_spline) for c in met_cols]

    # Scale Fourier periods to the data's sampling rate
    freq_td = df.index.to_series().diff().median()
    sph = max(1, round(pd.Timedelta('1h') / freq_td))
    config = TsgamEstimatorConfig(
        multi_periodic_config=make_constituent_multi_periodic(sph),
        exog_config=exog_config or None,
        trend_config=TsgamTrendConfig(
            trend_type=TrendType.LINEAR,
            grouping=24.0 * sph,
        ),
        solver_config=TsgamSolverConfig(solver='SCS', verbose=False),
        random_state=42,
    )

    print(f'\nTraining on {len(df_train)} samples ({train_start} to {train_end})')
    print(f'Testing on {len(df_test)} samples ({test_start} to {test_end})')
    print(f'Regressors: {met_cols if met_cols else "none (Fourier only)"}')

    est = TsgamEstimator(config=config)
    print('Fitting model ...')
    est.fit(X_train, y_train)
    print(f'Solver status: {est.problem_.status}')

    preds = est.predict(X_test)
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    mae = float(np.mean(np.abs(preds - y_test)))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print('\nTest metrics:')
    print(f'  RMSE: {rmse:.4f} m')
    print(f'  MAE:  {mae:.4f} m')
    print(f'  R²:   {r2:.4f}')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    ax = axes[0]
    ax.plot(df_test.index, y_test, 'k-', alpha=0.7, linewidth=0.5, label='Observed')
    ax.plot(df_test.index, preds, 'r-', alpha=0.8, linewidth=0.5, label='Predicted')
    ax.set_ylabel('Water level (m, MLLW)')
    ax.legend(loc='upper right')
    ax.set_title(
        f'Tidal Prediction — {DEFAULT_STATION_NAME} '
        f'(RMSE={rmse:.4f} m, R\u00b2={r2:.4f})',
    )

    ax = axes[1]
    n_detail = min(24 * 14, len(y_test))
    ax.plot(
        df_test.index[:n_detail], y_test[:n_detail],
        'k-', linewidth=1, label='Observed',
    )
    ax.plot(
        df_test.index[:n_detail], preds[:n_detail],
        'r-', linewidth=1, label='Predicted',
    )
    ax.set_ylabel('Water level (m)')
    ax.set_title('First two weeks (detail)')
    ax.legend(loc='upper right')

    ax = axes[2]
    residuals = y_test - preds
    ax.plot(df_test.index, residuals, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel('Residual (m)')
    ax.set_xlabel('Date')
    ax.set_title(f'Residuals (\u03c3 = {np.std(residuals):.4f} m)')

    fig.tight_layout()
    out = plots_dir / 'example_tidal.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved plot: {out}')
    print('Done.')


if __name__ == '__main__':
    main()
