#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Run script: Tidal Water Level Prediction — ablation study and report.

Predicts NOAA tide-gauge water levels using TSGAM with multi-periodic
Fourier terms for astronomical tides and meteorological regressors for
the non-tidal residual (storm surge, inverse barometer, wind setup).

Supports running a single station or all curated stations (``--station all``)
with per-station reports and a cross-station comparison summary.

Usage:
  uv run python examples/run_tidal.py                       # default station
  uv run python examples/run_tidal.py --station all          # all 6 stations
  uv run python examples/run_tidal.py --station 9414290      # San Francisco
Requires: uv sync --group examples
"""

import sys
from collections import OrderedDict
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
_project_root = _examples_dir.parent
sys.path.insert(0, str(_project_root / 'src'))
sys.path.insert(0, str(_examples_dir))

import click
import numpy as np
import pandas as pd

from common_cli import (
    add_common_data_options,
    add_n_jobs_option,
    add_no_download_option,
    console,
    default_output_dir,
    error,
    info,
    plot_ablation_bars,
    plot_ablation_comparison,
    plot_data_overview,
    plot_heatmap,
    plot_model_summary,
    plot_residual_heatmap,
    plot_scatter_train_test,
    plot_selected_days,
    print_ablation_table,
    quiet,
    run_ablation_parallel,
    savefig,
    section,
    set_journal_style,
    success,
    write_ablation_report,
    COLORS,
)
from example_tidal import (
    DEFAULT_STATION,
    DEFAULT_STATION_NAME,
    STATION_CATALOG,
    TIDAL_COMPONENT_LABELS,
    TIDAL_CONSTITUENT_PERIODS_HOURS,
    TIDE_TO_WEATHER,
    download_lcd_weather,
    download_tidal_data,
    load_lcd_weather,
    load_tidal_data,
    make_constituent_multi_periodic,
    merge_tidal_weather,
    resolve_tidal_cache_path,
)
from tidal_model_shared import (
    make_tidal_spline_configs,
    prepare_split_regressors,
    tidal_metrics,
    usable_lcd_columns,
)
from tsgam_estimator import (
    TrendType,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data' / 'tidal'
DEFAULT_TRAIN_START = '2022-01-01'
DEFAULT_TRAIN_END = '2023-12-31'
DEFAULT_TEST_START = '2024-01-01'
DEFAULT_TEST_END = '2024-03-31'


def _make_multi_periodic(samples_per_hour: float) -> TsgamMultiPeriodicConfig:
    """Build a constituent-aware Fourier config scaled to the data's sample grid."""
    return make_constituent_multi_periodic(samples_per_hour)


def _make_spline_configs(
    samples_per_hour: int,
) -> tuple[dict[str, TsgamSplineConfig], TsgamSplineConfig]:
    """Build spline configs with lags scaled to the data's sampling rate."""
    return make_tidal_spline_configs(samples_per_hour)


def _tidal_metrics(y_true, y_pred):
    """Compute RMSE, MAE, MAPE, R² for water-level data."""
    return tidal_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _fit_single_tidal(cfg):
    """Fit one tidal ablation configuration (top-level for ThreadPoolExecutor)."""
    name = cfg['name']
    try:
        variable_names = cfg['variable_names']
        spline_configs = cfg['spline_configs']
        default_spline = cfg['default_spline']
        exog_config = (
            [spline_configs.get(v, default_spline) for v in variable_names]
            or None
        )
        trend_config = cfg.get('trend_config')
        solver = cfg.get('solver', 'CLARABEL')
        est = TsgamEstimator(config=TsgamEstimatorConfig(
            multi_periodic_config=cfg['multi_periodic'],
            exog_config=exog_config,
            trend_config=trend_config,
            solver_config=TsgamSolverConfig(solver=solver, verbose=False),
            random_state=42,
        ))
        est.fit(cfg['X_train'], cfg['y_train'])
        y_pred_test = est.predict(cfg['X_test'])
        y_pred_train = est.predict(cfg['X_train'])
        metrics = _tidal_metrics(cfg['y_test'], y_pred_test)
        return {
            'name': name, **metrics,
            'y_pred': y_pred_test, 'y_true': cfg['y_test'],
            'y_train_pred': y_pred_train, 'y_train_true': cfg['y_train'],
            'error': None,
        }
    except Exception as exc:
        return {
            'name': name,
            'rmse': float('nan'), 'mae': float('nan'),
            'mape': float('nan'), 'r2': float('nan'),
            'y_pred': None, 'y_true': None,
            'y_train_pred': None, 'y_train_true': None,
            'error': f'{type(exc).__name__}: {exc}',
        }


def _build_tidal_configs(
    X_train, y_train, X_test, y_test, available_met,
    multi_periodic, spline_configs, default_spline,
    solver='CLARABEL',
    trend_grouping: float | None = None,
):
    """Build ablation config dicts, incrementally adding met regressors."""
    shared = dict(
        y_train=y_train, y_test=y_test,
        multi_periodic=multi_periodic,
        spline_configs=spline_configs, default_spline=default_spline,
        solver=solver,
    )

    configs = [{
        'name': 'Tidal (Fourier only)',
        'X_train': pd.DataFrame(index=X_train.index),
        'X_test': pd.DataFrame(index=X_test.index),
        'variable_names': [],
        **shared,
    }]

    groups = [
        (['pressure'], 'Pressure'),
        (['water_temp'], 'Water Temp'),
        (['wind_u', 'wind_v'], 'Wind'),
        (['air_temp'], 'Air Temp'),
    ]

    cumulative_vars: list[str] = []
    cumulative_names: list[str] = []

    for vars_in_group, group_name in groups:
        if all(v in available_met for v in vars_in_group):
            cumulative_vars.extend(vars_in_group)
            cumulative_names.append(group_name)
            configs.append({
                'name': '+ ' + ' + '.join(cumulative_names),
                'X_train': X_train[cumulative_vars].copy(),
                'X_test': X_test[cumulative_vars].copy(),
                'variable_names': list(cumulative_vars),
                **shared,
            })

    if cumulative_vars:
        configs.append({
            'name': '+ ' + ' + '.join(cumulative_names) + ' + Trend',
            'X_train': X_train[cumulative_vars].copy(),
            'X_test': X_test[cumulative_vars].copy(),
            'variable_names': list(cumulative_vars),
            'trend_config': TsgamTrendConfig(
                trend_type=TrendType.LINEAR,
                grouping=trend_grouping,
            ),
            **shared,
        })

    return configs


# ---------------------------------------------------------------------------
# Single-station analysis
# ---------------------------------------------------------------------------

def _run_station(
    station: str,
    weather_station: str | None,
    data_dir: Path,
    output_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    n_jobs: int,
    no_download: bool,
) -> dict | None:
    """
    Run the full ablation pipeline for one tide station.

    Returns a summary dict with the best-model metrics, or None on failure.
    """
    station_meta = STATION_CATALOG.get(station, {'name': station})
    station_name = station_meta['name']

    section(f'{station_name} ({station})')
    if 'tidal_regime' in station_meta:
        info(
            f'{station_meta["region"]} — {station_meta["tidal_regime"]} '
            f'— {station_meta.get("notes", "")}',
        )

    # ---- Download tide data ----
    if not no_download:
        begin = pd.Timestamp(train_start).strftime('%Y%m%d')
        end = pd.Timestamp(test_end).strftime('%Y%m%d')
        try:
            data_file = download_tidal_data(
                data_dir, station=station,
                begin_date=begin, end_date=end,
            )
        except Exception as e:
            error(f'Tide data download failed: {e}')
            return None
    else:
        data_file = resolve_tidal_cache_path(
            data_dir,
            station,
            pd.Timestamp(train_start).strftime('%Y%m%d'),
            pd.Timestamp(test_end).strftime('%Y%m%d'),
        )
        if not data_file.exists():
            error(f'Data file not found: {data_file}')
            return None

    with quiet():
        df = load_tidal_data(data_file, interpolate_missing=False)

    # ---- LCD weather data ----
    wx_id = weather_station
    if wx_id is None and station in TIDE_TO_WEATHER:
        wx_id, wx_name = TIDE_TO_WEATHER[station]
        info(f'Weather station: {wx_id} ({wx_name})')

    if wx_id:
        begin_year = pd.Timestamp(train_start).year
        end_year = pd.Timestamp(test_end).year
        try:
            if not no_download:
                download_lcd_weather(
                    data_dir, station_id=wx_id,
                    begin_year=begin_year, end_year=end_year,
                )
            with quiet():
                weather_df = load_lcd_weather(
                    data_dir, station_id=wx_id,
                    begin_date=train_start, end_date=test_end,
                    interpolate_missing=False,
                )
            merged_df = merge_tidal_weather(
                df,
                weather_df,
                interpolate_missing=False,
            )
            lcd_usable = usable_lcd_columns(
                weather_df=weather_df,
                merged_df=merged_df,
                candidate_columns=['air_temp', 'dewpoint', 'wind_speed', 'wind_u', 'wind_v', 'lcd_slp'],
            )
            if lcd_usable:
                overlap_usable = [col for col in lcd_usable if col in df.columns]
                if overlap_usable:
                    df = df.drop(columns=overlap_usable)
                df = df.join(merged_df[lcd_usable], how='left')
                info(f'Merged {len(weather_df)} LCD weather records')
            else:
                info('LCD weather loaded but had unusable overlap; keeping CO-OPS columns.')
        except Exception as e:
            info(f'LCD weather unavailable ({e}); continuing without.')

    # ---- Detect sampling rate ----
    freq_td = df.index.to_series().diff().median()
    samples_per_hour = max(1, round(pd.Timedelta('1h') / freq_td))
    freq_label = f'{int(freq_td.total_seconds() // 60)}min' if freq_td < pd.Timedelta('1h') else '1h'
    info(f'Sampling: {freq_label} ({samples_per_hour} samples/hour)')

    multi_periodic = _make_multi_periodic(samples_per_hour)

    # ---- Prepare train/test ----
    df_train = df[train_start:train_end].copy()
    df_test = df[test_start:test_end].copy()
    if len(df_train) == 0 or len(df_test) == 0:
        error('No data in train or test range')
        return None
    # Preserve freq on sliced index (pandas drops it on slice)
    for sub in (df_train, df_test):
        if sub.index.freq is None:
            sub.index.freq = pd.infer_freq(sub.index)
    info(f'Train: {len(df_train)} samples, Test: {len(df_test)} samples')

    y_train = df_train['water_level'].values
    y_test = df_test['water_level'].values

    regressor_candidates = [
        'pressure', 'water_temp', 'wind_u', 'wind_v', 'air_temp',
    ]
    X_train, X_test, regressor_cols, dropped_regressors = prepare_split_regressors(
        df_train=df_train,
        df_test=df_test,
        candidate_columns=[c for c in regressor_candidates if c in df.columns],
    )
    info(f'Regressors: {", ".join(regressor_cols) if regressor_cols else "none"}')
    if dropped_regressors:
        info(f'Dropped regressors: {", ".join(dropped_regressors)}')

    if not regressor_cols:
        X_train = pd.DataFrame(index=df_train.index)
        X_test = pd.DataFrame(index=df_test.index)

    # ---- Ablation ----
    spline_configs, default_spline = _make_spline_configs(samples_per_hour)
    solver = 'SCS'
    tidal_configs = _build_tidal_configs(
        X_train, y_train, X_test, y_test, set(regressor_cols),
        multi_periodic=multi_periodic,
        spline_configs=spline_configs, default_spline=default_spline,
        solver=solver,
        trend_grouping=24.0 * samples_per_hour,
    )
    results_list = run_ablation_parallel(
        _fit_single_tidal, tidal_configs, n_jobs=n_jobs,
    )
    print_ablation_table(
        results_list,
        title=f'{station_name} — Regressor ablation',
        baseline_name='Tidal (Fourier only)',
    )

    # ---- Per-station report + figures ----
    stn_dir = output_dir / station
    stn_dir.mkdir(parents=True, exist_ok=True)

    md_path, csv_path = write_ablation_report(
        results_list, stn_dir, 'tidal',
        baseline_name='Tidal (Fourier only)',
    )
    success(f'Report: {md_path}')

    df_full = pd.concat([df_train, df_test]).sort_index()

    series = OrderedDict([('Water level', (df_full['water_level'].values, 'm'))])
    if 'pressure' in df_full.columns:
        series['Pressure'] = (df_full['pressure'].values, 'hPa')
    if 'water_temp' in df_full.columns:
        series['Water temp'] = (df_full['water_temp'].values, '\u00b0C')
    if 'wind_speed' in df_full.columns and df_full['wind_speed'].notna().any():
        series['Wind speed'] = (df_full['wind_speed'].values, 'm/s')
    if 'air_temp' in df_full.columns and df_full['air_temp'].notna().any():
        series['Air temp'] = (df_full['air_temp'].values, '\u00b0C')

    with quiet():
        plot_data_overview(
            df_full.index, series,
            f'{station_name} — Data overview',
            stn_dir / 'tidal_data',
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
        )

    y_true = next(
        (r['y_true'] for r in results_list if r.get('y_true') is not None),
        None,
    )
    predictions = {
        r['name']: r['y_pred']
        for r in results_list if r.get('y_pred') is not None
    }
    if y_true is not None and predictions:
        with quiet():
            plot_model_summary(
                df_test.index, y_true, predictions,
                f'{station_name} — Model summary',
                stn_dir / 'tidal_model',
                'Water level (m)',
                results=results_list,
            )

    with quiet():
        plot_ablation_comparison(
            results_list,
            f'{station_name} — Ablation',
            stn_dir / 'tidal_ablation',
            metrics=('rmse', 'mae', 'r2'),
            baseline_name='Tidal (Fourier only)',
        )
        plot_ablation_bars(
            results_list, stn_dir, 'tidal',
            title=f'{station_name} — RMSE by configuration',
        )

    valid_results = [
        r for r in results_list
        if r.get('y_pred') is not None and r.get('y_train_pred') is not None
    ]
    best = min(valid_results, key=lambda r: r.get('rmse', np.inf), default=None)
    if best is not None:
        with quiet():
            plot_scatter_train_test(
                best['y_train_true'], best['y_train_pred'],
                best['y_true'], best['y_pred'],
                f'{station_name} — Actual vs Predicted',
                stn_dir / 'tidal_scatter',
                'Water level (m)', model_name=best['name'],
            )
            plot_selected_days(
                df_test.index, best['y_true'], best['y_pred'],
                f'{station_name} — Selected days',
                stn_dir / 'tidal_days',
                'Water level (m)', model_name=best['name'],
            )
            plot_residual_heatmap(
                df_test.index, best['y_true'], best['y_pred'],
                f'{station_name} — Residual heatmap',
                stn_dir / 'tidal_residual_heatmap',
                'Water level (m)',
            )

    with quiet():
        plot_heatmap(
            df_full.index,
            OrderedDict([('Water level', (df_full['water_level'].values, 'm'))]),
            f'{station_name} — Tidal heatmap',
            stn_dir / 'tidal_heatmap',
            cmap='RdBu_r',
        )

    success(f'{station_name}: done.')

    # Return summary for cross-station comparison
    baseline = next(
        (r for r in results_list if r['name'] == 'Tidal (Fourier only)'), None,
    )
    return {
        'station': station,
        'name': station_name,
        'region': station_meta.get('region', ''),
        'tidal_regime': station_meta.get('tidal_regime', ''),
        'n_train': len(df_train),
        'n_test': len(df_test),
        'n_regressors': len(regressor_cols),
        'regressors': ', '.join(regressor_cols),
        'baseline_rmse': baseline['rmse'] if baseline else np.nan,
        'baseline_r2': baseline['r2'] if baseline else np.nan,
        'best_name': best['name'] if best else '',
        'best_rmse': best['rmse'] if best else np.nan,
        'best_r2': best['r2'] if best else np.nan,
        'improvement_pct': (
            (baseline['rmse'] - best['rmse']) / baseline['rmse'] * 100
            if baseline and best
            and np.isfinite(baseline['rmse']) and baseline['rmse'] > 0
            else np.nan
        ),
    }


# ---------------------------------------------------------------------------
# Cross-station summary
# ---------------------------------------------------------------------------

def _write_cross_station_summary(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    """Write a cross-station comparison table and figure."""
    from rich.table import Table

    table = Table(
        title='Cross-station comparison — Best model',
        show_header=True, header_style='bold',
    )
    table.add_column('Station', style='cyan')
    table.add_column('Region')
    table.add_column('Regime')
    table.add_column('Baseline RMSE', justify='right')
    table.add_column('Best RMSE', justify='right')
    table.add_column('Best R\u00b2', justify='right')
    table.add_column('Improvement', justify='right')
    table.add_column('Best Config')

    for s in summaries:
        imp = f'{s["improvement_pct"]:+.1f}%' if np.isfinite(s['improvement_pct']) else '\u2014'
        table.add_row(
            s['name'],
            s['region'],
            s['tidal_regime'],
            f'{s["baseline_rmse"]:.4f} m' if np.isfinite(s['baseline_rmse']) else '\u2014',
            f'{s["best_rmse"]:.4f} m' if np.isfinite(s['best_rmse']) else '\u2014',
            f'{s["best_r2"]:.4f}' if np.isfinite(s['best_r2']) else '\u2014',
            imp,
            s['best_name'],
        )
    console.print(table)

    # Write CSV
    import csv
    csv_path = output_dir / 'cross_station_summary.csv'
    cols = [
        'station', 'name', 'region', 'tidal_regime',
        'n_train', 'n_test', 'n_regressors', 'regressors',
        'baseline_rmse', 'baseline_r2',
        'best_name', 'best_rmse', 'best_r2', 'improvement_pct',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    success(f'Summary CSV: {csv_path}')

    # Write markdown
    md_path = output_dir / 'cross_station_summary.md'
    lines = [
        '# Tidal Water Level Prediction — Cross-station comparison\n',
        '| Station | Region | Regime | Baseline RMSE (m) | Best RMSE (m) '
        '| Best R\u00b2 | Improvement | Best Config |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for s in summaries:
        imp = f'{s["improvement_pct"]:+.1f}%' if np.isfinite(s['improvement_pct']) else '\u2014'
        bl = f'{s["baseline_rmse"]:.4f}' if np.isfinite(s['baseline_rmse']) else '\u2014'
        br = f'{s["best_rmse"]:.4f}' if np.isfinite(s['best_rmse']) else '\u2014'
        r2 = f'{s["best_r2"]:.4f}' if np.isfinite(s['best_r2']) else '\u2014'
        lines.append(
            f'| {s["name"]} | {s["region"]} | {s["tidal_regime"]} '
            f'| {bl} | {br} | {r2} | {imp} | {s["best_name"]} |',
        )
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    success(f'Summary MD: {md_path}')

    # Bar chart comparing baseline vs best RMSE across stations
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    set_journal_style()
    names = [s['name'] for s in summaries]
    baseline_rmses = [s['baseline_rmse'] for s in summaries]
    best_rmses = [s['best_rmse'] for s in summaries]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.6)))
    ax.barh(x - width / 2, baseline_rmses, width,
            label='Fourier only', color=COLORS[0], alpha=0.8)
    ax.barh(x + width / 2, best_rmses, width,
            label='Best (+ met regressors)', color=COLORS[1], alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('RMSE (m)')
    ax.set_title('Cross-station: Fourier baseline vs best model')
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, output_dir / 'cross_station_rmse')
    success(f'Figure: {output_dir / "cross_station_rmse"}.pdf/.png')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@add_common_data_options
@add_n_jobs_option
@add_no_download_option
@click.option(
    '--station', default=DEFAULT_STATION,
    help=(
        f'NOAA tide station ID, or "all" for all curated stations '
        f'(default: {DEFAULT_STATION} = {DEFAULT_STATION_NAME}).'
    ),
)
@click.option(
    '--weather-station', default=None,
    help=(
        'LCD weather station ID for wind/air-temp data '
        '(default: auto-mapped from tide station).'
    ),
)
def main(
    data_dir: Path | None,
    output_dir: Path | None,
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
    n_jobs: int,
    no_download: bool,
    station: str,
    weather_station: str | None,
) -> None:
    """Run Tidal Water Level example with ablation study and write reports."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir or default_output_dir()
    train_start = train_start or DEFAULT_TRAIN_START
    train_end = train_end or DEFAULT_TRAIN_END
    test_start = test_start or DEFAULT_TEST_START
    test_end = test_end or DEFAULT_TEST_END

    section('Tidal Water Level Prediction')
    info(f'Data dir: {data_dir}')
    info(f'Output dir: {output_dir}')
    info(f'Train: {train_start} to {train_end}')
    info(f'Test: {test_start} to {test_end}')

    if station.lower() == 'all':
        stations = list(STATION_CATALOG.keys())
        info(f'Running all {len(stations)} curated stations')
    else:
        stations = [station]

    summaries: list[dict] = []

    for stn in stations:
        try:
            summary = _run_station(
                station=stn,
                weather_station=weather_station,
                data_dir=data_dir,
                output_dir=output_dir,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_jobs=n_jobs,
                no_download=no_download,
            )
            if summary is not None:
                summaries.append(summary)
        except Exception as e:
            error(f'Station {stn} failed: {e}')

    if len(summaries) > 1:
        section('Cross-station summary')
        _write_cross_station_summary(summaries, output_dir)

    success('All done.')


if __name__ == '__main__':
    main()
