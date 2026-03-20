#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Run script: LA Energy Demand example with ablation and report.
Usage: uv run python examples/run_la_energy.py [OPTIONS]
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
    compute_standard_metrics,
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
    section,
    success,
    write_ablation_report,
)

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSplineConfig,
    TsgamArConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
)

DEFAULT_DATA_DIR = _examples_dir / 'data' / 'energy'
DEFAULT_WEATHER_FILE = 'weather_CA_Los Angeles.csv'
DEFAULT_ENERGY_FILE = 'CA_Los Angeles_R.csv'
DEFAULT_TARGET = 'elec_total_MW'
# Default dates: one year of data, use last month as test
DEFAULT_TRAIN_START = '2018-01-01'
DEFAULT_TRAIN_END = '2018-11-30'
DEFAULT_TEST_START = '2018-12-01'
DEFAULT_TEST_END = '2018-12-31'

VAR_CONFIGS = {
    'temperature_degF': TsgamSplineConfig(n_knots=10, lags=[0, 1, 2, 3], reg_weight=6e-5, diff_reg_weight=0.5),
    'humidity_pc': TsgamSplineConfig(n_knots=8, lags=[0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
    'global_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
    'direct_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
    'diffuse_Wpms': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
}


def _build_exog_config(weather_cols: list[str]) -> list:
    out = []
    for c in weather_cols:
        out.append(VAR_CONFIGS.get(c, TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5)))
    return out


def _fit_single_la(cfg: dict) -> dict:
    """Fit one LA energy ablation config. Top-level for ThreadPoolExecutor."""
    name = cfg['name']
    try:
        est = TsgamEstimator(config=TsgamEstimatorConfig(
            multi_periodic_config=cfg['multi_periodic'],
            exog_config=cfg['exog'],
            ar_config=cfg['ar'],
            outlier_config=cfg['outlier'],
            solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
            random_state=42,
        ))
        est.fit(cfg['X_train'], cfg['y_tr'])
        pred_log = est.predict(cfg['X_test'])
        pred_train_log = est.predict(cfg['X_train'])
        take_log = cfg.get('take_log')
        pred = np.exp(pred_log) - 1.0 if take_log else pred_log
        pred_train = np.exp(pred_train_log) - 1.0 if take_log else pred_train_log
        y_true_orig = cfg['y_te']
        y_train_orig = np.exp(cfg['y_tr']) - 1.0 if take_log else cfg['y_tr']
        metrics = compute_standard_metrics(y_true_orig, pred)
        return {'name': name, **metrics, 'y_pred': pred, 'y_true': y_true_orig,
                'y_train_pred': pred_train, 'y_train_true': y_train_orig}
    except Exception:
        return {'name': name, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan,
                'y_pred': None, 'y_true': None, 'y_train_pred': None, 'y_train_true': None}


def _build_ablation_configs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    take_log: bool = True,
) -> list[dict]:
    """Build self-contained config dicts for parallel ablation."""
    weather_cols = [c for c in X_train.columns if c in VAR_CONFIGS or c.startswith('temperature') or c.startswith('humidity')]
    if not weather_cols:
        weather_cols = list(X_train.columns)[:2]

    y_tr = np.log(y_train + 1.0) if take_log else y_train.copy()
    X_train_empty = pd.DataFrame(index=X_train.index)
    X_test_empty = pd.DataFrame(index=X_test.index)

    shared = dict(y_tr=y_tr, y_te=y_test, take_log=take_log)

    configs: list[dict] = []

    def _add(name, *, multi_periodic=None, exog=None, ar=None, outlier=None, use_exog=False):
        configs.append({
            'name': name,
            'X_train': X_train if use_exog else X_train_empty,
            'X_test': X_test if use_exog else X_test_empty,
            'multi_periodic': multi_periodic, 'exog': exog,
            'ar': ar, 'outlier': outlier,
            **shared,
        })

    _add('Baseline (constant only)')

    all_harmonics = TsgamMultiPeriodicConfig(num_harmonics=[4, 4, 6], periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY], reg_weight=6e-5)

    _add('Harmonics: Yearly only',
         multi_periodic=TsgamMultiPeriodicConfig(num_harmonics=[4, 0, 0], periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY], reg_weight=6e-5))
    _add('Harmonics: Weekly only',
         multi_periodic=TsgamMultiPeriodicConfig(num_harmonics=[0, 4, 0], periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY], reg_weight=6e-5))
    _add('Harmonics: Daily only',
         multi_periodic=TsgamMultiPeriodicConfig(num_harmonics=[0, 0, 6], periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY], reg_weight=6e-5))
    _add('Harmonics: All', multi_periodic=all_harmonics)

    if weather_cols:
        exog_all = _build_exog_config(list(X_train.columns))
        _add('Exogenous only', exog=exog_all, use_exog=True)
        _add('Harmonics (all) + Exogenous', multi_periodic=all_harmonics, exog=exog_all, use_exog=True)
        _add('Harmonics + Exogenous + AR', multi_periodic=all_harmonics, exog=exog_all,
             ar=TsgamArConfig(lags=[1, 2, 3, 4], l1_constraint=0.97), use_exog=True)
        _add('Harmonics + Exogenous + Outlier', multi_periodic=all_harmonics, exog=exog_all,
             outlier=TsgamOutlierConfig(reg_weight=1e-4, period_hours=24.0), use_exog=True)

    return configs


@click.command()
@click.option('--weather-file', type=str, default=DEFAULT_WEATHER_FILE, help='Weather CSV filename.')
@click.option('--energy-file', type=str, default=DEFAULT_ENERGY_FILE, help='Energy CSV filename.')
@click.option('--target', type=str, default=DEFAULT_TARGET, help='Target column name.')
@add_common_data_options
@add_n_jobs_option
def main(
    weather_file: str,
    energy_file: str,
    target: str,
    data_dir: Path | None,
    output_dir: Path | None,
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
    n_jobs: int,
) -> None:
    """Run LA Energy Demand example with ablation study and write reports."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir or default_output_dir()
    train_start = train_start or DEFAULT_TRAIN_START
    train_end = train_end or DEFAULT_TRAIN_END
    test_start = test_start or DEFAULT_TEST_START
    test_end = test_end or DEFAULT_TEST_END

    section('LA Energy Demand — Ablation and report')
    info(f'Data dir: {data_dir}')
    info(f'Output dir: {output_dir}')
    info(f'Target: {target}')

    weather_path = data_dir / weather_file
    energy_path = data_dir / energy_file
    if not weather_path.exists():
        error(f'Weather file not found: {weather_path}')
        sys.exit(1)
    if not energy_path.exists():
        error(f'Energy file not found: {energy_path}')
        sys.exit(1)

    section('Loading data')
    df_weather = pd.read_csv(weather_path)
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather = df_weather.set_index('timestamp')
    df_energy = pd.read_csv(energy_path)
    df_energy['timestamp'] = pd.to_datetime(df_energy['timestamp'])
    df_energy = df_energy.set_index('timestamp')
    df = pd.merge(df_weather, df_energy, left_index=True, right_index=True, how='inner').sort_index()
    if target not in df.columns:
        error(f'Target column "{target}" not in data. Columns: {list(df.columns)}')
        sys.exit(1)
    df = df.loc[~df[target].isna()]
    df_train = df[train_start:train_end]
    df_test = df[test_start:test_end]
    if len(df_train) == 0 or len(df_test) == 0:
        error('No data in train or test range.')
        sys.exit(1)
    exog_cols = [c for c in df_weather.columns if c != 'timestamp' and c in df.columns]
    X_train = df_train[exog_cols].copy()
    X_test = df_test[exog_cols].copy()
    y_train = df_train[target].values
    y_test = df_test[target].values
    info(f'Training samples: {len(y_train)}')
    info(f'Test samples: {len(y_test)}')

    section('Running ablation study')
    ablation_configs = _build_ablation_configs(X_train, y_train, X_test, y_test)
    results_list = run_ablation_parallel(_fit_single_la, ablation_configs, n_jobs=n_jobs)
    print_ablation_table(
        results_list,
        title='LA Energy — Model ablation',
        baseline_name='Baseline (constant only)',
    )

    section('Writing report and plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = write_ablation_report(
        results_list,
        output_dir,
        'la_energy',
        baseline_name='Baseline (constant only)',
    )
    success(f'Report (markdown): {md_path}')
    success(f'Report (CSV): {csv_path}')

    # Publication-quality figures (PDF + PNG)
    df_full = pd.concat([df_train, df_test]).sort_index()
    def _unit(col: str) -> str:
        if 'temp' in col.lower():
            return '°F'
        if 'humid' in col.lower():
            return '%'
        return ''
    series = OrderedDict([(target, (df_full[target].values, 'MW'))])
    for c in exog_cols[:2]:
        series[c] = (df_full[c].values, _unit(c) or '-')
    with quiet():
        paths = plot_data_overview(
            df_full.index,
            series,
            'LA Energy — Data overview',
            output_dir / 'la_energy_data',
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
        )
    for p in paths:
        success(f'Figure: {p}')

    y_true = next((r['y_true'] for r in results_list if r.get('y_true') is not None), None)
    predictions = {r['name']: r['y_pred'] for r in results_list if r.get('y_pred') is not None}
    if y_true is not None and predictions:
        with quiet():
            paths = plot_model_summary(
                X_test.index,
                y_true,
                predictions,
                'LA Energy — Model summary',
                output_dir / 'la_energy_model',
                'MW',
                results=results_list,
            )
        for p in paths:
            success(f'Figure: {p}')

    with quiet():
        paths = plot_ablation_comparison(
            results_list,
            'LA Energy — Ablation comparison',
            output_dir / 'la_energy_ablation',
            metrics=('rmse', 'mae', 'r2'),
            baseline_name='Baseline (constant only)',
        )
    for p in paths:
        success(f'Figure: {p}')

    png_path = plot_ablation_bars(
        results_list,
        output_dir,
        'la_energy',
        title='LA Energy — RMSE by configuration',
    )
    if png_path:
        success(f'Plot: {png_path}')

    # Find best model for detailed plots
    valid_results = [r for r in results_list
                     if r.get('y_pred') is not None and r.get('y_train_pred') is not None]
    best = min(valid_results, key=lambda r: r.get('rmse', np.inf), default=None)
    if best is not None:
        best_name = best['name']

        # Train vs test scatter
        with quiet():
            paths = plot_scatter_train_test(
                best['y_train_true'], best['y_train_pred'],
                best['y_true'], best['y_pred'],
                'LA Energy — Actual vs Predicted', output_dir / 'la_energy_scatter',
                'MW', model_name=best_name,
            )
        for p in paths:
            success(f'Figure: {p}')

        # Selected-day overlays (auto-detect high vs low demand)
        with quiet():
            paths = plot_selected_days(
                X_test.index, best['y_true'], best['y_pred'],
                'LA Energy — Selected days', output_dir / 'la_energy_days',
                'MW', model_name=best_name,
            )
        for p in paths:
            success(f'Figure: {p}')

        # Residual heatmap
        with quiet():
            paths = plot_residual_heatmap(
                X_test.index, best['y_true'], best['y_pred'],
                'LA Energy — Residual heatmap', output_dir / 'la_energy_residual_heatmap',
                'MW',
            )
        for p in paths:
            success(f'Figure: {p}')

    # Data heatmap (full series -- shows daily + weekly + yearly patterns)
    heatmap_series = OrderedDict([(target, (df_full[target].values, 'MW'))])
    with quiet():
        paths = plot_heatmap(
            df_full.index, heatmap_series,
            'LA Energy — Data heatmap', output_dir / 'la_energy_heatmap',
        )
    for p in paths:
        success(f'Figure: {p}')

    success('Done.')


if __name__ == '__main__':
    main()
