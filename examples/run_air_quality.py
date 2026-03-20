#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Run script: Air Quality (Beijing PM2.5) example with ablation and report.
Usage: uv run python examples/run_air_quality.py [OPTIONS]
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
    TsgamSolverConfig,
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
)

from example_air_quality import (
    download_beijing_air_quality_data,
    load_beijing_air_quality,
    plot_ablation_results,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data' / 'air_quality'
DEFAULT_TRAIN_START = '2012-01-01'
DEFAULT_TRAIN_END = '2013-12-31'
DEFAULT_TEST_START = '2014-01-01'
DEFAULT_TEST_END = '2014-03-31'


_VAR_SPLINE_CONFIGS = {
    'temperature': TsgamSplineConfig(n_knots=12, lags=[-2, -1, 0, 1, 2], reg_weight=6e-5, diff_reg_weight=0.5),
    'dewpoint': TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
    'wind_speed': TsgamSplineConfig(n_knots=8, lags=[0, 1], reg_weight=6e-5, diff_reg_weight=0.5),
    'pressure': TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5),
}

_DEFAULT_SPLINE = TsgamSplineConfig(n_knots=8, lags=[0], reg_weight=6e-5, diff_reg_weight=0.5)


def _fit_single_aq(cfg: dict) -> dict:
    """Fit one air quality ablation config. Top-level for ThreadPoolExecutor."""
    name = cfg['name']
    try:
        variable_names = cfg['variable_names']
        exog_config = [_VAR_SPLINE_CONFIGS.get(v, _DEFAULT_SPLINE) for v in variable_names] or None
        est = TsgamEstimator(config=TsgamEstimatorConfig(
            multi_periodic_config=TsgamMultiPeriodicConfig(
                num_harmonics=[8, 6, 4],
                periods=[PERIOD_HOURLY_YEARLY, PERIOD_HOURLY_WEEKLY, PERIOD_HOURLY_DAILY],
                reg_weight=6e-5,
            ),
            exog_config=exog_config,
            solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
            random_state=42,
        ))
        est.fit(cfg['X_train'], cfg['y_train'])
        pred_log = est.predict(cfg['X_test'])
        pred_train_log = est.predict(cfg['X_train'])
        y_true_orig = np.exp(cfg['y_test']) - 1.0
        y_pred_orig = np.exp(pred_log) - 1.0
        y_train_true_orig = np.exp(cfg['y_train']) - 1.0
        y_train_pred_orig = np.exp(pred_train_log) - 1.0
        metrics = compute_standard_metrics(y_true_orig, y_pred_orig)
        return {'name': name, **metrics, 'y_pred': y_pred_orig, 'y_true': y_true_orig,
                'y_train_pred': y_train_pred_orig, 'y_train_true': y_train_true_orig}
    except Exception:
        return {'name': name, 'rmse': float('nan'), 'mae': float('nan'),
                'mape': float('nan'), 'r2': float('nan'),
                'y_pred': None, 'y_true': None,
                'y_train_pred': None, 'y_train_true': None}


def _build_aq_configs(X_train, y_train, X_test, y_test) -> list[dict]:
    """Build self-contained config dicts for parallel ablation."""
    shared = dict(y_train=y_train, y_test=y_test)
    return [
        {'name': 'None (seasonal only)',
         'X_train': pd.DataFrame(index=X_train.index),
         'X_test': pd.DataFrame(index=X_test.index),
         'variable_names': [], **shared},
        {'name': 'Temperature',
         'X_train': X_train[['temperature']],
         'X_test': X_test[['temperature']],
         'variable_names': ['temperature'], **shared},
        {'name': 'Temperature + Dewpoint',
         'X_train': X_train[['temperature', 'dewpoint']],
         'X_test': X_test[['temperature', 'dewpoint']],
         'variable_names': ['temperature', 'dewpoint'], **shared},
        {'name': 'Temperature + Dewpoint + Wind Speed',
         'X_train': X_train[['temperature', 'dewpoint', 'wind_speed']],
         'X_test': X_test[['temperature', 'dewpoint', 'wind_speed']],
         'variable_names': ['temperature', 'dewpoint', 'wind_speed'], **shared},
        {'name': 'All variables',
         'X_train': X_train,
         'X_test': X_test,
         'variable_names': list(X_train.columns), **shared},
    ]


@click.command()
@add_common_data_options
@add_n_jobs_option
@add_no_download_option
def main(
    data_dir: Path | None,
    output_dir: Path | None,
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
    n_jobs: int,
    no_download: bool,
) -> None:
    """Run Air Quality (Beijing PM2.5) example with ablation study and write reports."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir or default_output_dir()
    train_start = train_start or DEFAULT_TRAIN_START
    train_end = train_end or DEFAULT_TRAIN_END
    test_start = test_start or DEFAULT_TEST_START
    test_end = test_end or DEFAULT_TEST_END

    section('Air Quality (Beijing PM2.5) — Ablation and report')
    info(f'Data dir: {data_dir}')
    info(f'Output dir: {output_dir}')
    info(f'Train: {train_start} to {train_end}')
    info(f'Test: {test_start} to {test_end}')

    if not no_download:
        section('Downloading data (use --no-download to skip)')
        try:
            data_file = download_beijing_air_quality_data(data_dir)
            success(f'Data ready: {data_file}')
        except Exception as e:
            error(str(e))
            raise
    else:
        data_file = data_dir / 'PRSA_data_2010.1.1-2014.12.31.csv'
        if not data_file.exists():
            error(f'Data file not found: {data_file}. Run without --no-download to fetch.')
            sys.exit(1)

    section('Loading data')
    with quiet():
        df = load_beijing_air_quality(data_file)
    df_train = df[train_start:train_end].copy()
    df_test = df[test_start:test_end].copy()
    info(f'Training samples: {len(df_train)}')
    info(f'Test samples: {len(df_test)}')

    y_train = np.log(df_train['pm25'].values + 1.0)
    y_test = np.log(df_test['pm25'].values + 1.0)
    X_train = pd.DataFrame({
        'temperature': df_train['temperature'].values,
        'dewpoint': df_train['dewpoint'].values,
        'wind_speed': df_train['wind_speed'].values,
        'pressure': df_train['pressure'].values,
    }, index=df_train.index)
    X_test = pd.DataFrame({
        'temperature': df_test['temperature'].values,
        'dewpoint': df_test['dewpoint'].values,
        'wind_speed': df_test['wind_speed'].values,
        'pressure': df_test['pressure'].values,
    }, index=df_test.index)

    section('Running ablation study')
    aq_configs = _build_aq_configs(X_train, y_train, X_test, y_test)
    results_list = run_ablation_parallel(_fit_single_aq, aq_configs, n_jobs=n_jobs)
    print_ablation_table(
        results_list,
        title='Air Quality — Exogenous variable ablation',
        baseline_name='None (seasonal only)',
    )

    section('Writing report and plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = write_ablation_report(
        results_list,
        output_dir,
        'air_quality',
        baseline_name='None (seasonal only)',
    )
    success(f'Report (markdown): {md_path}')
    success(f'Report (CSV): {csv_path}')

    # Publication-quality figures (PDF + PNG)
    df_full = pd.concat([df_train, df_test]).sort_index()
    series = OrderedDict([
        ('PM2.5', (df_full['pm25'].values, 'μg/m³')),
        ('Temperature', (df_full['temperature'].values, '°C')),
        ('Dewpoint', (df_full['dewpoint'].values, '°C')),
        ('Wind speed', (df_full['wind_speed'].values, 'm/s')),
        ('Pressure', (df_full['pressure'].values, 'hPa')),
    ])
    with quiet():
        paths = plot_data_overview(
            df_full.index,
            series,
            'Air Quality — Data overview',
            output_dir / 'air_quality_data',
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
                'Air Quality — Model summary',
                output_dir / 'air_quality_model',
                'PM2.5 (μg/m³)',
                results=results_list,
            )
        for p in paths:
            success(f'Figure: {p}')

    with quiet():
        paths = plot_ablation_comparison(
            results_list,
            'Air Quality — Ablation comparison',
            output_dir / 'air_quality_ablation',
            metrics=('rmse', 'mae', 'r2'),
            baseline_name='None (seasonal only)',
        )
    for p in paths:
        success(f'Figure: {p}')

    ablation_dict = {r['name']: (r['rmse'], r['mae'], r['mape']) for r in results_list}
    with quiet():
        plot_ablation_results(ablation_dict, output_dir)
    png_path = plot_ablation_bars(
        results_list,
        output_dir,
        'air_quality',
        title='Air Quality — RMSE by configuration',
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
                'Air Quality — Actual vs Predicted', output_dir / 'air_quality_scatter',
                'PM2.5 (μg/m³)', model_name=best_name,
            )
        for p in paths:
            success(f'Figure: {p}')

        # Selected-day overlays (auto-detect high vs low pollution)
        with quiet():
            paths = plot_selected_days(
                X_test.index, best['y_true'], best['y_pred'],
                'Air Quality — Selected days', output_dir / 'air_quality_days',
                'PM2.5 (μg/m³)', model_name=best_name,
            )
        for p in paths:
            success(f'Figure: {p}')

        # Residual heatmap
        with quiet():
            paths = plot_residual_heatmap(
                X_test.index, best['y_true'], best['y_pred'],
                'Air Quality — Residual heatmap', output_dir / 'air_quality_residual_heatmap',
                'PM2.5 (μg/m³)',
            )
        for p in paths:
            success(f'Figure: {p}')

    # Data heatmap (full series -- shows daily + yearly patterns)
    heatmap_series = OrderedDict([
        ('PM2.5', (df_full['pm25'].values, 'μg/m³')),
    ])
    with quiet():
        paths = plot_heatmap(
            df_full.index, heatmap_series,
            'Air Quality — Data heatmap', output_dir / 'air_quality_heatmap',
            cmap='YlOrRd',
        )
    for p in paths:
        success(f'Figure: {p}')

    success('Done.')


if __name__ == '__main__':
    main()
