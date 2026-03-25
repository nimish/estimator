#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Run script: Outlier detector example (synthetic data) with ablation and report.
Usage: uv run python examples/run_outlier_detector.py [OPTIONS]
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
    add_n_jobs_option,
    compute_standard_metrics,
    default_output_dir,
    info,
    plot_ablation_bars,
    plot_ablation_comparison,
    plot_data_overview,
    plot_model_summary,
    plot_outlier_detection,
    print_ablation_table,
    quiet,
    run_ablation_parallel,
    section,
    success,
    write_ablation_report,
)

from example_outlier_detector import (
    generate_synthetic_data,
    fit_model_with_outlier_detector,
    fit_model_without_outlier_detector,
    verify_outlier_detection,
)


def _fit_single_outlier(cfg: dict) -> dict:
    """Fit one outlier ablation config. Top-level for ThreadPoolExecutor."""
    name = cfg['name']
    X, y_log, y_original = cfg['X'], cfg['y_log'], cfg['y_original']
    try:
        if cfg['use_outlier']:
            est = fit_model_with_outlier_detector(X, y_log, reg_weight=0.01)
        else:
            est = fit_model_without_outlier_detector(X, y_log)
        pred = np.exp(est.predict(X))
        metrics = compute_standard_metrics(y_original, pred)
        result: dict = {'name': name, **metrics, 'y_pred': pred, 'y_true': y_original}
        if cfg['use_outlier']:
            det = est.variables_['outlier'].value
            ver = verify_outlier_detection(
                cfg['true_outlier_values'], det, cfg['outlier_days'], tolerance=0.3,
            )
            result['mean_detection_error'] = ver['mean_error']
        return result
    except Exception:
        return {'name': name, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'y_pred': None, 'y_true': None}


def _build_outlier_configs(
    n_days: int = 60, outlier_days: list[int] | None = None, seed: int = 42,
) -> list[dict]:
    """Build self-contained config dicts for parallel outlier ablation."""
    if outlier_days is None:
        outlier_days = [10, 25, 40]
    timestamps, y_log, y_original, true_outlier_values = generate_synthetic_data(
        n_days=n_days,
        outlier_days=outlier_days,
        outlier_multipliers=[0.2, 2.0, 0.5],
        noise_scale=0.1,
        random_state=seed,
    )
    X = pd.DataFrame(index=timestamps)
    shared = dict(X=X, y_log=y_log, y_original=y_original,
                  true_outlier_values=true_outlier_values, outlier_days=outlier_days)
    return [
        {'name': 'Without outlier detector', 'use_outlier': False, **shared},
        {'name': 'With outlier detector', 'use_outlier': True, **shared},
    ]


@click.command()
@click.option('--output-dir', type=click.Path(path_type=Path), default=None, help='Output directory for reports.')
@click.option('--n-days', type=int, default=60, help='Number of days of synthetic data.')
@click.option('--seed', type=int, default=42, help='Random seed.')
@add_n_jobs_option
def main(output_dir: Path | None, n_days: int, seed: int, n_jobs: int) -> None:
    """Run Outlier Detector example (synthetic data) with ablation and report."""
    output_dir = output_dir or default_output_dir()

    section('Outlier detector (synthetic) — Ablation and report')
    info(f'Output dir: {output_dir}')
    info(f'n_days: {n_days}, seed: {seed}')

    section('Generating synthetic data and running ablation')
    outlier_configs = _build_outlier_configs(n_days=n_days, seed=seed)
    results_list = run_ablation_parallel(_fit_single_outlier, outlier_configs, n_jobs=n_jobs)
    print_ablation_table(
        results_list,
        title='Outlier detector — With vs without',
        baseline_name='Without outlier detector',
    )

    section('Writing report and plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = write_ablation_report(
        results_list,
        output_dir,
        'outlier_detector',
        baseline_name='Without outlier detector',
    )
    success(f'Report (markdown): {md_path}')
    success(f'Report (CSV): {csv_path}')

    cfg = outlier_configs[0]
    timestamps = cfg['X'].index
    y_log = cfg['y_log']
    y_original = cfg['y_original']
    true_outlier_values = cfg['true_outlier_values']
    outlier_days = cfg['outlier_days']
    n_days = len(true_outlier_values)

    # Clean signal (remove outlier effect from y_log)
    clean_log = y_log.copy()
    for d in outlier_days:
        if 0 <= d < n_days:
            start, end = d * 24, min((d + 1) * 24, len(clean_log))
            clean_log[start:end] -= true_outlier_values[d]
    clean_original = np.exp(clean_log)

    # Publication-quality figures (PDF + PNG)
    series = OrderedDict([
        ('Clean signal', (clean_original, '-')),
        ('Corrupted (observed)', (y_original, '-')),
    ])
    shade_ranges = []
    for d in outlier_days:
        if 0 <= d < n_days:
            start_idx = d * 24
            end_idx = min((d + 1) * 24, len(timestamps)) - 1
            if start_idx < len(timestamps) and end_idx >= start_idx:
                shade_ranges.append((timestamps[start_idx], timestamps[end_idx]))
    with quiet():
        paths = plot_data_overview(
            timestamps,
            series,
            'Outlier detector — Data overview',
            output_dir / 'outlier_data',
            shade_ranges=shade_ranges,
        )
    for p in paths:
        success(f'Figure: {p}')

    y_true = next((r['y_true'] for r in results_list if r.get('y_true') is not None), None)
    predictions = {r['name']: r['y_pred'] for r in results_list if r.get('y_pred') is not None}
    if y_true is not None and predictions:
        with quiet():
            paths = plot_model_summary(
                timestamps,
                y_true,
                predictions,
                'Outlier detector — Model summary',
                output_dir / 'outlier_model',
                'Value',
                results=results_list,
            )
        for p in paths:
            success(f'Figure: {p}')

    # Refit with outlier detector to get detected values for outlier detection figure
    with quiet():
        est_with = fit_model_with_outlier_detector(cfg['X'], cfg['y_log'], reg_weight=0.01)
    detected_outlier_values = est_with.variables_['outlier'].value
    day_indices = np.arange(n_days)
    with quiet():
        paths = plot_outlier_detection(
            day_indices,
            true_outlier_values,
            detected_outlier_values,
            'Outlier detector — True vs detected',
            output_dir / 'outlier_detection',
            outlier_days=outlier_days,
        )
    for p in paths:
        success(f'Figure: {p}')

    with quiet():
        paths = plot_ablation_comparison(
            results_list,
            'Outlier detector — Ablation comparison',
            output_dir / 'outlier_detector_ablation',
            metrics=('rmse', 'mae', 'r2'),
            baseline_name='Without outlier detector',
        )
    for p in paths:
        success(f'Figure: {p}')

    png_path = plot_ablation_bars(
        results_list,
        output_dir,
        'outlier_detector',
        metric='rmse',
        title='Outlier detector — RMSE',
    )
    if png_path:
        success(f'Plot: {png_path}')
    success('Done.')


if __name__ == '__main__':
    main()
