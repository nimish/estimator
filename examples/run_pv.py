#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Run script: PV/Solar example with trend ablation and report.
Usage: uv run python examples/run_pv.py [OPTIONS]
Requires: uv sync --group examples (includes solar-data-tools, matplotlib).
"""

import sys
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
_project_root = _examples_dir.parent
_pv_data_dir = _examples_dir / 'data' / 'pv'
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
    print_ablation_table,
    run_ablation_parallel,
    section,
    success,
    write_ablation_report,
    plot_ablation_bars,
)

from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSplineConfig,
    TsgamSolverConfig,
    TsgamTrendConfig,
    TrendType,
)

try:
    from solardatatools import DataHandler
except ImportError:
    DataHandler = None

DEFAULT_DATA_FILE = _pv_data_dir / '2107_data_combined.csv'
PRIMARY_COL = 'inv_03_ac_power_inv_149593'
MODULE_TEMP_COL = 'ambient_temperature_o_149575'
IRRAD_COL = 'poa_irradiance_o_149574'


def _load_pv_data(data_file: Path, primary_col: str, module_temp_col: str, irrad_col: str):
    """Load CSV, run DataHandler pipeline, return (X, y, y_max)."""
    df = pd.read_csv(data_file, parse_dates=[0], index_col=0)
    df = df.resample('15min').mean()
    dh = DataHandler(df)
    dh.fix_dst()
    extra_cols = [c for c in [module_temp_col, irrad_col] if c and c != primary_col]
    if extra_cols:
        dh.run_pipeline(power_col=primary_col, max_val=2000, extra_cols=extra_cols, linearity_threshold=0.1)
    else:
        dh.run_pipeline(power_col=primary_col, max_val=2000, linearity_threshold=0.1)
    if module_temp_col and module_temp_col in dh.extra_matrices:
        t = dh.extra_matrices[module_temp_col]
        t[t > 140] = np.nan
    if irrad_col and irrad_col in dh.extra_matrices:
        ir = dh.extra_matrices[irrad_col]
        ir[ir < 0] = 0
    data_end = dh.raw_data_matrix.shape[1] - 1
    _sel = np.s_[0:data_end + 1]
    y = np.copy(dh.raw_data_matrix)
    y_max = np.nanmax(y)
    y[:, ~dh.daily_flags.no_errors] = np.nan
    y[~dh.boolean_masks.daytime] = np.nan
    y[y < 0.01 * np.nanmax(y)] = np.nan
    y = y[:, _sel].ravel(order='F')
    y /= y_max
    y = np.log(y)
    x1 = np.copy(dh.extra_matrices[module_temp_col][:, _sel].ravel(order='F'))
    x1[np.isnan(x1)] = 0
    x1_max = np.max(x1) or 1.0
    x1 = x1 / x1_max
    x2 = np.copy(dh.extra_matrices[irrad_col][:, _sel].ravel(order='F'))
    x2[x2 < 0] = 0
    x2[np.isnan(x2)] = 0
    x2_max = np.max(x2) or 1.0
    x2 = x2 / x2_max
    valid = ~np.isnan(y) & np.isfinite(x1) & np.isfinite(x2)
    y = y[valid]
    x1 = x1[valid]
    x2 = x2[valid]
    np.nan_to_num(x1, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(x2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y)
    timestamps = pd.date_range(start=df.index[0], periods=n, freq='15min')
    X = pd.DataFrame({'temp': x1, 'irrad': x2}, index=timestamps)
    return X, y, y_max


TRAIN_FRACTION = 0.8


def _fit_single_pv(cfg: dict) -> dict:
    """Fit one PV trend ablation config. Top-level for ThreadPoolExecutor."""
    name = cfg['name']
    try:
        est = TsgamEstimator(config=cfg['estimator_config'])
        est.fit(cfg['X_train'], cfg['y_train'])
        pred_log = est.predict(cfg['X_test'])
        y_max = cfg['y_max']
        metrics = compute_standard_metrics(
            np.exp(cfg['y_test']) * y_max,
            np.exp(pred_log) * y_max,
        )
        slope = None
        if hasattr(est, 'variables_') and est.variables_ and 'trend_slope' in est.variables_:
            sl = est.variables_['trend_slope'].value
            slope = float(sl) if sl is not None else None
        return {'name': name, **metrics, 'trend_slope': slope}
    except Exception:
        return {'name': name, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan,
                'r2': np.nan, 'trend_slope': None}


def _build_pv_configs(
    X: pd.DataFrame, y: np.ndarray, y_max: float,
) -> list[dict]:
    """Build self-contained config dicts for parallel PV trend ablation."""
    split = int(len(y) * TRAIN_FRACTION)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    period_yearly = 365.2425 * 24.0
    period_daily = 24.0
    multi_periodic = TsgamMultiPeriodicConfig(
        num_harmonics=[6, 10],
        periods=[period_yearly, period_daily],
        reg_weight=1e-2,
    )
    exog_config = [
        TsgamSplineConfig(n_knots=10, lags=[0], reg_weight=1e-4),
        TsgamSplineConfig(n_knots=10, lags=[0], reg_weight=1e-4),
    ]

    shared = dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, y_max=y_max)
    configs: list[dict] = []
    for trend_name in ('none', 'linear', 'nonlinear'):
        trend_config = None
        if trend_name != 'none':
            trend_config = TsgamTrendConfig(
                trend_type=TrendType.LINEAR if trend_name == 'linear' else TrendType.NONLINEAR,
                grouping=24.0,
                reg_weight=10.0,
            )
        configs.append({
            'name': f'Trend: {trend_name}',
            'estimator_config': TsgamEstimatorConfig(
                multi_periodic_config=multi_periodic,
                exog_config=exog_config,
                trend_config=trend_config,
                solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
            ),
            **shared,
        })
    return configs


@click.command()
@click.option(
    '--data-file',
    type=click.Path(path_type=Path),
    default=None,
    help=f'Path to PV CSV. Default: {DEFAULT_DATA_FILE}',
)
@click.option('--primary-col', type=str, default=PRIMARY_COL, help='Power column name.')
@click.option('--module-temp-col', type=str, default=MODULE_TEMP_COL, help='Module temperature column.')
@click.option('--irrad-col', type=str, default=IRRAD_COL, help='Irradiance column.')
@add_common_data_options
@add_n_jobs_option
def main(
    data_file: Path | None,
    primary_col: str,
    module_temp_col: str,
    irrad_col: str,
    data_dir: Path | None,
    output_dir: Path | None,
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
    n_jobs: int,
) -> None:
    """Run PV/Solar example with trend ablation and write report."""
    if DataHandler is None:
        error('solar-data-tools not installed. Run: uv sync --group examples')
        sys.exit(1)
    data_file = data_file or DEFAULT_DATA_FILE
    output_dir = output_dir or default_output_dir()
    if not data_file.exists():
        error(f'Data file not found: {data_file}')
        sys.exit(1)

    section('PV/Solar — Trend ablation and report')
    info(f'Data file: {data_file}')
    info(f'Output dir: {output_dir}')

    section('Loading data')
    X, y, y_max = _load_pv_data(data_file, primary_col, module_temp_col, irrad_col)
    split = int(len(y) * TRAIN_FRACTION)
    info(f'Valid samples: {len(y)}  (train: {split}, test: {len(y) - split})')

    section('Running trend ablation')
    pv_configs = _build_pv_configs(X, y, y_max)
    results_list = run_ablation_parallel(_fit_single_pv, pv_configs, n_jobs=n_jobs)
    print_ablation_table(
        results_list,
        title='PV — Trend type comparison',
        baseline_name='Trend: none',
    )

    section('Writing report and plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = write_ablation_report(
        results_list,
        output_dir,
        'pv',
        baseline_name='Trend: none',
    )
    success(f'Report (markdown): {md_path}')
    success(f'Report (CSV): {csv_path}')
    png_path = plot_ablation_bars(
        results_list,
        output_dir,
        'pv',
        title='PV — RMSE by trend type',
    )
    if png_path:
        success(f'Plot: {png_path}')
    success('Done.')


if __name__ == '__main__':
    main()
