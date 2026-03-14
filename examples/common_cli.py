# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared CLI options and Rich-based report utilities for example run scripts.
Requires: uv sync --group examples (click, rich).
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Shared Click options (compose with @click.option decorators)
# ---------------------------------------------------------------------------

def add_common_data_options(f):
    """Add --data-dir, --output-dir, --train-start, --train-end, --test-start, --test-end."""
    f = click.option(
        '--data-dir',
        type=click.Path(path_type=Path),
        default=None,
        help='Data directory. Example-specific default if not set.',
    )(f)
    f = click.option(
        '--output-dir',
        type=click.Path(path_type=Path),
        default=None,
        help='Output directory for reports and plots. Default: examples/reports.',
    )(f)
    f = click.option(
        '--train-start',
        type=str,
        default=None,
        help='Training start date (YYYY-MM-DD). Example-specific default.',
    )(f)
    f = click.option(
        '--train-end',
        type=str,
        default=None,
        help='Training end date (YYYY-MM-DD). Example-specific default.',
    )(f)
    f = click.option(
        '--test-start',
        type=str,
        default=None,
        help='Test start date (YYYY-MM-DD). Example-specific default.',
    )(f)
    f = click.option(
        '--test-end',
        type=str,
        default=None,
        help='Test end date (YYYY-MM-DD). Example-specific default.',
    )(f)
    return f


def add_n_jobs_option(f):
    """Add --n-jobs for parallel ablation."""
    return click.option(
        '--n-jobs',
        type=int,
        default=4,
        help='Number of parallel jobs for ablation. Default: 4.',
    )(f)


def add_no_download_option(f):
    """Add --no-download (skip downloading data)."""
    return click.option(
        '--no-download',
        is_flag=True,
        default=False,
        help='Do not download data; use existing files only.',
    )(f)


def default_output_dir() -> Path:
    """Default output directory for reports (examples/reports)."""
    return Path(__file__).resolve().parent / 'reports'


STANDARD_METRICS: tuple[str, ...] = ('rmse', 'mae', 'mape', 'r2')


def compute_standard_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE, R² on finite positive pairs."""
    valid = np.isfinite(pred) & np.isfinite(y_true) & (y_true > 0)
    if not np.any(valid):
        return {k: np.nan for k in STANDARD_METRICS}
    y, p = y_true[valid], pred[valid]
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    mape = float(np.mean(np.abs((p - y) / y)) * 100)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

def section(title: str, style: str = 'bold cyan') -> None:
    """Print a section header with Rich."""
    console.print(Panel(title, style=style, expand=False))


def success(msg: str) -> None:
    """Print a success message in green."""
    console.print(f'[green]{msg}[/green]')


def error(msg: str) -> None:
    """Print an error message in red."""
    console.print(f'[red]{msg}[/red]')


def info(msg: str) -> None:
    """Print an info message in dim style."""
    console.print(f'[dim]{msg}[/dim]')


def print_ablation_table(
    results: list[dict[str, Any]],
    *,
    metric_columns: tuple[str, ...] = STANDARD_METRICS,
    title: str = 'Ablation results',
    show_improvement: bool = True,
    baseline_name: str | None = None,
) -> None:
    """
    Print ablation results as a Rich table.

    results : list of dicts with at least 'name' and metric keys (rmse, mae, mape, etc.).
    baseline_name : if set and show_improvement True, compute % improvement over this config.
    """
    table = Table(title=title, show_header=True, header_style='bold')
    table.add_column('Configuration', style='cyan')
    for col in metric_columns:
        table.add_column(col.upper(), justify='right')
    if show_improvement and baseline_name:
        table.add_column('Improvement', justify='right')

    baseline_val = None
    if baseline_name and results:
        for r in results:
            if r.get('name') == baseline_name:
                baseline_val = r.get('rmse')
                break

    for row in results:
        cells = [str(row.get('name', ''))]
        for col in metric_columns:
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                cells.append('—')
            elif isinstance(val, float):
                cells.append(f'{val:.4g}')
            else:
                cells.append(str(val))
        if show_improvement and baseline_name and baseline_val and baseline_val > 0:
            rmse = row.get('rmse')
            if rmse is not None and not (isinstance(rmse, float) and math.isnan(rmse)):
                pct = (baseline_val - rmse) / baseline_val * 100
                cells.append(f'{pct:+.1f}%')
            else:
                cells.append('—')
        table.add_row(*cells)

    console.print(table)


# ---------------------------------------------------------------------------
# Report file writers
# ---------------------------------------------------------------------------

def write_ablation_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    name: str,
    *,
    metric_columns: tuple[str, ...] = STANDARD_METRICS,
    baseline_name: str | None = None,
) -> tuple[Path, Path]:
    """
    Write ablation results to markdown and CSV in output_dir.
    name : e.g. 'air_quality', 'la_energy' -> ablation_air_quality.md, ablation_air_quality.csv.
    Returns (path_md, path_csv).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f'ablation_{name}'

    # Markdown
    baseline_val = None
    if baseline_name:
        for r in results:
            if r.get('name') == baseline_name:
                baseline_val = r.get('rmse')
                break
    md_path = base.with_suffix('.md')
    header_cols = ['Configuration'] + [c.upper() for c in metric_columns]
    if baseline_name and baseline_val is not None and baseline_val > 0:
        header_cols.append('Improvement')
    lines = [f'# Ablation report: {name}\n', '| ' + ' | '.join(header_cols) + ' |']
    lines.append('| ' + ' | '.join('---' for _ in header_cols) + ' |')
    for row in results:
        parts = [str(row.get('name', ''))]
        for col in metric_columns:
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                parts.append('—')
            elif isinstance(val, float):
                parts.append(f'{val:.4g}')
            else:
                parts.append(str(val))
        if baseline_val and baseline_val > 0:
            rmse = row.get('rmse')
            if rmse is not None and not (isinstance(rmse, float) and math.isnan(rmse)):
                pct = (baseline_val - rmse) / baseline_val * 100
                parts.append(f'{pct:+.1f}%')
        lines.append('| ' + ' | '.join(parts) + ' |')
    md_path.write_text('\n'.join(lines), encoding='utf-8')

    # CSV
    csv_path = base.with_suffix('.csv')
    import csv
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        cols = ['name'] + list(metric_columns)
        if baseline_name and baseline_val and baseline_val > 0:
            cols.append('improvement_pct')
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in results:
            out = dict(row)
            if baseline_name and baseline_val and baseline_val > 0 and 'improvement_pct' in cols:
                rmse = row.get('rmse')
                if rmse is not None and not (isinstance(rmse, float) and math.isnan(rmse)):
                    out['improvement_pct'] = (baseline_val - rmse) / baseline_val * 100
            w.writerow(out)

    return md_path, csv_path


def run_ablation_parallel(
    fit_fn: Callable[[dict], dict],
    configs: list[dict],
    n_jobs: int = 4,
) -> list[dict]:
    """Run fit_fn(config) for each config in parallel via ThreadPoolExecutor.

    ThreadPoolExecutor gives real parallelism here because the heavy work
    (CLARABEL solver in Rust, NumPy BLAS) releases the GIL.  Falls back to
    sequential execution on error or when n_jobs <= 1.
    """
    n = len(configs)
    if n_jobs <= 1 or n <= 1:
        results = []
        for i, c in enumerate(configs):
            info(f'  [{i + 1}/{n}] {c.get("name", "?")}')
            results.append(fit_fn(c))
        return results

    workers = min(n_jobs, n)
    info(f'  {n} configs, {workers} parallel workers')
    results: list[dict | None] = [None] * n

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_fn, c): i for i, c in enumerate(configs)}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                done += 1
                try:
                    result = future.result()
                except Exception as exc:
                    name = configs[idx].get('name', '?')
                    result = {'name': name, 'error': str(exc)}
                results[idx] = result
                name = result.get('name', configs[idx].get('name', '?'))
                info(f'  [{done}/{n}] {name}')
    except Exception:
        info('  ThreadPool failed, falling back to sequential')
        results = []
        for i, c in enumerate(configs):
            info(f'  [{i + 1}/{n}] {c.get("name", "?")}')
            results.append(fit_fn(c))

    return results  # type: ignore[return-value]


def plot_ablation_bars(
    results: list[dict[str, Any]],
    output_dir: Path,
    name: str,
    *,
    metric: str = 'rmse',
    title: str | None = None,
) -> Path | None:
    """
    Plot a bar chart of the chosen metric across configurations; save as PNG.
    Returns path to saved PNG, or None if matplotlib not available.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'ablation_{name}.png'
    names = [r.get('name', '') for r in results]
    values = [r.get(metric) for r in results]
    values = [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0 for v in values]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), values, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric.upper())
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
