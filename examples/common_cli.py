# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared CLI options and Rich-based report utilities for example run scripts.
Requires: uv sync --group examples (click, rich).
"""

import io
import math
import os
import sys
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(os.environ.get('TMPDIR', '/tmp'), 'mpl_cache'),
)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=UserWarning, module='cvxpy')
warnings.filterwarnings('ignore', message='SmallSampleWarning')
warnings.filterwarnings('ignore', message='After omitting NaNs')

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Okabe-Ito colorblind-safe palette (hex)
COLORS = (
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#CC79A7',  # pink
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#000000',  # black
)


def set_journal_style() -> None:
    """Configure matplotlib for publication-quality figures (serif, 300 DPI, grid)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })


def savefig(fig: 'Any', path_stem: Path) -> list[Path]:
    """
    Save figure as PDF and PNG. path_stem has no extension.
    Returns list of saved paths (e.g. [path_stem.pdf, path_stem.png]).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ('.pdf', '.png'):
        p = path_stem.with_suffix(ext)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        saved.append(p)
    plt.close(fig)
    return saved


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


@contextmanager
def quiet():
    """Suppress stdout/stderr from noisy library calls."""
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


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


# ---------------------------------------------------------------------------
# Publication-quality figures (journal submission)
# ---------------------------------------------------------------------------

def plot_data_overview(
    timestamps: np.ndarray,
    series: OrderedDict[str, tuple[np.ndarray, str]],
    title: str,
    path_stem: Path,
    *,
    test_start=None,
    test_end=None,
    shade_ranges: list[tuple[Any, Any]] | None = None,
) -> list[Path]:
    """
    Vertically stacked time-series subplots with shared x-axis.
    series: label -> (values, unit). Subplot labels (a), (b), ...
    If test_start/test_end are set (datetime-like), shade the test region.
    If shade_ranges is set, shade each (start, end) on all subplots (e.g. outlier days).
    Saves PDF and PNG; returns list of saved paths.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    set_journal_style()
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (label, (values, unit)) in enumerate(series.items()):
        ax = axes[i]
        ax.plot(timestamps, values, color=COLORS[0], linewidth=0.8, alpha=0.9)
        ax.set_ylabel(f'{label} ({unit})' if unit else label)
        ax.set_xlabel('')
        ax.text(0.02, 0.95, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        if test_start is not None and test_end is not None:
            try:
                ax.axvspan(test_start, test_end, alpha=0.15, color='gray', zorder=0)
            except Exception:
                pass
        if shade_ranges:
            for start, end in shade_ranges:
                try:
                    ax.axvspan(start, end, alpha=0.2, color='yellow', zorder=0)
                except Exception:
                    pass
    axes[-1].set_xlabel('Date')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def plot_model_summary(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str,
    path_stem: Path,
    y_unit: str,
    *,
    results: list[dict[str, Any]] | None = None,
    best_name: str | None = None,
) -> list[Path]:
    """
    2x2 figure: (a) forecast overlay, (b) scatter actual vs predicted (best),
    (c) residuals vs time, (d) residual histogram.
    If best_name is None, infer from results (lowest RMSE) or use first key in predictions.
    Saves PDF and PNG; returns list of saved paths.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        return []
    set_journal_style()
    if not predictions:
        return []
    if best_name is None and results:
        valid = [r for r in results if r.get('rmse') is not None and not (isinstance(r.get('rmse'), float) and np.isnan(r.get('rmse')))]
        best_row = min(valid, key=lambda r: r['rmse'], default=None) if valid else None
        best_name = best_row['name'] if best_row else None
    if best_name is None:
        best_name = next(iter(predictions), None)
    if best_name is None:
        return []
    y_pred_best = predictions.get(best_name)
    if y_pred_best is None or len(y_pred_best) != len(y_true):
        return []

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    # (a) Forecast overlay
    ax = axes[0, 0]
    ax.plot(timestamps, y_true, color='black', linewidth=0.8, alpha=0.8, label='Actual')
    for j, (name, pred) in enumerate(predictions.items()):
        if pred is None or len(pred) != len(y_true):
            continue
        c = COLORS[j % len(COLORS)]
        ax.plot(timestamps, pred, color=c, linewidth=0.6, alpha=0.7, label=name)
    ax.set_ylabel(f'Value ({y_unit})')
    ax.set_xlabel('Date')
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Forecast overlay')

    # (b) Scatter actual vs predicted (best)
    ax = axes[0, 1]
    valid = np.isfinite(y_true) & np.isfinite(y_pred_best)
    yt, yp = y_true[valid], y_pred_best[valid]
    ax.scatter(yt, yp, alpha=0.4, s=8, color=COLORS[0], edgecolors='none')
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='1:1')
    r2 = np.nan
    if len(yt) > 1 and np.var(yt) > 0:
        r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, va='top', fontsize=9)
    ax.set_xlabel(f'Actual ({y_unit})')
    ax.set_ylabel(f'Predicted ({y_unit})')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.set_title(f'Best: {best_name[:30]}' + ('...' if len(best_name) > 30 else ''))

    # (c) Residuals vs time
    ax = axes[1, 0]
    res = y_true - y_pred_best
    ax.plot(timestamps, res, color=COLORS[1], linewidth=0.5, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    rstd = np.nanstd(res)
    if np.isfinite(rstd) and rstd > 0:
        ax.axhline(2 * rstd, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
        ax.axhline(-2 * rstd, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_ylabel(f'Residual ({y_unit})')
    ax.set_xlabel('Date')
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.set_title('Residuals')

    # (d) Residual histogram
    ax = axes[1, 1]
    ax.hist(res[np.isfinite(res)], bins=min(50, max(20, len(res) // 20)), color=COLORS[0], alpha=0.7, density=True, edgecolor='white')
    xr = np.linspace(res.min(), res.max(), 100)
    mu, std = np.nanmean(res), np.nanstd(res)
    if np.isfinite(std) and std > 0:
        ax.plot(xr, stats.norm.pdf(xr, mu, std), 'k-', linewidth=1.2, label='Normal')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel(f'Residual ({y_unit})')
    ax.set_ylabel('Density')
    ax.text(0.02, 0.95, '(d)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.text(0.98, 0.95, f'$\\mu$={mu:.3g}\n$\\sigma$={std:.3g}', transform=ax.transAxes, va='top', ha='right', fontsize=8)
    ax.set_title('Residual distribution')

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def plot_ablation_comparison(
    results: list[dict[str, Any]],
    title: str,
    path_stem: Path,
    *,
    metrics: tuple[str, ...] = ('rmse', 'mae', 'r2'),
    baseline_name: str | None = None,
) -> list[Path]:
    """
    Grouped horizontal bar chart: one group per metric across configurations.
    Optionally annotate improvement over baseline_name. Saves PDF and PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    set_journal_style()
    names = [r.get('name', '') for r in results]
    n_cfg = len(names)
    n_met = len(metrics)
    baseline_val = None
    if baseline_name:
        for r in results:
            if r.get('name') == baseline_name:
                baseline_val = r.get('rmse')
                break

    x = np.arange(n_cfg)
    width = 0.8 / n_met
    fig, ax = plt.subplots(figsize=(7, max(4, n_cfg * 0.45)))
    for i, met in enumerate(metrics):
        values = []
        for r in results:
            v = r.get(met)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0
            values.append(v)
        offset = (i - n_met / 2 + 0.5) * width
        bars = ax.barh(x + offset, values, width, label=met.upper(), color=COLORS[i % len(COLORS)], alpha=0.85)
        if baseline_val and met == 'rmse' and baseline_val > 0:
            ax.axvline(baseline_val, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Metric value')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def plot_scatter_train_test(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    title: str,
    path_stem: Path,
    y_unit: str,
    *,
    model_name: str = '',
) -> list[Path]:
    """
    Side-by-side scatter: (a) train actual vs predicted, (b) test actual vs predicted.
    Each panel shows 1:1 line and R². Saves PDF and PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    set_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for i, (yt, yp, label) in enumerate([
        (y_train_true, y_train_pred, 'Training'),
        (y_test_true, y_test_pred, 'Test'),
    ]):
        ax = axes[i]
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt_v, yp_v = yt[valid], yp[valid]
        ax.scatter(yt_v, yp_v, alpha=0.3, s=6,
                   color=COLORS[0] if i == 0 else COLORS[1], edgecolors='none')
        lims = [min(yt_v.min(), yp_v.min()), max(yt_v.max(), yp_v.max())]
        ax.plot(lims, lims, 'k--', linewidth=1, label='1:1')
        r2 = np.nan
        if len(yt_v) > 1 and np.var(yt_v) > 0:
            r2 = 1 - np.sum((yt_v - yp_v) ** 2) / np.sum((yt_v - np.mean(yt_v)) ** 2)
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlabel(f'Actual ({y_unit})')
        ax.set_ylabel(f'Predicted ({y_unit})')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.text(0.02, 0.02, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='bottom')
        subtitle = f'{label}'
        if model_name:
            subtitle += f' — {model_name}'
        ax.set_title(subtitle, fontsize=9)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def _reshape_to_day_matrix(
    timestamps: Any,
    values: np.ndarray,
    samples_per_day: int | None = None,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Reshape a 1-D time series into a (time_slots, days) matrix.

    Returns (matrix, day_labels, time_fractions) where time_fractions
    are fractional hours [0, 24) for the y-axis.
    """
    import pandas as pd
    ts = pd.DatetimeIndex(timestamps)
    df = pd.DataFrame({'val': values}, index=ts)
    df['date'] = df.index.date
    df['time_frac'] = df.index.hour + df.index.minute / 60.0 + df.index.second / 3600.0

    if samples_per_day is None:
        samples_per_day = df.groupby('date').size().mode().iloc[0]

    pivot = df.pivot_table(values='val', index='time_frac', columns='date', aggfunc='mean')
    matrix = pivot.values
    day_labels = [str(d) for d in pivot.columns]
    time_fracs = pivot.index.values
    return matrix, day_labels, time_fracs


def plot_heatmap(
    timestamps: Any,
    variables: OrderedDict,
    title: str,
    path_stem: Path,
    *,
    cmap: str = 'viridis',
) -> list[Path]:
    """
    Day-by-time-of-day heatmap(s). Days as columns, time-of-day as rows.
    variables: label -> (values, unit). One subplot per variable.
    Saves PDF and PNG.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        return []
    set_journal_style()
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(max(7, 0.02 * 365), 2.5 * n))
    if n == 1:
        axes = [axes]

    for i, (label, (values, unit)) in enumerate(variables.items()):
        ax = axes[i]
        matrix, day_labels, time_fracs = _reshape_to_day_matrix(timestamps, values)
        im = ax.imshow(
            matrix, aspect='auto', origin='lower', cmap=cmap,
            extent=[0, matrix.shape[1], time_fracs[0], time_fracs[-1]],
            interpolation='nearest',
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f'{label} ({unit})' if unit else label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_ylabel('Hour of day')
        n_ticks = min(12, len(day_labels))
        tick_positions = np.linspace(0, len(day_labels) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([day_labels[j] for j in tick_positions], rotation=45,
                           ha='right', fontsize=6)
        ax.text(0.005, 0.95, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', color='white',
                bbox=dict(facecolor='black', alpha=0.5, pad=1, linewidth=0))
    axes[-1].set_xlabel('Date')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def plot_residual_heatmap(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path_stem: Path,
    y_unit: str,
) -> list[Path]:
    """
    Residual heatmap (days x time-of-day) with diverging colormap centered at zero.
    Shows systematic time-of-day or seasonal bias.  Saves PDF and PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    set_journal_style()
    residuals = y_true - y_pred
    matrix, day_labels, time_fracs = _reshape_to_day_matrix(timestamps, residuals)

    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(max(7, 0.02 * 365), 3))
    im = ax.imshow(
        matrix, aspect='auto', origin='lower', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        extent=[0, matrix.shape[1], time_fracs[0], time_fracs[-1]],
        interpolation='nearest',
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f'Residual ({y_unit})', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_ylabel('Hour of day')
    ax.set_xlabel('Date')
    n_ticks = min(12, len(day_labels))
    tick_positions = np.linspace(0, len(day_labels) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([day_labels[j] for j in tick_positions], rotation=45,
                       ha='right', fontsize=6)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def _select_diverse_days(
    timestamps: Any,
    values: np.ndarray,
    n_pairs: int = 1,
) -> list[tuple]:
    """Auto-select pairs of days with contrasting behavior (high vs low variance).

    For solar PV this picks a clear day vs an overcast day.
    For load data it picks a high-demand vs low-demand day.
    """
    import pandas as pd
    ts = pd.DatetimeIndex(timestamps)
    df = pd.DataFrame({'val': values}, index=ts)
    df['date'] = df.index.date
    daily = df.groupby('date')['val'].agg(['std', 'mean', 'count'])
    daily = daily[daily['count'] >= daily['count'].median() * 0.5]
    if len(daily) < 2:
        return []
    daily = daily.sort_values('std')
    pairs = []
    for k in range(n_pairs):
        lo_idx = min(k, len(daily) - 2)
        hi_idx = max(len(daily) - 1 - k, 1)
        pairs.append((daily.index[lo_idx], daily.index[hi_idx]))
    return pairs


def plot_selected_days(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path_stem: Path,
    y_unit: str,
    *,
    day_pairs: list[tuple] | None = None,
    n_pairs: int = 2,
    model_name: str = '',
) -> list[Path]:
    """
    Plot curated day-pair overlays: actual vs predicted on selected days.
    If day_pairs is None, auto-selects contrasting days (high vs low variance).
    Each pair gets two panels side by side; pairs stacked vertically.
    Saves PDF and PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    import pandas as pd

    set_journal_style()
    ts = pd.DatetimeIndex(timestamps)
    df = pd.DataFrame({'actual': y_true, 'predicted': y_pred}, index=ts)
    df['date'] = df.index.date
    df['hour'] = df.index.hour + df.index.minute / 60.0

    if day_pairs is None:
        day_pairs = _select_diverse_days(timestamps, y_true, n_pairs=n_pairs)
    if not day_pairs:
        return []

    n_rows = len(day_pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 2.5 * n_rows), squeeze=False)
    panel_idx = 0
    for row, (day_lo, day_hi) in enumerate(day_pairs):
        for col, day in enumerate([day_lo, day_hi]):
            ax = axes[row, col]
            mask = df['date'] == day
            if not mask.any():
                ax.text(0.5, 0.5, f'No data for {day}', ha='center', va='center',
                        transform=ax.transAxes)
                panel_idx += 1
                continue
            sub = df[mask].sort_index()
            ax.plot(sub['hour'], sub['actual'], color='black', linewidth=1.2,
                    alpha=0.9, label='Actual')
            ax.plot(sub['hour'], sub['predicted'], color=COLORS[0], linewidth=1.0,
                    alpha=0.8, linestyle='--', label='Predicted')
            ax.set_xlabel('Hour of day')
            ax.set_ylabel(f'Value ({y_unit})')
            ax.set_xlim(0, 24)
            variance_label = 'Low variability' if col == 0 else 'High variability'
            ax.set_title(f'{day}  ({variance_label})', fontsize=8)
            ax.text(0.02, 0.95, f'({chr(97 + panel_idx)})', transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='top')
            if panel_idx == 0:
                ax.legend(loc='upper right', fontsize=7)
            panel_idx += 1

    subtitle = title
    if model_name:
        subtitle += f' — {model_name}'
    fig.suptitle(subtitle, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))


def plot_outlier_detection(
    day_indices: np.ndarray,
    true_outlier_values: np.ndarray,
    detected_outlier_values: np.ndarray,
    title: str,
    path_stem: Path,
    *,
    outlier_days: list[int] | None = None,
) -> list[Path]:
    """
    Two-panel figure: (a) stem plot true vs detected outlier values per day;
    (b) optional overlay of correction on outlier days (if outlier_days set).
    Saves PDF and PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    set_journal_style()
    n_panels = 2 if outlier_days else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 if n_panels == 2 else 4, 3.5))
    if n_panels == 1:
        axes = [axes]
    ax = axes[0]
    ax.plot(day_indices, true_outlier_values, 'o-', color=COLORS[0], markersize=5, label='True')
    ax.plot(day_indices, detected_outlier_values, 's--', color=COLORS[1], markersize=5, label='Detected')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Day index')
    ax.set_ylabel('Outlier value (log)')
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    ax.legend(loc='best', fontsize=8)
    ax.set_title('True vs detected outliers')
    if n_panels == 2 and outlier_days:
        ax2 = axes[1]
        ax2.bar([d for d in outlier_days if d < len(detected_outlier_values)],
                [detected_outlier_values[d] for d in outlier_days if d < len(detected_outlier_values)],
                color=COLORS[0], alpha=0.8, label='Detected')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('Outlier day index')
        ax2.set_ylabel('Outlier value (log)')
        ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top')
        ax2.set_title('Correction on outlier days')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return savefig(fig, Path(path_stem))
