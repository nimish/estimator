#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable, Literal, TypedDict

_EXAMPLES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXAMPLES_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EXAMPLES_DIR))

import numpy as np
import pandas as pd
import click

from common_cli import (
    add_no_download_option,
    default_output_dir,
    info,
    savefig,
    section,
    set_journal_style,
    success,
)

DEFAULT_TRAIN_START = "2022-01-01"
DEFAULT_TRAIN_END = "2024-01-01"
DEFAULT_TEST_END = "2024-12-31"
RESULTS_FILENAME = "multi_station_results.csv"
EXCLUDED_FILENAME = "multi_station_excluded.csv"
SUMMARY_FILENAME = "multi_station_summary.md"
LOG_FILENAME = "multi_station.log"
RESULT_COLUMNS = (
    "station_id",
    "station_name",
    "tidal_regime",
    "region",
    "active_regs",
    "n_train",
    "n_validation",
    "train_rmse",
    "train_mae",
    "train_mape",
    "train_r2",
    "validation_rmse",
    "validation_mae",
    "validation_mape",
    "validation_r2",
    "validation_minus_train_rmse",
    "validation_minus_train_mae",
    "validation_minus_train_mape",
    "validation_minus_train_r2",
)
EXCLUSION_COLUMNS = ("station_id", "station_name", "category", "reason")
RANKED_SUMMARY_COLUMNS = (
    "rank",
    "station_name",
    "validation_mape",
    "validation_rmse",
    "validation_r2",
)
tidal_compact: object | None = None


class StationRow(TypedDict):
    station_id: str
    station_name: str
    tidal_regime: str
    region: str


class FixedModelConfig(TypedDict):
    harmonic_orders: dict[str, int]
    regressors: tuple[str, ...]
    lag_ranges: dict[str, tuple[int, int]]
    knot_presets: dict[str, str]
    fourier_reg_weight: float


class MetricSummary(TypedDict):
    rmse: float
    mae: float
    mape: float
    r2: float


class FitResultSummary(TypedDict):
    metrics_train: MetricSummary
    metrics_test: MetricSummary
    active_regs: list[str]
    n_train: int
    n_test: int


class StationFitSummary(FitResultSummary):
    te_index: pd.DatetimeIndex
    te_obs: np.ndarray
    te_pred: np.ndarray


class ExclusionRow(TypedDict):
    station_id: str
    station_name: str
    category: str
    reason: str


class ResultRow(TypedDict):
    station_id: str
    station_name: str
    tidal_regime: str
    region: str
    active_regs: str
    n_train: int
    n_validation: int
    train_rmse: float
    train_mae: float
    train_mape: float
    train_r2: float
    validation_rmse: float
    validation_mae: float
    validation_mape: float
    validation_r2: float
    validation_minus_train_rmse: float
    validation_minus_train_mae: float
    validation_minus_train_mape: float
    validation_minus_train_r2: float


class IncludedStationRun(TypedDict):
    status: Literal["included"]
    row: ResultRow
    station_figure_path: Path


class ExcludedStationRun(TypedDict):
    status: Literal["excluded"]
    row: ExclusionRow


StationRunResult = IncludedStationRun | ExcludedStationRun


def _get_station_catalog() -> dict[str, dict[str, object]]:
    from example_tidal import STATION_CATALOG

    return STATION_CATALOG


def _get_battery_grid():
    import run_tidal_battery_grid as battery_grid

    return battery_grid


def _get_tidal_compact():
    global tidal_compact
    if tidal_compact is None:
        import example_tidal_compact as tidal_compact_module

        tidal_compact = tidal_compact_module
    return tidal_compact


def build_station_rows() -> list[StationRow]:
    return [
        {
            "station_id": station_id,
            "station_name": str(meta["name"]),
            "tidal_regime": str(meta.get("tidal_regime", "")),
            "region": str(meta.get("region", "")),
        }
        for station_id, meta in _get_station_catalog().items()
    ]


def build_fixed_model_config() -> FixedModelConfig:
    battery_grid = _get_battery_grid()
    anchor = battery_grid.build_anchor_candidate()
    return {
        "harmonic_orders": battery_grid.build_harmonic_orders(
            anchor.mf_mm_order,
            anchor.annual_order,
        ),
        "regressors": anchor.regressors,
        "lag_ranges": battery_grid.build_lag_ranges(anchor),
        "knot_presets": battery_grid.build_knot_presets(anchor),
        "fourier_reg_weight": anchor.fourier_reg_weight,
    }


def metric_gap(left: float | int | None, right: float | int | None) -> float:
    if left is None or right is None:
        return float("nan")
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return float("nan")
    return left_value - right_value


def make_exclusion_row(
    *,
    station_id: str,
    station_name: str,
    category: str,
    reason: str,
) -> ExclusionRow:
    return {
        "station_id": station_id,
        "station_name": station_name,
        "category": category,
        "reason": reason,
    }


def make_result_row(
    *,
    station_row: StationRow,
    fit_result: FitResultSummary,
) -> ResultRow:
    train_metrics = fit_result["metrics_train"]
    validation_metrics = fit_result["metrics_test"]
    return {
        "station_id": station_row["station_id"],
        "station_name": station_row["station_name"],
        "tidal_regime": station_row["tidal_regime"],
        "region": station_row["region"],
        "active_regs": ",".join(fit_result["active_regs"]),
        "n_train": int(fit_result["n_train"]),
        "n_validation": int(fit_result["n_test"]),
        "train_rmse": float(train_metrics["rmse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_mape": float(train_metrics["mape"]),
        "train_r2": float(train_metrics["r2"]),
        "validation_rmse": float(validation_metrics["rmse"]),
        "validation_mae": float(validation_metrics["mae"]),
        "validation_mape": float(validation_metrics["mape"]),
        "validation_r2": float(validation_metrics["r2"]),
        "validation_minus_train_rmse": metric_gap(
            validation_metrics["rmse"], train_metrics["rmse"]
        ),
        "validation_minus_train_mae": metric_gap(
            validation_metrics["mae"], train_metrics["mae"]
        ),
        "validation_minus_train_mape": metric_gap(
            validation_metrics["mape"], train_metrics["mape"]
        ),
        "validation_minus_train_r2": metric_gap(
            validation_metrics["r2"], train_metrics["r2"]
        ),
    }


def _markdown_table(
    rows: list[ResultRow] | list[ExclusionRow], columns: list[str]
) -> str:
    return _markdown_table_rows(
        [dict(row) for row in rows],
        columns,
    )


def _markdown_table_rows(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_None._"

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| "
        + " | ".join(_format_markdown_value(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _format_markdown_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        if not math.isfinite(value_float):
            return str(value)
        return str(round(value_float, 12))
    return str(value)


def _report_relative_figure_path(path: Path, *, report_dir: Path | None) -> str:
    if not path.is_absolute():
        return path.as_posix()
    if report_dir is None:
        raise ValueError("report_dir is required for absolute figure paths")
    return Path(os.path.relpath(path, start=report_dir)).as_posix()


def _format_metric(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4g}"


def _train_window_end_inclusive() -> str:
    return (pd.Timestamp(DEFAULT_TRAIN_END) - pd.Timedelta(days=1)).date().isoformat()


def _held_out_window_start() -> str:
    return pd.Timestamp(DEFAULT_TRAIN_END).date().isoformat()


def _held_out_window_end_inclusive() -> str:
    return pd.Timestamp(DEFAULT_TEST_END).date().isoformat()


def _build_setup_lines() -> list[str]:
    fixed_config = build_fixed_model_config()
    harmonic_summary = ", ".join(
        f"{name}={order}" for name, order in fixed_config["harmonic_orders"].items()
    )
    lag_summary = ", ".join(
        f"{name} {lag_range}"
        for name, lag_range in fixed_config["lag_ranges"].items()
    )
    regressor_summary = ", ".join(fixed_config["regressors"])
    return [
        "Fixed model form shared across every station:",
        f"- Train window: `{DEFAULT_TRAIN_START}` to `{_train_window_end_inclusive()}`.",
        f"- Held-out window: `{_held_out_window_start()}` to `{_held_out_window_end_inclusive()}`.",
        f"- Harmonic orders: `{harmonic_summary}`.",
        (
            f"- Exogenous regressors: `{regressor_summary}` with lag windows "
            f"`{lag_summary}` and Fourier regularization "
            f"`{fixed_config['fourier_reg_weight']:.1e}`."
        ),
    ]


def _ranked_summary_rows(included_rows: list[ResultRow]) -> list[dict[str, object]]:
    ordered = sorted(
        included_rows,
        key=lambda row: (
            float(row["validation_mape"]),
            float(row["validation_rmse"]),
            -float(row["validation_r2"]),
            row["station_name"],
        ),
    )
    return [
        {
            "rank": rank,
            "station_name": row["station_name"],
            "validation_mape": row["validation_mape"],
            "validation_rmse": row["validation_rmse"],
            "validation_r2": row["validation_r2"],
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def _ranked_summary_lines(included_rows: list[ResultRow]) -> list[str]:
    ranked_rows = _ranked_summary_rows(included_rows)
    if not ranked_rows:
        return ["_No included stations to rank._"]

    best = ranked_rows[0]
    worst = ranked_rows[-1]
    return [
        (
            "Best held-out MAPE: "
            f"{best['station_name']} "
            f"(MAPE={_format_metric(best['validation_mape'])}, "
            f"RMSE={_format_metric(best['validation_rmse'])}, "
            f"R2={_format_metric(best['validation_r2'])})."
        ),
        (
            "Worst held-out MAPE: "
            f"{worst['station_name']} "
            f"(MAPE={_format_metric(worst['validation_mape'])}, "
            f"RMSE={_format_metric(worst['validation_rmse'])}, "
            f"R2={_format_metric(worst['validation_r2'])})."
        ),
        "",
        _markdown_table_rows(ranked_rows, list(RANKED_SUMMARY_COLUMNS)),
    ]


def build_summary_markdown(
    *,
    included_rows: list[ResultRow],
    excluded_rows: list[ExclusionRow],
    metric_figure_paths: list[Path],
    station_figure_paths: list[Path],
    report_dir: Path | None = None,
) -> str:
    lines = [
        "# Tidal Multi-Station Report",
        "",
        "## Setup",
        *_build_setup_lines(),
        "",
        "## Ranked held-out summary",
        *_ranked_summary_lines(included_rows),
        "",
        "## Included stations",
        _markdown_table(included_rows, list(RESULT_COLUMNS)),
        "",
        "## Excluded stations",
        _markdown_table(excluded_rows, list(EXCLUSION_COLUMNS)),
        "",
        "## Across-station charts",
    ]
    if metric_figure_paths:
        lines.extend(
            f"![{path.stem}]({_report_relative_figure_path(path, report_dir=report_dir)})"
            for path in metric_figure_paths
        )
    else:
        lines.append("_None._")
    lines.extend(["", "## Per-station figures"])
    if station_figure_paths:
        lines.extend(
            f"![{path.stem}]({_report_relative_figure_path(path, report_dir=report_dir)})"
            for path in station_figure_paths
        )
    else:
        lines.append("_None._")
    return "\n".join(lines) + "\n"


def _saved_png_path(saved_paths: list[Path]) -> Path:
    for path in saved_paths:
        if path.suffix == ".png":
            return path
    raise ValueError("savefig did not return a PNG path")


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _results_path(output_dir: Path) -> Path:
    return output_dir / RESULTS_FILENAME


def _excluded_path(output_dir: Path) -> Path:
    return output_dir / EXCLUDED_FILENAME


def _summary_path(output_dir: Path) -> Path:
    return output_dir / SUMMARY_FILENAME


def _log_path(output_dir: Path) -> Path:
    return output_dir / LOG_FILENAME


def _write_checkpoint_tables(
    *,
    output_dir: Path,
    included_rows: list[ResultRow],
    excluded_rows: list[ExclusionRow],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(output_dir)
    excluded_path = _excluded_path(output_dir)
    pd.DataFrame(included_rows, columns=list(RESULT_COLUMNS)).to_csv(
        results_path,
        index=False,
    )
    pd.DataFrame(excluded_rows, columns=list(EXCLUSION_COLUMNS)).to_csv(
        excluded_path,
        index=False,
    )
    return results_path, excluded_path


def build_station_figure(
    *,
    station_name: str,
    metrics_row: ResultRow,
    te_index: pd.DatetimeIndex,
    te_obs: np.ndarray,
    te_pred: np.ndarray,
    figure_dir: Path,
) -> Path:
    plt = _import_pyplot()
    set_journal_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.25))
    observed = np.asarray(te_obs, dtype=float)
    predicted = np.asarray(te_pred, dtype=float)

    ax.plot(te_index, observed, label="Observed", color="steelblue")
    ax.plot(te_index, predicted, label="Predicted", color="coral", alpha=0.9)
    ax.set_title(f"{station_name}: held-out observed vs predicted")
    ax.set_ylabel("Water level (m)")
    ax.legend(loc="upper right")

    metrics_text = (
        f"Train RMSE {metrics_row['train_rmse']:.3f}, MAPE {metrics_row['train_mape']:.1f}%, "
        f"R2 {metrics_row['train_r2']:.3f}\n"
        f"Held-out RMSE {metrics_row['validation_rmse']:.3f}, "
        f"MAPE {metrics_row['validation_mape']:.1f}%, "
        f"R2 {metrics_row['validation_r2']:.3f}"
    )
    ax.text(
        0.01,
        0.01,
        metrics_text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    saved_paths = savefig(fig, figure_dir / f"station_{metrics_row['station_id']}")
    return _saved_png_path(saved_paths)


def validate_fit_result(
    fit_result: StationFitSummary,
    *,
    expected_regressors: tuple[str, ...],
) -> str | None:
    if tuple(fit_result["active_regs"]) != expected_regressors:
        actual_regs = ",".join(fit_result["active_regs"]) or "(none)"
        expected_regs = ",".join(expected_regressors)
        return (
            "fit dropped or reordered the fixed regressor set: "
            f"expected {expected_regs}, got {actual_regs}"
        )

    metric_sources = (
        ("train", fit_result["metrics_train"]),
        ("validation", fit_result["metrics_test"]),
    )
    for split_name, metrics in metric_sources:
        for metric_name, value in metrics.items():
            if not np.isfinite(float(value)):
                return f"non-finite {split_name} {metric_name}: {value}"

    return None


def build_metric_figures(rows: list[ResultRow], figure_dir: Path) -> list[Path]:
    if not rows:
        return []

    plt = _import_pyplot()
    set_journal_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    figure_paths: list[Path] = []
    metric_specs = [
        ("mape", "MAPE (%)", True),
        ("rmse", "RMSE (m)", True),
        ("mae", "MAE (m)", True),
        ("r2", "R2", False),
    ]

    for metric_name, axis_label, ascending in metric_specs:
        validation_key = f"validation_{metric_name}"
        train_key = f"train_{metric_name}"
        ordered = frame.sort_values(validation_key, ascending=ascending).reset_index(drop=True)
        positions = np.arange(len(ordered), dtype=float)

        fig, ax = plt.subplots(
            figsize=(10, max(3.0, 0.45 * len(ordered) + 1.6)),
        )
        bar_height = 0.36
        ax.barh(
            positions - bar_height / 2,
            ordered[train_key],
            height=bar_height,
            label="Train",
            color="#4C72B0",
        )
        ax.barh(
            positions + bar_height / 2,
            ordered[validation_key],
            height=bar_height,
            label="Held-out",
            color="#DD8452",
        )
        ax.set_yticks(positions, ordered["station_name"])
        ax.set_xlabel(axis_label)
        ax.set_title(f"{axis_label} by station")
        ax.invert_yaxis()
        ax.legend(loc="best")

        saved_paths = savefig(fig, figure_dir / validation_key)
        figure_paths.append(_saved_png_path(saved_paths))

    return figure_paths


def run_station_fit(
    *,
    station_row: StationRow,
    output_dir: Path,
    no_download: bool,
) -> StationRunResult:
    tidal_compact_module = _get_tidal_compact()
    try:
        station_data = tidal_compact_module.load_station_frame(
            station_row["station_name"],
            use_weather=True,
            download_tidal=not no_download,
            download_weather=not no_download,
            data_start=pd.Timestamp(DEFAULT_TRAIN_START).date(),
            data_end=pd.Timestamp(DEFAULT_TEST_END).date(),
        )
    except FileNotFoundError as exc:
        return {
            "status": "excluded",
            "row": make_exclusion_row(
                station_id=station_row["station_id"],
                station_name=station_row["station_name"],
                category="coverage",
                reason=str(exc),
            ),
        }
    except Exception as exc:
        return {
            "status": "excluded",
            "row": make_exclusion_row(
                station_id=station_row["station_id"],
                station_name=station_row["station_name"],
                category="load",
                reason=str(exc),
            ),
        }

    fixed_config = build_fixed_model_config()
    component_mask = {
        **{name: True for name in fixed_config["harmonic_orders"]},
        **{name: True for name in fixed_config["regressors"]},
    }

    try:
        fit_result = tidal_compact_module.run_tidal_model(
            component_mask,
            df=station_data["df"],
            sph=station_data["sph"],
            harmonic_orders=fixed_config["harmonic_orders"],
            fourier_reg_weight=fixed_config["fourier_reg_weight"],
            lag_ranges=fixed_config["lag_ranges"],
            knot_presets=fixed_config["knot_presets"],
            interaction_pairs=[],
            train_start=DEFAULT_TRAIN_START,
            train_end=DEFAULT_TRAIN_END,
            test_end=DEFAULT_TEST_END,
            solver_verbose=False,
            debug=False,
        )
        validation_error = validate_fit_result(
            fit_result,
            expected_regressors=fixed_config["regressors"],
        )
        if validation_error is not None:
            return {
                "status": "excluded",
                "row": make_exclusion_row(
                    station_id=station_row["station_id"],
                    station_name=station_row["station_name"],
                    category="fit",
                    reason=validation_error,
                ),
            }
        result_row = make_result_row(
            station_row=station_row,
            fit_result=fit_result,
        )
        station_figure_path = build_station_figure(
            station_name=station_row["station_name"],
            metrics_row=result_row,
            te_index=fit_result["te_index"],
            te_obs=fit_result["te_obs"],
            te_pred=fit_result["te_pred"],
            figure_dir=output_dir / "figures" / "multi_station",
        )
    except Exception as exc:
        return {
            "status": "excluded",
            "row": make_exclusion_row(
                station_id=station_row["station_id"],
                station_name=station_row["station_name"],
                category="fit",
                reason=str(exc),
            ),
        }

    return {
        "status": "included",
        "row": result_row,
        "station_figure_path": station_figure_path,
    }


def collect_station_reports(
    *,
    output_dir: Path,
    no_download: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[list[ResultRow], list[ExclusionRow], list[Path], list[Path]]:
    included_rows: list[ResultRow] = []
    excluded_rows: list[ExclusionRow] = []
    station_figure_paths: list[Path] = []
    station_rows = build_station_rows()
    total = len(station_rows)

    for index, station_row in enumerate(station_rows, start=1):
        station_label = (
            f"{station_row['station_name']} ({station_row['station_id']})"
        )
        if log is not None:
            log(f"[{index}/{total}] Processing {station_label}")

        try:
            result = run_station_fit(
                station_row=station_row,
                output_dir=output_dir,
                no_download=no_download,
            )
        except Exception as exc:
            result = {
                "status": "excluded",
                "row": make_exclusion_row(
                    station_id=station_row["station_id"],
                    station_name=station_row["station_name"],
                    category="fit",
                    reason=str(exc),
                ),
            }

        if result["status"] == "included":
            included_rows.append(result["row"])
            station_figure_paths.append(result["station_figure_path"])
            if log is not None:
                log(
                    f"[{index}/{total}] Included {station_label}: "
                    f"held-out MAPE={_format_metric(result['row']['validation_mape'])}, "
                    f"RMSE={_format_metric(result['row']['validation_rmse'])}, "
                    f"R2={_format_metric(result['row']['validation_r2'])}"
                )
        else:
            excluded_rows.append(result["row"])
            if log is not None:
                log(
                    f"[{index}/{total}] Excluded {station_label}: "
                    f"{result['row']['category']} - {result['row']['reason']}"
                )
        _write_checkpoint_tables(
            output_dir=output_dir,
            included_rows=included_rows,
            excluded_rows=excluded_rows,
        )

    metric_figure_paths = (
        build_metric_figures(included_rows, output_dir / "figures" / "multi_station")
        if included_rows
        else []
    )
    if log is not None:
        if metric_figure_paths:
            log(f"Built {len(metric_figure_paths)} across-station metric figures.")
        else:
            log("No included stations; skipped across-station metric figures.")

    return (
        included_rows,
        excluded_rows,
        metric_figure_paths,
        station_figure_paths,
    )


def write_report_outputs(
    *,
    output_dir: Path,
    included_rows: list[ResultRow],
    excluded_rows: list[ExclusionRow],
    summary_markdown: str,
    log_lines: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path, excluded_path = _write_checkpoint_tables(
        output_dir=output_dir,
        included_rows=included_rows,
        excluded_rows=excluded_rows,
    )
    summary_path = _summary_path(output_dir)
    log_path = _log_path(output_dir)
    summary_path.write_text(summary_markdown, encoding="utf-8")

    log_text = "\n".join(log_lines or [])
    if log_text and not log_path.exists():
        log_path.write_text(log_text + "\n", encoding="utf-8")

    return results_path, excluded_path, summary_path


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Output directory for CSV, markdown summary, figures, and log files. "
        "Default: examples/reports/tidal_multi_station."
    ),
)
@add_no_download_option
def main(output_dir: Path | None, no_download: bool) -> None:
    """Run the fixed tidal multi-station transport report."""
    output_dir = output_dir or (default_output_dir() / "tidal_multi_station")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(output_dir)
    log_lines: list[str] = []

    def log(message: str) -> None:
        info(message)
        log_lines.append(message)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    section("Tidal multi-station report")
    log(f"Output dir: {output_dir}")
    log(f"No download: {no_download}")
    log(
        "Fixed window: "
        f"train {DEFAULT_TRAIN_START} to {_train_window_end_inclusive()}, "
        f"held-out {_held_out_window_start()} to {_held_out_window_end_inclusive()}"
    )

    included_rows, excluded_rows, metric_figure_paths, station_figure_paths = (
        collect_station_reports(
            output_dir=output_dir,
            no_download=no_download,
            log=log,
        )
    )
    log(f"Included stations: {len(included_rows)}")
    log(f"Excluded stations: {len(excluded_rows)}")

    summary_markdown = build_summary_markdown(
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        metric_figure_paths=metric_figure_paths,
        station_figure_paths=station_figure_paths,
        report_dir=output_dir,
    )
    results_path, excluded_path, summary_path = write_report_outputs(
        output_dir=output_dir,
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        summary_markdown=summary_markdown,
        log_lines=log_lines,
    )

    success(f"Results CSV: {results_path}")
    success(f"Exclusions CSV: {excluded_path}")
    success(f"Summary markdown: {summary_path}")
    success(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
