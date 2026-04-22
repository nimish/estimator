#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Grid-search long-period tidal hyperparameters for The Battery."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Protocol

_EXAMPLES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXAMPLES_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EXAMPLES_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl_cache"),
)

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn  # noqa: E402
from rich.table import Table  # noqa: E402

from common_cli import console, default_output_dir, error, info, section, success  # noqa: E402
import example_tidal_compact as tidal_compact  # noqa: E402

DEFAULT_STATION_NAME = "The Battery, NY"
DEFAULT_DATA_START = pd.Timestamp("2022-01-01").date()
DEFAULT_DATA_END = pd.Timestamp("2025-01-01").date()

SELECTION_TRAIN_START = "2022-01-01"
SELECTION_TRAIN_END = "2024-01-01"
SELECTION_TEST_END = "2025-01-01"

ANCHOR_REGRESSORS = ("pressure", "dp_dt", "wind_u")
ANCHOR_KNOT_PRESET = "med"
ANCHOR_FOURIER_REG_WEIGHT = 1.0e-4

KNOT_PRESETS = ("med", "high")
MF_MM_ORDERS = (0, 1, 2, 4, 8, 12, 16)
ANNUAL_ORDERS = (0, 2, 4, 8, 16, 24, 32)

LOW_COMPLEXITY_REG_WEIGHTS = (1.0e-4, 3.0e-4, 1.0e-3)
MEDIUM_COMPLEXITY_REG_WEIGHTS = (3.0e-4, 1.0e-3, 3.0e-3)
HIGH_COMPLEXITY_REG_WEIGHTS = (1.0e-3, 3.0e-3, 1.0e-2)
VERY_HIGH_COMPLEXITY_REG_WEIGHTS = (3.0e-3, 1.0e-2)

SHORT_PERIOD_DEFAULTS = {
    name: int(order)
    for name, order in tidal_compact.DEFAULT_HARMONICS.items()
    if name not in {"Mf", "Mm", "annual"}
}

METRIC_NAMES = ("rmse", "mae", "mape", "r2")


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    regressors: tuple[str, ...]
    knot_preset: str
    mf_mm_order: int
    annual_order: int
    fourier_reg_weight: float


class CandidateReporter(Protocol):
    def __call__(
        self,
        *,
        completed: int,
        total: int,
        stage_prefix: str,
        row: dict[str, Any],
        best_so_far: dict[str, Any] | None,
    ) -> None: ...


_CANDIDATE_WORKER_STATE: dict[str, Any] = {}


def build_anchor_candidate() -> SearchCandidate:
    return SearchCandidate(
        regressors=ANCHOR_REGRESSORS,
        knot_preset=ANCHOR_KNOT_PRESET,
        mf_mm_order=0,
        annual_order=0,
        fourier_reg_weight=ANCHOR_FOURIER_REG_WEIGHT,
    )


def is_anchor_candidate(candidate: SearchCandidate) -> bool:
    return candidate == build_anchor_candidate()


def minimum_reg_weight_for_orders(mf_mm_order: int, annual_order: int) -> float | None:
    if mf_mm_order >= 4:
        return 3.0e-3
    if mf_mm_order >= 2 and annual_order >= 8:
        return 3.0e-3
    return None


def reg_weight_grid_for_orders(mf_mm_order: int, annual_order: int) -> tuple[float, ...]:
    """Favor stronger regularization for aggressive long-period orders."""
    if mf_mm_order >= 12 or annual_order >= 24:
        base_grid = VERY_HIGH_COMPLEXITY_REG_WEIGHTS
    elif mf_mm_order >= 8 or annual_order >= 16:
        base_grid = HIGH_COMPLEXITY_REG_WEIGHTS
    elif mf_mm_order >= 4 or annual_order >= 8:
        base_grid = MEDIUM_COMPLEXITY_REG_WEIGHTS
    else:
        base_grid = LOW_COMPLEXITY_REG_WEIGHTS

    minimum_reg_weight = minimum_reg_weight_for_orders(mf_mm_order, annual_order)
    if minimum_reg_weight is None:
        return base_grid
    return tuple(weight for weight in base_grid if weight >= minimum_reg_weight)


def build_harmonic_orders(mf_mm_order: int, annual_order: int) -> dict[str, int]:
    """Keep short periods fixed while sweeping long-period flexibility."""
    harmonic_orders = dict(SHORT_PERIOD_DEFAULTS)
    harmonic_orders["Mf"] = mf_mm_order
    harmonic_orders["Mm"] = mf_mm_order
    harmonic_orders["annual"] = annual_order
    return harmonic_orders


def build_search_candidates() -> list[SearchCandidate]:
    """Build the anchored Battery-local search candidate list."""
    candidates = [build_anchor_candidate()]
    seen = {candidates[0]}
    for knot_preset in KNOT_PRESETS:
        for mf_mm_order in MF_MM_ORDERS:
            for annual_order in ANNUAL_ORDERS:
                for reg_weight in reg_weight_grid_for_orders(mf_mm_order, annual_order):
                    candidate = SearchCandidate(
                        regressors=ANCHOR_REGRESSORS,
                        knot_preset=knot_preset,
                        mf_mm_order=mf_mm_order,
                        annual_order=annual_order,
                        fourier_reg_weight=reg_weight,
                    )
                    if candidate in seen:
                        continue
                    candidates.append(candidate)
                    seen.add(candidate)
    return candidates


def build_candidate_label(candidate: SearchCandidate) -> str:
    regs = "+".join(candidate.regressors)
    label = (
        f"regs={regs} knots={candidate.knot_preset} "
        f"Mf/Mm={candidate.mf_mm_order} annual={candidate.annual_order} "
        f"reg={candidate.fourier_reg_weight:g}"
    )
    if is_anchor_candidate(candidate):
        return f"anchor {label}"
    return label


def candidate_key(candidate: SearchCandidate) -> tuple[tuple[str, ...], str, int, int, float]:
    return (
        candidate.regressors,
        candidate.knot_preset,
        candidate.mf_mm_order,
        candidate.annual_order,
        candidate.fourier_reg_weight,
    )


def row_key(row: dict[str, Any]) -> tuple[tuple[str, ...], str, int, int, float]:
    return (
        tuple(row["regressor_family"].split("+")),
        str(row["knot_preset"]),
        int(row["mf_mm_order"]),
        int(row["annual_order"]),
        float(row["fourier_reg_weight"]),
    )


def stage_train_prefix(stage_prefix: str) -> str:
    if stage_prefix == "validation":
        return "selection_train"
    if stage_prefix == "test":
        return "final_train"
    raise ValueError(f"Unsupported stage prefix: {stage_prefix!r}")


def base_candidate_row(candidate: SearchCandidate) -> dict[str, Any]:
    return {
        "candidate_label": build_candidate_label(candidate),
        "regressor_family": "+".join(candidate.regressors),
        "knot_preset": candidate.knot_preset,
        "mf_mm_order": candidate.mf_mm_order,
        "annual_order": candidate.annual_order,
        "fourier_reg_weight": candidate.fourier_reg_weight,
        "is_anchor": is_anchor_candidate(candidate),
    }


def numeric_metric_value(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float(np.nan)


def metric_values(prefix: str, metrics: Mapping[str, object]) -> dict[str, float]:
    return {
        f"{prefix}_{name}": numeric_metric_value(metrics.get(name, np.nan))
        for name in METRIC_NAMES
    }


def build_component_mask(candidate: SearchCandidate) -> dict[str, bool]:
    component_mask = {name: True for name in build_harmonic_orders(candidate.mf_mm_order, candidate.annual_order)}
    component_mask.update({name: True for name in candidate.regressors})
    return component_mask


def build_lag_ranges(candidate: SearchCandidate) -> dict[str, tuple[int, int]]:
    return {name: tidal_compact.LAG_DEFAULTS[name] for name in candidate.regressors}


def build_knot_presets(candidate: SearchCandidate) -> dict[str, str]:
    return {name: candidate.knot_preset for name in candidate.regressors}


def has_finite_metric_row(row: dict[str, Any], prefix: str) -> bool:
    return any(np.isfinite(float(row.get(f"{prefix}_{name}", np.nan))) for name in METRIC_NAMES)


def failed_candidate_row(
    candidate: SearchCandidate,
    stage_prefix: str,
    fit_error: str,
) -> dict[str, Any]:
    train_prefix = stage_train_prefix(stage_prefix)
    row = base_candidate_row(candidate)
    row.update(metric_values(train_prefix, {}))
    row.update(metric_values(stage_prefix, {}))
    row[f"{train_prefix}_mape_n"] = 0
    row[f"{stage_prefix}_mape_n"] = 0
    row[f"{train_prefix}_n"] = 0
    row[f"{stage_prefix}_n"] = 0
    row["active_regs"] = ",".join(candidate.regressors)
    row[f"{stage_prefix}_fit_status"] = "exception"
    row[f"{stage_prefix}_fit_error"] = fit_error
    return row


def evaluate_candidate(
    candidate: SearchCandidate,
    *,
    df: pd.DataFrame,
    sph: int,
    train_start: str,
    train_end: str,
    test_end: str,
    stage_prefix: str,
) -> dict[str, Any]:
    """Run one candidate through the compact notebook fitting path."""
    train_prefix = stage_train_prefix(stage_prefix)
    try:
        fit_result = tidal_compact.run_tidal_model(
            build_component_mask(candidate),
            df=df,
            sph=sph,
            harmonic_orders=build_harmonic_orders(candidate.mf_mm_order, candidate.annual_order),
            fourier_reg_weight=candidate.fourier_reg_weight,
            lag_ranges=build_lag_ranges(candidate),
            knot_presets=build_knot_presets(candidate),
            interaction_pairs=[],
            train_start=train_start,
            train_end=train_end,
            test_end=test_end,
            solver_verbose=False,
            debug=False,
        )
    except Exception as exc:
        return failed_candidate_row(candidate, stage_prefix, f"{type(exc).__name__}: {exc}")

    row = base_candidate_row(candidate)
    row.update(metric_values(train_prefix, fit_result["metrics_train"]))
    row.update(metric_values(stage_prefix, fit_result["metrics_test"]))
    row[f"{train_prefix}_mape_n"] = int(fit_result.get("tr_mape_n", 0))
    row[f"{stage_prefix}_mape_n"] = int(fit_result.get("te_mape_n", 0))
    row[f"{train_prefix}_n"] = int(fit_result.get("n_train", 0))
    row[f"{stage_prefix}_n"] = int(fit_result.get("n_test", 0))
    row["active_regs"] = ",".join(fit_result.get("active_regs", []))
    row[f"{stage_prefix}_fit_status"] = "ok"
    row[f"{stage_prefix}_fit_error"] = ""
    if not has_finite_metric_row(row, stage_prefix):
        row[f"{stage_prefix}_fit_status"] = "non_finite_metrics"
        row[f"{stage_prefix}_fit_error"] = "non-finite metrics"
    return row


def validation_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    mape = float(row.get("validation_mape", np.nan))
    rmse = float(row.get("validation_rmse", np.nan))
    r2 = float(row.get("validation_r2", np.nan))
    return (
        mape if np.isfinite(mape) else np.inf,
        rmse if np.isfinite(rmse) else np.inf,
        -r2 if np.isfinite(r2) else np.inf,
        float(row.get("fourier_reg_weight", np.inf)),
    )


def test_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    mape = float(row.get("test_mape", np.nan))
    rmse = float(row.get("test_rmse", np.nan))
    r2 = float(row.get("test_r2", np.nan))
    return (
        mape if np.isfinite(mape) else np.inf,
        rmse if np.isfinite(rmse) else np.inf,
        -r2 if np.isfinite(r2) else np.inf,
        float(row.get("fourier_reg_weight", np.inf)),
    )


def select_promoted_candidates(rows: list[dict[str, Any]], top_k: int = 12) -> list[dict[str, Any]]:
    """Legacy helper from the older two-stage search flow."""
    if not rows:
        return []

    ordered = sorted(rows, key=validation_sort_key)
    selected: dict[tuple[tuple[str, ...], str, int, int, float], dict[str, Any]] = {}

    def add_best(candidates: list[dict[str, Any]]) -> None:
        if not candidates:
            return
        valid = [row for row in candidates if has_finite_metric_row(row, "validation")]
        pool = valid or candidates
        best = min(pool, key=validation_sort_key)
        selected[row_key(best)] = best

    for row in ordered[:top_k]:
        selected[row_key(row)] = row

    for order in MF_MM_ORDERS:
        add_best([row for row in rows if int(row["mf_mm_order"]) == order])

    for order in ANNUAL_ORDERS:
        add_best([row for row in rows if int(row["annual_order"]) == order])

    families = {row["regressor_family"] for row in rows}
    for family in families:
        add_best([row for row in rows if row["regressor_family"] == family])

    return sorted(selected.values(), key=validation_sort_key)


def metric_gap(left: float | int | None, right: float | int | None) -> float:
    left_value = float(left) if left is not None else np.nan
    right_value = float(right) if right is not None else np.nan
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        return np.nan
    return left_value - right_value


def merge_rows(
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Combine validation and outer-test rows into one leaderboard frame."""
    merged_rows: list[dict[str, Any]] = []
    test_by_key = {row_key(row): row for row in test_rows}
    anchor_row = next((row for row in validation_rows if bool(row.get("is_anchor"))), None)
    anchor_validation_mape = anchor_row.get("validation_mape") if anchor_row is not None else np.nan
    anchor_validation_rmse = anchor_row.get("validation_rmse") if anchor_row is not None else np.nan
    anchor_validation_r2 = anchor_row.get("validation_r2") if anchor_row is not None else np.nan
    anchor_candidate_label = anchor_row.get("candidate_label", "") if anchor_row is not None else ""

    for validation_row in validation_rows:
        merged = dict(validation_row)
        test_row = test_by_key.get(row_key(validation_row))
        merged["promoted"] = test_row is not None
        merged["anchor_candidate_label"] = anchor_candidate_label
        merged["anchor_validation_mape"] = anchor_validation_mape
        merged["anchor_validation_rmse"] = anchor_validation_rmse
        merged["anchor_validation_r2"] = anchor_validation_r2
        if test_row is not None:
            for key, value in test_row.items():
                if key in merged and key not in {"active_regs"}:
                    continue
                merged[key] = value
        merged["validation_minus_anchor_mape"] = metric_gap(
            merged.get("validation_mape"),
            anchor_validation_mape,
        )
        merged["validation_minus_anchor_rmse"] = metric_gap(
            merged.get("validation_rmse"),
            anchor_validation_rmse,
        )
        merged["validation_minus_anchor_r2"] = metric_gap(
            merged.get("validation_r2"),
            anchor_validation_r2,
        )
        merged["validation_minus_selection_train_mape"] = metric_gap(
            merged.get("validation_mape"),
            merged.get("selection_train_mape"),
        )
        merged["validation_minus_selection_train_rmse"] = metric_gap(
            merged.get("validation_rmse"),
            merged.get("selection_train_rmse"),
        )
        merged["validation_minus_selection_train_r2"] = metric_gap(
            merged.get("validation_r2"),
            merged.get("selection_train_r2"),
        )
        merged["test_minus_final_train_mape"] = metric_gap(
            merged.get("test_mape"),
            merged.get("final_train_mape"),
        )
        merged["test_minus_final_train_rmse"] = metric_gap(
            merged.get("test_rmse"),
            merged.get("final_train_rmse"),
        )
        merged["test_minus_final_train_r2"] = metric_gap(
            merged.get("test_r2"),
            merged.get("final_train_r2"),
        )
        merged_rows.append(merged)

    frame = pd.DataFrame(merged_rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["validation_mape", "validation_rmse", "validation_r2", "annual_order", "mf_mm_order"],
        ascending=[True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def write_leaderboard(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return frame.to_dict(orient="records")


def resume_candidates(
    candidates: list[SearchCandidate],
    existing_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[SearchCandidate]]:
    candidate_keys = {candidate_key(candidate) for candidate in candidates}
    resumed_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[tuple[str, ...], str, int, int, float]] = set()

    for row in existing_rows:
        key = row_key(row)
        if key not in candidate_keys or key in completed_keys:
            continue
        resumed_rows.append(row)
        completed_keys.add(key)

    pending_candidates = [
        candidate for candidate in candidates if candidate_key(candidate) not in completed_keys
    ]
    return resumed_rows, pending_candidates


def init_candidate_worker(
    df: pd.DataFrame,
    sph: int,
    train_start: str,
    train_end: str,
    test_end: str,
    stage_prefix: str,
) -> None:
    global _CANDIDATE_WORKER_STATE
    _CANDIDATE_WORKER_STATE = {
        "df": df,
        "sph": sph,
        "train_start": train_start,
        "train_end": train_end,
        "test_end": test_end,
        "stage_prefix": stage_prefix,
    }


def run_candidate_job(candidate: SearchCandidate) -> dict[str, Any]:
    config = _CANDIDATE_WORKER_STATE
    stage_prefix = str(config.get("stage_prefix", "validation"))
    try:
        return evaluate_candidate(
            candidate,
            df=config["df"],
            sph=int(config["sph"]),
            train_start=str(config["train_start"]),
            train_end=str(config["train_end"]),
            test_end=str(config["test_end"]),
            stage_prefix=stage_prefix,
        )
    except Exception as exc:
        return failed_candidate_row(candidate, stage_prefix, f"{type(exc).__name__}: {exc}")


def make_candidate_pool(
    *,
    workers: int,
    df: pd.DataFrame,
    sph: int,
    train_start: str,
    train_end: str,
    test_end: str,
    stage_prefix: str,
):
    context = multiprocessing.get_context("spawn")
    return context.Pool(
        processes=workers,
        initializer=init_candidate_worker,
        initargs=(df, sph, train_start, train_end, test_end, stage_prefix),
    )


def make_file_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    return log


def make_plain_logger(log_path: Path):
    file_log = make_file_logger(log_path)

    def log(message: str) -> None:
        info(message)
        file_log(message)

    return log


def make_stage_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def handle_interrupt(pool: Any) -> None:
    pool.terminate()
    pool.join()


def format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4g}"


def should_use_live_display(console_obj: Any) -> bool:
    return bool(
        getattr(console_obj, "is_terminal", False)
        and not getattr(console_obj, "is_dumb_terminal", False)
    )


def build_stage_counts(rows: list[dict[str, Any]], stage_prefix: str) -> dict[str, int]:
    status_key = f"{stage_prefix}_fit_status"
    counts = {"ok": 0, "non_finite": 0, "exception": 0}
    for row in rows:
        status = row.get(status_key)
        if status == "ok":
            counts["ok"] += 1
        elif status == "non_finite_metrics":
            counts["non_finite"] += 1
        else:
            counts["exception"] += 1
    return counts


def trim_recent_rows(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return list(rows[-limit:])


def build_recent_results_table(
    rows: list[dict[str, Any]],
    stage_prefix: str,
    *,
    limit: int = 8,
) -> Table:
    table = Table(title="Recent results")
    table.add_column("Stage")
    table.add_column("Candidate")
    table.add_column("Status")
    table.add_column("MAPE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("R^2", justify="right")
    for row in trim_recent_rows(rows, limit=limit):
        table.add_row(
            stage_prefix,
            str(row.get("candidate_label", "")),
            str(row.get(f"{stage_prefix}_fit_status", "")),
            format_metric(row.get(f"{stage_prefix}_mape")),
            format_metric(row.get(f"{stage_prefix}_rmse")),
            format_metric(row.get(f"{stage_prefix}_r2")),
        )
    return table


def build_best_validation_panel(best_validation_row: dict[str, Any] | None) -> Panel:
    if best_validation_row is None:
        body = "No finite validation result yet."
    else:
        body = (
            f"{best_validation_row['candidate_label']}\n"
            f"MAPE={format_metric(best_validation_row['validation_mape'])}  "
            f"RMSE={format_metric(best_validation_row['validation_rmse'])}  "
            f"R^2={format_metric(best_validation_row['validation_r2'])}"
        )
    return Panel(body, title="Best validation so far")


def build_metadata_panel(
    *,
    station: str,
    stage_prefix: str,
    completed: int,
    total: int,
    n_jobs: int,
    output_dir: Path,
    counts: dict[str, int],
) -> Panel:
    body = (
        f"station={station}\n"
        f"stage={stage_prefix}\n"
        f"progress={completed}/{total}\n"
        f"workers={n_jobs}\n"
        f"ok={counts['ok']} non_finite={counts['non_finite']} exception={counts['exception']}\n"
        f"output_dir={output_dir}"
    )
    return Panel(body, title="Run status")


def build_live_renderable(
    *,
    metadata_panel: Panel,
    progress_renderable: Any,
    best_validation_panel: Panel,
    recent_results_table: Table,
) -> Group:
    return Group(
        metadata_panel,
        progress_renderable,
        best_validation_panel,
        recent_results_table,
    )


def split_start_date(split_boundary: str) -> str:
    return pd.Timestamp(split_boundary).date().isoformat()


def inclusive_train_end(split_boundary: str) -> str:
    return (pd.Timestamp(split_boundary) - pd.Timedelta(days=1)).date().isoformat()


def best_row(rows: list[dict[str, Any]], sort_key_func) -> dict[str, Any] | None:
    valid = [row for row in rows if sort_key_func(row)[0] != np.inf]
    if not valid:
        return None
    return min(valid, key=sort_key_func)


def is_high_long_period(row: dict[str, Any]) -> bool:
    return int(row["mf_mm_order"]) >= 8 or int(row["annual_order"]) >= 16


def build_summary_text(
    validation_rows: list[dict[str, Any]],
    *,
    n_jobs: int,
) -> str:
    best_validation = best_row(validation_rows, validation_sort_key)
    anchor_row = next((row for row in validation_rows if bool(row.get("is_anchor"))), None)
    high_rows = [row for row in validation_rows if is_high_long_period(row)]
    best_high = best_row(high_rows, validation_sort_key)

    lines = [
        "# Battery anchored tidal search",
        "",
        "## Setup",
        f"- Station: `{DEFAULT_STATION_NAME}`",
        f"- Search split: `{SELECTION_TRAIN_START}` to `{inclusive_train_end(SELECTION_TRAIN_END)}` train, "
        f"`{split_start_date(SELECTION_TRAIN_END)}` to `{SELECTION_TEST_END}` held-out",
        f"- Workers: `{n_jobs}`",
        f"- Candidates: `{len(validation_rows)}`",
        f"- Anchor candidate: `{build_candidate_label(build_anchor_candidate())}`",
        "",
        "## Anchor",
    ]
    if anchor_row is None or not has_finite_metric_row(anchor_row, "validation"):
        lines.append("- Anchor candidate did not produce finite held-out metrics.")
    else:
        lines.extend(
            [
                f"- Held-out MAPE/RMSE/R^2: `{format_metric(anchor_row['validation_mape'])}` / "
                f"`{format_metric(anchor_row['validation_rmse'])}` / "
                f"`{format_metric(anchor_row['validation_r2'])}`",
                f"- Train MAPE/RMSE/R^2: `{format_metric(anchor_row['selection_train_mape'])}` / "
                f"`{format_metric(anchor_row['selection_train_rmse'])}` / "
                f"`{format_metric(anchor_row['selection_train_r2'])}`",
            ]
        )

    lines.extend(["", "## Held-out results"])
    if best_validation is None:
        lines.append("- No finite held-out results.")
        return "\n".join(lines) + "\n"

    under_20 = float(best_validation["validation_mape"]) < 20.0
    lines.extend(
        [
            f"- Best held-out candidate: `{best_validation['candidate_label']}`",
            f"- Held-out MAPE/RMSE/R^2: `{format_metric(best_validation['validation_mape'])}` / "
            f"`{format_metric(best_validation['validation_rmse'])}` / "
            f"`{format_metric(best_validation['validation_r2'])}` "
            f"({'under 20%' if under_20 else 'did not reach 20%'})",
        ]
    )
    if anchor_row is not None and has_finite_metric_row(anchor_row, "validation"):
        lines.append(
            f"- Delta vs anchor (MAPE/RMSE/R^2): "
            f"`{format_metric(metric_gap(best_validation['validation_mape'], anchor_row['validation_mape']))}` / "
            f"`{format_metric(metric_gap(best_validation['validation_rmse'], anchor_row['validation_rmse']))}` / "
            f"`{format_metric(metric_gap(best_validation['validation_r2'], anchor_row['validation_r2']))}`"
        )
    lines.append(
        (
            f"- Best high-long-period candidate: `{best_high['candidate_label']}` "
            f"with held-out MAPE/RMSE/R^2 = `{format_metric(best_high['validation_mape'])}` / "
            f"`{format_metric(best_high['validation_rmse'])}` / `{format_metric(best_high['validation_r2'])}`"
        )
        if best_high is not None
        else "- No finite high-long-period candidate."
    )
    return "\n".join(lines) + "\n"


def run_candidates(
    candidates: list[SearchCandidate],
    *,
    df: pd.DataFrame,
    sph: int,
    train_start: str,
    train_end: str,
    test_end: str,
    stage_prefix: str,
    n_jobs: int,
    log,
    leaderboard_path: Path | None = None,
    validation_rows: list[dict[str, Any]] | None = None,
    existing_rows: list[dict[str, Any]] | None = None,
    reporter: CandidateReporter | None = None,
) -> list[dict[str, Any]]:
    """Run a stage and record each completed result."""
    rows: list[dict[str, Any]] = list(existing_rows or [])
    initial_completed = len(rows)
    pending_total = len(candidates)
    total = initial_completed + pending_total
    workers = min(max(1, n_jobs), pending_total) if pending_total else 0
    log(
        f"{stage_prefix}: {pending_total} pending candidates across {workers} worker(s)"
        + (f"; {initial_completed} already completed" if initial_completed else "")
    )

    if total == 0:
        return rows

    def evaluate(candidate: SearchCandidate) -> dict[str, Any]:
        return evaluate_candidate(
            candidate,
            df=df,
            sph=sph,
            train_start=train_start,
            train_end=train_end,
            test_end=test_end,
            stage_prefix=stage_prefix,
        )

    best_so_far = best_row(rows, validation_sort_key) if stage_prefix == "validation" else None

    def record_completed_row(row: dict[str, Any], *, completed: int) -> None:
        nonlocal best_so_far
        rows.append(row)
        best_so_far = log_completed_row(
            row,
            completed=completed,
            total=total,
            stage_prefix=stage_prefix,
            best_so_far=best_so_far,
            log=log,
        )
        if leaderboard_path is not None:
            stage_validation_rows = validation_rows if validation_rows is not None else rows
            write_leaderboard(
                merge_rows(stage_validation_rows, rows if stage_prefix == "test" else []),
                leaderboard_path,
            )
        if reporter is not None:
            reporter(
                completed=completed,
                total=total,
                stage_prefix=stage_prefix,
                row=row,
                best_so_far=best_so_far,
            )

    if pending_total == 0:
        return rows

    if workers <= 1:
        for completed, candidate in enumerate(candidates, start=initial_completed + 1):
            row = evaluate(candidate)
            record_completed_row(row, completed=completed)
        return rows

    pool = make_candidate_pool(
        workers=workers,
        df=df,
        sph=sph,
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
        stage_prefix=stage_prefix,
    )
    try:
        for completed, row in enumerate(
            pool.imap_unordered(run_candidate_job, candidates),
            start=initial_completed + 1,
        ):
            record_completed_row(row, completed=completed)
    except KeyboardInterrupt:
        handle_interrupt(pool)
        raise
    except Exception:
        handle_interrupt(pool)
        raise
    else:
        pool.close()
        pool.join()
    return rows


def log_completed_row(
    row: dict[str, Any],
    *,
    completed: int,
    total: int,
    stage_prefix: str,
    best_so_far: dict[str, Any] | None,
    log,
) -> dict[str, Any] | None:
    metric_prefix = "validation" if stage_prefix == "validation" else "test"
    log(
        f"[{stage_prefix} {completed}/{total}] {row['candidate_label']} "
        f"{metric_prefix}_mape={format_metric(row.get(f'{metric_prefix}_mape'))} "
        f"{metric_prefix}_rmse={format_metric(row.get(f'{metric_prefix}_rmse'))} "
        f"{metric_prefix}_r2={format_metric(row.get(f'{metric_prefix}_r2'))}"
    )
    if stage_prefix != "validation":
        return best_so_far
    if best_so_far is None or validation_sort_key(row) < validation_sort_key(best_so_far):
        log(
            "Best validation so far: "
            f"{row['candidate_label']} "
            f"(MAPE={format_metric(row['validation_mape'])}, "
            f"RMSE={format_metric(row['validation_rmse'])}, "
            f"R^2={format_metric(row['validation_r2'])})"
        )
        return row
    return best_so_far


@click.command()
@click.option(
    "--station",
    type=str,
    default=DEFAULT_STATION_NAME,
    show_default=True,
    help="Station name to load via the compact tidal helpers.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for CSV, markdown summary, and log files.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=6,
    show_default=True,
    help="Parallel worker count for the anchored local search.",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Ignore any partial CSV and start the grid from scratch.",
)
@click.option(
    "--no-download",
    is_flag=True,
    default=False,
    help="Use existing cached data only.",
)
def main(
    station: str,
    output_dir: Path | None,
    n_jobs: int,
    no_resume: bool,
    no_download: bool,
) -> None:
    """Run the Battery anchored long-period grid search."""
    output_dir = output_dir or (default_output_dir() / "tidal_battery_grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tidal_battery_grid.log"
    partial_path = output_dir / "tidal_battery_grid_partial.csv"
    results_path = output_dir / "tidal_battery_grid_results.csv"
    summary_path = output_dir / "tidal_battery_grid_summary.md"
    plain_log = make_plain_logger(log_path)
    file_log = make_file_logger(log_path)
    live_enabled = should_use_live_display(console)

    section("Battery anchored tidal search")
    try:
        plain_log(f"Station: {station}")
        plain_log(f"Output dir: {output_dir}")
        plain_log(f"n_jobs: {n_jobs}")
        plain_log(
            f"Search split: {SELECTION_TRAIN_START} to {inclusive_train_end(SELECTION_TRAIN_END)} train, "
            f"{split_start_date(SELECTION_TRAIN_END)} to {SELECTION_TEST_END} held-out"
        )

        station_data = tidal_compact.load_station_frame(
            station,
            use_weather=True,
            download_tidal=not no_download,
            download_weather=not no_download,
            data_start=DEFAULT_DATA_START,
            data_end=DEFAULT_DATA_END,
        )
        df = station_data["df"]
        sph = station_data["sph"]
        plain_log(station_data["status_message"])

        all_candidates = build_search_candidates()
        plain_log(f"Built {len(all_candidates)} anchored-search candidates")
        loaded_validation_rows = [] if no_resume else load_existing_rows(partial_path)
        existing_validation_rows, pending_candidates = resume_candidates(
            all_candidates,
            loaded_validation_rows,
        )
        skipped_partial_rows = len(loaded_validation_rows) - len(existing_validation_rows)
        if no_resume:
            plain_log("Resume disabled; ignoring any existing partial CSV.")
        elif loaded_validation_rows:
            plain_log(
                f"Resuming from {len(existing_validation_rows)} completed candidates in {partial_path}"
            )
            if skipped_partial_rows:
                plain_log(
                    f"Ignoring {skipped_partial_rows} saved rows that are outside the current pruned grid."
                )
        plain_log(
            f"{len(pending_candidates)} pending candidates remain after resume filtering."
        )

        validation_rows: list[dict[str, Any]]
        if live_enabled:
            progress = make_stage_progress()
            stage_task = progress.add_task(
                "validation",
                total=max(len(all_candidates), 1),
                completed=len(existing_validation_rows),
            )
            stage_rows: dict[str, list[dict[str, Any]]] = {"validation": list(existing_validation_rows)}
            best_validation_row = best_row(existing_validation_rows, validation_sort_key)

            def update_live(stage_prefix: str, *, completed: int, total: int) -> None:
                live.update(
                    build_live_renderable(
                        metadata_panel=build_metadata_panel(
                            station=station,
                            stage_prefix=stage_prefix,
                            completed=completed,
                            total=total,
                            n_jobs=n_jobs,
                            output_dir=output_dir,
                            counts=build_stage_counts(stage_rows[stage_prefix], stage_prefix),
                        ),
                        progress_renderable=progress,
                        best_validation_panel=build_best_validation_panel(best_validation_row),
                        recent_results_table=build_recent_results_table(stage_rows[stage_prefix], stage_prefix),
                    )
                )

            def reporter(
                *,
                completed: int,
                total: int,
                stage_prefix: str,
                row: dict[str, Any],
                best_so_far: dict[str, Any] | None,
            ) -> None:
                nonlocal best_validation_row
                progress.update(
                    stage_task,
                    description=stage_prefix,
                    total=max(total, 1),
                    completed=completed,
                )
                stage_rows[stage_prefix].append(row)
                if stage_prefix == "validation":
                    best_validation_row = best_so_far
                update_live(stage_prefix, completed=completed, total=total)

            with Live(console=console, refresh_per_second=8, transient=False) as live:
                progress.update(
                    stage_task,
                    description="validation",
                    total=max(len(all_candidates), 1),
                    completed=len(existing_validation_rows),
                )
                update_live(
                    "validation",
                    completed=len(existing_validation_rows),
                    total=len(all_candidates),
                )
                validation_rows = run_candidates(
                    pending_candidates,
                    df=df,
                    sph=sph,
                    train_start=SELECTION_TRAIN_START,
                    train_end=SELECTION_TRAIN_END,
                    test_end=SELECTION_TEST_END,
                    stage_prefix="validation",
                    n_jobs=n_jobs,
                    log=file_log,
                    leaderboard_path=partial_path,
                    existing_rows=existing_validation_rows,
                    reporter=reporter,
                )
        else:
            validation_rows = run_candidates(
                pending_candidates,
                df=df,
                sph=sph,
                train_start=SELECTION_TRAIN_START,
                train_end=SELECTION_TRAIN_END,
                test_end=SELECTION_TEST_END,
                stage_prefix="validation",
                n_jobs=n_jobs,
                log=plain_log,
                leaderboard_path=partial_path,
                existing_rows=existing_validation_rows,
            )
    except KeyboardInterrupt:
        plain_log("Interrupted by user; preserved any partial CSV and log output on disk.")
        raise SystemExit(130)

    leaderboard = merge_rows(validation_rows, [])
    write_leaderboard(leaderboard, results_path)
    write_leaderboard(leaderboard, partial_path)

    summary_text = build_summary_text(validation_rows, n_jobs=n_jobs)
    summary_path.write_text(summary_text, encoding="utf-8")

    best_validation = best_row(validation_rows, validation_sort_key)
    if best_validation is None:
        error("No finite held-out results were produced.")
    else:
        plain_log(
            "Best held-out candidate: "
            f"{best_validation['candidate_label']} "
            f"(MAPE={format_metric(best_validation['validation_mape'])}, "
            f"RMSE={format_metric(best_validation['validation_rmse'])}, "
            f"R^2={format_metric(best_validation['validation_r2'])})"
        )
    success(f"Leaderboard CSV: {results_path}")
    success(f"Summary markdown: {summary_path}")
    success(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
