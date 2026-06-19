# Battery Grid Rich Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `examples/run_tidal_battery_grid.py` to show a Rich live progress UI during interactive runs while preserving file logging, partial CSV updates, and plain-text fallback behavior.

**Architecture:** Keep all search logic in `examples/run_tidal_battery_grid.py`, but add a thin presentation layer of pure helper functions for UI-mode detection, stage counts, recent-results trimming, and Rich renderable construction. Replace the current thread-based parallel loop with an interruptible worker backend so `Ctrl-C` can stop the run promptly, then thread UI updates through a reporter callback on the main thread as results arrive.

**Tech Stack:** Python 3.13, click, rich, pandas, numpy, pytest

---

## File Map

- Modify: `examples/run_tidal_battery_grid.py`
  - Add Rich imports, live-mode helpers, render-state helpers, renderable builders, and reporter-driven live updates around an interruptible run loop
  - Replace the current `ThreadPoolExecutor` flow with a killable worker backend plus explicit `KeyboardInterrupt` handling
  - Split file-only logging from terminal printing so live mode can update widgets without spamming plain log lines
- Modify: `test/test_tidal_battery_grid.py`
  - Add helper-level tests for live-mode detection, stage counts, recent-results trimming, Rich table/panel builders, and the reporter callback seam in `run_candidates(...)`

### Task 1: Add Pure UI State Helpers

**Files:**
- Modify: `examples/run_tidal_battery_grid.py`
- Modify: `test/test_tidal_battery_grid.py`
- Test: `test/test_tidal_battery_grid.py`

- [ ] **Step 1: Write the failing helper tests**

```python
from types import SimpleNamespace


def test_should_use_live_display_requires_terminal_and_not_dumb():
    assert battery_grid.should_use_live_display(
        SimpleNamespace(is_terminal=True, is_dumb_terminal=False)
    ) is True
    assert battery_grid.should_use_live_display(
        SimpleNamespace(is_terminal=False, is_dumb_terminal=False)
    ) is False
    assert battery_grid.should_use_live_display(
        SimpleNamespace(is_terminal=True, is_dumb_terminal=True)
    ) is False


def test_build_stage_counts_counts_ok_non_finite_and_exception():
    rows = [
        {"validation_fit_status": "ok"},
        {"validation_fit_status": "ok"},
        {"validation_fit_status": "non_finite_metrics"},
        {"validation_fit_status": "exception"},
    ]

    assert battery_grid.build_stage_counts(rows, "validation") == {
        "ok": 2,
        "non_finite": 1,
        "exception": 1,
    }


def test_trim_recent_rows_keeps_latest_rows():
    rows = [{"candidate_label": f"row-{idx}"} for idx in range(6)]

    trimmed = battery_grid.trim_recent_rows(rows, limit=3)

    assert [row["candidate_label"] for row in trimmed] == ["row-3", "row-4", "row-5"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest test/test_tidal_battery_grid.py -k "should_use_live_display or build_stage_counts or trim_recent_rows" -v`

Expected: FAIL with missing helpers such as `AttributeError: module 'run_tidal_battery_grid' has no attribute 'should_use_live_display'`, `build_stage_counts`, or `trim_recent_rows`.

- [ ] **Step 3: Implement the pure UI helpers**

```python
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
    return list(rows[-limit:])
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run pytest test/test_tidal_battery_grid.py -k "should_use_live_display or build_stage_counts or trim_recent_rows" -v`

Expected: PASS for all three new helper tests.

- [ ] **Step 5: Commit the helper layer**

```bash
git add examples/run_tidal_battery_grid.py test/test_tidal_battery_grid.py
git commit -m "feat: add battery grid ui state helpers"
```

### Task 2: Add Rich Renderable Builders

**Files:**
- Modify: `examples/run_tidal_battery_grid.py`
- Modify: `test/test_tidal_battery_grid.py`
- Test: `test/test_tidal_battery_grid.py`

- [ ] **Step 1: Write the failing Rich render-helper tests**

```python
from rich.panel import Panel
from rich.table import Table


def test_build_recent_results_table_uses_latest_rows():
    rows = [
        {
            "candidate_label": "row-0",
            "validation_fit_status": "ok",
            "validation_mape": 12.0,
            "validation_rmse": 0.12,
            "validation_r2": 0.81,
        },
        {
            "candidate_label": "row-1",
            "validation_fit_status": "exception",
            "validation_mape": float("nan"),
            "validation_rmse": float("nan"),
            "validation_r2": float("nan"),
        },
    ]

    table = battery_grid.build_recent_results_table(rows, "validation", limit=2)

    assert isinstance(table, Table)
    assert table.title == "Recent results"
    assert table.row_count == 2
    assert [column.header for column in table.columns] == [
        "Stage",
        "Candidate",
        "Status",
        "MAPE",
        "RMSE",
        "R^2",
    ]


def test_build_best_validation_panel_handles_missing_and_present_best_row():
    empty_panel = battery_grid.build_best_validation_panel(None)
    full_panel = battery_grid.build_best_validation_panel(
        {
            "candidate_label": "regs=pressure+wind_u knots=med Mf/Mm=8 annual=16 reg=0.003",
            "validation_mape": 9.5,
            "validation_rmse": 0.11,
            "validation_r2": 0.87,
        }
    )

    assert isinstance(empty_panel, Panel)
    assert isinstance(full_panel, Panel)
    assert empty_panel.title == "Best validation so far"
    assert full_panel.title == "Best validation so far"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest test/test_tidal_battery_grid.py -k "recent_results_table or best_validation_panel" -v`

Expected: FAIL with missing helpers such as `AttributeError: module 'run_tidal_battery_grid' has no attribute 'build_recent_results_table'` or `build_best_validation_panel`.

- [ ] **Step 3: Implement the Rich renderable builders**

```python
from rich.console import Group
from rich.panel import Panel
from rich.table import Table


def build_recent_results_table(
    rows: list[dict[str, Any]],
    stage_prefix: str,
    *,
    limit: int = 8,
) -> Table:
    metric_prefix = "validation" if stage_prefix == "validation" else "test"
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
            row["candidate_label"],
            str(row.get(f"{stage_prefix}_fit_status", "")),
            format_metric(row.get(f"{metric_prefix}_mape")),
            format_metric(row.get(f"{metric_prefix}_rmse")),
            format_metric(row.get(f"{metric_prefix}_r2")),
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
    return Group(metadata_panel, progress_renderable, best_validation_panel, recent_results_table)
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run pytest test/test_tidal_battery_grid.py -k "recent_results_table or best_validation_panel" -v`

Expected: PASS for the new Rich render-helper tests.

- [ ] **Step 5: Commit the renderable builders**

```bash
git add examples/run_tidal_battery_grid.py test/test_tidal_battery_grid.py
git commit -m "feat: add battery grid rich render helpers"
```

### Task 3: Adopt Interruptible Workers And Wire Live Progress

**Files:**
- Modify: `examples/run_tidal_battery_grid.py`
- Modify: `test/test_tidal_battery_grid.py`
- Test: `test/test_tidal_battery_grid.py`

- [ ] **Step 1: Write the failing reporter-callback and interrupt regressions**

```python
def test_run_candidates_notifies_reporter_for_each_completed_row(monkeypatch):
    candidate = battery_grid.SearchCandidate(
        regressors=("pressure", "wind_u"),
        knot_preset="med",
        mf_mm_order=1,
        annual_order=2,
        fourier_reg_weight=3.0e-4,
    )
    reported: list[tuple[int, int, str]] = []

    def fake_evaluate_candidate(*args, **kwargs):
        return {
            "candidate_label": "demo",
            "validation_fit_status": "ok",
            "validation_mape": 8.0,
            "validation_rmse": 0.2,
            "validation_r2": 0.8,
        }

    monkeypatch.setattr(battery_grid, "evaluate_candidate", fake_evaluate_candidate)

    battery_grid.run_candidates(
        [candidate, candidate],
        df=pd.DataFrame({"water_level": [0.0]}, index=pd.date_range("2022-01-01", periods=1, freq="1h")),
        sph=1,
        train_start="2022-01-01",
        train_end="2023-01-01",
        test_end="2023-12-31",
        stage_prefix="validation",
        n_jobs=1,
        log=lambda _message: None,
        reporter=lambda *, completed, total, stage_prefix, row, best_so_far: reported.append(
            (completed, total, stage_prefix)
        ),
    )

    assert reported == [(1, 2, "validation"), (2, 2, "validation")]


def test_handle_interrupt_cancels_workers_and_preserves_interrupt_exit():
    shutdown_events: list[str] = []

    class FakePool:
        def terminate(self):
            shutdown_events.append("terminate")

        def join(self):
            shutdown_events.append("join")

    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            battery_grid.handle_interrupt(FakePool())
            raise

    assert shutdown_events == ["terminate", "join"]
```

- [ ] **Step 2: Run the targeted regression to verify it fails**

Run: `uv run pytest test/test_tidal_battery_grid.py -k "notifies_reporter_for_each_completed_row or handle_interrupt" -v`

Expected: FAIL with `TypeError: run_candidates() got an unexpected keyword argument 'reporter'`.

- [ ] **Step 3: Add file-only logging plus a live reporter callback**

```python
from common_cli import console, default_output_dir, error, info, section, success
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
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


def run_candidates(..., reporter=None) -> list[dict[str, Any]]:
    ...
    if workers <= 1:
        for completed, candidate in enumerate(candidates, start=1):
            row = evaluate(candidate)
            rows.append(row)
            best_so_far = log_completed_row(...)
            if reporter is not None:
                reporter(
                    completed=completed,
                    total=total,
                    stage_prefix=stage_prefix,
                    row=row,
                    best_so_far=best_so_far,
                )
            ...
    else:
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
            for completed, row in enumerate(pool.imap_unordered(run_candidate_job, candidates), start=1):
                rows.append(row)
                best_so_far = log_completed_row(...)
                if reporter is not None:
                    reporter(
                        completed=completed,
                        total=total,
                        stage_prefix=stage_prefix,
                        row=row,
                        best_so_far=best_so_far,
                    )
        except KeyboardInterrupt:
            handle_interrupt(pool)
            raise
        else:
            pool.close()
            pool.join()


def main(...):
    ...
    log = make_plain_logger(log_path)
    live_enabled = should_use_live_display(console)
    if live_enabled:
        file_log = make_file_logger(log_path)
        progress = make_stage_progress()
        stage_task = progress.add_task("validation", total=1)
        stage_rows: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
        best_validation_row: dict[str, Any] | None = None

        def reporter(*, completed: int, total: int, stage_prefix: str, row: dict[str, Any], best_so_far: dict[str, Any] | None):
            nonlocal best_validation_row
            progress.update(stage_task, description=stage_prefix, total=total, completed=completed)
            stage_rows[stage_prefix].append(row)
            recent_rows = trim_recent_rows(stage_rows[stage_prefix])
            if stage_prefix == "validation":
                best_validation_row = best_so_far
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
                    recent_results_table=build_recent_results_table(recent_rows, stage_prefix),
                )
            )
            file_log(
                f"[{stage_prefix} {completed}/{total}] {row['candidate_label']} "
                f"{stage_prefix}_mape={format_metric(row.get(f'{stage_prefix}_mape'))}"
            )

        with Live(console=console, refresh_per_second=8, transient=False) as live:
            ...
            validation_rows = run_candidates(..., log=lambda _message: None, reporter=reporter)
            ...
            test_rows = run_candidates(..., log=lambda _message: None, reporter=reporter)
    else:
        validation_rows = run_candidates(..., log=log)
        ...
        test_rows = run_candidates(..., log=log, validation_rows=validation_rows)
```

- [ ] **Step 4: Run the targeted reporter regression and the full script tests**

Run: `uv run pytest test/test_tidal_battery_grid.py -k "notifies_reporter_for_each_completed_row or handle_interrupt" -v && uv run pytest test/test_tidal_battery_grid.py -q && uv run pytest test/test_example_tidal_compact.py test/test_tidal_battery_grid.py -q`

Expected:
- PASS for the reporter regression;
- PASS for the focused battery-grid test file;
- PASS for the combined compact + battery-grid suite.

- [ ] **Step 5: Verify the CLI still boots and commit the integration**

Run: `uv run python examples/run_tidal_battery_grid.py --help`

Expected: exit `0` with the existing CLI options still listed.

```bash
git add examples/run_tidal_battery_grid.py test/test_tidal_battery_grid.py
git commit -m "feat: add rich live progress to battery grid search"
```

## Self-Review Checklist

- Spec coverage:
  - UI mode detection -> Task 1 + Task 3
  - live Rich layout -> Task 2 + Task 3
  - file logging + partial CSV preservation -> Task 3
  - main-thread-only live updates around the current executor loop -> Task 3
  - plain-text fallback -> Task 3
- Placeholder scan:
  - No `TODO`/`TBD` placeholders remain
  - Every test step includes concrete code and commands
- Type/shape consistency:
  - `stage_prefix` stays `validation` / `test`
  - `non_finite_metrics` from production rows maps to `non_finite` in the UI counters
  - `best_so_far` remains validation-only and is threaded through the reporter callback consistently
