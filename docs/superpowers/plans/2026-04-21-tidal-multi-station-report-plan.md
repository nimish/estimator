# Tidal Multi-Station Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone script that runs one fixed Battery-style tidal model across the notebook's `STATION_CATALOG` stations, skips incomplete stations, and writes CSV, Markdown, and figure outputs for a multi-station comparison report.

**Architecture:** Add a new script, `examples/run_tidal_multi_station_report.py`, that owns station enumeration, fixed-fit configuration, exclusion handling, plotting, and Markdown generation. Keep the single-station Battery search script unchanged; reuse existing station-loading and plotting/report utilities where they fit, and cover the new durable logic with focused tests in a new test file.

**Tech Stack:** Python, Click, pandas, numpy, matplotlib, Rich, existing example helpers from `example_tidal`, `example_tidal_compact`, and `common_cli`, pytest.

---

## File Map

- Create: `examples/run_tidal_multi_station_report.py`
  - New CLI entry point for the fixed-fit multi-station report.
  - Owns station list construction, per-station execution, exclusion handling, CSV writing, chart generation, and Markdown assembly.
- Create: `test/test_tidal_multi_station_report.py`
  - Focused tests for station enumeration, fixed configuration helpers, inclusion/exclusion row shaping, figure path references, and Markdown sections.
- Reuse without modification:
  - `examples/common_cli.py`
  - `examples/example_tidal.py`
  - `examples/example_tidal_compact.py`

## Task 1: Scaffold Fixed-Configuration And Station Helpers

**Files:**
- Create: `examples/run_tidal_multi_station_report.py`
- Create: `test/test_tidal_multi_station_report.py`
- Test: `test/test_tidal_multi_station_report.py`

- [ ] **Step 1: Write the failing helper tests**

```python
from pathlib import Path
import importlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

multi_station = importlib.import_module("run_tidal_multi_station_report")


def test_build_station_rows_uses_station_catalog_names_and_ids():
    rows = multi_station.build_station_rows()

    assert rows
    assert rows[0]["station_id"]
    assert rows[0]["station_name"]
    assert all("station_id" in row for row in rows)
    assert all("station_name" in row for row in rows)


def test_build_fixed_model_config_matches_battery_style_defaults():
    cfg = multi_station.build_fixed_model_config()

    assert cfg["harmonic_orders"] == {
        "M2": 4,
        "S2": 1,
        "N2": 1,
        "K1": 2,
        "O1": 1,
        "Mf": 0,
        "Mm": 0,
        "annual": 0,
    }
    assert cfg["regressors"] == ("pressure", "dp_dt", "wind_u")
    assert cfg["lag_ranges"] == {
        "pressure": (-2, 0),
        "dp_dt": (-2, 0),
        "wind_u": (-1, 0),
    }
    assert cfg["knot_presets"] == {
        "pressure": "med",
        "dp_dt": "med",
        "wind_u": "med",
    }
    assert cfg["fourier_reg_weight"] == pytest.approx(1.0e-4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "build_station_rows or fixed_model_config" -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'run_tidal_multi_station_report'`

- [ ] **Step 3: Write the minimal helper implementation**

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXAMPLES_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EXAMPLES_DIR))

from example_tidal import STATION_CATALOG

DEFAULT_TRAIN_START = "2022-01-01"
DEFAULT_TRAIN_END = "2024-01-01"
DEFAULT_TEST_END = "2025-01-01"


def build_station_rows() -> list[dict[str, str]]:
    return [
        {
            "station_id": station_id,
            "station_name": str(meta["name"]),
            "tidal_regime": str(meta.get("tidal_regime", "")),
            "region": str(meta.get("region", "")),
        }
        for station_id, meta in STATION_CATALOG.items()
    ]


def build_fixed_model_config() -> dict[str, object]:
    return {
        "harmonic_orders": {
            "M2": 4,
            "S2": 1,
            "N2": 1,
            "K1": 2,
            "O1": 1,
            "Mf": 0,
            "Mm": 0,
            "annual": 0,
        },
        "regressors": ("pressure", "dp_dt", "wind_u"),
        "lag_ranges": {
            "pressure": (-2, 0),
            "dp_dt": (-2, 0),
            "wind_u": (-1, 0),
        },
        "knot_presets": {
            "pressure": "med",
            "dp_dt": "med",
            "wind_u": "med",
        },
        "fourier_reg_weight": 1.0e-4,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "build_station_rows or fixed_model_config" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_tidal_multi_station_report.py examples/run_tidal_multi_station_report.py
git commit -m "feat: scaffold tidal multi-station report helpers"
```

## Task 2: Add Inclusion, Exclusion, And Result Row Shaping

**Files:**
- Modify: `examples/run_tidal_multi_station_report.py`
- Modify: `test/test_tidal_multi_station_report.py`
- Test: `test/test_tidal_multi_station_report.py`

- [ ] **Step 1: Write the failing row-shaping tests**

```python
def test_make_exclusion_row_records_station_and_reason():
    row = multi_station.make_exclusion_row(
        station_id="8518750",
        station_name="The Battery, NY",
        category="coverage",
        reason="missing wind_u over held-out window",
    )

    assert row == {
        "station_id": "8518750",
        "station_name": "The Battery, NY",
        "category": "coverage",
        "reason": "missing wind_u over held-out window",
    }


def test_make_result_row_includes_all_train_and_validation_metrics():
    row = multi_station.make_result_row(
        station_row={
            "station_id": "8518750",
            "station_name": "The Battery, NY",
            "tidal_regime": "Semi-diurnal",
            "region": "NY Harbor",
        },
        fit_result={
            "metrics_train": {"rmse": 0.1, "mae": 0.08, "mape": 9.0, "r2": 0.95},
            "metrics_test": {"rmse": 0.2, "mae": 0.16, "mape": 15.0, "r2": 0.85},
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 1000,
            "n_test": 500,
        },
    )

    assert row["station_id"] == "8518750"
    assert row["station_name"] == "The Battery, NY"
    assert row["train_rmse"] == pytest.approx(0.1)
    assert row["train_mae"] == pytest.approx(0.08)
    assert row["train_mape"] == pytest.approx(9.0)
    assert row["train_r2"] == pytest.approx(0.95)
    assert row["validation_rmse"] == pytest.approx(0.2)
    assert row["validation_mae"] == pytest.approx(0.16)
    assert row["validation_mape"] == pytest.approx(15.0)
    assert row["validation_r2"] == pytest.approx(0.85)
    assert row["active_regs"] == "pressure,dp_dt,wind_u"
    assert row["validation_minus_train_mape"] == pytest.approx(6.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "make_exclusion_row or make_result_row" -v`

Expected: FAIL with `AttributeError` for missing helper functions

- [ ] **Step 3: Write the minimal row-shaping implementation**

```python
def metric_gap(left: float | int | None, right: float | int | None) -> float:
    left_value = float(left) if left is not None else float("nan")
    right_value = float(right) if right is not None else float("nan")
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float("nan")
    return left_value - right_value


def make_exclusion_row(
    *,
    station_id: str,
    station_name: str,
    category: str,
    reason: str,
) -> dict[str, str]:
    return {
        "station_id": station_id,
        "station_name": station_name,
        "category": category,
        "reason": reason,
    }


def make_result_row(
    *,
    station_row: dict[str, str],
    fit_result: dict[str, object],
) -> dict[str, object]:
    train_metrics = fit_result["metrics_train"]
    test_metrics = fit_result["metrics_test"]
    return {
        "station_id": station_row["station_id"],
        "station_name": station_row["station_name"],
        "tidal_regime": station_row["tidal_regime"],
        "region": station_row["region"],
        "active_regs": ",".join(fit_result.get("active_regs", [])),
        "n_train": int(fit_result.get("n_train", 0)),
        "n_validation": int(fit_result.get("n_test", 0)),
        "train_rmse": float(train_metrics["rmse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_mape": float(train_metrics["mape"]),
        "train_r2": float(train_metrics["r2"]),
        "validation_rmse": float(test_metrics["rmse"]),
        "validation_mae": float(test_metrics["mae"]),
        "validation_mape": float(test_metrics["mape"]),
        "validation_r2": float(test_metrics["r2"]),
        "validation_minus_train_rmse": metric_gap(test_metrics["rmse"], train_metrics["rmse"]),
        "validation_minus_train_mae": metric_gap(test_metrics["mae"], train_metrics["mae"]),
        "validation_minus_train_mape": metric_gap(test_metrics["mape"], train_metrics["mape"]),
        "validation_minus_train_r2": metric_gap(test_metrics["r2"], train_metrics["r2"]),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "make_exclusion_row or make_result_row" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_tidal_multi_station_report.py examples/run_tidal_multi_station_report.py
git commit -m "feat: add multi-station result and exclusion rows"
```

## Task 3: Add Markdown And Figure Reference Builders

**Files:**
- Modify: `examples/run_tidal_multi_station_report.py`
- Modify: `test/test_tidal_multi_station_report.py`
- Test: `test/test_tidal_multi_station_report.py`

- [ ] **Step 1: Write the failing Markdown helper tests**

```python
def test_markdown_summary_includes_all_metrics_tables_and_relative_figures():
    summary = multi_station.build_summary_markdown(
        included_rows=[
            {
                "station_id": "8518750",
                "station_name": "The Battery, NY",
                "tidal_regime": "Semi-diurnal",
                "train_rmse": 0.10,
                "train_mae": 0.08,
                "train_mape": 9.0,
                "train_r2": 0.95,
                "validation_rmse": 0.20,
                "validation_mae": 0.16,
                "validation_mape": 15.0,
                "validation_r2": 0.85,
                "n_train": 1000,
                "n_validation": 500,
                "active_regs": "pressure,dp_dt,wind_u",
            }
        ],
        excluded_rows=[
            {
                "station_id": "8720218",
                "station_name": "Galveston Pier 21, TX",
                "category": "coverage",
                "reason": "missing wind_u over held-out window",
            }
        ],
        metric_figure_paths=[
            Path("figures/multi_station/validation_mape.png"),
            Path("figures/multi_station/validation_rmse.png"),
        ],
        station_figure_paths=[
            Path("figures/multi_station/station_8518750.png"),
        ],
    )

    assert "train_rmse" in summary
    assert "validation_mape" in summary
    assert "The Battery, NY" in summary
    assert "Galveston Pier 21, TX" in summary
    assert "![validation_mape](figures/multi_station/validation_mape.png)" in summary
    assert "![station_8518750](figures/multi_station/station_8518750.png)" in summary
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "summary_markdown" -v`

Expected: FAIL with `AttributeError: module 'run_tidal_multi_station_report' has no attribute 'build_summary_markdown'`

- [ ] **Step 3: Write the minimal Markdown implementation**

```python
def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def build_summary_markdown(
    *,
    included_rows: list[dict[str, object]],
    excluded_rows: list[dict[str, object]],
    metric_figure_paths: list[Path],
    station_figure_paths: list[Path],
) -> str:
    included_columns = [
        "station_id",
        "station_name",
        "tidal_regime",
        "train_rmse",
        "train_mae",
        "train_mape",
        "train_r2",
        "validation_rmse",
        "validation_mae",
        "validation_mape",
        "validation_r2",
        "n_train",
        "n_validation",
        "active_regs",
    ]
    excluded_columns = ["station_id", "station_name", "category", "reason"]
    lines = [
        "# Tidal Multi-Station Report",
        "",
        "## Included stations",
        _markdown_table(included_rows, included_columns),
        "",
        "## Excluded stations",
        _markdown_table(excluded_rows, excluded_columns),
        "",
        "## Across-station charts",
    ]
    lines.extend(
        f"![{path.stem}]({path.as_posix()})"
        for path in metric_figure_paths
    )
    lines.extend(["", "## Per-station figures"])
    lines.extend(
        f"![{path.stem}]({path.as_posix()})"
        for path in station_figure_paths
    )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "summary_markdown" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_tidal_multi_station_report.py examples/run_tidal_multi_station_report.py
git commit -m "feat: add multi-station markdown summary helpers"
```

## Task 4: Implement Per-Station Fixed Fit And Figure Builders

**Files:**
- Modify: `examples/run_tidal_multi_station_report.py`
- Modify: `test/test_tidal_multi_station_report.py`
- Test: `test/test_tidal_multi_station_report.py`

- [ ] **Step 1: Write the failing fit-and-figure tests**

```python
def test_run_station_fit_returns_included_row_and_station_figure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        multi_station.tidal_compact,
        "load_station_frame",
        lambda *args, **kwargs: {
            "df": pd.DataFrame(
                {
                    "water_level": [1.0, 1.1, 1.2, 1.3],
                    "pressure": [1010.0, 1011.0, 1012.0, 1011.5],
                    "dp_dt": [0.1, 0.2, -0.1, 0.0],
                    "wind_u": [1.0, 2.0, 3.0, 4.0],
                },
                index=pd.date_range("2022-01-01", periods=4, freq="1D"),
            ),
            "sph": 1,
            "status_message": "loaded",
        },
    )
    monkeypatch.setattr(
        multi_station.tidal_compact,
        "run_tidal_model",
        lambda *args, **kwargs: {
            "metrics_train": {"rmse": 0.1, "mae": 0.08, "mape": 9.0, "r2": 0.95},
            "metrics_test": {"rmse": 0.2, "mae": 0.16, "mape": 15.0, "r2": 0.85},
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 3,
            "n_test": 1,
            "df_test": pd.DataFrame(
                {"water_level": [1.3]},
                index=pd.date_range("2024-01-01", periods=1, freq="1D"),
            ),
            "test_pred": np.array([1.25]),
        },
    )

    result = multi_station.run_station_fit(
        station_row={
            "station_id": "8518750",
            "station_name": "The Battery, NY",
            "tidal_regime": "Semi-diurnal",
            "region": "NY Harbor",
        },
        output_dir=tmp_path,
        no_download=True,
    )

    assert result["status"] == "included"
    assert result["row"]["station_name"] == "The Battery, NY"
    assert result["row"]["validation_mape"] == pytest.approx(15.0)
    assert result["station_figure_path"].exists()


def test_build_metric_figures_writes_expected_pngs(tmp_path):
    rows = [
        {
            "station_name": "The Battery, NY",
            "validation_mape": 15.0,
            "validation_rmse": 0.20,
            "validation_mae": 0.16,
            "validation_r2": 0.85,
            "train_mape": 9.0,
            "train_rmse": 0.10,
            "train_mae": 0.08,
            "train_r2": 0.95,
        }
    ]

    figure_paths = multi_station.build_metric_figures(rows, tmp_path / "figures" / "multi_station")

    assert figure_paths
    assert all(path.suffix == ".png" for path in figure_paths)
    assert all(path.exists() for path in figure_paths)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "run_station_fit or build_metric_figures" -v`

Expected: FAIL with missing fit/figure helper functions

- [ ] **Step 3: Write the minimal fit-and-figure implementation**

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_cli import savefig, set_journal_style
import example_tidal_compact as tidal_compact


def build_station_figure(
    *,
    station_name: str,
    df_test: pd.DataFrame,
    test_pred: np.ndarray,
    metrics_row: dict[str, object],
    figure_dir: Path,
) -> Path:
    set_journal_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df_test.index, df_test["water_level"], label="Observed")
    ax.plot(df_test.index, test_pred, label="Predicted")
    ax.set_title(
        f"{station_name} | held-out MAPE={metrics_row['validation_mape']:.2f} "
        f"RMSE={metrics_row['validation_rmse']:.3f} R²={metrics_row['validation_r2']:.3f}"
    )
    ax.legend()
    saved = savefig(fig, figure_dir / f"station_{metrics_row['station_id']}")
    return next(path for path in saved if path.suffix == ".png")


def build_metric_figures(rows: list[dict[str, object]], figure_dir: Path) -> list[Path]:
    set_journal_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("validation_mape")
    figure_paths: list[Path] = []
    metric_specs = [
        ("validation_mape", "Held-out MAPE"),
        ("validation_rmse", "Held-out RMSE"),
        ("validation_mae", "Held-out MAE"),
        ("validation_r2", "Held-out R²"),
    ]
    for metric_key, title in metric_specs:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(frame["station_name"], frame[metric_key])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        saved = savefig(fig, figure_dir / metric_key)
        figure_paths.append(next(path for path in saved if path.suffix == ".png"))
    return figure_paths


def run_station_fit(
    *,
    station_row: dict[str, str],
    output_dir: Path,
    no_download: bool,
) -> dict[str, object]:
    try:
        station_data = tidal_compact.load_station_frame(
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

    cfg = build_fixed_model_config()
    component_mask = {name: True for name in cfg["harmonic_orders"]}
    component_mask.update({name: True for name in cfg["regressors"]})
    fit_result = tidal_compact.run_tidal_model(
        component_mask,
        df=station_data["df"],
        sph=station_data["sph"],
        harmonic_orders=cfg["harmonic_orders"],
        fourier_reg_weight=cfg["fourier_reg_weight"],
        lag_ranges=cfg["lag_ranges"],
        knot_presets=cfg["knot_presets"],
        interaction_pairs=[],
        train_start=DEFAULT_TRAIN_START,
        train_end=DEFAULT_TRAIN_END,
        test_end=DEFAULT_TEST_END,
        solver_verbose=False,
        debug=False,
    )
    result_row = make_result_row(station_row=station_row, fit_result=fit_result)
    station_figure_path = build_station_figure(
        station_name=station_row["station_name"],
        df_test=fit_result["df_test"],
        test_pred=fit_result["test_pred"],
        metrics_row=result_row,
        figure_dir=output_dir / "figures" / "multi_station",
    )
    return {
        "status": "included",
        "row": result_row,
        "station_figure_path": station_figure_path,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "run_station_fit or build_metric_figures" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_tidal_multi_station_report.py examples/run_tidal_multi_station_report.py
git commit -m "feat: add fixed-fit station execution and report figures"
```

## Task 5: Wire Full Report Collection, Output Writing, And CLI Verification

**Files:**
- Modify: `examples/run_tidal_multi_station_report.py`
- Modify: `test/test_tidal_multi_station_report.py`
- Test: `test/test_tidal_multi_station_report.py`

- [ ] **Step 1: Write the failing full-flow regression**

```python
def test_collect_station_reports_assembles_included_excluded_and_figures(monkeypatch, tmp_path):
    station_rows = [
        {
            "station_id": "8518750",
            "station_name": "The Battery, NY",
            "tidal_regime": "Semi-diurnal",
            "region": "NY Harbor",
        },
        {
            "station_id": "8720218",
            "station_name": "Galveston Pier 21, TX",
            "tidal_regime": "Diurnal",
            "region": "Gulf",
        },
    ]

    monkeypatch.setattr(multi_station, "build_station_rows", lambda: station_rows)
    monkeypatch.setattr(
        multi_station,
        "run_station_fit",
        lambda **kwargs: (
            {
                "status": "included",
                "row": {
                    "station_id": "8518750",
                    "station_name": "The Battery, NY",
                    "tidal_regime": "Semi-diurnal",
                    "train_rmse": 0.10,
                    "train_mae": 0.08,
                    "train_mape": 9.0,
                    "train_r2": 0.95,
                    "validation_rmse": 0.20,
                    "validation_mae": 0.16,
                    "validation_mape": 15.0,
                    "validation_r2": 0.85,
                    "n_train": 1000,
                    "n_validation": 500,
                    "active_regs": "pressure,dp_dt,wind_u",
                },
                "station_figure_path": tmp_path / "figures/multi_station/station_8518750.png",
            }
            if kwargs["station_row"]["station_id"] == "8518750"
            else {
                "status": "excluded",
                "row": {
                    "station_id": "8720218",
                    "station_name": "Galveston Pier 21, TX",
                    "category": "coverage",
                    "reason": "missing wind_u over held-out window",
                },
            }
        ),
    )
    monkeypatch.setattr(
        multi_station,
        "build_metric_figures",
        lambda rows, figure_dir: [tmp_path / "figures/multi_station/validation_mape.png"],
    )

    included_rows, excluded_rows, metric_paths, station_paths = multi_station.collect_station_reports(
        output_dir=tmp_path,
        no_download=True,
    )

    assert len(included_rows) == 1
    assert len(excluded_rows) == 1
    assert metric_paths == [tmp_path / "figures/multi_station/validation_mape.png"]
    assert station_paths == [tmp_path / "figures/multi_station/station_8518750.png"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "collect_station_reports" -v`

Expected: FAIL with missing `collect_station_reports`

- [ ] **Step 3: Write the minimal report collection and CLI implementation**

```python
def collect_station_reports(
    *,
    output_dir: Path,
    no_download: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path], list[Path]]:
    included_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    station_figure_paths: list[Path] = []

    for station_row in build_station_rows():
        result = run_station_fit(
            station_row=station_row,
            output_dir=output_dir,
            no_download=no_download,
        )
        if result["status"] == "included":
            included_rows.append(result["row"])
            station_figure_paths.append(result["station_figure_path"])
        else:
            excluded_rows.append(result["row"])
    metric_figure_paths = build_metric_figures(
        included_rows,
        output_dir / "figures" / "multi_station",
    ) if included_rows else []

    return included_rows, excluded_rows, metric_figure_paths, station_figure_paths


def write_report_outputs(
    *,
    output_dir: Path,
    included_rows: list[dict[str, object]],
    excluded_rows: list[dict[str, object]],
    summary_markdown: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "multi_station_results.csv"
    exclusions_path = output_dir / "multi_station_excluded.csv"
    summary_path = output_dir / "multi_station_summary.md"
    pd.DataFrame(included_rows).to_csv(results_path, index=False)
    pd.DataFrame(excluded_rows).to_csv(exclusions_path, index=False)
    summary_path.write_text(summary_markdown, encoding="utf-8")
    return results_path, exclusions_path, summary_path


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--no-download", is_flag=True, default=False)
def main(output_dir: Path | None, no_download: bool) -> None:
    target_dir = output_dir or (default_output_dir() / "tidal_multi_station")
    section("Tidal multi-station report")
    info(f"Output dir: {target_dir}")

    included_rows, excluded_rows, metric_paths, station_paths = collect_station_reports(
        output_dir=target_dir,
        no_download=no_download,
    )
    summary_markdown = build_summary_markdown(
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        metric_figure_paths=metric_paths,
        station_figure_paths=station_paths,
    )
    results_path, exclusions_path, summary_path = write_report_outputs(
        output_dir=target_dir,
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        summary_markdown=summary_markdown,
    )
    success(f"Results CSV: {results_path}")
    success(f"Exclusions CSV: {exclusions_path}")
    success(f"Summary markdown: {summary_path}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_tidal_multi_station_report.py -k "collect_station_reports" -v`

Expected: PASS

- [ ] **Step 5: Run final verification**

Run:

```bash
uv run pytest test/test_tidal_multi_station_report.py -q
uv run python examples/run_tidal_multi_station_report.py --help
```

Expected:

- pytest PASS
- CLI help renders with `--output-dir` and `--no-download`

- [ ] **Step 6: Run broader regression coverage**

Run:

```bash
uv run pytest test/test_example_tidal_compact.py test/test_tidal_multi_station_report.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add test/test_tidal_multi_station_report.py examples/run_tidal_multi_station_report.py
git commit -m "feat: add fixed-fit tidal multi-station report script"
```
