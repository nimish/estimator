import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

battery_grid = importlib.import_module("run_tidal_battery_grid")


def render_text(renderable: object, *, width: int = 120) -> str:
    console = Console(color_system=None, force_terminal=False, width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_default_split_targets_screenshot_window():
    assert battery_grid.DEFAULT_DATA_START == pd.Timestamp("2022-01-01").date()
    assert battery_grid.DEFAULT_DATA_END == pd.Timestamp("2025-01-01").date()
    assert battery_grid.SELECTION_TRAIN_START == "2022-01-01"
    assert battery_grid.SELECTION_TRAIN_END == "2024-01-01"
    assert battery_grid.SELECTION_TEST_END == "2025-01-01"


def test_reg_weight_grid_tracks_long_period_complexity():
    assert battery_grid.reg_weight_grid_for_orders(0, 0) == pytest.approx((1.0e-4, 3.0e-4, 1.0e-3))
    assert battery_grid.reg_weight_grid_for_orders(1, 2) == pytest.approx((1.0e-4, 3.0e-4, 1.0e-3))
    assert battery_grid.reg_weight_grid_for_orders(4, 8) == pytest.approx((3.0e-3,))
    assert battery_grid.reg_weight_grid_for_orders(8, 16) == pytest.approx((3.0e-3, 1.0e-2))
    assert battery_grid.reg_weight_grid_for_orders(12, 24) == pytest.approx((3.0e-3, 1.0e-2))


def test_build_harmonic_orders_only_changes_long_period_terms():
    harmonic_orders = battery_grid.build_harmonic_orders(mf_mm_order=12, annual_order=24)

    assert harmonic_orders["M2"] == 4
    assert harmonic_orders["S2"] == 1
    assert harmonic_orders["N2"] == 1
    assert harmonic_orders["K1"] == 2
    assert harmonic_orders["O1"] == 1
    assert harmonic_orders["Mf"] == 12
    assert harmonic_orders["Mm"] == 12
    assert harmonic_orders["annual"] == 24


def test_build_harmonic_orders_supports_zero_long_period_terms():
    harmonic_orders = battery_grid.build_harmonic_orders(mf_mm_order=0, annual_order=0)

    assert harmonic_orders["Mf"] == 0
    assert harmonic_orders["Mm"] == 0
    assert harmonic_orders["annual"] == 0


def test_build_search_candidates_is_local_anchor_sweep():
    candidates = battery_grid.build_search_candidates()

    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="med",
        mf_mm_order=0,
        annual_order=0,
        fourier_reg_weight=1.0e-4,
    ) in candidates
    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="high",
        mf_mm_order=16,
        annual_order=32,
        fourier_reg_weight=1.0e-2,
    ) in candidates
    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="low",
        mf_mm_order=0,
        annual_order=2,
        fourier_reg_weight=3.0e-4,
    ) not in candidates
    assert all(candidate.regressors == ("pressure", "dp_dt", "wind_u") for candidate in candidates)
    assert {candidate.knot_preset for candidate in candidates} == {"med", "high"}


def test_build_search_candidates_prunes_low_reg_for_unstable_high_complexity_combinations():
    candidates = battery_grid.build_search_candidates()

    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="med",
        mf_mm_order=4,
        annual_order=0,
        fourier_reg_weight=1.0e-3,
    ) not in candidates
    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="high",
        mf_mm_order=2,
        annual_order=8,
        fourier_reg_weight=1.0e-3,
    ) not in candidates
    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="med",
        mf_mm_order=4,
        annual_order=0,
        fourier_reg_weight=3.0e-3,
    ) in candidates
    assert battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="high",
        mf_mm_order=0,
        annual_order=16,
        fourier_reg_weight=1.0e-3,
    ) in candidates


def test_anchor_candidate_identifies_screenshot_starting_point():
    anchor = battery_grid.build_anchor_candidate()

    assert anchor == battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="med",
        mf_mm_order=0,
        annual_order=0,
        fourier_reg_weight=1.0e-4,
    )


def test_merge_rows_adds_anchor_reference_columns_and_deltas():
    rows = [
        {
            "candidate_label": "anchor",
            "regressor_family": "pressure+dp_dt+wind_u",
            "knot_preset": "med",
            "mf_mm_order": 0,
            "annual_order": 0,
            "fourier_reg_weight": 1.0e-4,
            "is_anchor": True,
            "selection_train_mape": 5.0,
            "selection_train_rmse": 0.10,
            "selection_train_r2": 0.90,
            "validation_mape": 10.0,
            "validation_rmse": 0.20,
            "validation_r2": 0.80,
        },
        {
            "candidate_label": "trial",
            "regressor_family": "pressure+dp_dt+wind_u",
            "knot_preset": "high",
            "mf_mm_order": 8,
            "annual_order": 16,
            "fourier_reg_weight": 1.0e-3,
            "is_anchor": False,
            "selection_train_mape": 4.0,
            "selection_train_rmse": 0.09,
            "selection_train_r2": 0.91,
            "validation_mape": 9.0,
            "validation_rmse": 0.18,
            "validation_r2": 0.82,
        },
    ]

    frame = battery_grid.merge_rows(rows, [])

    anchor_row = frame.loc[frame["candidate_label"] == "anchor"].iloc[0]
    trial_row = frame.loc[frame["candidate_label"] == "trial"].iloc[0]

    assert anchor_row["anchor_candidate_label"] == "anchor"
    assert anchor_row["anchor_validation_mape"] == pytest.approx(10.0)
    assert anchor_row["validation_minus_anchor_mape"] == pytest.approx(0.0)
    assert trial_row["anchor_candidate_label"] == "anchor"
    assert trial_row["validation_minus_anchor_mape"] == pytest.approx(-1.0)
    assert trial_row["validation_minus_anchor_rmse"] == pytest.approx(-0.02)
    assert trial_row["validation_minus_anchor_r2"] == pytest.approx(0.02)


def test_evaluate_candidate_threads_model_inputs_and_metrics(monkeypatch):
    candidate = battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="med",
        mf_mm_order=12,
        annual_order=24,
        fourier_reg_weight=3.0e-3,
    )
    captured: dict[str, object] = {}

    def fake_run_tidal_model(component_mask, **kwargs):
        captured["component_mask"] = component_mask
        captured["kwargs"] = kwargs
        return {
            "metrics_train": {"rmse": 0.1, "mae": 0.08, "mape": 5.0, "r2": 0.9},
            "metrics_test": {"rmse": 0.2, "mae": 0.15, "mape": 8.0, "r2": 0.8},
            "tr_mape_n": 100,
            "te_mape_n": 50,
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 200,
            "n_test": 100,
        }

    monkeypatch.setattr(battery_grid.tidal_compact, "run_tidal_model", fake_run_tidal_model)

    row = battery_grid.evaluate_candidate(
        candidate,
        df=pd.DataFrame({"water_level": [0.0]}, index=pd.date_range("2022-01-01", periods=1, freq="1h")),
        sph=1,
        train_start="2022-01-01",
        train_end="2024-01-01",
        test_end="2025-01-01",
        stage_prefix="validation",
    )

    kwargs = cast(dict[str, object], captured["kwargs"])
    harmonic_orders = cast(dict[str, int], kwargs["harmonic_orders"])
    lag_ranges = cast(dict[str, tuple[int, int]], kwargs["lag_ranges"])
    knot_presets = cast(dict[str, str], kwargs["knot_presets"])

    assert captured["component_mask"] == {
        "M2": True,
        "S2": True,
        "N2": True,
        "K1": True,
        "O1": True,
        "Mf": True,
        "Mm": True,
        "annual": True,
        "pressure": True,
        "dp_dt": True,
        "wind_u": True,
    }
    assert harmonic_orders["Mf"] == 12
    assert harmonic_orders["Mm"] == 12
    assert harmonic_orders["annual"] == 24
    assert lag_ranges == {"pressure": (-2, 0), "dp_dt": (-2, 0), "wind_u": (-1, 0)}
    assert knot_presets == {"pressure": "med", "dp_dt": "med", "wind_u": "med"}
    assert kwargs["fourier_reg_weight"] == pytest.approx(3.0e-3)
    assert kwargs["train_start"] == "2022-01-01"
    assert kwargs["train_end"] == "2024-01-01"
    assert kwargs["test_end"] == "2025-01-01"
    assert row["selection_train_rmse"] == pytest.approx(0.1)
    assert row["validation_mape"] == pytest.approx(8.0)
    assert row["validation_mape_n"] == 50
    assert row["active_regs"] == "pressure,dp_dt,wind_u"


def test_run_candidates_notifies_reporter_for_each_completed_row(monkeypatch):
    candidate = battery_grid.SearchCandidate(
        regressors=("pressure", "wind_u"),
        knot_preset="med",
        mf_mm_order=1,
        annual_order=2,
        fourier_reg_weight=3.0e-4,
    )
    reported: list[tuple[int, int, str, str]] = []
    call_count = 0

    def fake_evaluate_candidate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "candidate_label": f"demo-{call_count}",
            "validation_fit_status": "ok",
            "validation_mape": 8.0,
            "validation_rmse": 0.2,
            "validation_r2": 0.8,
        }

    monkeypatch.setattr(battery_grid, "evaluate_candidate", fake_evaluate_candidate)

    battery_grid.run_candidates(
        [candidate, candidate],
        df=pd.DataFrame(
            {"water_level": [0.0]},
            index=pd.date_range("2022-01-01", periods=1, freq="1h"),
        ),
        sph=1,
        train_start="2022-01-01",
        train_end="2023-01-01",
        test_end="2023-12-31",
        stage_prefix="validation",
        n_jobs=1,
        log=lambda _message: None,
        reporter=lambda *, completed, total, stage_prefix, row, best_so_far: reported.append(
            (completed, total, stage_prefix, str(row["candidate_label"]))
        ),
    )

    assert reported == [
        (1, 2, "validation", "demo-1"),
        (2, 2, "validation", "demo-2"),
    ]


def test_resume_candidates_reuses_matching_partial_rows_and_skips_completed_work():
    all_candidates = [
        battery_grid.SearchCandidate(
            regressors=("pressure", "dp_dt", "wind_u"),
            knot_preset="med",
            mf_mm_order=0,
            annual_order=0,
            fourier_reg_weight=1.0e-4,
        ),
        battery_grid.SearchCandidate(
            regressors=("pressure", "dp_dt", "wind_u"),
            knot_preset="med",
            mf_mm_order=1,
            annual_order=0,
            fourier_reg_weight=1.0e-4,
        ),
    ]
    existing_rows = [
        {
            "candidate_label": "anchor",
            "regressor_family": "pressure+dp_dt+wind_u",
            "knot_preset": "med",
            "mf_mm_order": 0,
            "annual_order": 0,
            "fourier_reg_weight": 1.0e-4,
            "validation_mape": 21.3,
        },
        {
            "candidate_label": "outside-grid",
            "regressor_family": "pressure+dp_dt+wind_u",
            "knot_preset": "med",
            "mf_mm_order": 16,
            "annual_order": 32,
            "fourier_reg_weight": 1.0e-2,
            "validation_mape": 40.0,
        },
    ]

    resumed_rows, pending_candidates = battery_grid.resume_candidates(all_candidates, existing_rows)

    assert [row["candidate_label"] for row in resumed_rows] == ["anchor"]
    assert pending_candidates == [all_candidates[1]]


def test_run_candidates_resumes_progress_from_existing_rows(monkeypatch):
    candidate = battery_grid.SearchCandidate(
        regressors=("pressure", "wind_u"),
        knot_preset="med",
        mf_mm_order=1,
        annual_order=2,
        fourier_reg_weight=3.0e-4,
    )
    reported: list[tuple[int, int, str, str]] = []

    def fake_evaluate_candidate(*args, **kwargs):
        return {
            "candidate_label": "demo-pending",
            "validation_fit_status": "ok",
            "validation_mape": 8.0,
            "validation_rmse": 0.2,
            "validation_r2": 0.8,
        }

    monkeypatch.setattr(battery_grid, "evaluate_candidate", fake_evaluate_candidate)

    rows = battery_grid.run_candidates(
        [candidate],
        df=pd.DataFrame(
            {"water_level": [0.0]},
            index=pd.date_range("2022-01-01", periods=1, freq="1h"),
        ),
        sph=1,
        train_start="2022-01-01",
        train_end="2023-01-01",
        test_end="2023-12-31",
        stage_prefix="validation",
        n_jobs=1,
        log=lambda _message: None,
        existing_rows=[
            {
                "candidate_label": "demo-existing",
                "regressor_family": "pressure+wind_u",
                "knot_preset": "med",
                "mf_mm_order": 0,
                "annual_order": 0,
                "fourier_reg_weight": 1.0e-4,
                "validation_mape": 9.0,
                "validation_rmse": 0.21,
                "validation_r2": 0.79,
                "validation_fit_status": "ok",
            }
        ],
        reporter=lambda *, completed, total, stage_prefix, row, best_so_far: reported.append(
            (completed, total, stage_prefix, str(row["candidate_label"]))
        ),
    )

    assert [row["candidate_label"] for row in rows] == ["demo-existing", "demo-pending"]
    assert reported == [(2, 2, "validation", "demo-pending")]


def test_handle_interrupt_terminates_and_joins_worker_backend():
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


def test_trim_recent_rows_returns_empty_for_non_positive_limit():
    rows = [{"candidate_label": f"row-{idx}"} for idx in range(6)]

    assert battery_grid.trim_recent_rows(rows, limit=0) == []
    assert battery_grid.trim_recent_rows(rows, limit=-2) == []


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
        {
            "candidate_label": "row-2",
            "validation_fit_status": "ok",
            "validation_mape": 8.5,
            "validation_rmse": 0.09,
            "validation_r2": 0.91,
        },
    ]

    table = battery_grid.build_recent_results_table(rows, "validation", limit=2)

    assert isinstance(table, Table)
    assert table.title == "Recent results"
    assert [column.header for column in table.columns] == [
        "Stage",
        "Candidate",
        "Status",
        "MAPE",
        "RMSE",
        "R^2",
    ]
    rendered = render_text(table)

    assert "Recent results" in rendered
    assert "Stage" in rendered
    assert "Candidate" in rendered
    assert "Status" in rendered
    assert "MAPE" in rendered
    assert "RMSE" in rendered
    assert "R^2" in rendered
    assert "row-0" not in rendered
    assert "row-1" in rendered
    assert "row-2" in rendered
    assert rendered.index("row-1") < rendered.index("row-2")
    assert "exception" in rendered
    assert "ok" in rendered
    assert "8.5" in rendered
    assert "0.09" in rendered
    assert "0.91" in rendered


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
    assert empty_panel.title == "Best validation so far"
    assert empty_panel.renderable == "No finite validation result yet."

    assert isinstance(full_panel, Panel)
    assert full_panel.title == "Best validation so far"
    assert full_panel.renderable == (
        "regs=pressure+wind_u knots=med Mf/Mm=8 annual=16 reg=0.003\n"
        "MAPE=9.5  RMSE=0.11  R^2=0.87"
    )
