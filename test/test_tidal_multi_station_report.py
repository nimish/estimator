from pathlib import Path
import importlib
import sys
from types import SimpleNamespace
from typing import get_args, get_origin, get_type_hints

from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import run_tidal_battery_grid as battery_grid  # noqa: E402
from example_tidal import STATION_CATALOG  # noqa: E402

multi_station = importlib.import_module("run_tidal_multi_station_report")


def _station_row() -> dict[str, str]:
    return {
        "station_id": "8518750",
        "station_name": "The Battery, NY",
        "tidal_regime": "Semi-diurnal",
        "region": "NY Harbor",
    }


def _loaded_station_frame() -> dict[str, object]:
    return {
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
    }


def _fit_result(
    *,
    metrics_train: dict[str, float] | None = None,
    metrics_test: dict[str, float] | None = None,
    active_regs: list[str] | None = None,
) -> dict[str, object]:
    return {
        "metrics_train": metrics_train
        or {
            "rmse": 0.1,
            "mae": 0.08,
            "mape": 9.0,
            "r2": 0.95,
        },
        "metrics_test": metrics_test
        or {
            "rmse": 0.2,
            "mae": 0.16,
            "mape": 15.0,
            "r2": 0.85,
        },
        "active_regs": active_regs or ["pressure", "dp_dt", "wind_u"],
        "n_train": 3,
        "n_test": 1,
        "te_index": pd.date_range("2024-01-01", periods=1, freq="1D"),
        "te_obs": np.array([1.3]),
        "te_pred": np.array([1.25]),
    }


def test_build_station_rows_uses_station_catalog_names_and_ids():
    rows = multi_station.build_station_rows()

    expected_rows = [
        {
            "station_id": station_id,
            "station_name": str(meta["name"]),
            "tidal_regime": str(meta.get("tidal_regime", "")),
            "region": str(meta.get("region", "")),
        }
        for station_id, meta in STATION_CATALOG.items()
    ]

    assert rows == expected_rows
    assert len(rows) == len(STATION_CATALOG)
    assert all(set(row) == {"station_id", "station_name", "tidal_regime", "region"} for row in rows)


def test_build_fixed_model_config_matches_battery_style_defaults():
    cfg = multi_station.build_fixed_model_config()
    anchor = battery_grid.build_anchor_candidate()

    assert cfg == {
        "harmonic_orders": battery_grid.build_harmonic_orders(
            anchor.mf_mm_order,
            anchor.annual_order,
        ),
        "regressors": anchor.regressors,
        "lag_ranges": battery_grid.build_lag_ranges(anchor),
        "knot_presets": battery_grid.build_knot_presets(anchor),
        "fourier_reg_weight": pytest.approx(anchor.fourier_reg_weight),
    }


def test_build_fixed_model_config_tracks_battery_source_of_truth(monkeypatch):
    patched_anchor = battery_grid.SearchCandidate(
        regressors=("pressure", "dp_dt", "wind_u"),
        knot_preset="high",
        mf_mm_order=1,
        annual_order=2,
        fourier_reg_weight=3.0e-4,
    )
    monkeypatch.setattr(battery_grid, "build_anchor_candidate", lambda: patched_anchor)

    cfg = multi_station.build_fixed_model_config()

    assert cfg == {
        "harmonic_orders": battery_grid.build_harmonic_orders(
            patched_anchor.mf_mm_order,
            patched_anchor.annual_order,
        ),
        "regressors": patched_anchor.regressors,
        "lag_ranges": battery_grid.build_lag_ranges(patched_anchor),
        "knot_presets": battery_grid.build_knot_presets(patched_anchor),
        "fourier_reg_weight": pytest.approx(patched_anchor.fourier_reg_weight),
    }


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
            "metrics_train": {
                "rmse": 0.10,
                "mae": 0.08,
                "mape": 9.0,
                "r2": 0.95,
            },
            "metrics_test": {
                "rmse": 0.20,
                "mae": 0.16,
                "mape": 15.0,
                "r2": 0.85,
            },
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 1000,
            "n_test": 500,
        },
    )

    assert row["station_id"] == "8518750"
    assert row["station_name"] == "The Battery, NY"
    assert row["tidal_regime"] == "Semi-diurnal"
    assert row["region"] == "NY Harbor"
    assert row["active_regs"] == "pressure,dp_dt,wind_u"
    assert row["n_train"] == 1000
    assert row["n_validation"] == 500

    assert row["train_rmse"] == pytest.approx(0.10)
    assert row["train_mae"] == pytest.approx(0.08)
    assert row["train_mape"] == pytest.approx(9.0)
    assert row["train_r2"] == pytest.approx(0.95)

    assert row["validation_rmse"] == pytest.approx(0.20)
    assert row["validation_mae"] == pytest.approx(0.16)
    assert row["validation_mape"] == pytest.approx(15.0)
    assert row["validation_r2"] == pytest.approx(0.85)

    assert row["validation_minus_train_rmse"] == pytest.approx(0.10)
    assert row["validation_minus_train_mae"] == pytest.approx(0.08)
    assert row["validation_minus_train_mape"] == pytest.approx(6.0)
    assert row["validation_minus_train_r2"] == pytest.approx(-0.10)


def test_build_summary_markdown_uses_typed_result_and_exclusion_rows_in_annotations():
    build_hints = get_type_hints(
        multi_station.build_summary_markdown,
        globalns=vars(multi_station),
    )
    table_hints = get_type_hints(
        multi_station._markdown_table,
        globalns=vars(multi_station),
    )

    assert get_origin(build_hints["included_rows"]) is list
    assert get_args(build_hints["included_rows"]) == (multi_station.ResultRow,)
    assert get_origin(build_hints["excluded_rows"]) is list
    assert get_args(build_hints["excluded_rows"]) == (multi_station.ExclusionRow,)

    table_row_args = get_args(table_hints["rows"])
    assert len(table_row_args) == 2
    assert table_row_args[0] == list[multi_station.ResultRow]
    assert table_row_args[1] == list[multi_station.ExclusionRow]


def test_format_markdown_value_rounds_binary_float_noise_for_report():
    assert multi_station._format_markdown_value(-0.09999999999999998) == "-0.1"
    assert multi_station._format_markdown_value(1000) == "1000"
    assert multi_station._format_markdown_value("The Battery, NY") == "The Battery, NY"


def test_build_summary_markdown_includes_all_metrics_tables_and_relative_figures():
    included_row = multi_station.make_result_row(
        station_row={
            "station_id": "8518750",
            "station_name": "The Battery, NY",
            "tidal_regime": "Semi-diurnal",
            "region": "NY Harbor",
        },
        fit_result={
            "metrics_train": {
                "rmse": 0.10,
                "mae": 0.08,
                "mape": 9.0,
                "r2": 0.95,
            },
            "metrics_test": {
                "rmse": 0.20,
                "mae": 0.16,
                "mape": 15.0,
                "r2": 0.85,
            },
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 1000,
            "n_test": 500,
        },
    )
    excluded_row = multi_station.make_exclusion_row(
        station_id="8771341",
        station_name="Galveston Pier 21, TX",
        category="coverage",
        reason="missing wind_u over held-out window",
    )

    summary = multi_station.build_summary_markdown(
        included_rows=[included_row],
        excluded_rows=[excluded_row],
        metric_figure_paths=[
            Path("figures/multi_station/validation_mape.png"),
            Path("figures/multi_station/validation_rmse.png"),
        ],
        station_figure_paths=[
            Path("figures/multi_station/station_8518750.png"),
        ],
    )
    lines = summary.splitlines()

    expected_included_header = (
        "| station_id | station_name | tidal_regime | region | active_regs | "
        "n_train | n_validation | train_rmse | train_mae | train_mape | "
        "train_r2 | validation_rmse | validation_mae | validation_mape | "
        "validation_r2 | validation_minus_train_rmse | "
        "validation_minus_train_mae | validation_minus_train_mape | "
        "validation_minus_train_r2 |"
    )
    expected_included_divider = (
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- | --- | --- |"
    )
    expected_included_row = (
        "| 8518750 | The Battery, NY | Semi-diurnal | NY Harbor | "
        "pressure,dp_dt,wind_u | 1000 | 500 | 0.1 | 0.08 | 9.0 | 0.95 | "
        "0.2 | 0.16 | 15.0 | 0.85 | 0.1 | 0.08 | 6.0 | -0.1 |"
    )
    expected_excluded_header = "| station_id | station_name | category | reason |"
    expected_excluded_divider = "| --- | --- | --- | --- |"
    expected_excluded_row = (
        "| 8771341 | Galveston Pier 21, TX | coverage | "
        "missing wind_u over held-out window |"
    )

    assert lines[0] == "# Tidal Multi-Station Report"
    assert "## Setup" in lines
    assert "Fixed model form shared across every station:" in summary
    assert "- Train window: `2022-01-01` to `2023-12-31`." in summary
    assert "- Held-out window: `2024-01-01` to `2024-12-31`." in summary
    assert "## Ranked held-out summary" in lines
    assert "| rank | station_name | validation_mape | validation_rmse | validation_r2 |" in summary
    assert "| 1 | The Battery, NY | 15.0 | 0.2 | 0.85 |" in summary

    included_idx = lines.index("## Included stations")
    assert lines[included_idx + 1 : included_idx + 4] == [
        expected_included_header,
        expected_included_divider,
        expected_included_row,
    ]

    excluded_idx = lines.index("## Excluded stations")
    assert lines[excluded_idx + 1 : excluded_idx + 4] == [
        expected_excluded_header,
        expected_excluded_divider,
        expected_excluded_row,
    ]

    across_idx = lines.index("## Across-station charts")
    assert lines[across_idx + 1 : across_idx + 3] == [
        "![validation_mape](figures/multi_station/validation_mape.png)",
        "![validation_rmse](figures/multi_station/validation_rmse.png)",
    ]

    station_idx = lines.index("## Per-station figures")
    assert lines[station_idx + 1 : station_idx + 2] == [
        "![station_8518750](figures/multi_station/station_8518750.png)"
    ]
    assert "-0.09999999999999998" not in summary
    assert summary.endswith("\n")


def test_build_summary_markdown_normalizes_absolute_figure_paths_relative_to_report_dir(
    tmp_path: Path,
):
    report_dir = tmp_path / "report"
    metric_figure_path = report_dir / "figures" / "multi_station" / "validation_mape.png"
    station_figure_path = report_dir / "figures" / "multi_station" / "station_8518750.png"

    summary = multi_station.build_summary_markdown(
        included_rows=[],
        excluded_rows=[],
        metric_figure_paths=[metric_figure_path],
        station_figure_paths=[station_figure_path],
        report_dir=report_dir,
    )

    assert "![validation_mape](figures/multi_station/validation_mape.png)" in summary
    assert "![station_8518750](figures/multi_station/station_8518750.png)" in summary
    assert str(metric_figure_path) not in summary
    assert str(station_figure_path) not in summary


def test_run_station_fit_returns_included_row_and_station_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        multi_station,
        "tidal_compact",
        SimpleNamespace(
            load_station_frame=lambda *args, **kwargs: _loaded_station_frame(),
            run_tidal_model=lambda *args, **kwargs: _fit_result(),
        ),
        raising=False,
    )

    result = multi_station.run_station_fit(
        station_row=_station_row(),
        output_dir=tmp_path,
        no_download=True,
    )

    assert result["status"] == "included"
    assert result["row"]["station_name"] == "The Battery, NY"
    assert result["row"]["validation_mape"] == pytest.approx(15.0)
    assert result["station_figure_path"].suffix == ".png"
    assert result["station_figure_path"].exists()


def test_run_station_fit_uses_spec_boundaries_for_load_and_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    def fake_load_station_frame(
        station_name: str,
        *,
        use_weather: bool,
        download_tidal: bool,
        download_weather: bool,
        data_start,
        data_end,
    ) -> dict[str, object]:
        captured["station_name"] = station_name
        captured["data_start"] = data_start
        captured["data_end"] = data_end
        captured["use_weather"] = use_weather
        captured["download_tidal"] = download_tidal
        captured["download_weather"] = download_weather
        return _loaded_station_frame()

    def fake_run_tidal_model(component_mask: dict[str, bool], **kwargs: object) -> dict[str, object]:
        captured["component_mask"] = component_mask
        captured["train_start"] = kwargs["train_start"]
        captured["train_end"] = kwargs["train_end"]
        captured["test_end"] = kwargs["test_end"]
        return _fit_result()

    monkeypatch.setattr(
        multi_station,
        "tidal_compact",
        SimpleNamespace(
            load_station_frame=fake_load_station_frame,
            run_tidal_model=fake_run_tidal_model,
        ),
        raising=False,
    )

    result = multi_station.run_station_fit(
        station_row=_station_row(),
        output_dir=tmp_path,
        no_download=True,
    )

    assert result["status"] == "included"
    assert captured["station_name"] == "The Battery, NY"
    assert captured["data_start"] == pd.Timestamp("2022-01-01").date()
    assert captured["data_end"] == pd.Timestamp("2024-12-31").date()
    assert captured["train_start"] == "2022-01-01"
    assert captured["train_end"] == "2024-01-01"
    assert captured["test_end"] == "2024-12-31"


def test_run_station_fit_returns_exclusion_row_on_load_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def raise_missing_data(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("missing wind_u over held-out window")

    monkeypatch.setattr(
        multi_station,
        "tidal_compact",
        SimpleNamespace(
            load_station_frame=raise_missing_data,
            run_tidal_model=lambda *args, **kwargs: pytest.fail(
                "run_tidal_model should not be called when loading fails"
            ),
        ),
        raising=False,
    )

    result = multi_station.run_station_fit(
        station_row=_station_row(),
        output_dir=tmp_path,
        no_download=True,
    )

    assert result == {
        "status": "excluded",
        "row": {
            "station_id": "8518750",
            "station_name": "The Battery, NY",
            "category": "coverage",
            "reason": "missing wind_u over held-out window",
        },
    }


def test_run_station_fit_returns_exclusion_row_on_non_file_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def raise_load_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(
        multi_station,
        "tidal_compact",
        SimpleNamespace(
            load_station_frame=raise_load_error,
            run_tidal_model=lambda *args, **kwargs: pytest.fail(
                "run_tidal_model should not be called when loading fails"
            ),
        ),
        raising=False,
    )

    result = multi_station.run_station_fit(
        station_row=_station_row(),
        output_dir=tmp_path,
        no_download=True,
    )

    assert result["status"] == "excluded"
    assert result["row"]["category"] == "load"
    assert "unexpected parser failure" in result["row"]["reason"]


def test_run_station_fit_excludes_degraded_fit_result_with_non_finite_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        multi_station,
        "tidal_compact",
        SimpleNamespace(
            load_station_frame=lambda *args, **kwargs: _loaded_station_frame(),
            run_tidal_model=lambda *args, **kwargs: _fit_result(
                metrics_test={
                    "rmse": np.nan,
                    "mae": 0.16,
                    "mape": 15.0,
                    "r2": 0.85,
                }
            ),
        ),
        raising=False,
    )

    result = multi_station.run_station_fit(
        station_row=_station_row(),
        output_dir=tmp_path,
        no_download=True,
    )

    assert result["status"] == "excluded"
    assert result["row"]["category"] == "fit"
    assert "non-finite" in result["row"]["reason"]


def test_run_station_fit_excludes_degraded_fit_result_with_dropped_regressors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        multi_station,
        "tidal_compact",
        SimpleNamespace(
            load_station_frame=lambda *args, **kwargs: _loaded_station_frame(),
            run_tidal_model=lambda *args, **kwargs: _fit_result(
                active_regs=["pressure", "dp_dt"]
            ),
        ),
        raising=False,
    )

    result = multi_station.run_station_fit(
        station_row=_station_row(),
        output_dir=tmp_path,
        no_download=True,
    )

    assert result["status"] == "excluded"
    assert result["row"]["category"] == "fit"
    assert "fixed regressor set" in result["row"]["reason"]


def test_build_metric_figures_writes_expected_pngs(tmp_path: Path):
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

    figure_paths = multi_station.build_metric_figures(
        rows,
        tmp_path / "figures" / "multi_station",
    )

    assert figure_paths
    assert all(path.suffix == ".png" for path in figure_paths)
    assert all(path.exists() for path in figure_paths)


def test_collect_station_reports_assembles_included_excluded_and_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    station_rows = [
        _station_row(),
        {
            "station_id": "8720218",
            "station_name": "Galveston Pier 21, TX",
            "tidal_regime": "Diurnal",
            "region": "Gulf",
        },
    ]
    included_row = multi_station.make_result_row(
        station_row=station_rows[0],
        fit_result={
            "metrics_train": {
                "rmse": 0.10,
                "mae": 0.08,
                "mape": 9.0,
                "r2": 0.95,
            },
            "metrics_test": {
                "rmse": 0.20,
                "mae": 0.16,
                "mape": 15.0,
                "r2": 0.85,
            },
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 1000,
            "n_test": 500,
        },
    )
    excluded_row = multi_station.make_exclusion_row(
        station_id="8720218",
        station_name="Galveston Pier 21, TX",
        category="coverage",
        reason="missing wind_u over held-out window",
    )
    station_figure_path = tmp_path / "figures" / "multi_station" / "station_8518750.png"
    metric_figure_paths = [
        tmp_path / "figures" / "multi_station" / "validation_mape.png",
        tmp_path / "figures" / "multi_station" / "validation_rmse.png",
    ]

    monkeypatch.setattr(multi_station, "build_station_rows", lambda: station_rows)

    def fake_run_station_fit(
        *, station_row: dict[str, str], output_dir: Path, no_download: bool
    ) -> dict[str, object]:
        assert output_dir == tmp_path
        assert no_download is True
        if station_row["station_id"] == "8518750":
            return {
                "status": "included",
                "row": included_row,
                "station_figure_path": station_figure_path,
            }
        return {
            "status": "excluded",
            "row": excluded_row,
        }

    monkeypatch.setattr(multi_station, "run_station_fit", fake_run_station_fit)
    monkeypatch.setattr(
        multi_station,
        "build_metric_figures",
        lambda rows, figure_dir: metric_figure_paths,
    )

    (
        included_rows,
        excluded_rows,
        built_metric_figure_paths,
        station_figure_paths,
    ) = multi_station.collect_station_reports(
        output_dir=tmp_path,
        no_download=True,
    )

    assert included_rows == [included_row]
    assert excluded_rows == [excluded_row]
    assert built_metric_figure_paths == metric_figure_paths
    assert station_figure_paths == [station_figure_path]


def test_write_report_outputs_writes_csvs_and_markdown(tmp_path: Path):
    output_dir = tmp_path / "report"
    included_rows = [
        multi_station.make_result_row(
            station_row=_station_row(),
            fit_result={
                "metrics_train": {
                    "rmse": 0.10,
                    "mae": 0.08,
                    "mape": 9.0,
                    "r2": 0.95,
                },
                "metrics_test": {
                    "rmse": 0.20,
                    "mae": 0.16,
                    "mape": 15.0,
                    "r2": 0.85,
                },
                "active_regs": ["pressure", "dp_dt", "wind_u"],
                "n_train": 1000,
                "n_test": 500,
            },
        )
    ]
    excluded_rows = [
        multi_station.make_exclusion_row(
            station_id="8720218",
            station_name="Galveston Pier 21, TX",
            category="coverage",
            reason="missing wind_u over held-out window",
        )
    ]
    summary_markdown = multi_station.build_summary_markdown(
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        metric_figure_paths=[
            output_dir / "figures" / "multi_station" / "validation_mape.png"
        ],
        station_figure_paths=[
            output_dir / "figures" / "multi_station" / "station_8518750.png"
        ],
        report_dir=output_dir,
    )

    results_path, excluded_path, summary_path = multi_station.write_report_outputs(
        output_dir=output_dir,
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        summary_markdown=summary_markdown,
    )

    assert results_path == output_dir / "multi_station_results.csv"
    assert excluded_path == output_dir / "multi_station_excluded.csv"
    assert summary_path == output_dir / "multi_station_summary.md"
    assert results_path.exists()
    assert excluded_path.exists()
    assert summary_path.exists()

    results_frame = pd.read_csv(results_path)
    excluded_frame = pd.read_csv(excluded_path)
    assert results_frame["station_id"].astype(str).tolist() == ["8518750"]
    assert excluded_frame["station_id"].astype(str).tolist() == ["8720218"]

    written_summary = summary_path.read_text(encoding="utf-8")
    assert "## Included stations" in written_summary
    assert "## Excluded stations" in written_summary
    assert "![validation_mape](figures/multi_station/validation_mape.png)" in written_summary
    assert "![station_8518750](figures/multi_station/station_8518750.png)" in written_summary


def test_main_persists_checkpoint_outputs_before_late_metric_figure_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    station_rows = [
        _station_row(),
        {
            "station_id": "8720218",
            "station_name": "Galveston Pier 21, TX",
            "tidal_regime": "Diurnal",
            "region": "Gulf",
        },
    ]
    included_row = multi_station.make_result_row(
        station_row=station_rows[0],
        fit_result={
            "metrics_train": {
                "rmse": 0.10,
                "mae": 0.08,
                "mape": 9.0,
                "r2": 0.95,
            },
            "metrics_test": {
                "rmse": 0.20,
                "mae": 0.16,
                "mape": 15.0,
                "r2": 0.85,
            },
            "active_regs": ["pressure", "dp_dt", "wind_u"],
            "n_train": 1000,
            "n_test": 500,
        },
    )
    excluded_row = multi_station.make_exclusion_row(
        station_id="8720218",
        station_name="Galveston Pier 21, TX",
        category="coverage",
        reason="missing wind_u over held-out window",
    )

    monkeypatch.setattr(multi_station, "build_station_rows", lambda: station_rows)

    def fake_run_station_fit(
        *, station_row: dict[str, str], output_dir: Path, no_download: bool
    ) -> dict[str, object]:
        assert output_dir == tmp_path
        assert no_download is True
        if station_row["station_id"] == "8518750":
            return {
                "status": "included",
                "row": included_row,
                "station_figure_path": tmp_path
                / "figures"
                / "multi_station"
                / "station_8518750.png",
            }
        return {
            "status": "excluded",
            "row": excluded_row,
        }

    monkeypatch.setattr(multi_station, "run_station_fit", fake_run_station_fit)

    def fail_metric_figures(
        rows: list[dict[str, object]], figure_dir: Path
    ) -> list[Path]:
        assert rows == [included_row]
        assert figure_dir == tmp_path / "figures" / "multi_station"
        raise RuntimeError("late metric figure failure")

    monkeypatch.setattr(multi_station, "build_metric_figures", fail_metric_figures)

    runner = CliRunner()
    result = runner.invoke(
        multi_station.main,
        ["--output-dir", str(tmp_path), "--no-download"],
        catch_exceptions=True,
    )

    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "late metric figure failure"

    results_path = tmp_path / "multi_station_results.csv"
    excluded_path = tmp_path / "multi_station_excluded.csv"
    log_path = tmp_path / "multi_station.log"
    summary_path = tmp_path / "multi_station_summary.md"

    assert results_path.exists()
    assert excluded_path.exists()
    assert log_path.exists()
    assert not summary_path.exists()

    results_frame = pd.read_csv(results_path)
    excluded_frame = pd.read_csv(excluded_path)
    assert results_frame["station_id"].astype(str).tolist() == ["8518750"]
    assert excluded_frame["station_id"].astype(str).tolist() == ["8720218"]

    log_text = log_path.read_text(encoding="utf-8")
    assert "Processing The Battery, NY (8518750)" in log_text
    assert "Included The Battery, NY (8518750)" in log_text
    assert "Excluded Galveston Pier 21, TX (8720218)" in log_text


def test_main_normalizes_relative_output_dir_for_report_relative_figure_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    included_rows = [
        multi_station.make_result_row(
            station_row=_station_row(),
            fit_result={
                "metrics_train": {
                    "rmse": 0.10,
                    "mae": 0.08,
                    "mape": 9.0,
                    "r2": 0.95,
                },
                "metrics_test": {
                    "rmse": 0.20,
                    "mae": 0.16,
                    "mape": 15.0,
                    "r2": 0.85,
                },
                "active_regs": ["pressure", "dp_dt", "wind_u"],
                "n_train": 1000,
                "n_test": 500,
            },
        )
    ]
    monkeypatch.chdir(tmp_path)

    def fake_collect_station_reports(
        *,
        output_dir: Path,
        no_download: bool,
        log=None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path], list[Path]]:
        assert output_dir.is_absolute()
        assert no_download is True
        return (
            included_rows,
            [],
            [output_dir / "figures" / "multi_station" / "validation_mape.png"],
            [output_dir / "figures" / "multi_station" / "station_8518750.png"],
        )

    monkeypatch.setattr(
        multi_station,
        "collect_station_reports",
        fake_collect_station_reports,
    )

    runner = CliRunner()
    result = runner.invoke(
        multi_station.main,
        ["--output-dir", "relative-report", "--no-download"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    summary_path = tmp_path / "relative-report" / "multi_station_summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "![validation_mape](figures/multi_station/validation_mape.png)" in summary
    assert "![station_8518750](figures/multi_station/station_8518750.png)" in summary
    assert "relative-report/figures" not in summary
