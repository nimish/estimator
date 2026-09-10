# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the real-data forecast benchmark support layer."""

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from forecast_real_data_support import (  # noqa: E402
    DatasetSpec,
    PreparedDataset,
    benchmark_dataset,
    compact_summary,
    plot_forecast_paths,
    split_and_alignment_table,
)


def _prepared_dataset() -> PreparedDataset:
    spec = DatasetSpec(
        name="synthetic",
        display_name="Synthetic benchmark",
        source="generated",
        target_label="target",
        feature_columns=("driver",),
        feature_labels=("driver at origin",),
        train_samples=48,
        eval_samples=12,
        horizon=3,
        feature_fill_limit=0,
        periods_in_samples=(24.0,),
        num_harmonics=(1,),
        coupling_weight=0.1,
        solver="CLARABEL",
        solver_opts=None,
        caveat="Synthetic test fixture.",
    )
    timestamps = pd.date_range("2025-01-01", periods=spec.required_samples, freq="1h")
    sample = np.arange(len(timestamps), dtype=float)
    driver = np.sin(sample / 5.0)
    target = 2.0 + 0.8 * driver + np.sin(2.0 * np.pi * sample / 24.0)
    frame = pd.DataFrame({"target": target, "driver": driver}, index=timestamps)
    return PreparedDataset(
        spec=spec,
        frame=frame,
        step=pd.Timedelta(hours=1),
        source_rows=len(frame),
        native_grid_rows=len(frame),
        filled_feature_cells=0,
    )


def test_real_data_benchmark_includes_aligned_nowcast():
    prepared = _prepared_dataset()

    result = benchmark_dataset(prepared)

    expected_columns = [f"horizon_{horizon}" for horizon in range(4)]
    assert list(result.actuals.columns) == expected_columns
    assert set(result.metrics["horizon"]) == {0, 1, 2, 3}
    for prediction in result.predictions.values():
        assert list(prediction.columns) == expected_columns
    summary = compact_summary({"synthetic": result})
    assert "independent RMSE h=0" in summary
    assert "coupled RMSE h=0" in summary


def test_real_data_tables_and_path_plot_show_horizon_zero():
    prepared = _prepared_dataset()
    result = benchmark_dataset(prepared)

    alignment = split_and_alignment_table({"synthetic": prepared})
    assert 0 in alignment["horizon steps"].to_numpy()

    figure = plot_forecast_paths({"synthetic": result})
    labels = {line.get_label() for line in figure.axes[0].lines}
    assert {"Independent", "Coupled", "Forecast origin"} <= labels
    plt.close(figure)
