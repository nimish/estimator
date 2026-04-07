#!/usr/bin/env python3
# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tidal modeling helpers used by scripts and notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsgam_estimator import TsgamSplineConfig

_VAR_LAG_HOURS: dict[str, list[int]] = {
    "pressure": [-2, -1, 0, 1],
    "water_temp": [0],
    "wind_u": [-1, 0, 1],
    "wind_v": [-1, 0, 1],
    "air_temp": [0],
}

_VAR_N_KNOTS: dict[str, int] = {
    "pressure": 10,
    "water_temp": 8,
    "wind_u": 8,
    "wind_v": 8,
    "air_temp": 8,
}


def make_tidal_spline_configs(
    samples_per_hour: int,
) -> tuple[dict[str, TsgamSplineConfig], TsgamSplineConfig]:
    """Build tidal spline configs with lags scaled to the sample grid."""
    configs = {}
    for variable, lag_hours in _VAR_LAG_HOURS.items():
        configs[variable] = TsgamSplineConfig(
            n_knots=_VAR_N_KNOTS.get(variable, 8),
            lags=[hour * samples_per_hour for hour in lag_hours],
            reg_weight=1e-5,
            diff_reg_weight=0.3,
        )
    default = TsgamSplineConfig(
        n_knots=8,
        lags=[0],
        reg_weight=1e-5,
        diff_reg_weight=0.3,
    )
    return configs, default


def tidal_metrics(y_true, y_pred):
    """Compute RMSE, MAE, MAPE, and R2 for tidal water levels."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}
    y, p = y_true[valid], y_pred[valid]
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    pos = np.abs(y) > 0.01
    mape = (
        float(np.mean(np.abs((p[pos] - y[pos]) / y[pos])) * 100)
        if np.any(pos)
        else np.nan
    )
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def usable_lcd_columns(
    weather_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    candidate_columns: list[str],
    min_coverage: float = 0.05,
) -> list[str]:
    """Return LCD-provided columns with usable overlap in the merged frame."""
    return [
        column
        for column in candidate_columns
        if column in weather_df.columns
        and column in merged_df.columns
        and merged_df[column].notna().mean() > min_coverage
    ]


def prepare_split_regressors(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    candidate_columns: list[str],
    min_raw_coverage: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Interpolate and standardize regressors within each split only."""
    candidate_columns = list(dict.fromkeys(candidate_columns))
    active_columns: list[str] = []
    dropped_columns: list[str] = []
    train_data: dict[str, pd.Series] = {}
    test_data: dict[str, pd.Series] = {}

    for column in candidate_columns:
        if column not in df_train.columns or column not in df_test.columns:
            dropped_columns.append(column)
            continue

        train_series = df_train[column].copy()
        test_series = df_test[column].copy()
        if train_series.notna().mean() < min_raw_coverage:
            dropped_columns.append(column)
            continue
        if test_series.notna().mean() < min_raw_coverage:
            dropped_columns.append(column)
            continue

        train_series = train_series.interpolate(limit_direction="both").ffill().bfill()
        test_series = test_series.interpolate(limit_direction="both").ffill().bfill()
        if train_series.isna().any() or test_series.isna().any():
            dropped_columns.append(column)
            continue
        if float(train_series.std()) < 1.0e-8:
            dropped_columns.append(column)
            continue

        train_data[column] = train_series
        test_data[column] = test_series
        active_columns.append(column)

    if not active_columns:
        return (
            pd.DataFrame(index=df_train.index),
            pd.DataFrame(index=df_test.index),
            [],
            dropped_columns,
        )

    X_train = pd.DataFrame(train_data, index=df_train.index)
    X_test = pd.DataFrame(test_data, index=df_test.index)
    reg_means = X_train.mean()
    reg_stds = X_train.std().replace(0, 1).fillna(1)
    X_train = (X_train - reg_means) / reg_stds
    X_test = (X_test - reg_means) / reg_stds
    return X_train, X_test, active_columns, dropped_columns
