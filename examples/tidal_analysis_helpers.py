#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable helpers for the tidal analysis notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from spcqe import make_basis_matrix
from spcqe.functions import cross_bases, initialize_arrays


def infer_samples_per_hour(index: pd.DatetimeIndex) -> int:
    """Infer samples per hour from a regular DatetimeIndex."""
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer sampling rate.")

    freq_td = index.to_series().diff().median()
    if pd.isna(freq_td) or freq_td <= pd.Timedelta(0):
        raise ValueError("Could not infer a positive sampling interval.")

    return max(1, round(pd.Timedelta("1h") / freq_td))


def build_day_hour_matrix(
    index: pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Aggregate a time series into a day x hour-of-day matrix."""
    if len(index) != len(values):
        raise ValueError("index and values must have the same length.")

    frame = pd.DataFrame(
        {"value": np.asarray(values, dtype=float)},
        index=pd.DatetimeIndex(index),
    )
    frame = frame[np.isfinite(frame["value"])]

    grouped = frame.groupby(
        [frame.index.normalize(), frame.index.hour],
    )["value"].mean()
    matrix = grouped.unstack()

    return matrix.reindex(columns=range(24))


def compute_periodogram(
    index: pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
    min_period_hours: float = 2.0,
    max_period_hours: float | None = None,
) -> pd.DataFrame:
    """Compute a simple FFT periodogram on a regularly sampled series."""
    series = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(index))
    series = series.interpolate(limit_direction="both").ffill().bfill()

    dt_hours = series.index.to_series().diff().median() / pd.Timedelta("1h")
    if pd.isna(dt_hours) or dt_hours <= 0:
        raise ValueError("Could not infer a positive sampling interval.")

    centered = series.to_numpy() - series.mean()
    freqs = np.fft.rfftfreq(len(centered), d=float(dt_hours))
    power = np.abs(np.fft.rfft(centered)) ** 2

    mask = freqs > 0
    spectrum = pd.DataFrame(
        {
            "frequency_per_hour": freqs[mask],
            "power": power[mask],
        },
    )
    spectrum["period_hours"] = 1.0 / spectrum["frequency_per_hour"]

    spectrum = spectrum[spectrum["period_hours"] >= min_period_hours]
    if max_period_hours is not None:
        spectrum = spectrum[spectrum["period_hours"] <= max_period_hours]

    return spectrum.sort_values("period_hours").reset_index(drop=True)


def compute_lagged_correlation(
    target: np.ndarray | pd.Series,
    feature: np.ndarray | pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """
    Correlate target with lagged versions of a feature.

    Positive lags mean the feature leads the target by that many samples.
    """
    y = np.asarray(target, dtype=float)
    x = np.asarray(feature, dtype=float)
    if len(y) != len(x):
        raise ValueError("target and feature must have the same length.")

    rows: list[dict[str, float | int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            y_lag = y[lag:]
            x_lag = x[:-lag]
        elif lag < 0:
            y_lag = y[:lag]
            x_lag = x[-lag:]
        else:
            y_lag = y
            x_lag = x

        valid = np.isfinite(y_lag) & np.isfinite(x_lag)
        if valid.sum() < 2:
            corr = np.nan
        else:
            corr = float(np.corrcoef(y_lag[valid], x_lag[valid])[0, 1])
        rows.append({"lag": lag, "correlation": corr})

    return pd.DataFrame(rows)


def extract_fourier_components(
    estimator: object,
    labels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct per-period Fourier contributions from a fitted estimator."""
    multi_periodic = estimator.config.multi_periodic_config
    if multi_periodic is None:
        raise ValueError("Estimator does not have a multi-periodic configuration.")

    fourier_coef = estimator.variables_["fourier_coef"].value
    if fourier_coef is None:
        raise ValueError("Estimator does not have fitted Fourier coefficients.")

    sort_idx, sorted_periods, sorted_harmonics, _ = initialize_arrays(
        multi_periodic.num_harmonics,
        multi_periodic.periods,
        False,
        None,
    )
    if labels is None:
        sorted_labels = [f"period_{idx + 1}" for idx in range(len(sorted_periods))]
    else:
        if len(labels) != len(sorted_periods):
            raise ValueError("labels must match the number of configured periods.")
        sorted_labels = [labels[idx] for idx in sort_idx]

    time_indices = np.asarray(estimator.time_indices_, dtype=int)
    max_idx = int(time_indices.max())

    components: dict[str, np.ndarray] = {}
    period_bases: list[np.ndarray] = []
    coef_idx = 0
    for label, period, n_harmonics in zip(
        sorted_labels,
        sorted_periods,
        sorted_harmonics,
        strict=True,
    ):
        n_coef = 2 * n_harmonics
        period_coef = fourier_coef[coef_idx:coef_idx + n_coef]
        period_basis = make_basis_matrix(
            num_harmonics=[n_harmonics],
            length=max_idx + 1,
            periods=[period],
        )[time_indices, 1:]
        components[label] = period_basis @ period_coef
        period_bases.append(period_basis)
        coef_idx += n_coef

    component_names = list(components.keys())
    for left_idx, left_basis in enumerate(period_bases):
        for right_idx in range(left_idx + 1, len(period_bases)):
            right_basis = period_bases[right_idx]
            cross_basis = cross_bases(left_basis, right_basis)
            n_coef = cross_basis.shape[1]
            cross_coef = fourier_coef[coef_idx:coef_idx + n_coef]
            cross_label = f"{component_names[left_idx]}_x_{component_names[right_idx]}"
            components[cross_label] = cross_basis @ cross_coef
            coef_idx += n_coef

    components["combined"] = np.sum(list(components.values()), axis=0)
    return components
