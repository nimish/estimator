# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, TypeGuard, cast, overload

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import spdiags, spmatrix
from sklearn.base import check_array
from sklearn.utils import check_X_y
from spcqe import make_basis_matrix
from spcqe.functions import initialize_arrays

if TYPE_CHECKING:
    from ._estimator import (
        TsgamEstimatorConfig,
        TsgamLinearConfig,
        TsgamSplineConfig,
    )


def _is_spline_config(exog_cfg: object) -> TypeGuard[TsgamSplineConfig]:
    from ._estimator import TsgamSplineConfig

    return isinstance(exog_cfg, TsgamSplineConfig)


def _to_pandas_timedelta_frequency(freq: str) -> str:
    """Return a frequency string accepted by ``pd.to_timedelta`` without deprecation warnings."""
    if freq.lower().endswith("d"):
        return f"{freq[:-1]}D"
    return freq


@dataclass
class _TsgamDesign:
    """Internal design matrices for one TSGAM regression target."""
    timestamps: pd.DatetimeIndex
    X_array: ndarray
    y: ndarray | None
    sample_weight: ndarray | None
    time_indices: ndarray
    valid_mask: ndarray
    exog_Hs: list[list[ndarray]]
    interaction_Hs: list[ndarray]
    interaction_pairs: list[tuple[int, int]]
    fourier_basis: ndarray | None


def _extract_timestamps(X: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract timestamps from supported DataFrame layouts."""
    if isinstance(X, pd.DataFrame):
        if isinstance(X.index, pd.DatetimeIndex):
            return X.index
        if len(X.columns) > 0 and pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
            return pd.DatetimeIndex(X.iloc[:, 0])
        raise ValueError(
            "X must have DatetimeIndex or first column must be datetime. "
            "Got DataFrame without datetime index or datetime column."
        )
    raise ValueError(
        "X must be a pandas DataFrame with DatetimeIndex or datetime column. "
        f"Got {type(X)} instead."
    )


def _ensure_timestamp_index(X: pd.DataFrame) -> tuple[pd.DatetimeIndex, ndarray]:
    """Return timestamps and the numeric X array without any timestamp column."""
    timestamps = _extract_timestamps(X)
    if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.DatetimeIndex):
        if pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
            return timestamps, X.iloc[:, 1:].values
        return timestamps, X.values
    return timestamps, X.values if isinstance(X, pd.DataFrame) else X


def normalize_X(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported timestamp layouts to a DatetimeIndex frame."""
    timestamps, X_array = _ensure_timestamp_index(X)
    if isinstance(X.index, pd.DatetimeIndex):
        columns = list(X.columns)
    else:
        columns = list(X.columns[1:])
    return pd.DataFrame(X_array, index=timestamps, columns=columns)


def _ensure_numeric_prefix(freq: str) -> str:
    """Ensure frequency string has a numeric prefix (e.g. ``'h'`` -> ``'1h'``)."""
    if freq and not freq[0].isdigit():
        return f"1{freq}"
    return freq


def _infer_frequency_from_differences(timestamps: pd.DatetimeIndex) -> str:
    """Infer the intended frequency from the most common timestamp difference."""
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to infer frequency")
    diffs = timestamps[1:] - timestamps[:-1]
    diff_seconds = np.array([d.total_seconds() for d in diffs])
    diff_seconds_rounded = np.round(diff_seconds).astype(int)
    unique_diffs, counts = np.unique(diff_seconds_rounded, return_counts=True)
    most_common_diff_seconds = unique_diffs[np.argmax(counts)]
    freq_mapping = {
        60: "1min",
        300: "5min",
        900: "15min",
        3600: "1h",
        86400: "1D",
    }
    if most_common_diff_seconds in freq_mapping:
        return freq_mapping[most_common_diff_seconds]
    for diff_sec, freq_str in freq_mapping.items():
        if abs(most_common_diff_seconds - diff_sec) / diff_sec < 0.01:
            return freq_str
    if most_common_diff_seconds < 60:
        return f"{most_common_diff_seconds}S"
    if most_common_diff_seconds < 3600:
        return f"{most_common_diff_seconds // 60}min"
    if most_common_diff_seconds < 86400:
        return f"{most_common_diff_seconds // 3600}h"
    return f"{most_common_diff_seconds // 86400}D"


def _validate_frequency(
    timestamps: pd.DatetimeIndex,
    expected_freq: str,
    allow_gaps: bool = False,
) -> None:
    """Validate timestamp frequency, optionally allowing gaps."""
    if len(timestamps) < 2:
        return
    inferred_freq = pd.infer_freq(timestamps)
    if inferred_freq is None:
        if allow_gaps:
            inferred_freq = _infer_frequency_from_differences(timestamps)
        else:
            raise ValueError(
                f"Could not infer frequency from timestamps. "
                f"Timestamps must be regularly spaced with frequency '{expected_freq}'."
            )
    normalized_inferred = _ensure_numeric_prefix(inferred_freq).lower()
    normalized_expected = _ensure_numeric_prefix(expected_freq).lower()
    if normalized_inferred == normalized_expected:
        return
    if allow_gaps:
        try:
            inferred_step = pd.Timedelta(pd.tseries.frequencies.to_offset(inferred_freq)).total_seconds()
            expected_step = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_freq)).total_seconds()
        except ValueError:
            inferred_step = expected_step = None
        if (
            inferred_step is not None
            and expected_step is not None
            and inferred_step >= expected_step
            and np.isclose(inferred_step % expected_step, 0.0)
        ):
            return
    raise ValueError(
        f"Timestamps frequency '{inferred_freq}' does not match "
        f"expected frequency '{expected_freq}'."
    )


def _timestamps_to_indices(
    timestamps: pd.DatetimeIndex,
    reference: pd.Timestamp,
    freq: str | None = None,
) -> ndarray:
    """Convert timestamps to integer sample offsets from a reference timestamp."""
    if freq is None:
        freq = pd.infer_freq(timestamps)
        if freq is None:
            freq = _infer_frequency_from_differences(timestamps)
    freq = _ensure_numeric_prefix(freq)
    return (
        (timestamps - reference)
        / pd.to_timedelta(_to_pandas_timedelta_frequency(freq))
    ).astype(int)


@overload
def _ensure_sorted_index(
    X: pd.DataFrame,
    *,
    sort_index: bool,
    y: ndarray,
    sample_weight: ndarray | None = None,
) -> tuple[pd.DataFrame, ndarray, ndarray | None]: ...


@overload
def _ensure_sorted_index(
    X: pd.DataFrame,
    *,
    sort_index: bool,
    y: None = None,
    sample_weight: None = None,
) -> tuple[pd.DataFrame]: ...


def _ensure_sorted_index(
    X: pd.DataFrame,
    *,
    sort_index: bool,
    y: ndarray | None = None,
    sample_weight: ndarray | None = None,
) -> tuple[pd.DataFrame, ndarray, ndarray | None] | tuple[pd.DataFrame]:
    """Sort or validate a timestamp-indexed input frame and aligned arrays."""
    timestamps = _extract_timestamps(X)
    if sort_index:
        sort_idx = np.argsort(timestamps)
        X = X.iloc[sort_idx]
        if y is not None:
            y = np.asarray(y)[sort_idx]
            sw = (
                np.asarray(sample_weight)[sort_idx]
                if sample_weight is not None
                else None
            )
            return X, y, sw
        return (X,)
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Data index is not sorted chronologically. Sort the DataFrame by "
            "its datetime index (e.g. X = X.sort_index()) or set "
            "config.sort_index=True to sort automatically."
        )
    if y is not None:
        sw = np.asarray(sample_weight) if sample_weight is not None else None
        return X, y, sw
    return (X,)


def sort_fit_inputs(
    X: pd.DataFrame,
    *,
    sort_index: bool,
    y: ndarray,
    sample_weight: ndarray | None,
) -> tuple[pd.DataFrame, ndarray, ndarray | None]:
    return _ensure_sorted_index(
        X,
        sort_index=sort_index,
        y=y,
        sample_weight=sample_weight,
    )


def sort_predict_X(X: pd.DataFrame, *, sort_index: bool) -> pd.DataFrame:
    (X,) = _ensure_sorted_index(X, sort_index=sort_index)
    return X


def _make_regularization_matrix(
    num_harmonics: list[int],
    weight: float,
    periods: list[float],
    drop_constant: bool = False,
    standing_wave: bool | list[bool] = False,
    trend: bool = False,
    max_cross_k: int | None = None,
    custom_basis: dict[int, ndarray] | None = None,
) -> spmatrix:
    """Create the Fourier coefficient regularization matrix."""
    sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
        num_harmonics, periods, standing_wave, custom_basis
    )
    ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]
    i_value_list = []
    for ix, nh in enumerate(num_harmonics):
        if standing_wave[ix]:
            i_value_list.append(np.arange(1, nh + 1))
        else:
            i_value_list.append(np.repeat(np.arange(1, nh + 1), 2))
    blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
    if custom_basis is not None:
        for ix, val in custom_basis.items():
            ixt = np.where(sort_idx == ix)[0][0]
            blocks_original[ixt] = ls_original[ixt] * np.arange(1, val.shape[1] + 1)
    if max_cross_k is not None:
        max_cross_k *= 2
    blocks_cross = [
        [l2 for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]]
        for c in combinations(blocks_original, 2)
    ]
    first_block = [np.zeros(1)] if trend is False else [np.zeros(2)]
    if drop_constant:
        first_block = first_block[1:]
    coeff_i = np.concatenate(first_block + blocks_original + blocks_cross)
    return spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)


def _make_spline_H(x: ndarray, knots: ndarray, include_offset: bool = False) -> ndarray:
    """Create a natural cubic spline basis matrix."""
    def d_func(x: ndarray, k: float, k_max: float) -> ndarray:
        n1 = np.clip(np.power(x - k, 3), 0, np.inf)
        n2 = np.clip(np.power(x - k_max, 3), 0, np.inf)
        return (n1 - n2) / (k_max - k)

    nK = len(knots)
    H = np.ones((len(x), nK), dtype=float)
    H[:, 1] = x
    for _i in range(nK - 2):
        _j = _i + 2
        H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(
            x, knots[-2], knots[-1]
        )
    return H if include_offset else H[:, 1:]


def _make_offset_H(H: ndarray, offset: int) -> ndarray:
    """Create a lead/lag version of a basis matrix."""
    newH = np.roll(np.copy(H), -offset, axis=0)
    if offset > 0:
        newH[-offset:] = np.nan
    elif offset < 0:
        newH[:-offset] = np.nan
    return newH


def _build_exog_Hs(
    exog_cfg: TsgamSplineConfig | TsgamLinearConfig,
    exog_var: ndarray,
    knots: ndarray | None = None,
) -> list[ndarray]:
    """Build basis matrices for one exogenous variable across configured lags."""
    Hs = []
    for lag in exog_cfg.lags:
        if _is_spline_config(exog_cfg):
            if knots is None:
                raise ValueError("knots must be provided for TsgamSplineConfig")
            H0 = _make_spline_H(exog_var, knots, include_offset=False)
        else:
            H0 = exog_var.reshape(-1, 1)
        Hs.append(_make_offset_H(H0, lag))
    return Hs


def _process_exog_config(
    exog_cfg: TsgamSplineConfig | TsgamLinearConfig,
    exog_var: ndarray,
    knots: ndarray | None = None,
) -> tuple[ndarray, list[ndarray]]:
    """Build exogenous lag matrices and the boundary-valid sample mask."""
    if knots is None and _is_spline_config(exog_cfg):
        cfg_knots = np.asarray(exog_cfg.knots) if exog_cfg.knots is not None else np.array([])
        if len(cfg_knots) == 0:
            if exog_cfg.n_knots:
                knots = np.linspace(np.min(exog_var), np.max(exog_var), exog_cfg.n_knots)
            else:
                raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
        else:
            knots = cfg_knots
    if knots is not None:
        knots = np.asarray(knots)
    Hs = _build_exog_Hs(exog_cfg, exog_var, knots)
    valid_mask = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)
    return valid_mask, Hs


def _get_zero_lag_H(
    exog_cfg: TsgamSplineConfig | TsgamLinearConfig,
    Hs: list[ndarray],
) -> ndarray:
    """Return the current-index basis block for an exogenous term."""
    try:
        zero_lag_ix = exog_cfg.lags.index(0)
    except ValueError as exc:
        raise ValueError(
            "Interaction pairs require each referenced exogenous factor to include lag=0."
        ) from exc
    return Hs[zero_lag_ix]


def _outer_column_product(arr1: ndarray, arr2: ndarray) -> ndarray:
    """Build a q*r interaction design block from two response matrices."""
    return (arr1[:, :, None] * arr2[:, None, :]).reshape(arr1.shape[0], -1)


def _interaction_contribution_from_blocks(
    arr1: ndarray,
    arr2: ndarray,
    interaction_coef: ndarray,
    *,
    nan_to_zero: bool = False,
) -> ndarray:
    """Contract two response matrices against flattened interaction coefficients."""
    if nan_to_zero:
        arr1 = np.nan_to_num(arr1, nan=0.0)
        arr2 = np.nan_to_num(arr2, nan=0.0)
    coef_matrix = interaction_coef.reshape(arr1.shape[1], arr2.shape[1])
    return np.sum((arr1 @ coef_matrix) * arr2, axis=1)


def _normalize_interaction_pairs(config: TsgamEstimatorConfig) -> list[tuple[int, int]]:
    """Validate and normalize configured exogenous interaction pairs."""
    if not config.interaction_pairs:
        return []
    if not config.exog_config:
        raise ValueError("interaction_pairs requires a non-empty exog_config.")
    normalized_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    n_exog = len(config.exog_config)
    for pair in config.interaction_pairs:
        try:
            left_ix, right_ix = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each interaction pair must contain exactly two exogenous indices."
            ) from exc
        left_is_integer_index = (
            isinstance(left_ix, (int, np.integer)) and not isinstance(left_ix, bool)
        )
        right_is_integer_index = (
            isinstance(right_ix, (int, np.integer)) and not isinstance(right_ix, bool)
        )
        if not left_is_integer_index or not right_is_integer_index:
            raise ValueError("Each interaction pair index must be an integer exogenous index.")
        if left_ix == right_ix:
            raise ValueError("interaction_pairs cannot contain self-pairs.")
        if not 0 <= left_ix < n_exog or not 0 <= right_ix < n_exog:
            raise ValueError(
                f"interaction pair {(left_ix, right_ix)} is out of range for "
                f"{n_exog} exogenous terms."
            )
        normalized_pair = (left_ix, right_ix)
        if right_ix < left_ix:
            normalized_pair = (right_ix, left_ix)
        if normalized_pair in seen_pairs:
            raise ValueError(f"Duplicate interaction pair detected: {normalized_pair}.")
        for exog_ix in normalized_pair:
            if 0 not in config.exog_config[exog_ix].lags:
                raise ValueError(
                    "Interaction pairs require each referenced exogenous factor "
                    "to include lag=0."
                )
        seen_pairs.add(normalized_pair)
        normalized_pairs.append(normalized_pair)
    return normalized_pairs


def _min_samples_required(config: TsgamEstimatorConfig) -> int:
    """Calculate minimum samples required by exogenous and AR lags."""
    all_exog_lags = []
    for exog_cfg in config.exog_config or []:
        all_exog_lags.extend(exog_cfg.lags)
    max_positive_lag = 0
    min_negative_lag = 0
    for lag in all_exog_lags:
        if lag > 0:
            max_positive_lag = max(max_positive_lag, lag)
        elif lag < 0:
            min_negative_lag = min(min_negative_lag, lag)
    max_ar_lag = 0
    if config.ar_config is not None:
        max_ar_lag = max(config.ar_config.lags) if config.ar_config.lags else 0
    return max(max_positive_lag, max_ar_lag) + abs(min_negative_lag) + 1


def _resolve_exog_knots(
    config: TsgamEstimatorConfig,
    X_array: ndarray,
) -> list[ndarray | None]:
    """Resolve shared spline knots from a fitting matrix."""
    if not config.exog_config:
        return []
    knots_by_exog: list[ndarray | None] = []
    for ix, exog_cfg in enumerate(config.exog_config):
        if _is_spline_config(exog_cfg):
            cfg_knots = np.asarray(exog_cfg.knots) if exog_cfg.knots is not None else np.array([])
            if len(cfg_knots) == 0:
                if not exog_cfg.n_knots:
                    raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                knots_by_exog.append(
                    np.linspace(np.min(X_array[:, ix]), np.max(X_array[:, ix]), exog_cfg.n_knots)
                )
            else:
                knots_by_exog.append(None)
        else:
            knots_by_exog.append(None)
    return knots_by_exog


def resolve_exog_knots(
    config: TsgamEstimatorConfig,
    X: pd.DataFrame,
) -> list[ndarray | None]:
    if not config.exog_config:
        return []
    _, X_array = _ensure_timestamp_index(X)
    return _resolve_exog_knots(config, X_array)


def _make_fourier_basis(
    config: TsgamEstimatorConfig,
    time_indices: ndarray,
) -> ndarray | None:
    if config.multi_periodic_config is None:
        return None
    max_idx = int(np.max(time_indices))
    min_idx = int(np.min(time_indices))
    if min_idx < 0:
        offset = -min_idx
        adjusted_indices = time_indices.astype(int) + offset
        basis_length = max_idx + offset + 1
    else:
        adjusted_indices = time_indices.astype(int)
        basis_length = max_idx + 1
    if np.any(adjusted_indices < 0) or np.any(adjusted_indices >= basis_length):
        raise ValueError(
            f"Adjusted indices out of bounds: min={adjusted_indices.min()}, "
            f"max={adjusted_indices.max()}, basis_length={basis_length}"
        )
    F_full = make_basis_matrix(
        num_harmonics=config.multi_periodic_config.num_harmonics,
        length=basis_length,
        periods=config.multi_periodic_config.periods,
    )
    if np.any(np.isnan(F_full)):
        raise ValueError(
            f"Basis matrix contains NaN. basis_length={basis_length}, "
            f"F_full shape: {F_full.shape}, "
            f"time_indices range: [{min_idx}, {max_idx}]"
        )
    fourier_basis = F_full[adjusted_indices, 1:]
    if np.any(np.isnan(fourier_basis)):
        raise ValueError(
            f"Indexed basis matrix F contains NaN. "
            f"F shape: {fourier_basis.shape}, "
            f"adjusted_indices range: [{adjusted_indices.min()}, {adjusted_indices.max()}]"
        )
    return fourier_basis


def _build_tsgam_design(
    config: TsgamEstimatorConfig,
    X: pd.DataFrame,
    *,
    y: ndarray | None = None,
    sample_weight: ndarray | None = None,
    knots_by_exog: list[ndarray | None] | None = None,
    reference: pd.Timestamp | None = None,
    freq: str | None = None,
) -> _TsgamDesign:
    """Build shared TSGAM design matrices for fit or predict paths."""
    timestamps, X_array = _ensure_timestamp_index(X)
    if freq is None:
        freq = pd.infer_freq(timestamps)
        if freq is None:
            freq = _infer_frequency_from_differences(timestamps)
    freq = _ensure_numeric_prefix(freq).lower()
    if reference is None:
        reference = timestamps[0]
    time_indices = _timestamps_to_indices(timestamps, reference, freq)
    if y is None:
        X_array = check_array(X_array, ensure_min_features=len(config.exog_config or []))
        y_array = None
        valid_mask = np.ones(len(X_array), dtype=bool)
    else:
        X_array, y_array = check_X_y(
            X_array,
            y,
            ensure_min_features=len(config.exog_config or []),
            ensure_min_samples=_min_samples_required(config),
        )
        valid_mask = ~np.isnan(y_array)
        y_array = cast(ndarray, y_array)
    if knots_by_exog is None:
        knots_by_exog = _resolve_exog_knots(config, X_array)
    exog_Hs: list[list[ndarray]] = []
    interaction_Hs: list[ndarray] = []
    zero_lag_Hs: dict[int, ndarray] = {}
    interaction_pairs = _normalize_interaction_pairs(config)
    interaction_parent_indices = {ix for pair in interaction_pairs for ix in pair}
    if config.exog_config:
        for ix, exog_cfg in enumerate(config.exog_config):
            stored_knots = knots_by_exog[ix] if _is_spline_config(exog_cfg) else None
            exog_valid_mask, Hs = _process_exog_config(
                exog_cfg, X_array[:, ix], knots=stored_knots
            )
            exog_Hs.append(Hs)
            if y is not None:
                valid_mask &= exog_valid_mask
            if ix in interaction_parent_indices:
                zero_lag_Hs[ix] = _get_zero_lag_H(exog_cfg, Hs)
        for left_ix, right_ix in interaction_pairs:
            interaction_H = _outer_column_product(
                zero_lag_Hs[left_ix],
                zero_lag_Hs[right_ix],
            )
            interaction_Hs.append(interaction_H)
            if y is not None:
                valid_mask &= np.all(~np.isnan(interaction_H), axis=1)
    fourier_basis = _make_fourier_basis(config, time_indices)
    weights = None
    if y_array is not None:
        if sample_weight is None:
            weights = np.ones(len(y_array), dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(y_array),):
                raise ValueError(
                    f"sample_weight must have shape (n_samples,) = ({len(y_array)},), got {weights.shape}"
                )
            if np.any(weights < 0):
                raise ValueError("sample_weight must be non-negative")
            if np.sum(weights) <= 0:
                raise ValueError("sample_weight must have positive sum")
    return _TsgamDesign(
        timestamps=timestamps,
        X_array=X_array,
        y=y_array,
        sample_weight=weights,
        time_indices=time_indices,
        valid_mask=valid_mask,
        exog_Hs=exog_Hs,
        interaction_Hs=interaction_Hs,
        interaction_pairs=interaction_pairs,
        fourier_basis=fourier_basis,
    )


def build_tsgam_design(
    config: TsgamEstimatorConfig,
    X: pd.DataFrame,
    y: ndarray | None = None,
    sample_weight: ndarray | None = None,
    *,
    knots_by_exog: list[ndarray | None],
    reference: pd.Timestamp,
    freq: str,
) -> _TsgamDesign:
    return _build_tsgam_design(
        config,
        X,
        y=y,
        sample_weight=sample_weight,
        knots_by_exog=knots_by_exog,
        reference=reference,
        freq=freq,
    )


def infer_fit_frequency(timestamps: pd.DatetimeIndex) -> str:
    inferred_freq = pd.infer_freq(timestamps)
    if inferred_freq is None:
        inferred_freq = _infer_frequency_from_differences(timestamps)
    _validate_frequency(timestamps, inferred_freq, allow_gaps=True)
    return _ensure_numeric_prefix(inferred_freq).lower()


def validate_predict_frequency(
    timestamps: pd.DatetimeIndex,
    expected_freq: str,
) -> None:
    _validate_frequency(timestamps, expected_freq, allow_gaps=False)


def step_timedelta(freq: str) -> pd.Timedelta:
    return pd.to_timedelta(_to_pandas_timedelta_frequency(freq))
