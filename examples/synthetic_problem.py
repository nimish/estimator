from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from tsgam_estimator import (
    TrendType,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
)


class SyntheticRegressorRelationship(StrEnum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class SyntheticTrendKind(StrEnum):
    NONE = "none"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


@dataclass(frozen=True)
class SyntheticPeriodicComponent:
    name: str
    period_hours: float
    harmonics: int = 1
    amplitude: float = 1.0
    phase: float = 0.0
    harmonic_decay: float = 1.0
    phase_step: float = 0.35


@dataclass(frozen=True)
class SyntheticRegressorSpec:
    name: str
    relationship: SyntheticRegressorRelationship = SyntheticRegressorRelationship.LINEAR
    effect_scale: float = 1.0
    driver_period_hours: float = 24.0
    driver_harmonics: int = 2
    driver_noise_scale: float = 0.2
    phase: float = 0.2
    lags: tuple[int, ...] = (0,)
    n_knots: int = 7
    harmonic_decay: float = 1.0
    nonlinear_scale: float = 1.25


@dataclass(frozen=True)
class SyntheticTrendSpec:
    kind: SyntheticTrendKind = SyntheticTrendKind.NONE
    amplitude: float = 0.0
    grouping_hours: float = 24.0


@dataclass(frozen=True)
class SyntheticProblemConfig:
    start: str = "2024-01-01"
    n_samples: int = 24 * 30
    freq: str = "1h"
    train_fraction: float = 0.75
    seed: int = 0
    noise_scale: float = 0.1
    periodic_components: tuple[SyntheticPeriodicComponent, ...] = field(
        default_factory=lambda: (
            SyntheticPeriodicComponent(
                name="daily",
                period_hours=24.0,
                harmonics=2,
                amplitude=1.0,
            ),
        )
    )
    regressors: tuple[SyntheticRegressorSpec, ...] = field(default_factory=tuple)
    trend: SyntheticTrendSpec = field(default_factory=SyntheticTrendSpec)


@dataclass(frozen=True)
class SyntheticProblemResult:
    config: SyntheticProblemConfig
    X: pd.DataFrame
    y: pd.Series
    truth_components: pd.DataFrame
    signal: pd.Series
    noise: pd.Series


@dataclass(frozen=True)
class SyntheticProblemSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _freq_timedelta(freq: str) -> pd.Timedelta:
    try:
        freq_delta = pd.to_timedelta(freq)
    except ValueError as exc:
        raise ValueError(f"Unsupported frequency string: {freq}") from exc
    if freq_delta <= pd.Timedelta(0):
        raise ValueError(f"Frequency must be positive: {freq}")
    return freq_delta


def samples_per_hour(freq: str) -> float:
    return pd.Timedelta("1h") / _freq_timedelta(freq)


def sample_hours(freq: str) -> float:
    return _freq_timedelta(freq) / pd.Timedelta("1h")


def grouping_samples(grouping_hours: float, freq: str) -> float:
    grouping = grouping_hours * samples_per_hour(freq)
    return float(max(1, int(round(grouping))))


def _time_hours(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=float)
    deltas = index - index[0]
    return deltas / pd.Timedelta("1h")


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    scale = np.std(centered, ddof=0)
    if np.isclose(scale, 0.0):
        return np.zeros_like(centered)
    return centered / scale


def _shift_with_nan(values: np.ndarray, offset: int) -> np.ndarray:
    shifted = np.roll(values.copy(), -offset)
    if offset > 0:
        shifted[-offset:] = np.nan
    elif offset < 0:
        shifted[:-offset] = np.nan
    return shifted


def _lag_weights(lags: tuple[int, ...]) -> np.ndarray:
    weights = np.array([1.0 / (abs(lag) + 1.0) for lag in lags], dtype=float)
    return weights / weights.sum()


def _periodic_component_series(
    index: pd.DatetimeIndex,
    component: SyntheticPeriodicComponent,
) -> pd.Series:
    time_hours = _time_hours(index)
    values = np.zeros(len(index), dtype=float)
    for harmonic in range(1, component.harmonics + 1):
        amplitude = component.amplitude / (harmonic ** component.harmonic_decay)
        phase = component.phase + harmonic * component.phase_step
        values += amplitude * np.sin(
            (2.0 * np.pi * harmonic * time_hours / component.period_hours) + phase
        )
    return pd.Series(values, index=index, name=f"periodic:{component.name}")


def _regressor_driver(
    index: pd.DatetimeIndex,
    spec: SyntheticRegressorSpec,
    rng: np.random.Generator,
) -> pd.Series:
    time_hours = _time_hours(index)
    driver = np.zeros(len(index), dtype=float)
    for harmonic in range(1, spec.driver_harmonics + 1):
        amplitude = 1.0 / (harmonic ** spec.harmonic_decay)
        phase = spec.phase + 0.4 * harmonic
        driver += amplitude * np.cos(
            (2.0 * np.pi * harmonic * time_hours / spec.driver_period_hours) + phase
        )
    driver += spec.driver_noise_scale * rng.standard_normal(len(index))
    standardized = _standardize(driver)
    return pd.Series(standardized, index=index, name=spec.name)


def _regressor_effect(values: np.ndarray, spec: SyntheticRegressorSpec) -> np.ndarray:
    lag_weights = _lag_weights(spec.lags)
    effect = np.zeros_like(values)
    for lag_weight, lag in zip(lag_weights, spec.lags, strict=True):
        lagged = np.nan_to_num(_shift_with_nan(values, lag), nan=0.0)
        if spec.relationship == SyntheticRegressorRelationship.LINEAR:
            effect += lag_weight * spec.effect_scale * lagged
        else:
            effect += lag_weight * spec.effect_scale * np.tanh(
                spec.nonlinear_scale * lagged
            )
    return effect


def _trend_series(
    index: pd.DatetimeIndex,
    trend: SyntheticTrendSpec,
    freq: str,
) -> pd.Series:
    if trend.kind == SyntheticTrendKind.NONE or np.isclose(trend.amplitude, 0.0):
        values = np.zeros(len(index), dtype=float)
        return pd.Series(values, index=index, name="trend")

    time_hours = _time_hours(index)
    total_hours = max(sample_hours(freq), time_hours[-1] + sample_hours(freq))
    position = np.clip(time_hours / total_hours, 0.0, 1.0)
    if trend.kind == SyntheticTrendKind.LINEAR:
        continuous = trend.amplitude * (position - 0.5)
    else:
        continuous = trend.amplitude * (0.5 - np.sqrt(position))

    group_size = int(grouping_samples(trend.grouping_hours, freq))
    group_ids = np.arange(len(index)) // group_size
    grouped = pd.Series(continuous).groupby(group_ids).transform("mean").to_numpy()
    return pd.Series(grouped, index=index, name="trend")


def generate_synthetic_problem(config: SyntheticProblemConfig) -> SyntheticProblemResult:
    if config.n_samples < 8:
        raise ValueError("Synthetic problems require at least 8 samples.")
    if not 0.1 < config.train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.1 and 0.9.")

    index = pd.date_range(config.start, periods=config.n_samples, freq=config.freq)
    rng = np.random.default_rng(config.seed)

    X_columns: dict[str, pd.Series] = {}
    component_columns: dict[str, pd.Series] = {}

    for component in config.periodic_components:
        series = _periodic_component_series(index, component)
        component_columns[series.name] = series

    for spec in config.regressors:
        driver = _regressor_driver(index, spec, rng)
        X_columns[spec.name] = driver
        effect = _regressor_effect(driver.to_numpy(), spec)
        component_columns[f"regressor:{spec.name}"] = pd.Series(
            effect,
            index=index,
            name=f"regressor:{spec.name}",
        )

    trend_series = _trend_series(index, config.trend, config.freq)
    component_columns["trend"] = trend_series

    X = pd.DataFrame(X_columns, index=index)
    truth_components = pd.DataFrame(component_columns, index=index)
    signal = truth_components.sum(axis=1).rename("signal")
    noise = pd.Series(
        config.noise_scale * rng.standard_normal(config.n_samples),
        index=index,
        name="noise",
    )
    y = (signal + noise).rename("target")

    return SyntheticProblemResult(
        config=config,
        X=X,
        y=y,
        truth_components=truth_components,
        signal=signal,
        noise=noise,
    )


def split_problem_frames(problem: SyntheticProblemResult) -> SyntheticProblemSplit:
    split_at = int(problem.config.n_samples * problem.config.train_fraction)
    split_at = min(max(split_at, 1), problem.config.n_samples - 1)
    return SyntheticProblemSplit(
        X_train=problem.X.iloc[:split_at].copy(),
        y_train=problem.y.iloc[:split_at].copy(),
        X_test=problem.X.iloc[split_at:].copy(),
        y_test=problem.y.iloc[split_at:].copy(),
    )


def build_estimator_config(
    config: SyntheticProblemConfig,
    *,
    solver_name: str = "CLARABEL",
    fourier_reg_weight: float = 1.0e-5,
    linear_reg_weight: float = 1.0e-4,
    spline_reg_weight: float = 1.0e-4,
    spline_diff_reg_weight: float = 1.0,
    trend_reg_weight: float = 10.0,
) -> TsgamEstimatorConfig:
    multi_periodic_config = None
    if config.periodic_components:
        multi_periodic_config = TsgamMultiPeriodicConfig(
            num_harmonics=[component.harmonics for component in config.periodic_components],
            periods=[component.period_hours for component in config.periodic_components],
            reg_weight=fourier_reg_weight,
        )

    exog_config: list[TsgamLinearConfig | TsgamSplineConfig] | None = None
    if config.regressors:
        exog_config = []
        for spec in config.regressors:
            lags = list(spec.lags)
            if spec.relationship == SyntheticRegressorRelationship.LINEAR:
                exog_config.append(
                    TsgamLinearConfig(
                        lags=lags,
                        reg_weight=linear_reg_weight,
                        diff_reg_weight=spline_diff_reg_weight,
                    )
                )
            else:
                exog_config.append(
                    TsgamSplineConfig(
                        n_knots=spec.n_knots,
                        lags=lags,
                        reg_weight=spline_reg_weight,
                        diff_reg_weight=spline_diff_reg_weight,
                    )
                )

    trend_config = None
    if config.trend.kind != SyntheticTrendKind.NONE:
        trend_type = (
            TrendType.LINEAR
            if config.trend.kind == SyntheticTrendKind.LINEAR
            else TrendType.NONLINEAR
        )
        trend_config = TsgamTrendConfig(
            trend_type=trend_type,
            grouping=grouping_samples(config.trend.grouping_hours, config.freq),
            reg_weight=trend_reg_weight,
        )

    return TsgamEstimatorConfig(
        multi_periodic_config=multi_periodic_config,
        exog_config=exog_config,
        trend_config=trend_config,
        solver_config=TsgamSolverConfig(solver=solver_name, verbose=False),
        random_state=config.seed,
    )


def synthetic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residual = y_true_array - y_pred_array
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered = y_true_array - np.mean(y_true_array)
    denom = float(np.sum(centered**2))
    if np.isclose(denom, 0.0):
        r2 = 1.0 if np.isclose(np.sum(residual**2), 0.0) else 0.0
    else:
        r2 = float(1.0 - (np.sum(residual**2) / denom))
    return {"mae": mae, "rmse": rmse, "r2": r2}
