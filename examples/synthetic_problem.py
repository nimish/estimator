from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

import numpy as np
import pandas as pd
from spcqe import make_basis_matrix

from tsgam_estimator import (
    TrendType,
    TsgamEstimator,
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


class SyntheticNonlinearCurve(StrEnum):
    TANH = "tanh"
    SIGMOID = "sigmoid"
    BELL = "bell"
    POLY = "poly"


class SyntheticTrendKind(StrEnum):
    NONE = "none"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    NONLINEAR_INC = "nonlinear_inc"
    NONLINEAR_DEC = "nonlinear_dec"


class SyntheticHarmonicProfile(StrEnum):
    POWER = "power"
    FLAT = "flat"
    ALTERNATING = "alternating"
    SQUARE_WAVE = "square_wave"
    SAWTOOTH = "sawtooth"
    TRIANGLE_WAVE = "triangle_wave"


class SyntheticDriverNoiseDistribution(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    STUDENT_T = "student_t"


@dataclass(frozen=True)
class SyntheticPeriodicComponent:
    name: str
    period_hours: float
    harmonics: int = 1
    amplitude: float = 1.0
    phase: float = 0.0
    harmonic_decay: float = 1.0
    harmonic_profile: SyntheticHarmonicProfile = SyntheticHarmonicProfile.POWER
    phase_step: float = 0.35


@dataclass(frozen=True)
class SyntheticPeriodicInteractionSpec:
    left: str
    right: str
    effect_scale: float = 0.25


@dataclass(frozen=True)
class SyntheticRegressorSpec:
    name: str
    relationship: SyntheticRegressorRelationship = SyntheticRegressorRelationship.LINEAR
    effect_scale: float = 1.0
    driver_period_hours: float = 24.0
    driver_harmonics: int = 2
    driver_noise_scale: float = 0.2
    driver_noise_distribution: SyntheticDriverNoiseDistribution = (
        SyntheticDriverNoiseDistribution.GAUSSIAN
    )
    phase: float = 0.2
    lags: tuple[int, ...] = (0,)
    n_knots: int = 7
    harmonic_decay: float = 1.0
    driver_harmonic_profile: SyntheticHarmonicProfile = SyntheticHarmonicProfile.POWER
    nonlinear_scale: float = 1.25
    nonlinear_curve: SyntheticNonlinearCurve = SyntheticNonlinearCurve.TANH


@dataclass(frozen=True)
class SyntheticTrendSpec:
    kind: SyntheticTrendKind = SyntheticTrendKind.NONE
    amplitude: float = 0.0
    grouping_hours: float = 24.0
    breakpoints: int = 4


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
    periodic_interactions: tuple[SyntheticPeriodicInteractionSpec, ...] = field(
        default_factory=tuple
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


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def problem_summary_rows(config: SyntheticProblemConfig) -> list[dict[str, str]]:
    train_percent = int(round(config.train_fraction * 100))
    test_percent = 100 - train_percent
    rows = [
        {
            "section": "Data",
            "setting": "Window",
            "value": (
                f"{config.n_samples} samples at {config.freq}; "
                f"{train_percent}% train / {test_percent}% test"
            ),
        },
        {
            "section": "Data",
            "setting": "Noise",
            "value": f"Gaussian noise scale {config.noise_scale:.2f}, seed {config.seed}",
        },
    ]

    if config.periodic_components:
        rows.extend(
            {
                "section": "Periodic truth",
                "setting": component.name,
                "value": (
                    f"{component.period_hours:.1f}h period, "
                    f"{component.harmonics} harmonics, "
                    f"amplitude {component.amplitude:.2f}"
                ),
            }
            for component in config.periodic_components
        )
    else:
        rows.append(
            {
                "section": "Periodic truth",
                "setting": "none",
                "value": "No periodic terms enabled",
            }
        )

    if config.periodic_interactions:
        rows.extend(
            {
                "section": "Periodic cross terms",
                "setting": f"{interaction.left} x {interaction.right}",
                "value": f"cross-term effect {interaction.effect_scale:.2f}",
            }
            for interaction in config.periodic_interactions
        )
    else:
        rows.append(
            {
                "section": "Periodic cross terms",
                "setting": "none",
                "value": "No periodic cross terms enabled",
            }
        )

    if config.regressors:
        def _regressor_value(spec: SyntheticRegressorSpec) -> str:
            value = (
                f"{spec.relationship.value} effect {spec.effect_scale:.2f}, "
                f"{spec.driver_period_hours:.1f}h driver"
            )
            if spec.relationship == SyntheticRegressorRelationship.NONLINEAR:
                value = (
                    f"{value}, {spec.nonlinear_curve.value} truth, "
                    f"{spec.n_knots} spline knots"
                )
            return value

        rows.extend(
            {
                "section": "Regressors",
                "setting": spec.name,
                "value": _regressor_value(spec),
            }
            for spec in config.regressors
        )
    else:
        rows.append(
            {
                "section": "Regressors",
                "setting": "none",
                "value": "No exogenous regressors enabled",
            }
        )

    rows.append(
        {
            "section": "Trend",
            "setting": config.trend.kind.value,
            "value": _trend_summary_value(config.trend),
        }
    )
    return rows


def _trend_summary_value(trend: SyntheticTrendSpec) -> str:
    base = (
        f"amplitude {trend.amplitude:.2f}, "
        f"grouped every {trend.grouping_hours:.1f}h"
    )
    if trend.kind in (
        SyntheticTrendKind.NONLINEAR,
        SyntheticTrendKind.NONLINEAR_INC,
        SyntheticTrendKind.NONLINEAR_DEC,
    ):
        return f"{base}, {trend.breakpoints} seeded irregular jump breakpoints"
    return base


def describe_problem_config(config: SyntheticProblemConfig) -> str:
    periodic_text = _plural(len(config.periodic_components), "periodic term")
    cross_text = _plural(len(config.periodic_interactions), "periodic cross term")
    regressor_text = _plural(len(config.regressors), "regressor")
    return (
        f"{config.n_samples} samples at {config.freq}; "
        f"{periodic_text}, {cross_text}, {regressor_text}, "
        f"{config.trend.kind.value} trend, noise {config.noise_scale:.2f}."
    )


def _format_compact_float(value: float) -> str:
    return f"{float(value):.6g}"


def _rms(values: np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if np.isclose(denom, 0.0):
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def _relative_rmse(rmse: float, truth: np.ndarray) -> float:
    truth_rms = _rms(truth)
    scale = truth_rms if not np.isclose(truth_rms, 0.0) else 1.0
    return float(rmse / scale)


def problem_dashboard_rows(
    problem: SyntheticProblemResult,
    split: SyntheticProblemSplit,
) -> list[dict[str, str]]:
    """Compact high-level rows for the synthetic data dashboard."""
    config = problem.config
    regressors = ", ".join(problem.X.columns) if len(problem.X.columns) else "none"
    periodic = (
        ", ".join(component.name for component in config.periodic_components)
        if config.periodic_components
        else "none"
    )
    periodic_cross = (
        ", ".join(
            f"{interaction.left} x {interaction.right}"
            for interaction in config.periodic_interactions
        )
        if config.periodic_interactions
        else "none"
    )
    return [
        {"metric": "Samples", "value": f"{config.n_samples}"},
        {"metric": "Train / test", "value": f"{len(split.y_train)} / {len(split.y_test)}"},
        {"metric": "Frequency", "value": config.freq},
        {"metric": "Periodic terms", "value": periodic},
        {"metric": "Periodic cross terms", "value": periodic_cross},
        {"metric": "Regressors", "value": regressors},
        {"metric": "Trend", "value": config.trend.kind.value},
        {"metric": "Signal RMS", "value": f"{_rms(problem.signal):.4f}"},
        {"metric": "Noise RMS", "value": f"{_rms(problem.noise):.4f}"},
    ]


def _component_detail(config: SyntheticProblemConfig, component_name: str) -> tuple[str, str]:
    if component_name.startswith("periodic:"):
        name = component_name.removeprefix("periodic:")
        component = next(
            item for item in config.periodic_components if item.name == name
        )
        return "periodic", f"{component.period_hours:.1f}h, {component.harmonics} harmonics"
    if component_name.startswith("periodic_cross:"):
        name = component_name.removeprefix("periodic_cross:")
        interaction = next(
            item
            for item in config.periodic_interactions
            if f"{item.left} x {item.right}" == name
        )
        return (
            "periodic cross",
            f"{interaction.left} x {interaction.right}, effect {interaction.effect_scale:.2f}",
        )
    if component_name.startswith("regressor:"):
        name = component_name.removeprefix("regressor:")
        spec = next(item for item in config.regressors if item.name == name)
        return "regressor", f"{spec.relationship.value}, {spec.driver_period_hours:.1f}h driver"
    return "trend", f"{config.trend.kind.value}, grouped {config.trend.grouping_hours:.1f}h"


def component_summary_rows(
    config: SyntheticProblemConfig,
    problem: SyntheticProblemResult,
) -> list[dict[str, str | float]]:
    """Summarize true additive component scale and metadata."""
    rows: list[dict[str, str | float]] = []
    for component_name in problem.truth_components.columns:
        values = problem.truth_components[component_name]
        kind, detail = _component_detail(config, component_name)
        rows.append(
            {
                "component": component_name,
                "kind": kind,
                "detail": detail,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "rms": _rms(values),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return rows


def regressor_inspection_frame(problem: SyntheticProblemResult) -> pd.DataFrame:
    """Long-form regressor drivers and true contributions for plotting."""
    rows = []
    for spec in problem.config.regressors:
        regressor = spec.name
        if regressor not in problem.X.columns:
            continue
        contribution_name = f"regressor:{regressor}"
        if contribution_name not in problem.truth_components.columns:
            continue
        curve = _regressor_curve_label(spec)
        rows.append(
            pd.DataFrame(
                {
                    "datetime": problem.X.index,
                    "regressor": regressor,
                    "relationship": spec.relationship.value,
                    "curve": curve,
                    "series": "driver",
                    "value": problem.X[regressor].to_numpy(dtype=float),
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "datetime": problem.truth_components.index,
                    "regressor": regressor,
                    "relationship": spec.relationship.value,
                    "curve": curve,
                    "series": "true contribution",
                    "value": problem.truth_components[contribution_name].to_numpy(
                        dtype=float
                    ),
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "regressor",
                "relationship",
                "curve",
                "series",
                "value",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def estimator_config_rows(
    config: SyntheticProblemConfig,
    estimator_config: TsgamEstimatorConfig,
) -> list[dict[str, str]]:
    """Describe the estimator form that will be fit for a synthetic config."""
    rows = [
        {
            "section": "Solver",
            "term": "solver",
            "value": estimator_config.solver_config.solver,
        }
    ]
    if estimator_config.multi_periodic_config is None:
        rows.append({"section": "Periodic", "term": "periods", "value": "none"})
    else:
        periods = ", ".join(
            f"{period:.1f}h" for period in estimator_config.multi_periodic_config.periods
        )
        harmonics = ", ".join(
            str(value) for value in estimator_config.multi_periodic_config.num_harmonics
        )
        rows.extend(
            [
                {"section": "Periodic", "term": "periods", "value": periods},
                {"section": "Periodic", "term": "harmonics", "value": harmonics},
                {
                    "section": "Periodic",
                    "term": "cross terms",
                    "value": (
                        "Fourier pair-products"
                        if len(config.periodic_components) > 1
                        else "none"
                    ),
                },
                {
                    "section": "Periodic",
                    "term": "regularization",
                    "value": _format_compact_float(
                        estimator_config.multi_periodic_config.reg_weight
                    ),
                },
            ]
        )

    exog_config = estimator_config.exog_config or []
    if not exog_config:
        rows.append({"section": "Regressors", "term": "none", "value": "none"})
    for spec, exog_cfg in zip(config.regressors, exog_config, strict=True):
        lags = f"lags {exog_cfg.lags}"
        reg = f"reg {_format_compact_float(exog_cfg.reg_weight)}"
        if isinstance(exog_cfg, TsgamSplineConfig):
            value = f"spline, {exog_cfg.n_knots} knots, {lags}, {reg}"
        else:
            value = f"linear, {lags}, {reg}"
        rows.append({"section": "Regressors", "term": spec.name, "value": value})

    if estimator_config.trend_config is None:
        rows.append({"section": "Trend", "term": "trend", "value": "none"})
    else:
        trend_config = estimator_config.trend_config
        grouping = _format_compact_float(float(trend_config.grouping or 0.0))
        value = (
            f"{trend_config.trend_type.value}, grouping {grouping} samples, "
            f"reg {_format_compact_float(trend_config.reg_weight)}"
        )
        rows.append({"section": "Trend", "term": "trend", "value": value})
    return rows


def residual_summary_rows(
    residual_train: np.ndarray,
    residual_test: np.ndarray,
) -> list[dict[str, str | int | float]]:
    """Summarize train/test residual distributions."""
    rows: list[dict[str, str | int | float]] = []
    for split_name, residual in (
        ("train", np.asarray(residual_train, dtype=float)),
        ("test", np.asarray(residual_test, dtype=float)),
    ):
        rows.append(
            {
                "split": split_name,
                "n": int(len(residual)),
                "mean": float(np.mean(residual)),
                "std": float(np.std(residual, ddof=0)),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": _rms(residual),
                "min": float(np.min(residual)),
                "max": float(np.max(residual)),
            }
        )
    return rows


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
    return _freq_timedelta(freq).total_seconds() / pd.Timedelta("1h").total_seconds()


def grouping_samples(grouping_hours: float, freq: str) -> float:
    grouping = grouping_hours * samples_per_hour(freq)
    return float(max(1, int(round(grouping))))


def _period_sample_count(period_hours: float, freq: str) -> float:
    return float(period_hours * samples_per_hour(freq))


def _validate_periodic_component_sampling(config: SyntheticProblemConfig) -> None:
    for component in config.periodic_components:
        period_samples = _period_sample_count(component.period_hours, config.freq)
        max_harmonics = int(np.floor(period_samples / 2.0))
        if component.harmonics > max_harmonics:
            raise ValueError(
                f"Periodic component {component.name!r} requests {component.harmonics} "
                f"harmonics for a {component.period_hours:.6g}h period at {config.freq} "
                f"({period_samples:.6g} samples/period). Maximum supported harmonics: "
                f"{max_harmonics}."
            )


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


def _harmonic_weight(
    harmonic: int,
    profile: SyntheticHarmonicProfile,
    decay: float,
) -> float:
    if harmonic < 1:
        raise ValueError("harmonic must be one-indexed.")

    if profile == SyntheticHarmonicProfile.FLAT:
        return 1.0

    magnitude = 1.0 / (harmonic**decay)
    if profile == SyntheticHarmonicProfile.POWER:
        return magnitude
    if profile == SyntheticHarmonicProfile.ALTERNATING:
        sign = 1.0 if harmonic % 2 == 1 else -1.0
        return sign * magnitude
    if profile == SyntheticHarmonicProfile.SQUARE_WAVE:
        if harmonic % 2 == 0:
            return 0.0
        return 4.0 / (np.pi * harmonic)
    if profile == SyntheticHarmonicProfile.SAWTOOTH:
        sign = 1.0 if harmonic % 2 == 1 else -1.0
        return 2.0 * sign / (np.pi * harmonic)
    if profile == SyntheticHarmonicProfile.TRIANGLE_WAVE:
        if harmonic % 2 == 0:
            return 0.0
        sign = 1.0 if harmonic % 4 == 1 else -1.0
        return 8.0 * sign / (np.pi**2 * harmonic**2)
    raise ValueError(f"Unsupported harmonic profile: {profile}")


def _driver_noise(
    rng: np.random.Generator,
    distribution: SyntheticDriverNoiseDistribution,
    n_samples: int,
) -> np.ndarray:
    if distribution == SyntheticDriverNoiseDistribution.GAUSSIAN:
        return rng.standard_normal(n_samples)
    if distribution == SyntheticDriverNoiseDistribution.UNIFORM:
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n_samples)
    if distribution == SyntheticDriverNoiseDistribution.STUDENT_T:
        return rng.standard_t(df=3.0, size=n_samples) / np.sqrt(3.0)
    raise ValueError(f"Unsupported driver noise distribution: {distribution}")


def _periodic_component_series(
    index: pd.DatetimeIndex,
    component: SyntheticPeriodicComponent,
) -> pd.Series:
    time_hours = _time_hours(index)
    values = np.zeros(len(index), dtype=float)
    for harmonic in range(1, component.harmonics + 1):
        amplitude = component.amplitude * _harmonic_weight(
            harmonic,
            component.harmonic_profile,
            component.harmonic_decay,
        )
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
        amplitude = _harmonic_weight(
            harmonic,
            spec.driver_harmonic_profile,
            spec.harmonic_decay,
        )
        phase = spec.phase + 0.4 * harmonic
        driver += amplitude * np.cos(
            (2.0 * np.pi * harmonic * time_hours / spec.driver_period_hours) + phase
        )
    driver += spec.driver_noise_scale * _driver_noise(
        rng,
        spec.driver_noise_distribution,
        len(index),
    )
    standardized = _standardize(driver)
    return pd.Series(standardized, index=index, name=spec.name)


def _regressor_effect(values: np.ndarray, spec: SyntheticRegressorSpec) -> np.ndarray:
    lag_weights = _lag_weights(spec.lags)
    effect = np.zeros_like(values)
    for lag_weight, lag in zip(lag_weights, spec.lags, strict=True):
        lagged = np.nan_to_num(_shift_with_nan(values, lag), nan=0.0)
        effect += lag_weight * _true_regressor_response(lagged, spec)
    return effect


def _trend_series(
    index: pd.DatetimeIndex,
    trend: SyntheticTrendSpec,
    freq: str,
    rng: np.random.Generator,
) -> pd.Series:
    if trend.kind == SyntheticTrendKind.NONE or np.isclose(trend.amplitude, 0.0):
        values = np.zeros(len(index), dtype=float)
        return pd.Series(values, index=index, name="trend")

    group_size = int(grouping_samples(trend.grouping_hours, freq))
    group_ids = np.arange(len(index)) // group_size
    time_hours = _time_hours(index)
    total_hours = max(sample_hours(freq), time_hours[-1] + sample_hours(freq))
    position = np.clip(time_hours / total_hours, 0.0, 1.0)
    if trend.kind == SyntheticTrendKind.LINEAR:
        continuous = trend.amplitude * (position - 0.5)
        grouped = pd.Series(continuous).groupby(group_ids).transform("mean").to_numpy()
    elif trend.kind in (SyntheticTrendKind.NONLINEAR, SyntheticTrendKind.NONLINEAR_DEC):
        grouped = _nonlinear_jump_trend(
            group_ids.max() + 1,
            trend,
            rng=rng,
            direction=-1.0,
        )[group_ids]
    elif trend.kind == SyntheticTrendKind.NONLINEAR_INC:
        grouped = _nonlinear_jump_trend(
            group_ids.max() + 1,
            trend,
            rng=rng,
            direction=1.0,
        )[group_ids]
    else:
        raise ValueError(f"Unsupported trend kind: {trend.kind}")

    return pd.Series(grouped, index=index, name="trend")


def _nonlinear_jump_trend(
    n_samples: int,
    trend: SyntheticTrendSpec,
    *,
    rng: np.random.Generator,
    direction: float,
) -> np.ndarray:
    n_breakpoints = min(int(max(0, trend.breakpoints)), max(0, n_samples - 1))
    n_segments = n_breakpoints + 1
    values = np.linspace(
        -0.5 * trend.amplitude * direction,
        0.5 * trend.amplitude * direction,
        n_segments,
    )
    if n_breakpoints:
        breakpoints = np.sort(
            rng.choice(np.arange(1, n_samples), size=n_breakpoints, replace=False)
        )
        boundaries = np.r_[0, breakpoints, n_samples]
    else:
        boundaries = np.array([0, n_samples])
    out = np.zeros(n_samples, dtype=float)
    for segment_ix, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        out[start:stop] = values[segment_ix]
    return out


def generate_synthetic_problem(config: SyntheticProblemConfig) -> SyntheticProblemResult:
    if config.n_samples < 8:
        raise ValueError("Synthetic problems require at least 8 samples.")
    if not 0.1 < config.train_fraction <= 0.9:
        raise ValueError("train_fraction must be greater than 0.1 and at most 0.9.")

    index = pd.date_range(config.start, periods=config.n_samples, freq=config.freq)
    rng = np.random.default_rng(config.seed)

    X_columns: dict[str, pd.Series] = {}
    component_columns: dict[str, pd.Series] = {}

    for component in config.periodic_components:
        series = _periodic_component_series(index, component)
        component_columns[str(series.name)] = series

    periodic_series = {
        component.name: component_columns[f"periodic:{component.name}"]
        for component in config.periodic_components
    }
    for interaction in config.periodic_interactions:
        if interaction.left == interaction.right:
            raise ValueError("Periodic cross terms cannot pair a component with itself.")
        try:
            left = periodic_series[interaction.left]
            right = periodic_series[interaction.right]
        except KeyError as exc:
            available = ", ".join(periodic_series) or "none"
            raise ValueError(
                "Periodic cross term references an unknown component: "
                f"{interaction.left} x {interaction.right}. Available: {available}."
            ) from exc
        name = f"periodic_cross:{interaction.left} x {interaction.right}"
        component_columns[name] = pd.Series(
            interaction.effect_scale * left.to_numpy() * right.to_numpy(),
            index=index,
            name=name,
        )

    for spec in config.regressors:
        driver = _regressor_driver(index, spec, rng)
        X_columns[spec.name] = driver
        effect = _regressor_effect(driver.to_numpy(), spec)
        component_columns[f"regressor:{spec.name}"] = pd.Series(
            effect,
            index=index,
            name=f"regressor:{spec.name}",
        )

    trend_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 17]))
    trend_series = _trend_series(index, config.trend, config.freq, trend_rng)
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
    solver_verbose: bool = False,
    fourier_reg_weight: float = 1.0e-5,
    linear_reg_weight: float = 1.0e-4,
    spline_reg_weight: float = 1.0e-4,
    spline_diff_reg_weight: float = 1.0,
    trend_reg_weight: float = 0.1,
) -> TsgamEstimatorConfig:
    multi_periodic_config = None
    if config.periodic_components:
        _validate_periodic_component_sampling(config)
        sample_periods = [
            _period_sample_count(component.period_hours, config.freq)
            for component in config.periodic_components
        ]
        multi_periodic_config = TsgamMultiPeriodicConfig(
            num_harmonics=[component.harmonics for component in config.periodic_components],
            periods=sample_periods,
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
        if config.trend.kind == SyntheticTrendKind.LINEAR:
            trend_type = TrendType.LINEAR
        elif config.trend.kind == SyntheticTrendKind.NONLINEAR_INC:
            trend_type = TrendType.NONLINEAR_INC
        elif config.trend.kind == SyntheticTrendKind.NONLINEAR_DEC:
            trend_type = TrendType.NONLINEAR_DEC
        else:
            trend_type = TrendType.NONLINEAR
        trend_config = TsgamTrendConfig(
            trend_type=trend_type,
            grouping=grouping_samples(config.trend.grouping_hours, config.freq),
            reg_weight=trend_reg_weight,
        )

    return TsgamEstimatorConfig(
        multi_periodic_config=multi_periodic_config,
        exog_config=exog_config,
        trend_config=trend_config,
        solver_config=TsgamSolverConfig(solver=solver_name, verbose=solver_verbose),
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


def _coefficient_value(value: object, label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is None. Fit may not have converged.")
    coefficients = np.asarray(value, dtype=float)
    if np.any(np.isnan(coefficients)):
        raise ValueError(f"{label} contains NaN.")
    return coefficients


def _fourier_contribution(estimator: TsgamEstimator, time_indices: np.ndarray) -> np.ndarray:
    if estimator.config.multi_periodic_config is None:
        return np.zeros(len(time_indices), dtype=float)

    max_idx = int(np.max(time_indices))
    min_idx = int(np.min(time_indices))
    if min_idx < 0:
        offset = -min_idx
        adjusted_indices = time_indices.astype(int) + offset
        basis_length = max_idx + offset + 1
    else:
        adjusted_indices = time_indices.astype(int)
        basis_length = max_idx + 1

    basis = make_basis_matrix(
        num_harmonics=estimator.config.multi_periodic_config.num_harmonics,
        length=basis_length,
        periods=estimator.config.multi_periodic_config.periods,
    )[adjusted_indices, 1:]
    coefficients = _coefficient_value(
        estimator.variables_["fourier_coef"].value,
        "Fourier coefficients",
    )
    return basis @ coefficients


def _periodic_truth_coefficients(
    component: SyntheticPeriodicComponent,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for harmonic in range(1, component.harmonics + 1):
        amplitude = component.amplitude * _harmonic_weight(
            harmonic,
            component.harmonic_profile,
            component.harmonic_decay,
        )
        phase = component.phase + harmonic * component.phase_step
        rows.extend(
            [
                {
                    "component": component.name,
                    "period_hours": component.period_hours,
                    "harmonic": harmonic,
                    "term": "cos",
                    "truth_coefficient": float(amplitude * np.sin(phase)),
                },
                {
                    "component": component.name,
                    "period_hours": component.period_hours,
                    "harmonic": harmonic,
                    "term": "sin",
                    "truth_coefficient": float(amplitude * np.cos(phase)),
                },
            ]
        )
    return rows


def _periodic_basis_blocks(
    config: SyntheticProblemConfig,
) -> list[tuple[SyntheticPeriodicComponent, float, list[dict[str, str | int | float]]]]:
    sample_periods = np.array(
        [_period_sample_count(component.period_hours, config.freq) for component in config.periodic_components],
        dtype=float,
    )
    return [
        (
            config.periodic_components[ix],
            float(sample_periods[ix]),
            _periodic_truth_coefficients(config.periodic_components[ix]),
        )
        for ix in np.argsort(-sample_periods)
    ]


def fourier_coefficient_frame(
    config: SyntheticProblemConfig,
    estimator: TsgamEstimator,
) -> pd.DataFrame:
    """Compare synthetic truth Fourier coefficients with fitted coefficients."""
    columns = [
        "component",
        "period_hours",
        "harmonic",
        "term",
        "truth_coefficient",
        "fitted_coefficient",
        "difference",
    ]
    if not config.periodic_components or estimator.config.multi_periodic_config is None:
        return pd.DataFrame(columns=columns)

    fitted_coefficients = _coefficient_value(
        estimator.variables_["fourier_coef"].value,
        "Fourier coefficients",
    )
    rows: list[dict[str, str | int | float]] = []
    coefficient_ix = 0
    for _, _, truth_rows in _periodic_basis_blocks(config):
        for row in truth_rows:
            fitted = (
                float(fitted_coefficients[coefficient_ix])
                if coefficient_ix < len(fitted_coefficients)
                else np.nan
            )
            truth = float(row["truth_coefficient"])
            rows.append(
                {
                    **row,
                    "fitted_coefficient": fitted,
                    "difference": fitted - truth if not np.isnan(fitted) else np.nan,
                }
            )
            coefficient_ix += 1

    return pd.DataFrame(rows, columns=columns)


def cross_basis_coefficient_frame(
    config: SyntheticProblemConfig,
    estimator: TsgamEstimator,
) -> pd.DataFrame:
    """Compare synthetic periodic cross-term coefficients with fitted cross basis."""
    columns = [
        "left_component",
        "right_component",
        "left_period_hours",
        "right_period_hours",
        "left_harmonic",
        "left_term",
        "right_harmonic",
        "right_term",
        "truth_interaction",
        "truth_coefficient",
        "fitted_coefficient",
        "difference",
    ]
    if len(config.periodic_components) < 2 or estimator.config.multi_periodic_config is None:
        return pd.DataFrame(columns=columns)

    fitted_coefficients = _coefficient_value(
        estimator.variables_["fourier_coef"].value,
        "Fourier coefficients",
    )
    basis_blocks = _periodic_basis_blocks(config)
    coefficient_ix = sum(len(rows) for _, _, rows in basis_blocks)

    interaction_by_pair = {
        tuple(sorted((interaction.left, interaction.right))): interaction
        for interaction in config.periodic_interactions
    }
    rows: list[dict[str, str | int | float]] = []
    for (left, _, left_rows), (right, _, right_rows) in combinations(basis_blocks, 2):
        interaction = interaction_by_pair.get(tuple(sorted((left.name, right.name))))
        effect_scale = 0.0 if interaction is None else interaction.effect_scale
        interaction_label = (
            "none"
            if interaction is None
            else f"{interaction.left} x {interaction.right}"
        )
        for left_row in left_rows:
            for right_row in right_rows:
                fitted = (
                    float(fitted_coefficients[coefficient_ix])
                    if coefficient_ix < len(fitted_coefficients)
                    else np.nan
                )
                truth = (
                    effect_scale
                    * float(left_row["truth_coefficient"])
                    * float(right_row["truth_coefficient"])
                )
                rows.append(
                    {
                        "left_component": left.name,
                        "right_component": right.name,
                        "left_period_hours": left.period_hours,
                        "right_period_hours": right.period_hours,
                        "left_harmonic": int(left_row["harmonic"]),
                        "left_term": str(left_row["term"]),
                        "right_harmonic": int(right_row["harmonic"]),
                        "right_term": str(right_row["term"]),
                        "truth_interaction": interaction_label,
                        "truth_coefficient": truth,
                        "fitted_coefficient": fitted,
                        "difference": fitted - truth if not np.isnan(fitted) else np.nan,
                    }
                )
                coefficient_ix += 1

    return pd.DataFrame(rows, columns=columns)


def _true_regressor_response(
    values: np.ndarray,
    spec: SyntheticRegressorSpec,
) -> np.ndarray:
    if spec.relationship == SyntheticRegressorRelationship.LINEAR:
        return spec.effect_scale * values
    scaled = spec.nonlinear_scale * values
    if spec.nonlinear_curve == SyntheticNonlinearCurve.TANH:
        response = np.tanh(scaled)
    elif spec.nonlinear_curve == SyntheticNonlinearCurve.SIGMOID:
        response = 2.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0))) - 1.0
    elif spec.nonlinear_curve == SyntheticNonlinearCurve.BELL:
        response = np.exp(-0.5 * scaled**2)
    elif spec.nonlinear_curve == SyntheticNonlinearCurve.POLY:
        response = scaled**2 - 1.0
    else:
        raise ValueError(f"Unsupported nonlinear curve: {spec.nonlinear_curve}")
    return spec.effect_scale * response


def _regressor_curve_label(spec: SyntheticRegressorSpec) -> str:
    if spec.relationship == SyntheticRegressorRelationship.LINEAR:
        return "linear"
    return spec.nonlinear_curve.value


def _regressor_response_grid(driver: np.ndarray, grid_size: int) -> np.ndarray:
    max_abs = float(np.max(np.abs(driver)))
    if np.isclose(max_abs, 0.0):
        return np.linspace(-1.0, 1.0, grid_size)
    return np.linspace(-max_abs, max_abs, grid_size)


def true_regressor_response_frame(
    problem: SyntheticProblemResult,
    *,
    grid_size: int = 100,
) -> pd.DataFrame:
    """Evaluate generated true regressor response curves before fitting."""
    columns = ["regressor", "relationship", "curve", "x", "source", "value"]
    if not problem.config.regressors or grid_size < 2:
        return pd.DataFrame(columns=columns)

    rows: list[pd.DataFrame] = []
    for spec in problem.config.regressors:
        if spec.name not in problem.X.columns:
            continue

        driver = problem.X[spec.name].to_numpy(dtype=float)
        grid = _regressor_response_grid(driver, grid_size)
        rows.append(
            pd.DataFrame(
                {
                    "regressor": spec.name,
                    "relationship": spec.relationship.value,
                    "curve": _regressor_curve_label(spec),
                    "x": grid,
                    "source": "synthetic truth",
                    "value": _true_regressor_response(grid, spec),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def regressor_response_frame(
    estimator: TsgamEstimator,
    problem: SyntheticProblemResult,
    *,
    grid_size: int = 100,
) -> pd.DataFrame:
    """Evaluate true and fitted regressor response curves on standardized grids."""
    columns = ["regressor", "relationship", "curve", "x", "source", "value"]
    if not problem.config.regressors or grid_size < 2:
        return pd.DataFrame(columns=columns)

    exog_config = estimator.config.exog_config or []
    rows: list[pd.DataFrame] = [
        true_regressor_response_frame(problem, grid_size=grid_size)
    ]
    for ix, spec in enumerate(problem.config.regressors):
        if spec.name not in problem.X.columns:
            continue

        driver = problem.X[spec.name].to_numpy(dtype=float)
        grid = _regressor_response_grid(driver, grid_size)
        if ix >= len(exog_config) or 0 not in exog_config[ix].lags:
            continue
        exog_cfg = exog_config[ix]
        stored_knots = (
            estimator.exog_knots_[ix]
            if isinstance(exog_cfg, TsgamSplineConfig)
            else None
        )
        _, basis_blocks = estimator._process_exog_config(
            exog_cfg,
            grid,
            knots=stored_knots,
        )
        zero_lag_ix = exog_cfg.lags.index(0)
        coefficients = _coefficient_value(
            estimator.variables_[f"exog_coef_{ix}"].value,
            f"Exogenous coefficients for {spec.name}",
        )
        fitted = basis_blocks[zero_lag_ix] @ coefficients[:, zero_lag_ix]
        rows.append(
            pd.DataFrame(
                {
                    "regressor": spec.name,
                    "relationship": spec.relationship.value,
                    "curve": _regressor_curve_label(spec),
                    "x": grid,
                    "source": "fitted model",
                    "value": fitted,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def _trend_contribution(
    estimator: TsgamEstimator,
    time_indices: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    trend_config = estimator.config.trend_config
    if (
        trend_config is None
        or trend_config.trend_type == TrendType.NONE
        or "trend" not in estimator.variables_
    ):
        return np.zeros(n_samples, dtype=float)

    trend = _coefficient_value(estimator.variables_["trend"].value, "Trend coefficients")
    period_hours = getattr(estimator, "trend_period_hours_", trend_config.grouping)
    if period_hours is None:
        raise ValueError("Trend period is unavailable. Fit may not have converged.")

    period_indices = (time_indices / period_hours).astype(int)
    if len(period_indices) == 0:
        return np.zeros(n_samples, dtype=float)
    n_periods_pred = int(period_indices.max()) + 1
    if n_periods_pred <= 0:
        return np.zeros(n_samples, dtype=float)

    trend_extended = np.zeros(n_periods_pred, dtype=float)
    n_periods_fit = len(trend)
    trend_extended[: min(n_periods_fit, n_periods_pred)] = trend[:n_periods_pred]
    if n_periods_pred > n_periods_fit:
        trend_extended[:n_periods_fit] = trend
        slope_variable = estimator.variables_.get("trend_slope")
        slope = None if slope_variable is None else slope_variable.value
        if trend_config.trend_type == TrendType.LINEAR and slope is not None:
            slope_value = float(np.asarray(slope))
            for ix in range(n_periods_fit, n_periods_pred):
                trend_extended[ix] = trend[-1] + slope_value * (ix - n_periods_fit + 1)
        else:
            trend_extended[n_periods_fit:] = trend[-1]

    contribution = np.zeros(n_samples, dtype=float)
    valid_mask = period_indices >= 0
    contribution[valid_mask] = trend_extended[period_indices[valid_mask]]
    return contribution


def fitted_component_frame(
    estimator: TsgamEstimator,
    X: pd.DataFrame,
    *,
    regressor_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return fitted additive component contributions for a fitted estimator."""
    (X_sorted,) = estimator._ensure_sorted_index(X)
    timestamps, X_array = estimator._ensure_timestamp_index(X_sorted)
    time_indices = estimator._timestamps_to_indices(timestamps, estimator.time_reference_)
    n_samples = len(X_array)

    constant = float(_coefficient_value(estimator.variables_["constant"].value, "Constant"))
    components: dict[str, np.ndarray] = {
        "constant": np.full(n_samples, constant, dtype=float),
    }

    exog_config = estimator.config.exog_config or []
    names = list(regressor_names) if regressor_names is not None else list(X_sorted.columns)
    if len(names) < len(exog_config):
        names.extend(f"x{ix}" for ix in range(len(names), len(exog_config)))

    zero_lag_Hs: dict[int, np.ndarray] = {}
    interaction_parent_indices = {
        exog_ix
        for pair in getattr(estimator, "interaction_pairs_", [])
        for exog_ix in pair
    }
    for ix, exog_cfg in enumerate(exog_config):
        stored_knots = (
            estimator.exog_knots_[ix]
            if isinstance(exog_cfg, TsgamSplineConfig)
            else None
        )
        _, basis_blocks = estimator._process_exog_config(
            exog_cfg,
            X_array[:, ix],
            knots=stored_knots,
        )
        if ix in interaction_parent_indices:
            zero_lag_Hs[ix] = estimator._get_zero_lag_H(exog_cfg, basis_blocks)

        coefficients = _coefficient_value(
            estimator.variables_[f"exog_coef_{ix}"].value,
            f"Exogenous coefficients for {names[ix]}",
        )
        contribution = np.zeros(n_samples, dtype=float)
        for lag_ix, basis in enumerate(basis_blocks):
            contribution += np.nan_to_num(basis, nan=0.0) @ coefficients[:, lag_ix]
        components[f"regressor:{names[ix]}"] = contribution

    for pair_ix, (left_ix, right_ix) in enumerate(getattr(estimator, "interaction_pairs_", [])):
        coefficients = _coefficient_value(
            estimator.variables_[f"interaction_coef_{pair_ix}"].value,
            f"Interaction coefficients for pair {pair_ix}",
        )
        left_name = names[left_ix]
        right_name = names[right_ix]
        components[f"interaction:{left_name} x {right_name}"] = (
            estimator._interaction_contribution_from_blocks(
                zero_lag_Hs[left_ix],
                zero_lag_Hs[right_ix],
                coefficients,
                nan_to_zero=True,
            )
        )

    components["periodic"] = _fourier_contribution(estimator, time_indices)
    components["trend"] = _trend_contribution(estimator, time_indices, n_samples)

    component_frame = pd.DataFrame(components, index=timestamps)
    component_frame["fitted"] = component_frame.sum(axis=1)
    return component_frame


def _component_truth_series(
    truth_components: pd.DataFrame,
    component: str,
) -> pd.Series | None:
    if component == "periodic":
        periodic_columns = [
            column
            for column in truth_components.columns
            if column.startswith("periodic:") or column.startswith("periodic_cross:")
        ]
        if not periodic_columns:
            return None
        return truth_components[periodic_columns].sum(axis=1).rename("periodic")
    if component in truth_components.columns:
        return truth_components[component]
    return None


def _component_score_row(
    *,
    component: str,
    truth_term: str,
    model_term: str,
    truth: pd.Series,
    fitted_train: pd.DataFrame,
    fitted_test: pd.DataFrame,
) -> dict[str, str | float]:
    train_truth = truth.loc[fitted_train.index].to_numpy(dtype=float)
    train_fit = fitted_train[component].to_numpy(dtype=float)
    test_truth = truth.loc[fitted_test.index].to_numpy(dtype=float)
    test_fit = fitted_test[component].to_numpy(dtype=float)
    train_metrics = synthetic_metrics(train_truth, train_fit)
    test_metrics = synthetic_metrics(test_truth, test_fit)
    train_mean_offset = float(np.mean(train_fit) - np.mean(train_truth))
    test_mean_offset = float(np.mean(test_fit) - np.mean(test_truth))
    return {
        "component": component,
        "truth_term": truth_term,
        "model_term": model_term,
        "train_mean_offset": train_mean_offset,
        "test_mean_offset": test_mean_offset,
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_correlation": _safe_correlation(train_truth, train_fit),
        "test_correlation": _safe_correlation(test_truth, test_fit),
        "train_relative_rmse": _relative_rmse(train_metrics["rmse"], train_truth),
        "test_relative_rmse": _relative_rmse(test_metrics["rmse"], test_truth),
    }


def component_fit_stat_rows(
    quality_rows: Sequence[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    """Convert component quality rows into a split-by-component stats table."""

    rows: list[dict[str, str | float]] = []
    for row in quality_rows:
        for split in ("train", "test"):
            rows.append(
                {
                    "component": str(row["component"]),
                    "split": split,
                    "truth": str(row["truth_term"]),
                    "model": str(row["model_term"]),
                    "rmse": float(row[f"{split}_rmse"]),
                    "mae": float(row[f"{split}_mae"]),
                    "r2": float(row[f"{split}_r2"]),
                    "correlation": float(row[f"{split}_correlation"]),
                    "relative_rmse": float(row[f"{split}_relative_rmse"]),
                    "mean_offset": float(row[f"{split}_mean_offset"]),
                }
            )
    return rows


def component_fit_quality_rows(
    *,
    config: SyntheticProblemConfig,
    truth_components: pd.DataFrame,
    fitted_train: pd.DataFrame,
    fitted_test: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Score fitted additive components against the known synthetic truth."""
    component_specs: list[tuple[str, str, str]] = []
    if config.periodic_components:
        truth_terms = len(config.periodic_components) + len(config.periodic_interactions)
        truth_term = _plural(truth_terms, "periodic truth term")
        model_term = _plural(
            sum(component.harmonics for component in config.periodic_components),
            "Fourier harmonic",
        )
        if len(config.periodic_components) > 1:
            model_term = f"{model_term} + cross basis"
        component_specs.append(("periodic", truth_term, model_term))

    component_specs.extend(
        (
            f"regressor:{spec.name}",
            spec.relationship.value,
            "linear"
            if spec.relationship == SyntheticRegressorRelationship.LINEAR
            else "spline",
        )
        for spec in config.regressors
    )

    if config.trend.kind != SyntheticTrendKind.NONE:
        component_specs.append(("trend", config.trend.kind.value, config.trend.kind.value))

    rows: list[dict[str, str | float]] = []
    for component, truth_term, model_term in component_specs:
        if component not in fitted_train.columns or component not in fitted_test.columns:
            continue
        truth = _component_truth_series(truth_components, component)
        if truth is None:
            continue
        rows.append(
            _component_score_row(
                component=component,
                truth_term=truth_term,
                model_term=model_term,
                truth=truth,
                fitted_train=fitted_train,
                fitted_test=fitted_test,
            )
        )
    return rows
