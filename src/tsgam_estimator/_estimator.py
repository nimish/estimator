# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import cvxpy
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import RandomState
from scipy import signal, stats
from signaldecomp import make_offset_basis, solve
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted
from sklearn.utils import check_random_state

from ._design import (
    _extract_timestamps,
    _to_pandas_timedelta_frequency,
    build_tsgam_design,
    infer_fit_frequency,
    sort_fit_inputs,
    sort_predict_X,
    validate_predict_frequency,
)
from ._problem import (
    evaluate_single_output_prediction,
    build_single_output_decomposition,
)
from ._sklearn import SklearnConfigMixin


@dataclass
class TsgamMultiPeriodicConfig(SklearnConfigMixin):
    """
    Configuration for multi-periodic Fourier basis functions.

    This config defines the seasonal/periodic patterns in the time series using
    Fourier basis functions with multiple harmonics and periods. Each period
    can have multiple harmonics to capture complex seasonal patterns.

    Parameters
    ----------
    num_harmonics : list[int]
        Number of harmonics for each period. Each element corresponds to a period.
        For example, [6, 4, 3] means 6 harmonics for the first period,
        4 for the second, and 3 for the third.
    periods : list[float]
        Periods for each harmonic block, in hours. Must have same length as
        num_harmonics. Common values:
        - 24: daily pattern
        - 168 (7*24): weekly pattern
        - 8766 (365.2425*24): yearly pattern
    reg_weight : float, default=1.0e-4
        Regularization weight for Fourier coefficients. Higher values increase
        smoothness of the seasonal patterns. Typical range: 1e-5 to 1e-3.

    Examples
    --------
    >>> config = TsgamMultiPeriodicConfig(
    ...     num_harmonics=[6, 4, 3],
    ...     periods=[365.2425 * 24, 7 * 24, 24]  # yearly, weekly, daily
    ... )
    """
    num_harmonics: list[int]
    periods: list[float]
    reg_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        if len(self.num_harmonics) != len(self.periods):
            raise ValueError("num_harmonics and periods must have the same length.")
        for ix, (harmonics, period) in enumerate(zip(self.num_harmonics, self.periods, strict=True)):
            if (
                not isinstance(harmonics, (int, np.integer))
                or isinstance(harmonics, bool)
                or harmonics < 0
            ):
                raise ValueError(
                    f"num_harmonics[{ix}] must be a non-negative integer, got {harmonics!r}."
                )
            if not np.isfinite(period) or period <= 0:
                raise ValueError(f"periods[{ix}] must be positive and finite, got {period!r}.")
            max_harmonics = int(np.floor(float(period) / 2.0))
            if harmonics > max_harmonics:
                raise ValueError(
                    f"num_harmonics[{ix}]={harmonics} exceeds the Nyquist limit "
                    f"{max_harmonics} for period {float(period):.6g} samples."
                )

@dataclass
class TsgamSplineConfig(SklearnConfigMixin):
    """
    Configuration for cubic spline basis functions for exogenous variables.

    This config defines how an exogenous variable (e.g., temperature) is modeled
    using cubic splines with optional lead/lag terms. Splines allow for non-linear
    relationships between the exogenous variable and the target.

    Parameters
    ----------
    n_knots : int or None
        Number of knots for the spline basis. Knots will be evenly spaced between
        min and max of the variable. If None, knots must be provided explicitly.
        Ignored if knots is non-empty.
    lags : list[int], default=[0]
        Time offsets for the exogenous variable, evaluated as ``x[t + offset]``.
        Negative values are lags (past values), and positive values are leads
        (future values). For example, ``[-3, -2, -1, 0]`` includes the current
        value and the previous three samples.
    reg_weight : float, default=1.0e-4
        Regularization weight for spline coefficients. Higher values increase
        smoothness. Typical range: 1e-5 to 1e-3.
    diff_reg_weight : float, default=1.0
        Regularization weight for differences between coefficients at different
        lags. This encourages smooth transitions across lags. Higher values make
        lag coefficients more similar.
    knots : list[float], default=[]
        Explicit knot locations for the spline. If empty list, knots will be
        auto-generated using n_knots. If provided, n_knots is ignored.

    Examples
    --------
    >>> # Auto-generate 10 knots
    >>> config = TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1])
    >>>
    >>> # Use explicit knots
    >>> config = TsgamSplineConfig(
    ...     knots=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    ...     lags=[0]
    ... )
    """
    n_knots: int | None = None
    lags: list[int] = field(default_factory=lambda:[0])
    reg_weight: float = 1.0e-4
    diff_reg_weight: float = 1.0
    knots: ndarray | list[float] = field(default_factory=list)

@dataclass
class TsgamLinearConfig(SklearnConfigMixin):
    """
    Configuration for linear basis functions for exogenous variables.

    This config defines how an exogenous variable is modeled using simple linear
    terms with optional lead/lag. Use this instead of TsgamSplineConfig when
    you expect a linear relationship.

    Parameters
    ----------
    lags : list[int], default=[0]
        Time offsets for the exogenous variable, evaluated as ``x[t + offset]``.
        Negative values are lags (past values), and positive values are leads
        (future values). For example, ``[-1, 0]`` includes the current value and
        the previous sample.
    reg_weight : float, default=1.0e-4
        Regularization weight for linear coefficients. Higher values increase
        regularization. Typical range: 1e-5 to 1e-3.
    diff_reg_weight : float, default=1.0
        Regularization weight for differences between coefficients at different
        lags. This encourages smooth transitions across lags. Higher values make
        lag coefficients more similar.

    Examples
    --------
    >>> config = TsgamLinearConfig(lags=[-2, -1, 0, 1, 2])
    """
    lags: list[int] = field(default_factory=lambda:[0])
    reg_weight: float = 1.0e-4
    diff_reg_weight: float = 1.0

@dataclass
class TsgamArConfig(SklearnConfigMixin):
    """
    Configuration for autoregressive (AR) residual modeling.

    After fitting the baseline model (Fourier + exogenous), this config enables
    fitting an AR model on the residuals to capture remaining temporal dependencies.
    The AR model uses L1 regularization to encourage sparsity.

    Parameters
    ----------
    lags : list[int]
        AR lags to include in the model. Typically [1] for AR(1), [1, 2] for AR(2), etc.
        Only positive lags are meaningful (looking back in time).
    l1_constraint : float, default=0.95
        L1 norm constraint for AR coefficients. This controls sparsity - lower values
        allow fewer non-zero coefficients. Typical range: 0.5 to 1.0.

    Examples
    --------
    >>> # AR(1) model
    >>> config = TsgamArConfig(lags=[1], l1_constraint=0.95)
    >>>
    >>> # AR(2) model with tighter constraint
    >>> config = TsgamArConfig(lags=[1, 2], l1_constraint=0.8)
    """
    lags: list[int]
    l1_constraint: float = 0.95


class TrendType(StrEnum):
    NONE = 'none'
    LINEAR = 'linear'
    NONLINEAR = 'nonlinear'
    NONLINEAR_DECREASING = 'nonlinear_decreasing'
    NONLINEAR_INCREASING = 'nonlinear_increasing'
    NONLINEAR_DEC = 'nonlinear_decreasing'
    NONLINEAR_INC = 'nonlinear_increasing'

@dataclass
class TsgamTrendConfig(SklearnConfigMixin):
    """
    Configuration for trend term in the model.

    The trend is constant per period (e.g., per day for hourly data). This allows
    modeling long-term changes that are constant within each period but can vary
    across periods.

    Parameters
    ----------
    trend_type : str, default='none'
        Type of trend to fit:
        - 'none': No trend (trend = 0)
        - 'linear': Linear trend with constant slope
        - 'nonlinear': Non-linear monotonic decreasing trend
    period_hours : float or None, default=None
        Period length in hours. If None, will be inferred from data frequency
        (defaults to daily: 24 hours for hourly data, 1 day for daily data, etc.).
        For example:
        - Hourly data: 24.0 for daily trend, 168.0 for weekly trend
        - 15-minute data: 24.0 for daily trend (96 samples per day)
        - Daily data: 7.0 for weekly trend, 365.2425 for yearly trend
    reg_weight : float, default=10.0
        Regularization weight for trend differences. Higher values encourage
        smoother trends. Typical range: 1.0 to 100.0.

    Examples
    --------
    >>> # Daily trend for hourly data (default)
    >>> config = TsgamTrendConfig(trend_type='linear')
    >>>
    >>> # Weekly trend for hourly data
    >>> config = TsgamTrendConfig(trend_type='nonlinear', period_hours=168.0)
    >>>
    >>> # No trend
    >>> config = TsgamTrendConfig(trend_type='none')
    """
    trend_type: TrendType = TrendType.NONE
    grouping: float | None = None # todo: rename this to something better
    reg_weight: float = 10.0

@dataclass
class TsgamOutlierConfig(SklearnConfigMixin):
    """
    Configuration for the outlier detector component.

    The outlier detector identifies anomalous periods (e.g., days) in the time series
    with sparse multiplicative corrections. This component is particularly useful for
    detecting days with unusual patterns that deviate from the normal seasonal and
    trend behavior, such as holidays, special events, or data quality issues.

    How it works:

    - The detector assigns one correction value per period (e.g., per day)
    - Corrections are constant across all samples within a period
    - L1 regularization encourages sparsity: most periods have no correction (≈0),
      while only outlier periods have non-zero corrections
    - The correction is additive in log space, which translates to a multiplicative
      effect in the original scale (e.g., 0.2 in log space ≈ 0.82x multiplier,
      0.5 in log space ≈ 1.65x multiplier)

    Mathematical formulation:

    The outlier term is added to the model as: ``T @ outlier``, where:

    - ``T`` is a binary matrix mapping each sample to its period
    - ``outlier`` is a sparse vector of period-level corrections
    - The L1 penalty ``reg_weight * ||outlier||_1`` encourages sparsity

    Parameters
    ----------
    reg_weight : float
        L1 regularization weight controlling sparsity of outlier detection.
        Higher values encourage more sparsity (fewer outliers detected).
        Lower values allow more outliers to be detected.

        Typical ranges:

        - Very sparse (few outliers): 0.01 to 0.1
        - Moderate sparsity: 0.001 to 0.01
        - More sensitive (more outliers): 0.0001 to 0.001

        Guidelines:

        - Start with 0.002 and adjust based on results
        - If too many outliers detected (>10% of periods), increase reg_weight
        - If no outliers detected, decrease reg_weight
        - For real-world data, values between 0.001 and 0.01 are often appropriate

    period_hours : float or None, default=None
        Period length in hours for which the outlier correction is constant.
        If None, defaults to 24.0 hours (daily outliers) for hourly data.

        Examples:

        - Hourly data, daily outliers: ``period_hours=24.0`` (default)
        - 15-minute data, daily outliers: ``period_hours=24.0`` (96 samples per day)
        - Hourly data, weekly outliers: ``period_hours=168.0`` (7 days)
        - Daily data, weekly outliers: ``period_hours=7.0``
        - Hourly data, hourly outliers: ``period_hours=1.0`` (not recommended, use AR instead)

    Notes
    -----
    - The outlier detector works best when the target is log-transformed, as it
      naturally models multiplicative effects
    - Outlier corrections are applied during both fit and predict
    - For prediction periods beyond the training data, outlier corrections default
      to 0 (no correction)
    - The detector is most effective when combined with other components (seasonality,
      trend, exogenous variables) that explain normal variation, leaving outliers
      as the residual anomaly

    Examples
    --------
    >>> # Daily outlier detector for hourly data (default, moderate sparsity)
    >>> config = TsgamOutlierConfig(reg_weight=0.002)
    >>>
    >>> # Weekly outlier detector with higher sparsity (fewer outliers)
    >>> config = TsgamOutlierConfig(reg_weight=0.01, period_hours=168.0)
    >>>
    >>> # Daily outlier detector with lower sparsity (more outliers detected)
    >>> config = TsgamOutlierConfig(reg_weight=0.001)
    >>>
    >>> # Very sparse detector (only extreme outliers)
    >>> config = TsgamOutlierConfig(reg_weight=0.1)
    >>>
    >>> # Use in full estimator configuration
    >>> from tsgam_estimator import TsgamEstimatorConfig
    >>> estimator_config = TsgamEstimatorConfig(
    ...     outlier_config=TsgamOutlierConfig(reg_weight=0.002)
    ... )
    """
    reg_weight: float
    period_hours: float | None = None

type SolverOptionValue = int | float | bool | str | dict[str, SolverOptionValue]

@dataclass
class TsgamSolverConfig(SklearnConfigMixin):
    """
    Configuration for the CVXPY solver used in optimization.

    Parameters
    ----------
    solver : str, default='CLARABEL'
        CVXPY solver name. Common options:
        - 'CLARABEL': Fast, modern solver (recommended)
        - 'ECOS': Reliable, slower
        - 'OSQP': Good for quadratic problems
        - 'SCS': General purpose
    verbose : bool, default=True
        Whether to print solver output during optimization. Useful for debugging
        but can be verbose for large problems.
    warm_start : bool, default=True
        Whether to warm-start the solver using cached results from a previous
        solve. Can significantly speed up repeated solves with similar data.
    solver_opts : dict[str, SolverOptionValue] | None, default=None
        Additional keyword arguments forwarded to ``cvxpy.Problem.solve()``.
        Each solver accepts its own options; see
        https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options

        CLARABEL options include ``max_iter`` (default 50) and
        ``time_limit`` (default 0.0, no limit).

        MOSEK options are passed via a ``mosek_params`` dict with string
        parameter names, e.g.
        ``{"mosek_params": {"MSK_IPAR_INTPNT_MAX_ITERATIONS": 400}}``.

    Examples
    --------
    >>> config = TsgamSolverConfig(solver='CLARABEL', verbose=False)
    >>> config = TsgamSolverConfig(
    ...     solver='CLARABEL',
    ...     solver_opts={"max_iter": 200, "time_limit": 60.0},
    ... )
    >>> config = TsgamSolverConfig(
    ...     solver='MOSEK',
    ...     solver_opts={
    ...         "mosek_params": {"MSK_IPAR_INTPNT_MAX_ITERATIONS": 400}
    ...     },
    ... )
    """
    solver: str = 'CLARABEL'
    verbose: bool = True
    warm_start: bool = True
    solver_opts: dict[str, SolverOptionValue] | None = None

    _RESERVED_KEYS = frozenset({"solver", "verbose", "warm_start"})

    def _solve_kwargs(self) -> dict[str, SolverOptionValue]:
        """Build the extra kwargs dict for ``Problem.solve()``, validating no reserved keys."""
        opts = dict(self.solver_opts or {})
        conflict = self._RESERVED_KEYS & opts.keys()
        if conflict:
            raise ValueError(
                f"solver_opts must not contain keys that are passed explicitly: "
                f"{sorted(conflict)}"
            )
        return opts

@dataclass
class TsgamEstimatorConfig(SklearnConfigMixin):
    """
    Main configuration for TsgamEstimator.

    This config combines all component configurations (Fourier, exogenous, AR)
    and solver settings into a single configuration object.

    Parameters
    ----------
    multi_periodic_config : TsgamMultiPeriodicConfig or None
        Configuration for multi-periodic Fourier basis functions. If None,
        no time-based seasonal patterns are modeled.
    exog_config : list of TsgamSplineConfig or TsgamLinearConfig, or None
        List of configurations for exogenous variables. Each element corresponds
        to one exogenous variable in X. Order must match column order in X.
        If None, no exogenous variables are used.
    interaction_pairs : list[tuple[int, int]] or None, default=None
        Exact 2-way interaction pairs between exogenous terms. Each tuple refers
        to positions in ``exog_config`` and the matching X column order. When
        interactions are enabled, they use only each factor's current-index
        response block (``lag=0``), even if the corresponding main effect also
        includes additional lagged terms.
    ar_config : TsgamArConfig or None, default=None
        Configuration for AR residual modeling. If None, no AR model is fitted.
    trend_config : TsgamTrendConfig or None, default=None
        Configuration for trend term. If None, no trend is fitted (equivalent to
        trend_type='none'). The trend is constant per period and can be linear,
        nonlinear (monotonic decreasing), or none.
    outlier_config : TsgamOutlierConfig or None, default=None
        Configuration for outlier detector component. If None, no outlier detector
        is fitted.

        The outlier detector identifies anomalous periods (e.g., days) with sparse
        multiplicative corrections. It uses L1 regularization to encourage sparsity,
        meaning most periods will have no correction (≈0), while only outlier
        periods will have non-zero corrections. The corrections are constant per
        period and additive in log space (multiplicative in original scale).

        See :class:`TsgamOutlierConfig` for detailed documentation and parameter
        tuning guidelines.
    solver_config : TsgamSolverConfig, default=TsgamSolverConfig()
        Solver configuration for CVXPY optimization.
    sort_index : bool, default=True
        If True, sort the data by its datetime index before fit/predict so that
        row order matches time order. If False, require the index to already be
        sorted (chronologically); raise ValueError if not.
    random_state : int, RandomState instance or None, default=None
        Random seed/state for reproducible stochastic sampling. Integer seeds are
        convenient for shared configs, while ``RandomState`` instances allow
        callers to manage RNG state explicitly.
    debug : bool, default=False
        If True, stores additional debug attributes (e.g., _baseline_residuals_,
        _B_running_view_) for inspection.

    Examples
    --------
    >>> multi_periodic = TsgamMultiPeriodicConfig(
    ...     num_harmonics=[6, 4, 3],
    ...     periods=[365.2425 * 24, 7 * 24, 24]
    ... )
    >>> exog = [TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1])]
    >>> ar = TsgamArConfig(lags=[1])
    >>> config = TsgamEstimatorConfig(
    ...     multi_periodic_config=multi_periodic,
    ...     exog_config=exog,
    ...     ar_config=ar
    ... )
    """
    multi_periodic_config: TsgamMultiPeriodicConfig | None
    exog_config: list[TsgamSplineConfig | TsgamLinearConfig] | None
    interaction_pairs: list[tuple[int, int]] | None = None # ensure this works with linear and splines
    ar_config: TsgamArConfig | None = None
    trend_config: TsgamTrendConfig | None = None
    outlier_config: TsgamOutlierConfig | None = None
    solver_config: TsgamSolverConfig = field(default_factory=TsgamSolverConfig)
    sort_index: bool = True
    random_state: RandomState | int | None = None
    debug: bool = False


PERIOD_HOURLY_DAILY = 24
PERIOD_HOURLY_WEEKLY = 24 * 7
PERIOD_HOURLY_YEARLY = 24 * 365.2425

PERIOD_DAILY_YEARLY = 365.2425
PERIOD_WEEKLY_YEARLY = 52.1775

PERIOD_MONTHLY_YEARLY = 12
PERIOD_QUARTERLY_YEARLY = 4
PERIOD_YEARLY_YEARLY = 1

# common periods: 1m, 5m, 15m, 60m/1h
# todo(nimish): helper functions to set proper periods based on data's inferred frequency
# infer frequency of data and then compute values for periods automatically


def get_recommended_periods(X: pd.DataFrame, include_harmonics: bool = False) -> list[float] | tuple[list[float], list[int]]:
    """
    Get recommended periods for Fourier basis based on data frequency.

    This function infers the frequency of the input time series data and returns
    recommended periods (in hours) that are appropriate for capturing seasonal
    patterns at that time scale. Periods are calculated as multiples of the
    data's base frequency, then converted to hours.

    Parameters
    ----------
    X : pd.DataFrame
        Input data with DatetimeIndex or first column containing datetime values.
    include_harmonics : bool, default=False
        If True, also returns recommended number of harmonics for each period.

    Returns
    -------
    periods : list[float]
        Recommended periods in hours. Periods are calculated as multiples of the
        data's base frequency, then converted to hours. For example:
        - For 5-minute data: multiples [1, 3, 12, 288, 2016] of 5-minute intervals
        - For hourly data: multiples [24, 168, 8765.82] of 1-hour intervals
        - For daily data: multiples [7, 365.2425] of 1-day intervals

        The periods capture:
        - Short-term patterns (small multiples: 1x, 3x, 5x, etc.)
        - Daily patterns (multiples corresponding to ~24 hours)
        - Weekly patterns (multiples corresponding to ~168 hours)
        - Yearly patterns (multiples corresponding to ~8766 hours) when appropriate
    num_harmonics : list[int], optional
        Recommended number of harmonics for each period. Only returned if
        include_harmonics=True. Higher harmonics capture more complex patterns.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tsgam_estimator import get_recommended_periods
    >>>
    >>> # For 1-minute data: periods are multiples of 1-minute intervals
    >>> dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    >>> X = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)
    >>> periods = get_recommended_periods(X)
    >>> # Returns periods like [1/60, 5/60, 15/60, 1, 24, 168] (hours)
    >>> # These correspond to 1, 5, 15, 60, 1440, 10080 minutes
    >>>
    >>> # For 5-minute data: periods are multiples of 5-minute intervals
    >>> dates = pd.date_range('2020-01-01', periods=1000, freq='5min')
    >>> X = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)
    >>> periods = get_recommended_periods(X)
    >>> # Returns periods like [5/60, 15/60, 1, 24, 168] (hours)
    >>> # These correspond to 1, 3, 12, 288, 2016 five-minute intervals
    >>>
    >>> # For hourly data: periods are multiples of 1-hour intervals
    >>> dates = pd.date_range('2020-01-01', periods=1000, freq='h')
    >>> X = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)
    >>> periods, harmonics = get_recommended_periods(X, include_harmonics=True)
    >>> # Returns periods like [24, 168, 8766] (hours)
    >>> # These correspond to 24, 168, 8766 hourly intervals
    """
    # Extract timestamps
    if isinstance(X, pd.DataFrame):
        if isinstance(X.index, pd.DatetimeIndex):
            timestamps = X.index
        elif len(X.columns) > 0 and pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
            timestamps = pd.DatetimeIndex(X.iloc[:, 0])
        else:
            raise ValueError(
                "X must have DatetimeIndex or first column must be datetime. "
                "Got DataFrame without datetime index or datetime column."
            )
    else:
        raise ValueError(
            "X must be a pandas DataFrame with DatetimeIndex or datetime column. "
            f"Got {type(X)} instead."
        )

    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to infer frequency.")

    # Infer frequency and calculate base time step
    inferred_freq = pd.infer_freq(timestamps)
    if inferred_freq is None:
        # Try to infer from differences
        diffs = timestamps[1:] - timestamps[:-1]
        median_diff = diffs.median()
        base_step_hours = median_diff.total_seconds() / 3600.0
        # Convert to approximate frequency string
        if median_diff <= pd.Timedelta(minutes=1):
            inferred_freq = '1min'
        elif median_diff <= pd.Timedelta(minutes=5):
            inferred_freq = '5min'
        elif median_diff <= pd.Timedelta(minutes=15):
            inferred_freq = '15min'
        elif median_diff <= pd.Timedelta(hours=1):
            inferred_freq = 'h'
            base_step_hours = 1.0
        elif median_diff <= pd.Timedelta(days=1):
            inferred_freq = 'D'
            base_step_hours = 24.0
        else:
            raise ValueError(
                "Could not infer frequency from timestamps. "
                "Timestamps must be regularly spaced."
            )
    else:
        # Calculate base step from frequency string using pd.to_timedelta
        try:
            freq_td_str = inferred_freq if inferred_freq[0].isdigit() else f'1{inferred_freq}'
            base_step_hours = pd.to_timedelta(_to_pandas_timedelta_frequency(freq_td_str)).total_seconds() / 3600.0
        except (ValueError, IndexError):
            diffs = timestamps[1:] - timestamps[:-1]
            base_step_hours = diffs.median().total_seconds() / 3600.0

    # Determine periods as multiples of base frequency, then convert to hours.
    # Use base_step_hours ranges to select appropriate period multiples,
    # independent of the particular frequency string format pandas returns.
    periods = []
    num_harmonics = []

    if base_step_hours < 1 / 60:  # Sub-minute frequency
        period_multiples = [1, 5, 15, 60, 1440, 10080]
        num_harmonics = [4, 3, 3, 6, 4, 3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif base_step_hours < 1:  # Sub-hourly (minute-level) frequency
        minutes = round(base_step_hours * 60)
        if minutes == 1:
            period_multiples = [1, 5, 15, 60, 1440, 10080]
            num_harmonics = [4, 3, 3, 6, 4, 3]
        elif minutes == 5:
            period_multiples = [1, 3, 12, 288, 2016]
            num_harmonics = [3, 3, 6, 4, 3]
        elif minutes == 15:
            period_multiples = [1, 4, 96, 672]
            num_harmonics = [3, 6, 4, 3]
        else:
            periods_per_day = (24 * 60) / minutes
            periods_per_week = (7 * 24 * 60) / minutes
            period_multiples = [1, 3, int(periods_per_day / 24), int(periods_per_day), int(periods_per_week)]
            num_harmonics = [3, 3, 6, 4, 3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif abs(base_step_hours - 1.0) < 0.01:  # Hourly
        period_multiples = [24, 168, PERIOD_HOURLY_YEARLY]
        num_harmonics = [6, 4, 3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif abs(base_step_hours - 24.0) < 0.01:  # Daily
        period_multiples = [7, PERIOD_DAILY_YEARLY]
        num_harmonics = [4, 3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif abs(base_step_hours - 168.0) < 0.5:  # Weekly
        period_multiples = [PERIOD_WEEKLY_YEARLY]
        num_harmonics = [3]
        periods = [mult * base_step_hours for mult in period_multiples]
    else:
        # Unknown frequency - provide generic recommendations
        # Try to estimate from median time difference
        diffs = timestamps[1:] - timestamps[:-1]
        median_diff_hours = diffs.median().total_seconds() / 3600.0

        if median_diff_hours < 1/60:  # Sub-minute frequency
            # Use multiples appropriate for minute-level data
            period_multiples = [1, 5, 15, 60, 1440, 10080]
            num_harmonics = [4, 3, 3, 6, 4, 3]
            periods = [mult * base_step_hours for mult in period_multiples]
        elif median_diff_hours < 1:  # Sub-hourly frequency
            # Calculate multiples for daily and weekly patterns
            periods_per_day = 24.0 / base_step_hours
            periods_per_week = 168.0 / base_step_hours
            period_multiples = [int(periods_per_day), int(periods_per_week)]
            num_harmonics = [6, 4, 3]
            periods = [mult * base_step_hours for mult in period_multiples]
        elif median_diff_hours < 24:  # Sub-daily frequency
            # Calculate multiples for daily, weekly, and yearly patterns
            periods_per_day = 24.0 / base_step_hours
            periods_per_week = 168.0 / base_step_hours
            periods_per_year = 365.2425 * 24.0 / base_step_hours
            period_multiples = [int(periods_per_day), int(periods_per_week), int(periods_per_year)]
            num_harmonics = [6, 4, 3]
            periods = [mult * base_step_hours for mult in period_multiples]
        else:  # Daily or longer frequency
            # Calculate multiples for weekly and yearly patterns
            periods_per_week = 7.0 / (base_step_hours / 24.0)
            periods_per_year = 365.2425 / (base_step_hours / 24.0)
            period_multiples = [int(periods_per_week), int(periods_per_year)]
            num_harmonics = [4, 3]
            periods = [mult * base_step_hours for mult in period_multiples]

    if include_harmonics:
        return periods, num_harmonics
    else:
        return periods


class TsgamEstimator(RegressorMixin, BaseEstimator):
    """
    Time Series Generalized Additive Model (TSGAM) Estimator.

    This estimator fits a GAM model for time series forecasting that combines:

    - Multi-periodic Fourier basis functions for seasonal patterns
    - Cubic spline or linear basis functions for exogenous variables with lead/lag
    - Optional trend term (constant per period, linear or nonlinear)
    - Optional outlier detector (sparse multiplicative corrections per period)
    - Optional autoregressive (AR) modeling of residuals

    The model composes its structural components with SignalDecomp and uses
    CVXPY to fit coefficients.
    While the model can work with targets in any scale, log transformation is
    commonly used when components are multiplicative rather than additive.

    Parameters
    ----------
    config : TsgamEstimatorConfig
        Configuration object containing all model settings.

    Attributes
    ----------
    freq_ : str
        Inferred frequency of the time series (e.g., 'h' for hourly).
    time_reference_ : Timestamp
        Reference timestamp used for phase alignment (first timestamp from fit).
    time_indices_ : ndarray
        Numeric time indices (hours since reference) used during fit.
    decomposition_ : dict
        Native SignalDecomp solve result. Fitted components and coefficients are
        available by role under ``decomposition_["values"]``.
    exog_knots_ : list
        List of knot locations for spline exogenous variables (auto-computed
        during fit, reused during predict).
    trend_period_hours_ : float or None
        Period length in hours used for trend (if trend_config provided).
    outlier_period_hours_ : float or None
        Period length in hours used for outlier detector (if outlier_config provided).
    ar_coef_ : ndarray or None
        Fitted AR coefficients (if ar_config provided and model converged).
    ar_intercept_ : float or None
        Fitted AR intercept (if ar_config provided and model converged).
    ar_noise_loc_ : float or None
        Location parameter of Laplace noise distribution for AR model.
    ar_noise_scale_ : float or None
        Scale parameter of Laplace noise distribution for AR model.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tsgam_estimator import (
    ...     TsgamEstimator, TsgamEstimatorConfig,
    ...     TsgamMultiPeriodicConfig, TsgamSplineConfig, TsgamOutlierConfig
    ... )
    >>>
    >>> # Create configuration with outlier detector
    >>> multi_periodic = TsgamMultiPeriodicConfig(
    ...     num_harmonics=[6, 4, 3],
    ...     periods=[365.2425 * 24, 7 * 24, 24]  # yearly, weekly, daily
    ... )
    >>> exog_config = [TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1])]
    >>> outlier_config = TsgamOutlierConfig(reg_weight=0.002)  # Daily outliers
    >>> config = TsgamEstimatorConfig(
    ...     multi_periodic_config=multi_periodic,
    ...     exog_config=exog_config,
    ...     outlier_config=outlier_config
    ... )
    >>>
    >>> # Create estimator
    >>> estimator = TsgamEstimator(config=config)
    >>>
    >>> # Prepare data (X must be DataFrame with DatetimeIndex)
    >>> dates = pd.date_range('2020-01-01', periods=1000, freq='h')
    >>> X = pd.DataFrame({'temp': np.random.randn(1000)}, index=dates)
    >>> y = np.log(np.random.rand(1000) * 100 + 50)  # log-transform recommended
    >>>
    >>> # Fit model
    >>> estimator.fit(X, y)
    >>>
    >>> # Access detected outliers
    >>> outlier_values = estimator.decomposition_["values"]["outlier_group_values"]
    >>> print(f"Detected {np.sum(np.abs(outlier_values) > 0.1)} outlier days")
    >>>
    >>> # Make predictions
    >>> X_pred = pd.DataFrame({'temp': np.random.randn(100)},
    ...                       index=pd.date_range('2021-01-01', periods=100, freq='h'))
    >>> predictions = estimator.predict(X_pred)
    """
    def __init__(self, config: TsgamEstimatorConfig) -> None:
        self.config = config

    def fit(self, X: pd.DataFrame, y: ndarray, sample_weight: ndarray | None = None) -> "TsgamEstimator":
        """
        Fit the TSGAM model to training data.

        This method:
        1. Extracts and validates timestamps from X
        2. Builds Fourier basis matrices for seasonal patterns
        3. Builds spline/linear basis matrices for exogenous variables
        4. Solves the regularized optimization problem
        5. Optionally fits an AR model on residuals

        Parameters
        ----------
        X : DataFrame
            Training data with exogenous variables. Must have DatetimeIndex or
            first column must be datetime. Remaining columns are exogenous variables
            (e.g., temperature). Column order must match exog_config order.
        y : array-like of shape (n_samples,)
            Target values. Can be in any scale, though log transformation is
            commonly used for multiplicative components. Must not contain NaN.
        sample_weight : array-like of shape (n_samples,), default=None
            Optional sample weights for weighted least squares. Must be non-negative
            and match the length of y. If None, all samples are weighted equally (ones).

        Returns
        -------
        self : TsgamEstimator
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If X doesn't have proper timestamp index/column, if frequency doesn't
            match, or if insufficient samples for configured lags.

        Examples
        --------
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=1000, freq='h')
        >>> X = pd.DataFrame({'temp': np.random.randn(1000)}, index=dates)
        >>> y = np.log(np.random.rand(1000) * 100 + 50)
        >>> estimator.fit(X, y)
        TsgamEstimator(...)
        """
        # Validate sample_weight shape before sort (must match X/y length)
        if sample_weight is not None:
            w = np.asarray(sample_weight)
            if w.ndim != 1 or w.shape[0] != len(y):
                raise ValueError(
                    f"sample_weight must have shape (n_samples,) = ({len(y)},), got {w.shape}"
                )
        X, y, sample_weight = sort_fit_inputs(
            X,
            sort_index=self.config.sort_index,
            y=y,
            sample_weight=sample_weight,
        )
        self.n_features_in_ = X.shape[1]
        if all(isinstance(column, str) for column in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        timestamps = _extract_timestamps(X)
        self.freq_ = infer_fit_frequency(timestamps)
        self.time_reference_ = timestamps[0]
        design = build_tsgam_design(
            self.config,
            X,
            y,
            sample_weight,
            reference=self.time_reference_,
            freq=self.freq_,
        )
        assert design.y is not None
        assert design.sample_weight is not None
        time_indices = design.time_indices
        self.time_indices_ = time_indices

        trend_period = None
        if self.config.trend_config is not None and self.config.trend_config.trend_type != TrendType.NONE:
            trend_period = self.config.trend_config.grouping or 24.0
            self.trend_period_hours_ = trend_period

        outlier_period = None
        if self.config.outlier_config is not None:
            outlier_period = self.config.outlier_config.period_hours or 24.0
            self.outlier_period_hours_ = outlier_period

        built = build_single_output_decomposition(
            self.config,
            design,
            trend_period=trend_period,
            outlier_period=outlier_period,
        )
        decomposition = solve(
            built, solver=self.config.solver_config.solver, verify_dcp=True,
            verbose=self.config.solver_config.verbose,
            warm_start=self.config.solver_config.warm_start,
            **self.config.solver_config._solve_kwargs(),
        )
        self.decomposition_ = decomposition
        values = cast(dict[str, ndarray | float], decomposition["values"])
        metadata = cast(dict[str, dict[str, object]], decomposition["component_metadata"])
        self.exog_knots_ = [
            np.asarray(metadata[f"exog_{ix}"]["knots"])
            if isinstance(exog_cfg, TsgamSplineConfig)
            else None
            for ix, exog_cfg in enumerate(self.config.exog_config or [])
        ]

        # Check that constant term is valid
        if np.isnan(cast(ndarray, values["intercept_group_values"])[0]):
            raise ValueError(
                f"Constant term is NaN after optimization. Status: {decomposition['status']}"
            )

        # Fit AR model if configured
        if self.config.ar_config is not None:
            self._fit_ar_model()

        return self

    def _fit_ar_model(self) -> None:
        """Fit the AR model on the SignalDecomp residual."""
        # Compute residuals on valid samples
        fit_mask = cast(ndarray, self.decomposition_["fit_mask"])
        values = cast(dict[str, ndarray | float], self.decomposition_["values"])
        residuals = np.where(fit_mask, values["residual"], np.nan)

        # Build AR design matrix
        if self.config.ar_config is None:
            return
        ar_config = self.config.ar_config
        ar_lags = len(ar_config.lags)
        basis = make_offset_basis(residuals, offsets=tuple(reversed(ar_config.lags)))
        B = basis.design
        ar_valid_mask = fit_mask & basis.valid_mask

        if self.config.debug:
            self._B_running_view_ = B
            self._ar_valid_mask_ = ar_valid_mask
            self._baseline_residuals_ = residuals

        if not np.any(ar_valid_mask):
            # Not enough data for AR model
            self.ar_coef_ = None
            self.ar_intercept_ = None
            self.ar_noise_loc_ = None
            self.ar_noise_scale_ = None
            return

        # Fit AR model using CVXPY
        theta = cvxpy.Variable(ar_lags)
        constant = cvxpy.Variable()

        ar_problem = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.sum_squares(residuals[ar_valid_mask] - B[ar_valid_mask] @ theta - constant)),
            [cvxpy.norm1(theta) <= ar_config.l1_constraint]
        )
        ar_problem.solve(
            solver=self.config.solver_config.solver,
            verbose=self.config.solver_config.verbose,
            warm_start=self.config.solver_config.warm_start,
            **self.config.solver_config._solve_kwargs(),
        )

        if ar_problem.status in ["optimal", "optimal_inaccurate"]:
            assert theta.value is not None, "AR coefficients should be set"
            assert constant.value is not None, "AR intercept should be set"
            self.ar_coef_ = theta.value
            self.ar_intercept_ = constant.value

            # Fit Laplace distribution to AR model residuals
            ar_model = B[ar_valid_mask] @ theta.value + constant.value
            ar_residuals = residuals[ar_valid_mask] - ar_model
            self.ar_noise_loc_, self.ar_noise_scale_ = stats.laplace.fit(ar_residuals)
        else:
            # AR model failed to solve
            self.ar_coef_ = None
            self.ar_intercept_ = None
            self.ar_noise_loc_ = None
            self.ar_noise_scale_ = None

    def predict(self, X: pd.DataFrame,
                remove_periodic : bool = False, remove_exogenous : bool = False,
                remove_trend : bool = False) -> ndarray:
        """
        Predict target values for new data.

        Predictions are made using the fitted model components:
        - Constant term
        - Fourier basis (seasonal patterns)
        - Exogenous variable basis (splines/linear)
        - Trend term (if configured)
        - AR model is NOT included in predictions (use sample() for AR noise)

        Parameters
        ----------
        X : DataFrame
            Input data with exogenous variables. Must have DatetimeIndex or
            first column must be datetime. Must have same frequency as training data.
            Column order must match training data. Rows lacking source history
            for an exogenous offset return NaN; include source padding to predict them.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values in the same scale as training data. If training data
            was log-transformed, predictions will be in log space and can be
            converted back using np.exp(predictions).

        Raises
        ------
        ValueError
            If model not fitted, if X doesn't have proper timestamp index/column,
            if frequency doesn't match training data, or if model didn't converge.

        Examples
        --------
        >>> # After fitting
        >>> X_pred = pd.DataFrame({'temp': np.random.randn(100)},
        ...                       index=pd.date_range('2021-01-01', periods=100, freq='h'))
        >>> predictions = estimator.predict(X_pred)
        >>> # Convert back to original scale
        >>> predictions_original = np.exp(predictions)
        """
        check_is_fitted(self, ['decomposition_', 'time_reference_', 'freq_'])

        X = sort_predict_X(X, sort_index=self.config.sort_index)
        timestamps = _extract_timestamps(X)
        validate_predict_frequency(timestamps, self.freq_)
        design = build_tsgam_design(
            self.config,
            X,
            y=None,
            sample_weight=None,
            knots_by_exog=self.exog_knots_,
            reference=self.time_reference_,
            freq=self.freq_,
        )
        time_indices = design.time_indices
        predictions = evaluate_single_output_prediction(
            self.config,
            design,
            cast(dict[str, ndarray | float], self.decomposition_["values"]),
            remove_periodic=remove_periodic,
            remove_exogenous=remove_exogenous,
        )

        values = cast(dict[str, ndarray | float], self.decomposition_["values"])
        if (
            not remove_trend
            and self.config.trend_config is not None
            and self.config.trend_config.trend_type != TrendType.NONE
        ):
            trend = cast(ndarray, values["trend_group_values"])
            period_indices = (time_indices / self.trend_period_hours_).astype(int)
            valid = period_indices >= 0
            selected = period_indices[valid]
            contribution = trend[np.minimum(selected, len(trend) - 1)].copy()
            beyond = selected >= len(trend)
            if self.config.trend_config.trend_type == TrendType.LINEAR:
                contribution[beyond] += float(values["trend_slope"]) * (
                    selected[beyond] - len(trend) + 1
                )
            predictions[valid] += contribution

        # Final check for NaN in predictions
        if np.any(np.isnan(predictions) & design.valid_mask):
            nan_count = np.sum(np.isnan(predictions))
            nan_indices = np.where(np.isnan(predictions))[0]
            raise ValueError(
                f"Predictions contain {nan_count} NaN values out of {len(predictions)}. "
                f"First few NaN indices: {nan_indices[:10] if len(nan_indices) > 0 else []}. "
                f"Constant value: {cast(ndarray, values['intercept_group_values'])[0]}, "
                f"Time indices range: [{time_indices.min():.1f}, {time_indices.max():.1f}]"
            )

        return predictions

    def sample(self, X: pd.DataFrame, n_samples: int = 1, random_state: RandomState | int | None = None) -> ndarray:
        """
        Generate sample predictions with AR noise rollout.

        This method generates multiple sample paths by adding noise to baseline
        predictions. If an AR model was fitted, it uses AR noise rollout to generate
        temporally correlated noise. Otherwise, it adds independent Laplace noise.

        The AR noise rollout:
        1. Initializes with random noise from fitted Laplace distribution
        2. Generates AR noise using: noise[t] = AR_coef @ noise[t-lags] + intercept + new_noise
        3. Adds burn-in period before using samples

        Parameters
        ----------
        X : DataFrame
            Input data with timestamps. Same format as predict().
        n_samples : int, default=1
            Number of sample paths to generate.
        random_state : int, RandomState instance or None, default=None
            Random state for reproducible results. If None, uses estimator's
            random_state from config.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_pred_samples)
            Sample predictions in the same scale as training data. Each row is one
            sample path. If AR model is fitted, includes temporally correlated AR
            noise. Otherwise, adds independent small Laplace noise (scale=0.1).

        Raises
        ------
        ValueError
            If model not fitted or if AR model was configured but didn't converge.

        Examples
        --------
        >>> # Generate 100 sample paths
        >>> samples = estimator.sample(X_pred, n_samples=100, random_state=42)
        >>> # samples shape: (100, n_pred_samples)
        >>> # If data was log-transformed, convert back to original scale
        >>> samples_original = np.exp(samples)
        >>> # Compute percentiles
        >>> p5 = np.percentile(samples_original, 5, axis=0)
        >>> p95 = np.percentile(samples_original, 95, axis=0)
        """
        check_is_fitted(self, ['decomposition_', 'time_reference_', 'freq_'])
        if random_state is None:
            random_state = self.config.random_state
        random_state = check_random_state(random_state)

        # Get baseline predictions
        baseline_pred = self.predict(X)

        if self.config.ar_config is not None and hasattr(self, 'ar_coef_') and self.ar_coef_ is not None:
            samples = self._generate_ar_samples(baseline_pred, n_samples, random_state)
        else:
            # No AR model, just add small noise
            noise = stats.laplace.rvs(
                loc=0, scale=0.1, size=(n_samples, len(baseline_pred)),
                random_state=random_state
            )
            samples = baseline_pred + noise

        return samples

    def _generate_ar_samples(self, baseline_pred: ndarray, n_samples: int, random_state: RandomState) -> ndarray:
        """
        Generate samples with AR noise rollout using residuals.

        Parameters
        ----------
        baseline_pred : ndarray
            Baseline predictions (same scale as training data).
        n_samples : int
            Number of samples to generate.
        random_state : RandomState
            Random state for reproducible results.

        Returns
        -------
        samples : ndarray of shape (n_samples, len(baseline_pred))
            Sample predictions with AR noise (same scale as training data).
        """
        assert self.ar_coef_ is not None and self.ar_intercept_ is not None, \
            "AR coefficients must be set before generating samples"
        assert self.ar_noise_loc_ is not None and self.ar_noise_scale_ is not None, \
            "AR noise distribution parameters must be set before generating samples"

        assert self.config.ar_config is not None
        ar_lags = max(self.config.ar_config.lags)
        length = len(baseline_pred)
        a = np.zeros(ar_lags + 1)
        a[0] = 1.0
        a[self.config.ar_config.lags] = -self.ar_coef_[::-1]
        noise = stats.laplace.rvs(
            loc=self.ar_noise_loc_, scale=self.ar_noise_scale_,
            size=(n_samples, length + 2 * ar_lags), random_state=random_state,
        )
        # Preserve each path's original filter state and burn-in.
        ar_noise, _ = signal.lfilter(
            [1.0], a, self.ar_intercept_ + noise[:, ar_lags:], axis=1,
            zi=noise[:, :ar_lags][:, ::-1].copy(),
        )
        return baseline_pred + ar_noise[:, -length:]



__all__ = [
    "TsgamEstimator",
    "TsgamEstimatorConfig",
    "TsgamMultiPeriodicConfig",
    "TsgamSplineConfig",
    "TsgamLinearConfig",
    "TsgamArConfig",
    "TsgamTrendConfig",
    "TsgamOutlierConfig",
    "TsgamSolverConfig",
    "SolverOptionValue",
    "TrendType",
    "get_recommended_periods",
    "PERIOD_HOURLY_DAILY",
    "PERIOD_HOURLY_WEEKLY",
    "PERIOD_HOURLY_YEARLY",
    "PERIOD_DAILY_YEARLY",
    "PERIOD_WEEKLY_YEARLY",
    "PERIOD_MONTHLY_YEARLY",
    "PERIOD_QUARTERLY_YEARLY",
    "PERIOD_YEARLY_YEARLY",
]
