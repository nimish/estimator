# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from enum import StrEnum
from typing import overload
from itertools import combinations
from numpy import ndarray
import numpy as np
import cvxpy
from numpy.random import RandomState
from scipy import stats, signal
from scipy.sparse import spdiags, spmatrix
from sklearn.base import RegressorMixin, BaseEstimator, check_array, check_is_fitted
from sklearn.utils import check_X_y, check_random_state
from spcqe import make_basis_matrix
from spcqe.functions import initialize_arrays
import pandas as pd

@dataclass
class TsgamMultiPeriodicConfig:
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

@dataclass
class TsgamSplineConfig:
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
        Lead/lag offsets for the exogenous variable. Positive values = lag
        (looking back), negative values = lead (looking forward). For example,
        [-3, -2, -1, 0, 1, 2, 3] includes 3 hours ahead, current, and 3 hours back.
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
    n_knots: int | None
    lags: list[int] = field(default_factory=lambda:[0])
    reg_weight: float = 1.0e-4
    diff_reg_weight: float = 1.0
    knots: ndarray | list[float] = field(default_factory=list)

@dataclass
class TsgamLinearConfig:
    """
    Configuration for linear basis functions for exogenous variables.

    This config defines how an exogenous variable is modeled using simple linear
    terms with optional lead/lag. Use this instead of TsgamSplineConfig when
    you expect a linear relationship.

    Parameters
    ----------
    lags : list[int], default=[0]
        Lead/lag offsets for the exogenous variable. Positive values = lag
        (looking back), negative values = lead (looking forward). For example,
        [-1, 0, 1] includes 1 hour ahead, current, and 1 hour back.
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
class TsgamArConfig:
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

@dataclass
class TsgamTrendConfig:
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
class TsgamOutlierConfig:
    """
    Configuration for the outlier detector component.

    The outlier detector identifies anomalous periods (e.g., days) in the time series
    with sparse multiplicative corrections. This component is particularly useful for
    detecting days with unusual patterns that deviate from the normal seasonal and
    trend behavior, such as holidays, special events, or data quality issues.

    **How it works:**
    - The detector assigns one correction value per period (e.g., per day)
    - Corrections are constant across all samples within a period
    - L1 regularization encourages sparsity: most periods have no correction (≈0),
      while only outlier periods have non-zero corrections
    - The correction is additive in log space, which translates to a multiplicative
      effect in the original scale (e.g., 0.2 in log space ≈ 0.82x multiplier,
      0.5 in log space ≈ 1.65x multiplier)

    **Mathematical formulation:**
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

        **Typical ranges:**
        - Very sparse (few outliers): 0.01 to 0.1
        - Moderate sparsity: 0.001 to 0.01
        - More sensitive (more outliers): 0.0001 to 0.001

        **Guidelines:**
        - Start with 0.002 and adjust based on results
        - If too many outliers detected (>10% of periods), increase reg_weight
        - If no outliers detected, decrease reg_weight
        - For real-world data, values between 0.001 and 0.01 are often appropriate

    period_hours : float or None, default=None
        Period length in hours for which the outlier correction is constant.
        If None, defaults to 24.0 hours (daily outliers) for hourly data.

        **Examples:**
        - Hourly data, daily outliers: ``period_hours=24.0`` (default)
        - 15-minute data, daily outliers: ``period_hours=24.0`` (96 samples per day)
        - Hourly data, weekly outliers: ``period_hours=168.0`` (7 days)
        - Daily data, weekly outliers: ``period_hours=7.0``
        - Hourly data, hourly outliers: ``period_hours=1.0`` (not recommended, use AR instead)

    Attributes
    ----------
    reg_weight : float
        L1 regularization weight.
    period_hours : float or None
        Period length in hours.

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
class TsgamSolverConfig:
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

@dataclass
class TsgamEstimatorConfig:
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
    random_state : RandomState or None, default=None
        Random state for reproducible results. Used in AR sampling if ar_config
        is provided.
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
    ar_config: TsgamArConfig | None = None
    trend_config: TsgamTrendConfig | None = None
    outlier_config: TsgamOutlierConfig | None = None
    solver_config: TsgamSolverConfig = field(default_factory=TsgamSolverConfig)
    sort_index: bool = True
    random_state: RandomState | None = None
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
            base_step_hours = pd.to_timedelta(freq_td_str).total_seconds() / 3600.0
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


class TsgamEstimator(BaseEstimator, RegressorMixin):
    """
    Time Series Generalized Additive Model (TSGAM) Estimator.

    This estimator fits a GAM model for time series forecasting that combines:

    - Multi-periodic Fourier basis functions for seasonal patterns
    - Cubic spline or linear basis functions for exogenous variables with lead/lag
    - Optional trend term (constant per period, linear or nonlinear)
    - Optional outlier detector (sparse multiplicative corrections per period)
    - Optional autoregressive (AR) modeling of residuals

    The model uses regularized optimization via CVXPY to fit coefficients.
    While the model can work with targets in any scale, log transformation is
    commonly used when components are multiplicative rather than additive.

    Parameters
    ----------
    config : TsgamEstimatorConfig
        Configuration object containing all model settings.

    Attributes
    ----------
    problem_ : cvxpy.Problem
        The solved optimization problem. Check `problem_.status` to verify
        convergence (should be 'optimal' or 'optimal_inaccurate').
    freq_ : str
        Inferred frequency of the time series (e.g., 'h' for hourly).
    time_reference_ : Timestamp
        Reference timestamp used for phase alignment (first timestamp from fit).
    time_indices_ : ndarray
        Numeric time indices (hours since reference) used during fit.
    variables_ : dict
        Dictionary of CVXPY variables containing fitted coefficients:
        - 'constant': intercept term
        - 'fourier_coef': Fourier coefficients (if multi_periodic_config provided)
        - 'exog_coef_{i}': Exogenous variable coefficients for variable i
        - 'trend': Trend coefficients (if trend_config provided)
        - 'trend_slope': Trend slope (if trend_type='linear')
        - 'outlier': Outlier coefficients (if outlier_config provided)
    exog_knots_ : list
        List of knot locations for spline exogenous variables (auto-computed
        during fit, reused during predict).
    trend_T_matrix_ : ndarray or None
        Matrix mapping samples to periods for trend term (if trend_config provided).
    trend_period_hours_ : float or None
        Period length in hours used for trend (if trend_config provided).
    outlier_T_matrix_ : ndarray or None
        Matrix mapping samples to periods for outlier detector (if outlier_config provided).
    outlier_period_hours_ : float or None
        Period length in hours used for outlier detector (if outlier_config provided).
    combined_valid_mask_ : ndarray
        Boolean mask indicating valid samples (no NaN from lead/lag operations).
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
    >>> outlier_values = estimator.variables_['outlier'].value
    >>> print(f"Detected {np.sum(np.abs(outlier_values) > 0.1)} outlier days")
    >>>
    >>> # Make predictions
    >>> X_pred = pd.DataFrame({'temp': np.random.randn(100)},
    ...                       index=pd.date_range('2021-01-01', periods=100, freq='h'))
    >>> predictions = estimator.predict(X_pred)
    """
    def __init__(self, config: TsgamEstimatorConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

    def _extract_timestamps(self, X: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Extract timestamps from X.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data. If DataFrame with DatetimeIndex, extracts index.
            If DataFrame, checks first column for datetime.
            Otherwise raises ValueError.

        Returns
        -------
        timestamps : DatetimeIndex
            Extracted timestamps.
        """
        if isinstance(X, pd.DataFrame):
            # Check if index is DatetimeIndex
            if isinstance(X.index, pd.DatetimeIndex):
                return X.index
            # Check if first column is datetime
            elif len(X.columns) > 0 and pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
                return pd.DatetimeIndex(X.iloc[:, 0])
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

    @staticmethod
    def _ensure_numeric_prefix(freq: str) -> str:
        """Ensure frequency string has a numeric prefix (e.g. ``'h'`` -> ``'1h'``)."""
        if freq and not freq[0].isdigit():
            return f'1{freq}'
        return freq

    def _timestamps_to_indices(self, timestamps: pd.DatetimeIndex, reference: pd.Timestamp) -> ndarray:
        """
        Convert timestamps to numeric indices (hours since reference).

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps to convert.
        reference : Timestamp
            Reference timestamp (time 0).

        Returns
        -------
        indices : ndarray
            Numeric indices in hours since reference.
        """
        freq = getattr(self, 'freq_', None)
        if freq is None:
            freq = pd.infer_freq(timestamps)
            if freq is None:
                freq = self._infer_frequency_from_differences(timestamps)
            freq = self._ensure_numeric_prefix(freq)
        return ((timestamps - reference) / pd.to_timedelta(freq)).astype(int)

    def _get_trend_period_hours(self, timestamps: pd.DatetimeIndex, period_hours: float | None = None) -> tuple[float, float]:
        """
        Determine trend period in hours from data frequency.

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps from the data.
        period_hours : float or None, default=None
            Explicit period in hours. If None, defaults to daily (24 hours for
            sub-daily data, 1 day for daily data, etc.).

        Returns
        -------
        period_hours : float
            Period length in hours.
        samples_per_period : float
            Number of samples per period (for creating T matrix).
        """
        if period_hours is not None:
            # Use explicit period
            # Calculate samples per period from data frequency
            if len(timestamps) < 2:
                raise ValueError("Need at least 2 timestamps to infer frequency.")
            diffs = timestamps[1:] - timestamps[:-1]
            median_diff_hours = diffs.median().total_seconds() / 3600.0
            samples_per_period = period_hours / median_diff_hours
            return period_hours, samples_per_period

        # Default to daily period
        # Infer frequency and calculate base time step
        inferred_freq = pd.infer_freq(timestamps)
        if inferred_freq is None:
            # Try to infer from differences
            diffs = timestamps[1:] - timestamps[:-1]
            median_diff = diffs.median()
            base_step_hours = median_diff.total_seconds() / 3600.0
        else:
            try:
                freq_td_str = inferred_freq if inferred_freq[0].isdigit() else f'1{inferred_freq}'
                base_step_hours = pd.to_timedelta(freq_td_str).total_seconds() / 3600.0
            except (ValueError, IndexError):
                diffs = timestamps[1:] - timestamps[:-1]
                base_step_hours = diffs.median().total_seconds() / 3600.0

        # Default period: daily (24 hours)
        period_hours = 24.0
        samples_per_period = period_hours / base_step_hours

        return period_hours, samples_per_period

    def _infer_frequency_from_differences(self, timestamps: pd.DatetimeIndex) -> str:
        """
        Infer the intended frequency from time differences, even when there are gaps.

        This method finds the most common time difference between consecutive timestamps
        and maps it to a pandas frequency string.

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps (may have gaps)

        Returns
        -------
        freq : str
            Inferred frequency string (e.g., 'h', '15min', 'D')
        """
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 timestamps to infer frequency")

        # Compute time differences
        diffs = timestamps[1:] - timestamps[:-1]
        diff_seconds = np.array([d.total_seconds() for d in diffs])

        # Find the most common difference (mode)
        # Use histogram to find the most frequent difference
        # Round to nearest second to handle floating point issues
        diff_seconds_rounded = np.round(diff_seconds).astype(int)
        unique_diffs, counts = np.unique(diff_seconds_rounded, return_counts=True)
        most_common_diff_seconds = unique_diffs[np.argmax(counts)]

        # Map to pandas frequency string
        # Common mappings:
        freq_mapping = {
            60: '1min',
            300: '5min',
            900: '15min',
            3600: '1h',
            86400: '1D',
        }

        # Try exact match first
        if most_common_diff_seconds in freq_mapping:
            return freq_mapping[most_common_diff_seconds]

        # Try approximate match (within 1% tolerance)
        for diff_sec, freq_str in freq_mapping.items():
            if abs(most_common_diff_seconds - diff_sec) / diff_sec < 0.01:
                return freq_str

        # If no match, try to construct frequency from seconds
        if most_common_diff_seconds < 60:
            return f'{most_common_diff_seconds}S'
        elif most_common_diff_seconds < 3600:
            minutes = most_common_diff_seconds // 60
            return f'{minutes}min'
        elif most_common_diff_seconds < 86400:
            hours = most_common_diff_seconds // 3600
            return f'{hours}h'
        else:
            days = most_common_diff_seconds // 86400
            return f'{days}D'

    def _validate_frequency(self, timestamps: pd.DatetimeIndex, expected_freq: str, allow_gaps: bool = False) -> None:
        """
        Validate that timestamps match expected frequency, optionally allowing gaps.

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps to validate.
        expected_freq : str
            Expected pandas frequency string (e.g., 'h' for hourly, 'H' also accepted).
        allow_gaps : bool, default=False
            If True, allow gaps in timestamps and infer base frequency from
            time differences.  If False, require perfectly regular timestamps.

        Raises
        ------
        ValueError
            If frequency doesn't match or timestamps are not regular (when allow_gaps=False).
        """
        if len(timestamps) < 2:
            return  # Can't validate frequency with < 2 samples

        inferred_freq = pd.infer_freq(timestamps)

        if inferred_freq is None:
            if allow_gaps:
                inferred_freq = self._infer_frequency_from_differences(timestamps)
            else:
                raise ValueError(
                    f"Could not infer frequency from timestamps. "
                    f"Timestamps must be regularly spaced with frequency '{expected_freq}'."
                )

        if inferred_freq is None:
            raise ValueError(
                f"Could not infer frequency from timestamps. "
                f"Timestamps must be regularly spaced with frequency '{expected_freq}'."
            )

        if self._ensure_numeric_prefix(inferred_freq).lower() != self._ensure_numeric_prefix(expected_freq).lower():
            raise ValueError(
                f"Timestamps frequency '{inferred_freq}' does not match "
                f"expected frequency '{expected_freq}'."
            )

    def _ensure_timestamp_index(self, X: pd.DataFrame) -> tuple[pd.DatetimeIndex, ndarray]:
        """
        Ensure X has proper timestamp index/column, extracting timestamps.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data.

        Returns
        -------
        timestamps : DatetimeIndex
            Extracted timestamps.
        X_array : ndarray
            X as array without timestamp column if it was extracted.
        """
        timestamps = self._extract_timestamps(X)

        # If X is DataFrame and first column was datetime, remove it
        if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.DatetimeIndex):
            if pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
                X_array = X.iloc[:, 1:].values
            else:
                X_array = X.values
        else:
            X_array = X.values if isinstance(X, pd.DataFrame) else X

        return timestamps, X_array

    @overload
    def _ensure_sorted_index(
        self, X: pd.DataFrame, y: ndarray, sample_weight: ndarray | None = None
    ) -> tuple[pd.DataFrame, ndarray, ndarray | None]: ...
    @overload
    def _ensure_sorted_index(
        self, X: pd.DataFrame, y: None = None, sample_weight: None = None
    ) -> tuple[pd.DataFrame]: ...

    def _ensure_sorted_index(
        self,
        X: pd.DataFrame,
        y: ndarray | None = None,
        sample_weight: ndarray | None = None,
    ) -> tuple[pd.DataFrame, ndarray, ndarray | None] | tuple[pd.DataFrame]:
        """
        Sort X (and y, sample_weight if provided) by datetime index, or require sorted.

        If config.sort_index is True, sort by timestamps so row order matches
        time order. If False, require the index to be chronologically sorted
        and raise ValueError if not.

        Parameters
        ----------
        X : DataFrame
            Input data with DatetimeIndex or datetime column.
        y : array-like of shape (n_samples,) or None
            Target values (fit only). If provided, reordered in the same way as X.
        sample_weight : array-like of shape (n_samples,) or None
            Sample weights (fit only). If provided, reordered with X and y.

        Returns
        -------
        If y is None: (X_sorted,)
        If y is not None: (X_sorted, y_sorted, sample_weight_sorted or None)
        """
        timestamps = self._extract_timestamps(X)
        if self.config.sort_index:
            sort_idx = np.argsort(timestamps)
            X = X.iloc[sort_idx]
            if y is not None:
                y = np.asarray(y)[sort_idx]
                sw = (
                    np.asarray(sample_weight)[sort_idx]
                    if sample_weight is not None
                    else None
                )
                return (X, y, sw)
            return (X,)
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                "Data index is not sorted chronologically. Sort the DataFrame by "
                "its datetime index (e.g. X = X.sort_index()) or set "
                "config.sort_index=True to sort automatically."
            )
        if y is not None:
            sw = np.asarray(sample_weight) if sample_weight is not None else None
            return (X, y, sw)
        return (X,)

    def _make_regularization_matrix(self, num_harmonics: list[int],
                                   weight: float,
                                   periods: list[float],
                                   drop_constant: bool = False,
                                   standing_wave: bool | list[bool] = False,
                                   trend: bool = False,
                                   max_cross_k: int | None = None,
                                   custom_basis: dict[int, ndarray] | None = None) -> spmatrix:
        """
        Create regularization matrix for Fourier coefficients.

        Parameters
        ----------
        num_harmonics : int or array-like
            Number of harmonics for each period.
        weight : float
            Regularization weight.
        periods : float or array-like
            Periods for each harmonic block.
        standing_wave : bool or array-like, default=False
            Whether to use standing wave basis.
        trend : bool, default=False
            Whether to include trend term.
        max_cross_k : int or None, default=None
            Maximum cross terms.
        custom_basis : dict or None, default=None
            Custom basis matrices.

        Returns
        -------
        D : sparse matrix
            Regularization matrix.
        """
        sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
            num_harmonics, periods, standing_wave, custom_basis
        )
        ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]

        # Create sequence of values from 1 to K
        i_value_list = []
        for ix, nh in enumerate(num_harmonics):
            if standing_wave[ix]:
                i_value_list.append(np.arange(1, nh + 1))
            else:
                i_value_list.append(np.repeat(np.arange(1, nh + 1), 2))

        # Create blocks of coefficients
        blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
        if custom_basis is not None:
            for ix, val in custom_basis.items():
                ixt = np.where(sort_idx == ix)[0][0]
                blocks_original[ixt] = ls_original[ixt] * np.arange(1, val.shape[1] + 1)

        if max_cross_k is not None:
            max_cross_k *= 2

        # Compute cross-term penalties
        blocks_cross = [
            [l2 for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]]
            for c in combinations(blocks_original, 2)
        ]

        # Combine blocks
        if trend is False:
            first_block = [np.zeros(1)]
        else:
            first_block = [np.zeros(2)]

        if drop_constant:
            first_block = first_block[1:]

        coeff_i = np.concatenate(first_block + blocks_original + blocks_cross)

        D = spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)
        return D



    def _make_H(self, x: ndarray, knots: ndarray, include_offset: bool = False) -> ndarray:
        """
        Create cubic spline basis matrix.

        Parameters
        ----------
        x : array-like
            Input values.
        knots : array-like
            Knot locations.
        include_offset : bool, default=False
            Whether to include constant term.

        Returns
        -------
        H : ndarray
            Basis matrix.
        """
        def d_func(x, k, k_max):
            n1 = np.clip(np.power(x - k, 3), 0, np.inf)
            n2 = np.clip(np.power(x - k_max, 3), 0, np.inf)
            d1 = k_max - k
            out = (n1 - n2) / d1
            return out

        nK = len(knots)
        H = np.ones((len(x), nK), dtype=float)
        H[:, 1] = x
        for _i in range(nK - 2):
            _j = _i + 2
            H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(
                x, knots[-2], knots[-1]
            )
        if include_offset:
            return H
        else:
            return H[:, 1:]

    def _make_offset_H(self, H: ndarray, offset: int) -> ndarray:
        """
        Create lead/lag version of basis matrix.

        Parameters
        ----------
        H : ndarray
            Original basis matrix.
        offset : int
            Lead/lag offset (positive = lag, negative = lead).

        Returns
        -------
        newH : ndarray
            Offset basis matrix with NaN padding.
        """
        newH = np.roll(np.copy(H), -offset, axis=0)
        if offset > 0:
            newH[-offset:] = np.nan
        elif offset < 0:
            newH[:-offset] = np.nan
        return newH

    def _running_view(self, arr: ndarray, window: int, lag: int = 1, axis: int = -1) -> ndarray:
        """
        Create running view of array for AR terms.

        Parameters
        ----------
        arr : array-like
            Input array.
        window : int
            Window size (number of AR lags).
        lag : int, default=1
            Lag offset (typically 1 for standard AR).
        axis : int, default=-1
            Axis along which to create running view.

        Returns
        -------
        view : ndarray
            Running view with extra dimension of shape (len(arr), window).
        """
        mod_arr = np.r_[np.ones(window + lag - 1) * np.nan, arr[:-1]]
        shape = list(mod_arr.shape)
        shape[axis] -= (window - 1)
        assert shape[axis] > 0, f"Array too short for window={window}, lag={lag}"
        return np.lib.stride_tricks.as_strided(
            mod_arr,
            shape=shape + [window],
            strides=mod_arr.strides + (mod_arr.strides[axis],)
        )

    def _build_exog_Hs(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray, knots: ndarray | None = None) -> list[ndarray]:
        """
        Build basis matrices for an exogenous variable with lead/lag.

        This is a helper method that can be reused in both fit and predict.

        Parameters
        ----------
        exog_cfg : TsgamSplineConfig or TsgamLinearConfig
            Configuration for the exogenous variable.
        exog_var : ndarray
            Single exogenous variable column (shape: (n_samples,)).
        knots : ndarray or None, default=None
            Knot locations for spline (if None and spline config, will be computed or error).

        Returns
        -------
        Hs : list of ndarray
            List of basis matrices, one for each lag in exog_cfg.lags.
        """
        Hs = []

        for lag in exog_cfg.lags:
            if isinstance(exog_cfg, TsgamSplineConfig):
                if knots is None:
                    raise ValueError("knots must be provided for TsgamSplineConfig")
                H0 = self._make_H(exog_var, knots, include_offset=False)
                H_lag = self._make_offset_H(H0, lag)
            else:  # TsgamLinearConfig
                H0 = exog_var.reshape(-1, 1)
                H_lag = self._make_offset_H(H0, lag)

            Hs.append(H_lag)

        return Hs

    def _process_exog_config(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray, knots: ndarray | None = None) -> tuple[ndarray, list[ndarray]]:
        """
        Process an exogenous variable configuration to build basis matrices.

        Parameters
        ----------
        exog_cfg : TsgamSplineConfig or TsgamLinearConfig
            Configuration for the exogenous variable.
        exog_var : ndarray
            Single exogenous variable column (shape: (n_samples,)).
        knots : ndarray or None, optional
            Pre-computed knots to use (for prediction). If None, computes from config or data.

        Returns
        -------
        valid_mask : ndarray
            Boolean mask indicating valid samples (no NaN from lead/lag operations).
        Hs : list of ndarray
            List of basis matrices, one for each lag in exog_cfg.lags.
        """
        # Get knots if spline config
        if knots is None:
            if isinstance(exog_cfg, TsgamSplineConfig):
                cfg_knots = np.asarray(exog_cfg.knots) if exog_cfg.knots is not None else np.array([])
                if len(cfg_knots) == 0:
                    if exog_cfg.n_knots:
                        knots = np.linspace(np.min(exog_var), np.max(exog_var), exog_cfg.n_knots)
                    else:
                        raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                else:
                    knots = cfg_knots
            else:
                knots = None

        # Build Hs using helper method
        # Ensure knots is a numpy array if provided
        if knots is not None:
            knots = np.asarray(knots)
        Hs = self._build_exog_Hs(exog_cfg, exog_var, knots)

        # Find valid samples (no NaN from lead/lag operations)
        valid_mask = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)

        return valid_mask, Hs

    def _get_min_samples_required(self) -> int:
        """
        Calculate minimum number of samples required based on lags.

        For positive lags (looking back), we need at least that many samples.
        For negative lags (leads/looking forward), we need at least abs(lag) samples.
        For AR lags, we need at least max(ar_lags) samples.

        Returns
        -------
        min_samples : int
            Minimum number of samples required.
        """
        all_exog_lags = []
        for exog_cfg in self.config.exog_config or []:
            all_exog_lags.extend(exog_cfg.lags)

        max_positive_lag = 0
        min_negative_lag = 0
        for lag in all_exog_lags:
            if lag > 0:
                max_positive_lag = max(max_positive_lag, lag)
            elif lag < 0:
                min_negative_lag = min(min_negative_lag, lag)
        max_negative_lag = abs(min_negative_lag)

        # AR lags are typically positive (looking back)
        max_ar_lag = 0
        if self.config.ar_config is not None:
            max_ar_lag = max(self.config.ar_config.lags) if self.config.ar_config.lags else 0

        # Need enough samples for the maximum backward-looking lag (exog or AR)
        # plus enough samples for the maximum forward-looking lag
        # plus 1 for at least one valid sample where all requirements overlap
        min_samples = max(max_positive_lag, max_ar_lag) + max_negative_lag + 1

        return min_samples

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
        # Sort by index or require sorted (config.sort_index); reorder y and sample_weight too
        X, y, sample_weight = self._ensure_sorted_index(X, y, sample_weight)

        # Extract timestamps before check_X_y converts DataFrame to array
        timestamps, X_array = self._ensure_timestamp_index(X)

        # Try to infer frequency - use gap-tolerant method if regular inference fails
        inferred_freq = pd.infer_freq(timestamps)
        if inferred_freq is None:
            # Infer from time differences (handles gaps)
            inferred_freq = self._infer_frequency_from_differences(timestamps)

        # Validate frequency (allow gaps)
        self._validate_frequency(timestamps, inferred_freq, allow_gaps=True)

        # Store frequency and reference timestamp
        self.freq_ = self._ensure_numeric_prefix(inferred_freq).lower()
        self.time_reference_ = timestamps[0]

        # Convert timestamps to numeric indices (hours since reference)
        time_indices = self._timestamps_to_indices(timestamps, self.time_reference_)
        self.time_indices_ = time_indices

        # Now validate X and y with the array version - check_X_y will reject NaN's
        X_array, y = check_X_y(X_array, y,
            ensure_min_features=len(self.config.exog_config or []),
            ensure_min_samples=self._get_min_samples_required())

        # Validate and store sample weights for weighted least squares (main loss only)
        if sample_weight is None:
            self._sample_weight_ = np.ones(len(y), dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != (len(y),):
                raise ValueError(
                    f"sample_weight must have shape (n_samples,) = ({len(y)},), got {w.shape}"
                )
            if np.any(w < 0):
                raise ValueError("sample_weight must be non-negative")
            if np.sum(w) <= 0:
                raise ValueError("sample_weight must have positive sum")
            self._sample_weight_ = w

        self.variables_ = {
            'constant': cvxpy.Variable(),
        }
        self.exog_knots_ = []  # Store knots only when auto-computed from training data
        model_term = self.variables_['constant']
        regularization_term = 0
        # Start with mask excluding NaN's in y (defensive programming - check_X_y should have rejected them)
        self.combined_valid_mask_ = ~np.isnan(y)

        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                valid_mask, Hs = self._process_exog_config(exog_cfg, X_array[:, ix])

                # Store knots only if auto-computed (not provided in config)
                if isinstance(exog_cfg, TsgamSplineConfig):
                    cfg_knots = np.asarray(exog_cfg.knots) if exog_cfg.knots is not None else np.array([])
                    if len(cfg_knots) == 0:
                        if exog_cfg.n_knots:
                            knots = np.linspace(np.min(X_array[:, ix]), np.max(X_array[:, ix]), exog_cfg.n_knots)
                            self.exog_knots_.append(knots)
                        else:
                            raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                    else:
                        self.exog_knots_.append(None)
                else:
                    self.exog_knots_.append(None)

                # Create CVXPY variable for coefficients
                # Shape: (basis_dim, num_lags)
                basis_dim = Hs[0].shape[1]
                num_lags = len(exog_cfg.lags)
                exog_coef = cvxpy.Variable((basis_dim, num_lags))

                self.variables_[f'exog_coef_{ix}'] = exog_coef
                regularization_term += cvxpy.sum_squares(exog_coef) * exog_cfg.reg_weight
                if len(exog_cfg.lags) > 1:
                    regularization_term += cvxpy.sum_squares(cvxpy.diff(exog_coef, axis=1)) * exog_cfg.diff_reg_weight
                self.combined_valid_mask_ &= valid_mask

            for ix, exog_cfg in enumerate(self.config.exog_config):
                # Rebuild Hs to build model term (Hs are only needed during fit)
                valid_mask, Hs = self._process_exog_config(exog_cfg, X_array[:, ix])
                # Sum over lags: H @ exog_coef[:, lag_ix] for each lag
                model_term += cvxpy.sum(expr=[H[self.combined_valid_mask_] @ self.variables_[f'exog_coef_{ix}'][:, lag_ix] for lag_ix, H in enumerate(Hs)])



        if self.config.multi_periodic_config:
            # Generate basis matrix for max index + 1, then index with time_indices
            # This ensures correct phase alignment (as shown in notebook)
            max_idx = int(np.max(time_indices))
            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_periodic_config.num_harmonics,
                length=max_idx + 1,
                periods=self.config.multi_periodic_config.periods
            )
            # Index with time_indices to get correct rows
            F = F_full[time_indices.astype(int), 1:]  # Drop constant column

            Wf = self._make_regularization_matrix(
                num_harmonics=self.config.multi_periodic_config.num_harmonics,
                weight=1.0,
                periods=self.config.multi_periodic_config.periods,
                drop_constant=True
            )
            self.variables_['fourier_coef'] = cvxpy.Variable(F.shape[1])
            regularization_term += self.config.multi_periodic_config.reg_weight * cvxpy.sum_squares(Wf @ self.variables_['fourier_coef'])
            model_term += F[self.combined_valid_mask_] @ self.variables_['fourier_coef']

        # Add trend term if configured
        trend_term = None
        constraints = []
        if self.config.trend_config is not None and self.config.trend_config.trend_type != TrendType.NONE:
            trend_config = self.config.trend_config

            # Determine period and samples per period
            period_hours, samples_per_period = self._get_trend_period_hours(
                timestamps, trend_config.grouping
            )

            # Calculate number of periods
            # Use time_indices to determine which period each sample belongs to
            period_indices = (time_indices / period_hours).astype(int)
            n_periods = period_indices.max() + 1

            # Create T matrix: maps each sample to its period
            # T[i, j] = 1 if sample i belongs to period j, else 0
            T = np.zeros((len(y), n_periods))
            # Use numpy advanced indexing: T[i, period_indices[i]] = 1.0 for all i
            T[np.arange(len(period_indices)), period_indices] = 1.0

            # Create trend variable (one value per period)
            trend = cvxpy.Variable(n_periods)
            self.variables_['trend'] = trend
            self.trend_T_matrix_ = T  # Store for prediction
            self.trend_period_hours_ = period_hours  # Store period for prediction

            # Add trend term to model
            trend_term = T @ trend
            model_term += trend_term[self.combined_valid_mask_]

            # Add regularization for trend differences
            regularization_term += trend_config.reg_weight * cvxpy.sum_squares(cvxpy.diff(trend))

            # Add constraints based on trend type
            constraints.append(trend[0] == 0)  # Baseline constraint

            if trend_config.trend_type == TrendType.LINEAR:
                # Linear trend: constant slope
                slope = cvxpy.Variable()
                self.variables_['trend_slope'] = slope
                constraints.append(cvxpy.diff(trend) == slope)
            elif trend_config.trend_type == TrendType.NONLINEAR:
                # Nonlinear monotonic decreasing trend
                constraints.append(cvxpy.diff(trend) <= 0)
            # For 'none', trend_term is None so it won't be added

        # Add outlier detector term if configured
        if self.config.outlier_config is not None:
            outlier_config = self.config.outlier_config
            # Determine period (default to 24 hours for daily)
            if outlier_config.period_hours is not None:
                period_hours = outlier_config.period_hours
            else:
                # Default to daily (24 hours)
                period_hours = 24.0

            # Calculate number of periods
            # Use time_indices to determine which period each sample belongs to
            period_indices = (time_indices / period_hours).astype(int)
            n_periods = period_indices.max() + 1

            # Create T matrix: maps each sample to its period
            # T[i, j] = 1 if sample i belongs to period j, else 0
            T = np.zeros((len(y), n_periods))
            # Use numpy advanced indexing: T[i, period_indices[i]] = 1.0 for all i
            T[np.arange(len(period_indices)), period_indices] = 1.0

            # Create outlier variable (one value per period)
            outlier = cvxpy.Variable(n_periods)
            self.variables_['outlier'] = outlier
            self.outlier_T_matrix_ = T  # Store for prediction
            self.outlier_period_hours_ = period_hours  # Store period for prediction

            # Add outlier term to model (additive in log space, multiplicative in original scale)
            outlier_term = T @ outlier
            model_term += outlier_term[self.combined_valid_mask_]

            # Add L1 regularization to encourage sparsity
            regularization_term += outlier_config.reg_weight * cvxpy.norm1(outlier)

        # Weighted least squares: sum(w_i * r_i^2) / sum(w_i) on valid samples
        residual = y[self.combined_valid_mask_] - model_term
        weight_valid = self._sample_weight_[self.combined_valid_mask_]
        error = cvxpy.sum_squares(cvxpy.multiply(np.sqrt(weight_valid), residual)) / np.sum(weight_valid)
        self.problem_ = cvxpy.Problem(cvxpy.Minimize(error + regularization_term), constraints)
        self.problem_.solve(
            solver=self.config.solver_config.solver,
            verbose=self.config.solver_config.verbose,
            warm_start=self.config.solver_config.warm_start,
            **(self.config.solver_config.solver_opts or {}),
        )

        # Check convergence
        if self.problem_.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(
                f"Optimization problem did not converge. Status: {self.problem_.status}. "
                f"This may cause NaN predictions. Check your data and model configuration."
            )

        # Check that constant term is valid
        if self.variables_['constant'].value is None or np.isnan(self.variables_['constant'].value):
            raise ValueError(
                f"Constant term is None or NaN after optimization. Problem status: {self.problem_.status}"
            )

        # Fit AR model if configured
        if self.config.ar_config is not None:
            self._fit_ar_model(X_array, y, time_indices)

        return self

    def _fit_ar_model(self, X_array: ndarray, y: ndarray, time_indices: ndarray) -> None:
        """
        Fit AR model on baseline residuals.

        Parameters
        ----------
        X_array : ndarray
            Exogenous variables array.
        y : ndarray
            Target values.
        time_indices : ndarray
            Time indices for Fourier basis.
        """
        # Get baseline predictions
        baseline_pred = np.full(len(y), self.variables_['constant'].value)

        # Add exogenous terms if present
        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                exog_var = X_array[:, ix]
                stored_knots = self.exog_knots_[ix] if isinstance(exog_cfg, TsgamSplineConfig) else None
                _, Hs = self._process_exog_config(exog_cfg, exog_var, knots=stored_knots)
                exog_coef = self.variables_[f'exog_coef_{ix}'].value
                if exog_coef is not None:
                    exog_pred = np.sum([H @ exog_coef[:, lag_ix] for lag_ix, H in enumerate(Hs)], axis=0)
                    baseline_pred += exog_pred

        # Add Fourier terms if present
        if self.config.multi_periodic_config:
            max_idx = int(np.max(time_indices))
            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_periodic_config.num_harmonics,
                length=max_idx + 1,
                periods=self.config.multi_periodic_config.periods
            )
            F = F_full[time_indices.astype(int), 1:]  # Drop constant column
            fourier_coef = self.variables_['fourier_coef'].value
            if fourier_coef is not None:
                baseline_pred += F @ fourier_coef

        # Add trend term if present
        if self.config.trend_config is not None and self.config.trend_config.trend_type != TrendType.NONE and 'trend' in self.variables_:
            trend = self.variables_['trend'].value
            if trend is not None and hasattr(self, 'trend_T_matrix_'):
                T = self.trend_T_matrix_
                baseline_pred += T @ trend

        # Compute residuals on valid samples
        residuals = y[self.combined_valid_mask_] - baseline_pred[self.combined_valid_mask_]

        # Build AR design matrix
        if self.config.ar_config is None:
            return
        ar_config = self.config.ar_config
        ar_lags = len(ar_config.lags)
        B = self._running_view(residuals, ar_lags)
        ar_valid_mask = np.all(~np.isnan(B), axis=1)

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
            **(self.config.solver_config.solver_opts or {}),
        )

        if ar_problem.status not in ["infeasible", "unbounded"]:
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

    def predict(self, X: pd.DataFrame) -> ndarray:
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
            Column order must match training data.

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
        check_is_fitted(self, ['problem_', 'time_reference_', 'freq_'])

        # Sort by index or require sorted (config.sort_index)
        (X,) = self._ensure_sorted_index(X)

        # todo: check for nan in predict provided data

        # Extract timestamps and validate
        timestamps, X_array = self._ensure_timestamp_index(X)

        # Prediction data must be regularly spaced with no gaps
        self._validate_frequency(timestamps, self.freq_)

        # Convert timestamps to indices using stored reference
        time_indices = self._timestamps_to_indices(timestamps, self.time_reference_)

        # Validate X_array shape
        X_array = check_array(X_array, ensure_min_features=len(self.config.exog_config or []))

        # Initialize prediction with constant term
        constant_value = self.variables_['constant'].value
        if constant_value is None or np.isnan(constant_value):
            raise ValueError(f"Constant term is None or NaN: {constant_value}")
        predictions = np.full(len(X_array), constant_value)

        # Add exogenous terms if present
        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                exog_var = X_array[:, ix]

                # Get stored knots if available (auto-computed during fit), otherwise None
                stored_knots = self.exog_knots_[ix] if isinstance(exog_cfg, TsgamSplineConfig) else None

                # Use _process_exog_config with stored knots (will use config knots if stored_knots is None)
                valid_mask_pred, Hs_pred = self._process_exog_config(exog_cfg, exog_var, knots=stored_knots)

                # Check for NaN in input variable (this is a real problem, not just boundary effects)
                if np.any(np.isnan(exog_var)):
                    raise ValueError(
                        f"Exogenous variable {ix} contains NaN values. "
                        f"NaN count: {np.sum(np.isnan(exog_var))} out of {len(exog_var)}"
                    )

                # Compute exogenous prediction
                exog_coef = self.variables_[f'exog_coef_{ix}'].value
                if exog_coef is None:
                    raise ValueError(f"Exogenous coefficients for variable {ix} are None. Model may not have converged.")
                if np.any(np.isnan(exog_coef)):
                    raise ValueError(f"Exogenous coefficients for variable {ix} contain NaN.")

                # Compute prediction - handle NaN from lead/lag operations gracefully
                # NaN in basis matrices at boundaries is expected and handled by valid_mask
                exog_pred = np.zeros(len(exog_var))
                for lag_ix, H in enumerate(Hs_pred):
                    # For each lag, compute contribution only for valid samples
                    # Samples with NaN in basis matrix (from lead/lag boundaries) get 0 contribution
                    H_clean = np.nan_to_num(H, nan=0.0)  # Replace NaN with 0 for matrix multiplication
                    lag_contribution = H_clean @ exog_coef[:, lag_ix]
                    exog_pred += lag_contribution

                # Final check - should not have NaN after this
                if np.any(np.isnan(exog_pred)):
                    raise ValueError(
                        f"Exogenous prediction for variable {ix} contains NaN after computation. "
                        f"H shapes: {[H.shape for H in Hs_pred]}, exog_coef shape: {exog_coef.shape}, "
                        f"valid_mask has {np.sum(valid_mask_pred)} valid samples out of {len(exog_var)}"
                    )
                predictions += exog_pred

        # Add Fourier terms if present
        if self.config.multi_periodic_config:
            # Check for NaN in time_indices
            if np.any(np.isnan(time_indices)):
                raise ValueError("Time indices contain NaN. Check timestamp conversion.")

            # Generate basis matrix for max index + 1, then index with time_indices
            max_idx = int(np.max(time_indices))
            min_idx = int(np.min(time_indices))

            # Handle negative indices (prediction before fit period)
            # Generate basis matrix from 0 to max_idx, then adjust indices
            if min_idx < 0:
                # Generate enough basis matrix to cover negative indices
                # We'll shift indices to be non-negative
                offset = -min_idx
                adjusted_indices = time_indices.astype(int) + offset
                basis_length = max_idx + offset + 1
            else:
                adjusted_indices = time_indices.astype(int)
                basis_length = max_idx + 1

            # Validate indices are within bounds
            if np.any(adjusted_indices < 0) or np.any(adjusted_indices >= basis_length):
                raise ValueError(
                    f"Adjusted indices out of bounds: min={adjusted_indices.min()}, "
                    f"max={adjusted_indices.max()}, basis_length={basis_length}"
                )

            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_periodic_config.num_harmonics,
                length=basis_length,
                periods=self.config.multi_periodic_config.periods
            )

            # Check for NaN in basis matrix
            if np.any(np.isnan(F_full)):
                raise ValueError(
                    f"Basis matrix contains NaN. basis_length={basis_length}, "
                    f"F_full shape: {F_full.shape}, "
                    f"time_indices range: [{min_idx}, {max_idx}]"
                )

            # Index with adjusted_indices to get correct rows
            F = F_full[adjusted_indices, 1:]  # Drop constant column

            # Check for NaN in indexed basis matrix
            if np.any(np.isnan(F)):
                raise ValueError(
                    f"Indexed basis matrix F contains NaN. "
                    f"F shape: {F.shape}, adjusted_indices range: [{adjusted_indices.min()}, {adjusted_indices.max()}]"
                )

            fourier_coef = self.variables_['fourier_coef'].value
            if fourier_coef is None:
                raise ValueError("Fourier coefficients are None. Model may not have converged.")

            # Check for NaN in Fourier contribution
            fourier_contrib = F @ fourier_coef
            if np.any(np.isnan(fourier_contrib)):
                raise ValueError(
                    f"Fourier contribution contains NaN. F shape: {F.shape}, "
                    f"fourier_coef shape: {fourier_coef.shape}, "
                    f"adjusted_indices range: [{adjusted_indices.min()}, {adjusted_indices.max()}]"
                )

            predictions += fourier_contrib

        # Add trend term if present
        if self.config.trend_config is not None and self.config.trend_config.trend_type != TrendType.NONE and 'trend' in self.variables_:
            trend = self.variables_['trend'].value
            if trend is None:
                raise ValueError("Trend coefficients are None. Model may not have converged.")

            # Use stored period_hours from fit (or recalculate if not stored)
            if hasattr(self, 'trend_period_hours_'):
                period_hours = self.trend_period_hours_
            else:
                # Fallback: recalculate (shouldn't happen if fit was called first)
                period_hours, _ = self._get_trend_period_hours(
                    timestamps, self.config.trend_config.grouping
                )

            # Calculate period indices for prediction timestamps
            period_indices = (time_indices / period_hours).astype(int)
            n_periods_fit = len(trend)
            n_periods_pred = period_indices.max() + 1

            # Create T matrix for predictions
            T_pred = np.zeros((len(predictions), n_periods_pred))
            # Use numpy advanced indexing for efficiency
            # Filter out negative indices (can occur if predicting before training data)
            valid_mask = period_indices >= 0
            T_pred[np.arange(len(period_indices))[valid_mask], period_indices[valid_mask]] = 1.0

            # Extend trend if prediction extends beyond training data
            if n_periods_pred > n_periods_fit:
                # Extend trend using the last value or extrapolate based on trend type
                trend_extended = np.zeros(n_periods_pred)
                trend_extended[:n_periods_fit] = trend

                if self.config.trend_config.trend_type == TrendType.LINEAR and self.variables_['trend_slope'].value is not None:
                    for i in range(n_periods_fit, n_periods_pred):
                        trend_extended[i] = trend[-1] + self.variables_['trend_slope'].value * (i - n_periods_fit + 1)
                else:
                    # fallback: use last value
                    trend_extended[n_periods_fit:] = trend[-1]

                trend = trend_extended

            # Add trend term to predictions
            predictions += T_pred @ trend

        # Final check for NaN in predictions
        if np.any(np.isnan(predictions)):
            nan_count = np.sum(np.isnan(predictions))
            nan_indices = np.where(np.isnan(predictions))[0]
            raise ValueError(
                f"Predictions contain {nan_count} NaN values out of {len(predictions)}. "
                f"First few NaN indices: {nan_indices[:10] if len(nan_indices) > 0 else []}. "
                f"Constant value: {self.variables_['constant'].value}, "
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
        check_is_fitted(self, ['problem_', 'time_reference_', 'freq_'])
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

        if random_state is not None:
            if isinstance(random_state, np.random.RandomState):
                rng = random_state
            else:
                rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState()

        ar_coef = self.ar_coef_
        ar_intercept = self.ar_intercept_
        ar_noise_loc = self.ar_noise_loc_
        ar_noise_scale = self.ar_noise_scale_
        ar_lags = len(ar_coef)
        length = len(baseline_pred)
        nvals = length + ar_lags * 2
        samples = np.zeros((n_samples, len(baseline_pred)))
        # Prepare filter coefficients
        a = np.concatenate([[1], -ar_coef[::-1]])
        b = np.array([1])
        for i in range(n_samples):
            # Generate i.i.d. noise for the entire sequence
            noise = stats.laplace.rvs(
                loc=ar_noise_loc,
                scale=ar_noise_scale,
                size=nvals,
                random_state=rng
            )

            # Input to the filter
            x = ar_intercept + noise

            # Initialize the filter state with the first ar_lags noise values
            # This matches the original "window" initialization
            initial_window = noise[:ar_lags]

            # Convert initial window to filter initial conditions
            # For an AR process, we need to set zi such that the first outputs match our window
            if ar_lags > 0:
                zi = np.zeros(ar_lags)
                # Work backwards through the initial window to set up the state
                for j in range(ar_lags):
                    zi[j] = initial_window[ar_lags - 1 - j]
            else:
                zi = None

            # Apply the AR filter starting after the initial window
            if zi is not None:
                ar_noise, _ = signal.lfilter(b, a, x[ar_lags:], zi=zi)
                # Prepend the initial window
                ar_noise = np.concatenate([initial_window, ar_noise])
            else:
                ar_noise, _ = signal.lfilter(b, a, x)

            # Use last length values (after burn-in)
            ar_noise = ar_noise[-length:]
            samples[i] = baseline_pred + ar_noise
        return samples



if __name__ == "__main__":
    """
    Baseline configuration replicating the notebook baseline model.

    Configuration:
    - Multi-periodic: [6, 4, 3] harmonics for periods [365.2425*24, 7*24, 24]
    - Temperature spline: 10 knots, lags [-3, -2, -1, 0, 1, 2, 3]
    - Regularization: 1e-4 for Fourier and exog, 1.0 for exog diff
    - Solver: CLARABEL with verbose=True
    - No AR model (baseline only)
    """
    import pandas as pd
    from pathlib import Path

    # Load data from same place as notebook
    def load_notebook_data(sheet: str = 'RI', years: list[int] | None = None) -> pd.DataFrame:
        if years is None:
            years = [2020, 2021]
        df_list = []
        for year in years:
            fp = Path(__file__).resolve().parent.parent.parent / 'examples' / 'data' / 'iso' / f'{year}_smd_hourly.xlsx'
            df = pd.read_excel(fp, sheet_name=sheet)
            df['year'] = year
            df.index = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hr_End'].map(lambda x: f"{x-1}:00:00")) + pd.Timedelta(hours=1)
            df_list.append(df)
        return pd.concat(df_list, axis=0)

    # Load data
    print("Loading data...")
    df = load_notebook_data(sheet='RI', years=[2020, 2021])

    # Prepare y (log-transformed RT_Demand) and X (temperature only)
    df_subset = df.loc["2020":"2021"]
    y = np.log(df_subset["RT_Demand"]).values
    X = pd.DataFrame({'temp': df_subset["Dry_Bulb"].values}, index=df_subset.index)

    # Multi-periodic configuration for time features
    multi_periodic_config = TsgamMultiPeriodicConfig(
        num_harmonics=[6, 4, 3],
        periods=[365.2425 * 24, 7 * 24, 24]
    )

    # Spline configuration for temperature (exogenous variable)
    exog_config: list[TsgamSplineConfig | TsgamLinearConfig] = [
        TsgamSplineConfig(
            knots=[],  # Empty list means knots will be auto-generated from data
            n_knots=10,  # Number of knots to generate
            lags=[-3, -2, -1, 0, 1, 2, 3],
            reg_weight=1e-4,  # Regularization weight for coefficients
            diff_reg_weight=1.0  # Regularization weight for differences between lags
        )
    ]

    # No AR model in baseline (AR is added later in the notebook)
    ar_config = None

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver='CLARABEL',
        verbose=True
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_periodic_config=multi_periodic_config,
        exog_config=exog_config,
        ar_config=ar_config,
        solver_config=solver_config,
        random_state=None,
        debug=False
    )

    # Create estimator
    print("\nCreating estimator...")
    estimator = TsgamEstimator(config=config)

    print("\nConfiguration:")
    if config.multi_periodic_config:
        print(f"  Multi-periodic: {config.multi_periodic_config.num_harmonics} harmonics")
        print(f"  Periods: {config.multi_periodic_config.periods}")
    if config.exog_config:
        print(f"  Exog config: {len(config.exog_config)} exogenous variable(s)")
        for ix, exog_cfg in enumerate(config.exog_config):
            if isinstance(exog_cfg, TsgamSplineConfig):
                print(f"    [{ix}] Type: Spline")
                print(f"        n_knots: {exog_cfg.n_knots}")
            else:
                print(f"    [{ix}] Type: Linear")
            print(f"        lags: {exog_cfg.lags}")
            print(f"        reg_weight: {exog_cfg.reg_weight}")
            print(f"        diff_reg_weight: {exog_cfg.diff_reg_weight}")
    print(f"  Solver: {config.solver_config.solver} (verbose={config.solver_config.verbose})")

    # Fit the model
    print("\nFitting model...")
    estimator.fit(X, y)

    print("\nFitting complete!")
    print(f"Problem status: {estimator.problem_.status}")
    if estimator.problem_.status in ["optimal", "optimal_inaccurate"]:
        print(f"Optimal value: {estimator.problem_.value:.6e}")
