# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from numpy import ndarray
import numpy as np
import cvxpy
from numpy.random import RandomState
from scipy import stats
from scipy.sparse import spdiags
from sklearn.base import RegressorMixin, BaseEstimator, check_array, check_is_fitted
from sklearn.utils import check_X_y, check_random_state
from spcqe import make_basis_matrix
from spcqe.functions import initialize_arrays
import pandas as pd

@dataclass
class TsgamMultiHarmonicConfig:
    """
    Configuration for multi-harmonic Fourier basis functions.

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
    >>> config = TsgamMultiHarmonicConfig(
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
    knots: list[float] = field(default_factory=list)

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
    type: TrendType = TrendType.NONE
    grouping: float | None = None # todo: rename this to something better
    reg_weight: float = 10.0

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

    Examples
    --------
    >>> config = TsgamSolverConfig(solver='CLARABEL', verbose=False)
    """
    solver: str = 'CLARABEL'
    verbose: bool = True

@dataclass
class TsgamEstimatorConfig:
    """
    Main configuration for TsgamEstimator.

    This config combines all component configurations (Fourier, exogenous, AR)
    and solver settings into a single configuration object.

    Parameters
    ----------
    multi_harmonic_config : TsgamMultiHarmonicConfig or None
        Configuration for multi-harmonic Fourier basis functions. If None,
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
    solver_config : TsgamSolverConfig, default=TsgamSolverConfig()
        Solver configuration for CVXPY optimization.
    random_state : RandomState or None, default=None
        Random state for reproducible results. Used in AR sampling if ar_config
        is provided.
    debug : bool, default=False
        If True, stores additional debug attributes (e.g., _baseline_residuals_,
        _B_running_view_) for inspection.

    Examples
    --------
    >>> multi_harmonic = TsgamMultiHarmonicConfig(
    ...     num_harmonics=[6, 4, 3],
    ...     periods=[365.2425 * 24, 7 * 24, 24]
    ... )
    >>> exog = [TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1])]
    >>> ar = TsgamArConfig(lags=[1])
    >>> config = TsgamEstimatorConfig(
    ...     multi_harmonic_config=multi_harmonic,
    ...     exog_config=exog,
    ...     ar_config=ar
    ... )
    """
    multi_harmonic_config: TsgamMultiHarmonicConfig | None
    exog_config: list[TsgamSplineConfig | TsgamLinearConfig] | None
    ar_config: TsgamArConfig | None = None
    trend_config: TsgamTrendConfig | None = None
    solver_config: TsgamSolverConfig = field(default_factory=TsgamSolverConfig)
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


def get_recommended_periods(X, include_harmonics=False) -> tuple[list[float], list[int]]:
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
        # Calculate base step from frequency string
        freq_str = inferred_freq.lower() if inferred_freq == 'H' else inferred_freq
        if freq_str.endswith('min') or freq_str == 'T':
            # Parse minutes
            minutes_str = freq_str.replace('min', '').replace('T', '')
            minutes = int(minutes_str) if minutes_str else 1
            base_step_hours = minutes / 60.0
        elif freq_str == 'h' or freq_str == 'hourly':
            base_step_hours = 1.0
        elif freq_str == 'd' or freq_str == 'daily':
            base_step_hours = 24.0
        elif freq_str == 'w' or freq_str == 'weekly':
            base_step_hours = 24.0 * 7
        elif freq_str == 'm' or freq_str == 'monthly':
            base_step_hours = 24.0 * 30.44  # Approximate
        elif freq_str == 'q' or freq_str == 'quarterly':
            base_step_hours = 24.0 * 91.31  # Approximate
        else:
            # Fallback: calculate from actual differences
            diffs = timestamps[1:] - timestamps[:-1]
            base_step_hours = diffs.median().total_seconds() / 3600.0

    # Normalize frequency string for period selection logic
    freq_str = inferred_freq.lower() if inferred_freq == 'H' else inferred_freq

    # Determine periods as multiples of base frequency, then convert to hours
    periods = []
    num_harmonics = []

    # Parse frequency to determine appropriate multiples
    # Handle pandas frequency strings like 'min', '5min', '15min', 'h', 'D', etc.
    if freq_str.endswith('min') or freq_str == 'T':
        # Minute-level data: periods are multiples of the minute interval
        if freq_str == 'min' or freq_str == '1min' or freq_str == 'T':
            # 1-minute data: recommend multiples [1, 5, 15, 60, 1440, 10080]
            # These capture short-term, hourly, daily, and weekly patterns
            period_multiples = [1, 5, 15, 60, 1440, 10080]
            num_harmonics = [4, 3, 3, 6, 4, 3]
        elif freq_str == '5min':
            # 5-minute data: recommend multiples [1, 3, 12, 288, 2016]
            # These capture short-term, hourly, daily, and weekly patterns
            period_multiples = [1, 3, 12, 288, 2016]
            num_harmonics = [3, 3, 6, 4, 3]
        elif freq_str == '15min':
            # 15-minute data: recommend multiples [1, 4, 96, 672]
            # These capture short-term, hourly, daily, and weekly patterns
            period_multiples = [1, 4, 96, 672]
            num_harmonics = [3, 6, 4, 3]
        else:
            # Other minute frequencies: try to parse
            try:
                minutes_str = freq_str.replace('min', '').replace('T', '')
                minutes = int(minutes_str) if minutes_str else 1
                # Recommend periods that are multiples of the base frequency
                # Use common multiples: 1x, 3x, then multiples for daily/weekly patterns
                periods_per_day = (24 * 60) / minutes
                periods_per_week = (7 * 24 * 60) / minutes
                period_multiples = [1, 3, int(periods_per_day / 24), int(periods_per_day), int(periods_per_week)]
                num_harmonics = [3, 3, 6, 4, 3]
            except ValueError:
                # Fallback: calculate multiples from base step
                periods_per_day = 24.0 / base_step_hours
                periods_per_week = 168.0 / base_step_hours
                period_multiples = [int(periods_per_day), int(periods_per_week)]
                num_harmonics = [6, 4, 3]
        # Convert multiples to hours
        periods = [mult * base_step_hours for mult in period_multiples]
    elif freq_str == 'h' or freq_str == 'hourly':
        # Hourly data: recommend multiples [24, 168, 8765.82] (daily, weekly, yearly)
        period_multiples = [24, 168, PERIOD_HOURLY_YEARLY]
        num_harmonics = [6, 4, 3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif freq_str == 'd' or freq_str == 'daily':
        # Daily data: recommend multiples [7, 365.2425] (weekly, yearly)
        period_multiples = [7, PERIOD_DAILY_YEARLY]
        num_harmonics = [4, 3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif freq_str == 'w' or freq_str == 'weekly':
        # Weekly data: recommend yearly pattern
        period_multiples = [PERIOD_WEEKLY_YEARLY]
        num_harmonics = [3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif freq_str == 'm' or freq_str == 'monthly':
        # Monthly data: recommend yearly pattern
        period_multiples = [PERIOD_MONTHLY_YEARLY]
        num_harmonics = [3]
        periods = [mult * base_step_hours for mult in period_multiples]
    elif freq_str == 'q' or freq_str == 'quarterly':
        # Quarterly data: recommend yearly pattern
        period_multiples = [PERIOD_QUARTERLY_YEARLY]
        num_harmonics = [2]
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

    - Multi-harmonic Fourier basis functions for seasonal patterns
    - Cubic spline or linear basis functions for exogenous variables with lead/lag
    - Optional trend term (constant per period, linear or nonlinear)
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
        - 'fourier_coef': Fourier coefficients (if multi_harmonic_config provided)
        - 'exog_coef_{i}': Exogenous variable coefficients for variable i
        - 'trend': Trend coefficients (if trend_config provided)
        - 'trend_slope': Trend slope (if trend_type='linear')
    exog_knots_ : list
        List of knot locations for spline exogenous variables (auto-computed
        during fit, reused during predict).
    trend_T_matrix_ : ndarray or None
        Matrix mapping samples to periods for trend term (if trend_config provided).
    trend_period_hours_ : float or None
        Period length in hours used for trend (if trend_config provided).
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
    ...     TsgamMultiHarmonicConfig, TsgamSplineConfig
    ... )
    >>>
    >>> # Create configuration
    >>> multi_harmonic = TsgamMultiHarmonicConfig(
    ...     num_harmonics=[6, 4, 3],
    ...     periods=[365.2425 * 24, 7 * 24, 24]  # yearly, weekly, daily
    ... )
    >>> exog_config = [TsgamSplineConfig(n_knots=10, lags=[-1, 0, 1])]
    >>> config = TsgamEstimatorConfig(
    ...     multi_harmonic_config=multi_harmonic,
    ...     exog_config=exog_config
    ... )
    >>>
    >>> # Create estimator
    >>> estimator = TsgamEstimator(config=config)
    >>>
    >>> # Prepare data (X must be DataFrame with DatetimeIndex)
    >>> dates = pd.date_range('2020-01-01', periods=1000, freq='h')
    >>> X = pd.DataFrame({'temp': np.random.randn(1000)}, index=dates)
    >>> y = np.log(np.random.rand(1000) * 100 + 50)  # log-transform optional
    >>>
    >>> # Fit model
    >>> estimator.fit(X, y)
    >>>
    >>> # Make predictions
    >>> X_pred = pd.DataFrame({'temp': np.random.randn(100)},
    ...                       index=pd.date_range('2021-01-01', periods=100, freq='h'))
    >>> predictions = estimator.predict(X_pred)
    """
    def __init__(self, config: TsgamEstimatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _extract_timestamps(self, X):
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

    def _timestamps_to_indices(self, timestamps, reference):
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
        return (timestamps - reference).total_seconds() / 3600.0

    def _get_trend_period_hours(self, timestamps, period_hours=None):
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
            # Calculate base step from frequency string
            freq_str = inferred_freq.lower() if inferred_freq == 'H' else inferred_freq
            if freq_str.endswith('min') or freq_str == 'T':
                minutes_str = freq_str.replace('min', '').replace('T', '')
                minutes = int(minutes_str) if minutes_str else 1
                base_step_hours = minutes / 60.0
            elif freq_str == 'h' or freq_str == 'hourly':
                base_step_hours = 1.0
            elif freq_str == 'd' or freq_str == 'daily':
                base_step_hours = 24.0
            else:
                # Fallback: calculate from actual differences
                diffs = timestamps[1:] - timestamps[:-1]
                base_step_hours = diffs.median().total_seconds() / 3600.0

        # Default period: daily (24 hours)
        period_hours = 24.0
        samples_per_period = period_hours / base_step_hours

        return period_hours, samples_per_period

    def _validate_frequency(self, timestamps, expected_freq):
        """
        Validate that timestamps match expected frequency.

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps to validate.
        expected_freq : str
            Expected pandas frequency string (e.g., 'h' for hourly, 'H' also accepted).

        Raises
        ------
        ValueError
            If frequency doesn't match or timestamps are not regular.
        """
        if len(timestamps) < 2:
            return  # Can't validate frequency with < 2 samples

        # Normalize 'H' to 'h' for backward compatibility
        normalized_freq = expected_freq.lower() if expected_freq == 'H' else expected_freq

        # Infer frequency from timestamps
        inferred_freq = pd.infer_freq(timestamps)

        if inferred_freq is None:
            raise ValueError(
                f"Could not infer frequency from timestamps. "
                f"Timestamps must be regularly spaced with frequency '{expected_freq}'."
            )

        # Normalize inferred frequency for comparison ('H' -> 'h')
        normalized_inferred = inferred_freq.lower() if inferred_freq == 'H' else inferred_freq

        if normalized_inferred != normalized_freq:
            # Try to create expected range and compare
            try:
                expected_range = pd.date_range(
                    start=timestamps[0],
                    periods=len(timestamps),
                    freq=normalized_freq
                )
                if not timestamps.equals(expected_range):
                    raise ValueError(
                        f"Timestamps frequency '{inferred_freq}' does not match "
                        f"expected frequency '{expected_freq}'."
                    )
            except Exception as e:
                raise ValueError(
                    f"Timestamps frequency '{inferred_freq}' does not match "
                    f"expected frequency '{expected_freq}': {e}"
                )

    def _ensure_timestamp_index(self, X):
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

    def _make_regularization_matrix(self, num_harmonics,
                                   weight: float,
                                   periods: list[float],
                                   drop_constant: bool = False,
                                   standing_wave=False,
                                   trend=False,
                                   max_cross_k=None,
                                   custom_basis=None):
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



    def _make_H(self, x, knots, include_offset=False):
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

    def _make_offset_H(self, H, offset):
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

    def _running_view(self, arr, window, lag=1, axis=-1):
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

    def _build_exog_Hs(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray, knots: ndarray | None = None):
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
                H0 = exog_var
                H_lag = self._make_offset_H(H0, lag)

            Hs.append(H_lag)

        return Hs

    def _process_exog_config(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray, knots: ndarray | None = None):
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
                # Empty list means knots not specified, need to compute from n_knots
                if not exog_cfg.knots:  # Handles both None and empty list
                    if exog_cfg.n_knots:
                        knots = np.linspace(np.min(exog_var), np.max(exog_var), exog_cfg.n_knots)
                    else:
                        raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                else:
                    knots = np.asarray(exog_cfg.knots)
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

    def _get_min_samples_required(self):
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

    def fit(self, X, y, sample_weight=None):
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
            Sample weights. Currently not used but included for sklearn compatibility.

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
        # Extract timestamps before check_X_y converts DataFrame to array
        timestamps, X_array = self._ensure_timestamp_index(X)
        inferred_freq = pd.infer_freq(timestamps)
        # Validate frequency
        self._validate_frequency(timestamps, inferred_freq)

        # Store frequency and reference timestamp
        # Normalize 'H' to 'h' for consistency
        normalized_freq = inferred_freq.lower() if inferred_freq == 'H' else inferred_freq
        self.freq_ = normalized_freq
        self.time_reference_ = timestamps[0]

        # Convert timestamps to numeric indices (hours since reference)
        time_indices = self._timestamps_to_indices(timestamps, self.time_reference_)
        self.time_indices_ = time_indices

        # Now validate X and y with the array version - check_X_y will reject NaN's
        X_array, y = check_X_y(X_array, y,
            ensure_min_features=len(self.config.exog_config or []),
            ensure_min_samples=self._get_min_samples_required())

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
                    # Empty list means knots not specified, need to compute from n_knots
                    if not exog_cfg.knots:  # Handles both None and empty list
                        if exog_cfg.n_knots:
                            knots = np.linspace(np.min(X_array[:, ix]), np.max(X_array[:, ix]), exog_cfg.n_knots)
                            # Store auto-computed knots (need to reuse for prediction)
                            self.exog_knots_.append(knots)
                        else:
                            raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                    else:
                        # Knots provided in config, don't need to store
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



        if self.config.multi_harmonic_config:
            # Generate basis matrix for max index + 1, then index with time_indices
            # This ensures correct phase alignment (as shown in notebook)
            max_idx = int(np.max(time_indices))
            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=max_idx + 1,
                periods=self.config.multi_harmonic_config.periods
            )
            # Index with time_indices to get correct rows
            F = F_full[time_indices.astype(int), 1:]  # Drop constant column

            Wf = self._make_regularization_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                weight=1.0,
                periods=self.config.multi_harmonic_config.periods,
                drop_constant=True
            )
            self.variables_['fourier_coef'] = cvxpy.Variable(F.shape[1])
            regularization_term += self.config.multi_harmonic_config.reg_weight * cvxpy.sum_squares(Wf @ self.variables_['fourier_coef'])
            model_term += F[self.combined_valid_mask_] @ self.variables_['fourier_coef']

        # Add trend term if configured
        trend_term = None
        constraints = []
        if self.config.trend_config is not None and self.config.trend_config.type != TrendType.NONE:
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

            if trend_config.type == TrendType.LINEAR:
                # Linear trend: constant slope
                slope = cvxpy.Variable()
                self.variables_['trend_slope'] = slope
                constraints.append(cvxpy.diff(trend) == slope)
            elif trend_config.type == TrendType.NONLINEAR:
                # Nonlinear monotonic decreasing trend
                constraints.append(cvxpy.diff(trend) <= 0)
            # For 'none', trend_term is None so it won't be added

        error = cvxpy.sum_squares(y[self.combined_valid_mask_] - model_term) / np.sum(self.combined_valid_mask_)
        self.problem_ = cvxpy.Problem(cvxpy.Minimize(error + regularization_term), constraints)
        self.problem_.solve(solver=self.config.solver_config.solver, verbose=self.config.solver_config.verbose)

        # Fit AR model if configured
        if self.config.ar_config is not None:
            self._fit_ar_model(X_array, y, time_indices)

        return self

    def _fit_ar_model(self, X_array, y, time_indices):
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
        if self.config.multi_harmonic_config:
            max_idx = int(np.max(time_indices))
            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=max_idx + 1,
                periods=self.config.multi_harmonic_config.periods
            )
            F = F_full[time_indices.astype(int), 1:]  # Drop constant column
            fourier_coef = self.variables_['fourier_coef'].value
            if fourier_coef is not None:
                baseline_pred += F @ fourier_coef

        # Add trend term if present
        if self.config.trend_config is not None and self.config.trend_config.type != TrendType.NONE and 'trend' in self.variables_:
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
        ar_problem.solve(solver=self.config.solver_config.solver, verbose=self.config.solver_config.verbose)

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

    def predict(self, X):
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

        # todo: check for nan in predict provided data

        # Extract timestamps and validate
        timestamps, X_array = self._ensure_timestamp_index(X)

        # Validate frequency matches
        self._validate_frequency(timestamps, self.freq_)

        # Convert timestamps to indices using stored reference
        time_indices = self._timestamps_to_indices(timestamps, self.time_reference_)

        # Validate X_array shape
        X_array = check_array(X_array, ensure_min_features=len(self.config.exog_config or []))

        # Initialize prediction with constant term
        predictions = np.full(len(X_array), self.variables_['constant'].value)

        # Add exogenous terms if present
        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                exog_var = X_array[:, ix]

                # Get stored knots if available (auto-computed during fit), otherwise None
                stored_knots = self.exog_knots_[ix] if isinstance(exog_cfg, TsgamSplineConfig) else None

                # Use _process_exog_config with stored knots (will use config knots if stored_knots is None)
                _, Hs_pred = self._process_exog_config(exog_cfg, exog_var, knots=stored_knots)

                # Compute exogenous prediction
                exog_coef = self.variables_[f'exog_coef_{ix}'].value
                if exog_coef is None:
                    raise ValueError(f"Exogenous coefficients for variable {ix} are None. Model may not have converged.")
                exog_pred = np.sum([H @ exog_coef[:, lag_ix] for lag_ix, H in enumerate(Hs_pred)], axis=0)
                predictions += exog_pred

        # Add Fourier terms if present
        if self.config.multi_harmonic_config:
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

            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=basis_length,
                periods=self.config.multi_harmonic_config.periods
            )
            # Index with adjusted_indices to get correct rows
            F = F_full[adjusted_indices, 1:]  # Drop constant column

            fourier_coef = self.variables_['fourier_coef'].value
            if fourier_coef is None:
                raise ValueError("Fourier coefficients are None. Model may not have converged.")
            predictions += F @ fourier_coef

        # Add trend term if present
        if self.config.trend_config is not None and self.config.trend_config.type != TrendType.NONE and 'trend' in self.variables_:
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

                if self.config.trend_config.type == TrendType.LINEAR and self.variables_['trend_slope'].value is not None:
                    for i in range(n_periods_fit, n_periods_pred):
                        trend_extended[i] = trend[-1] + self.variables_['trend_slope'].value * (i - n_periods_fit + 1)
                else:
                    # fallback: use last value
                    trend_extended[n_periods_fit:] = trend[-1]

                trend = trend_extended

            # Add trend term to predictions
            predictions += T_pred @ trend

        return predictions

    def sample(self, X, n_samples=1, random_state=None):
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

    def _generate_ar_samples(self, baseline_pred, n_samples, random_state):
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

        ar_coef = self.ar_coef_
        ar_intercept = self.ar_intercept_
        ar_lags = len(ar_coef)

        samples = np.zeros((n_samples, len(baseline_pred)))
        for i in range(n_samples):
            # Initialize window with random noise (as in notebook)
            window = stats.laplace.rvs(
                loc=self.ar_noise_loc_,
                scale=self.ar_noise_scale_,
                size=ar_lags,
                random_state=random_state
            )
            # Generate AR noise with burn-in period (matching notebook)
            # Notebook generates length + ar_lags * 2 values, then uses last length values
            length = len(baseline_pred)
            nvals = length + ar_lags * 2
            gen_data = np.empty(nvals, dtype=float)
            for it in range(nvals):
                # Generate AR value: ar_coef @ window + intercept + noise
                ar_val = ar_coef @ window + ar_intercept + stats.laplace.rvs(
                    loc=self.ar_noise_loc_,
                    scale=self.ar_noise_scale_,
                    random_state=random_state
                )
                gen_data[it] = ar_val
                # Update window: roll and replace last element
                window = np.roll(window, -1)
                window[-1] = ar_val
            # Use last length values (after burn-in)
            ar_noise = gen_data[-length:]
            samples[i] = baseline_pred + ar_noise
        return samples



if __name__ == "__main__":
    """
    Baseline configuration replicating the notebook baseline model.

    Configuration:
    - Multi-harmonic: [6, 4, 3] harmonics for periods [365.2425*24, 7*24, 24]
    - Temperature spline: 10 knots, lags [-3, -2, -1, 0, 1, 2, 3]
    - Regularization: 1e-4 for Fourier and exog, 1.0 for exog diff
    - Solver: CLARABEL with verbose=True
    - No AR model (baseline only)
    """
    import pandas as pd
    from pathlib import Path

    # Load data from same place as notebook
    def load_notebook_data(sheet='RI', years=[2020, 2021]):
        df_list = []
        for year in years:
            fp = Path('.') / 'ISO_Data' / f'{year}_smd_hourly.xlsx'
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

    # Multi-harmonic configuration for time features
    multi_harmonic_config = TsgamMultiHarmonicConfig(
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
        multi_harmonic_config=multi_harmonic_config,
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
    if config.multi_harmonic_config:
        print(f"  Multi-harmonic: {config.multi_harmonic_config.num_harmonics} harmonics")
        print(f"  Periods: {config.multi_harmonic_config.periods}")
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
