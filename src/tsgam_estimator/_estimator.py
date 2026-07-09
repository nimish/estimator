# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from enum import StrEnum
from typing import overload

from numpy import ndarray
import numpy as np
import cvxpy
from numpy.random import RandomState
from scipy import stats, signal
from scipy.sparse import spmatrix
from sklearn.base import RegressorMixin, BaseEstimator, check_array, check_is_fitted
from sklearn.utils import check_X_y, check_random_state
from spcqe import make_basis_matrix
import pandas as pd

from ._design import (
    _build_exog_Hs,
    _ensure_numeric_prefix,
    _ensure_sorted_index,
    _ensure_timestamp_index,
    _extract_timestamps,
    _get_zero_lag_H,
    _infer_frequency_from_differences,
    _interaction_contribution_from_blocks,
    _make_offset_H,
    _make_regularization_matrix,
    _make_spline_H,
    _min_samples_required,
    _normalize_interaction_pairs,
    _outer_column_product,
    _process_exog_config,
    _timestamps_to_indices,
    _to_pandas_timedelta_frequency,
    _validate_frequency,
    build_tsgam_design,
    infer_fit_frequency,
    resolve_exog_knots,
    sort_fit_inputs,
    sort_predict_X,
    validate_predict_frequency,
)
from ._problem import (
    evaluate_single_output_prediction,
    make_single_output_standard_variables,
    single_output_prediction_expression,
    solve_problem,
    weighted_squared_loss,
)


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
    NONLINEAR_DECREASING = 'nonlinear_decreasing'
    NONLINEAR_INCREASING = 'nonlinear_increasing'
    NONLINEAR_DEC = 'nonlinear_decreasing'
    NONLINEAR_INC = 'nonlinear_increasing'

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
        return _extract_timestamps(X)

    @staticmethod
    def _ensure_numeric_prefix(freq: str) -> str:
        """Ensure frequency string has a numeric prefix (e.g. ``'h'`` -> ``'1h'``)."""
        return _ensure_numeric_prefix(freq)

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
        return _timestamps_to_indices(timestamps, reference, getattr(self, 'freq_', None))

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
                base_step_hours = pd.to_timedelta(_to_pandas_timedelta_frequency(freq_td_str)).total_seconds() / 3600.0
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
        return _infer_frequency_from_differences(timestamps)

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
        _validate_frequency(timestamps, expected_freq, allow_gaps=allow_gaps)

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
        return _ensure_timestamp_index(X)

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
        if y is None:
            return _ensure_sorted_index(
                X,
                sort_index=self.config.sort_index,
            )
        return _ensure_sorted_index(
            X,
            sort_index=self.config.sort_index,
            y=y,
            sample_weight=sample_weight,
        )

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
        return _make_regularization_matrix(
            num_harmonics,
            weight,
            periods,
            drop_constant=drop_constant,
            standing_wave=standing_wave,
            trend=trend,
            max_cross_k=max_cross_k,
            custom_basis=custom_basis,
        )



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
        return _make_spline_H(x, knots, include_offset=include_offset)

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
        return _make_offset_H(H, offset)

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
        return _build_exog_Hs(exog_cfg, exog_var, knots)

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
        return _process_exog_config(exog_cfg, exog_var, knots=knots)

    def _get_zero_lag_H(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, Hs: list[ndarray]) -> ndarray:
        """Return the current-index basis block for an exogenous term."""
        return _get_zero_lag_H(exog_cfg, Hs)

    def _outer_column_product(self, arr1: ndarray, arr2: ndarray) -> ndarray:
        """Build a q*r interaction design block from two response matrices."""
        return _outer_column_product(arr1, arr2)

    def _interaction_contribution_from_blocks(
        self,
        arr1: ndarray,
        arr2: ndarray,
        interaction_coef: ndarray,
        *,
        nan_to_zero: bool = False,
    ) -> ndarray:
        """Contract two response matrices against flattened interaction coefficients."""
        return _interaction_contribution_from_blocks(
            arr1,
            arr2,
            interaction_coef,
            nan_to_zero=nan_to_zero,
        )

    def _normalize_interaction_pairs(self) -> list[tuple[int, int]]:
        """Validate and normalize configured exogenous interaction pairs."""
        return _normalize_interaction_pairs(self.config)

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
        return _min_samples_required(self.config)

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
        timestamps = _extract_timestamps(X)
        self.freq_ = infer_fit_frequency(timestamps)
        self.time_reference_ = timestamps[0]
        self.exog_knots_ = resolve_exog_knots(self.config, X)
        design = build_tsgam_design(
            self.config,
            X,
            y,
            sample_weight,
            knots_by_exog=self.exog_knots_,
            reference=self.time_reference_,
            freq=self.freq_,
        )
        assert design.y is not None
        assert design.sample_weight is not None
        X_array = design.X_array
        fit_y = design.y
        time_indices = design.time_indices
        self.time_indices_ = time_indices
        self._sample_weight_ = design.sample_weight
        self.combined_valid_mask_ = design.valid_mask
        self.interaction_pairs_ = design.interaction_pairs
        self._fit_design_ = design
        self.variables_, regularization_term = make_single_output_standard_variables(
            self.config,
            design,
        )
        model_term = single_output_prediction_expression(
            self.config,
            design,
            self.variables_,
            self.combined_valid_mask_,
        )

        # Add trend term if configured
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
            T = np.zeros((len(fit_y), n_periods))
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
            elif trend_config.trend_type in (
                TrendType.NONLINEAR,
                TrendType.NONLINEAR_DEC,
            ):
                # Nonlinear monotonic decreasing trend
                constraints.append(cvxpy.diff(trend) <= 0)
            elif trend_config.trend_type == TrendType.NONLINEAR_INC:
                constraints.append(cvxpy.diff(trend) >= 0)
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
            T = np.zeros((len(fit_y), n_periods))
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
        y_valid = fit_y[self.combined_valid_mask_]
        weight_valid = self._sample_weight_[self.combined_valid_mask_]
        error = weighted_squared_loss(y_valid, model_term, weight_valid)
        self.problem_ = cvxpy.Problem(cvxpy.Minimize(error + regularization_term), constraints)
        solve_problem(
            self.problem_,
            self.config.solver_config,
            failure_message=(
                "Optimization problem did not converge. "
                "This may cause NaN predictions. Check your data and model configuration."
            ),
        )

        # Check that constant term is valid
        if self.variables_['constant'].value is None or np.isnan(self.variables_['constant'].value):
            raise ValueError(
                f"Constant term is None or NaN after optimization. Problem status: {self.problem_.status}"
            )

        # Fit AR model if configured
        if self.config.ar_config is not None:
            self._fit_ar_model(X_array, fit_y, time_indices)

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
        baseline_pred = evaluate_single_output_prediction(
            self.config,
            self._fit_design_,
            self.variables_,
        )

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
            **self.config.solver_config._solve_kwargs(),
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
            self.variables_,
            remove_periodic=remove_periodic,
            remove_exogenous=remove_exogenous,
        )

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
            else:
                trend = trend[:n_periods_pred]

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
