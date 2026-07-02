# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import PackageNotFoundError, version

from ._estimator import (
    PERIOD_DAILY_YEARLY,
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
    PERIOD_MONTHLY_YEARLY,
    PERIOD_QUARTERLY_YEARLY,
    PERIOD_WEEKLY_YEARLY,
    PERIOD_YEARLY_YEARLY,
    SolverOptionValue,
    TrendType,
    TsgamArConfig,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    TsgamTrendConfig,
    get_recommended_periods,
)
from ._forecast import (
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
)

try:
    __version__ = version("tsgam-estimator")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "TsgamEstimator",
    "TsgamForecastEstimator",
    "TsgamEstimatorConfig",
    "TsgamForecastConfig",
    "TsgamForecastCouplingConfig",
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
