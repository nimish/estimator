# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Time Series Generalized Additive Model (TSGAM) Estimator.

This package provides a GAM model for time series forecasting that combines:
- Multi-harmonic Fourier basis functions for seasonal patterns
- Cubic spline or linear basis functions for exogenous variables with lead/lag
- Optional trend term (constant per period, linear or nonlinear)
- Optional outlier detector (sparse multiplicative corrections per period)
- Optional autoregressive (AR) modeling of residuals
"""

from .tsgam_estimator import (
    # Main estimator class
    TsgamEstimator,
    # Configuration classes
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSplineConfig,
    TsgamLinearConfig,
    TsgamArConfig,
    TsgamTrendConfig,
    TsgamOutlierConfig,
    TsgamSolverConfig,
    # Enums
    TrendType,
    # Utility functions
    get_recommended_periods,
    # Constants
    PERIOD_HOURLY_DAILY,
    PERIOD_HOURLY_WEEKLY,
    PERIOD_HOURLY_YEARLY,
    PERIOD_DAILY_YEARLY,
    PERIOD_WEEKLY_YEARLY,
    PERIOD_MONTHLY_YEARLY,
    PERIOD_QUARTERLY_YEARLY,
    PERIOD_YEARLY_YEARLY,
)

__all__ = [
    "TsgamEstimator",
    "TsgamEstimatorConfig",
    "TsgamMultiHarmonicConfig",
    "TsgamSplineConfig",
    "TsgamLinearConfig",
    "TsgamArConfig",
    "TsgamTrendConfig",
    "TsgamOutlierConfig",
    "TsgamSolverConfig",
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

__version__ = "0.1.0"
