.. Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
.. SPDX-License-Identifier: BSD-3-Clause

API Reference
=============

Import both estimators and their configuration classes from ``tsgam``.

Time-series decomposition
-------------------------

Use ``TsgamEstimator`` to fit a series and inspect its fitted components
through ``decomposition_``.

.. autoclass:: tsgam.TsgamEstimator
   :members: fit, predict, sample, score
   :undoc-members:

Direct multi-horizon forecasting
--------------------------------

Use ``TsgamForecastEstimator`` to predict several horizons from each forecast
origin. Each output row corresponds to one origin, with a column per horizon.

.. autoclass:: tsgam.TsgamForecastEstimator
   :members: fit, predict
   :undoc-members:

Configuration and utilities
---------------------------

.. automodule:: tsgam
   :members:
   :exclude-members: TsgamEstimator, TsgamForecastEstimator
   :undoc-members:
   :show-inheritance:

