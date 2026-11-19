.. Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
.. SPDX-License-Identifier: BSD-3-Clause

TSGAM
=====

TSGAM fits time series as a sum of components, including periodic patterns,
trends, and responses to external variables such as temperature. The two
notebooks below illustrate different questions you can ask of hourly load.

`What explains hourly load? <https://github.com/cvxgrp/tsgam/blob/main/examples/example_la_energy_marimo.py>`_
   Estimate held-out Los Angeles electricity demand from observed weather and
   calendar patterns. Inspect the fitted contributions and compare estimates
   with observations from the last week of each month.

`What can I forecast from this hour? <https://github.com/cvxgrp/tsgam/blob/main/examples/example_simple_dmh_forecast_marimo.py>`_
   Use a synthetic weather observation available at the forecast origin to
   predict demand at that hour and one, two, and three hours ahead.

To run either notebook locally, follow the
`example guide <https://github.com/cvxgrp/tsgam/blob/main/examples/README_RUN_SCRIPTS.md>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

