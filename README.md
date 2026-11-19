# Time-series generalized additive model (TSGAM)

TSGAM fits a time series as a sum of interpretable components. For electricity demand, those components might include a daily load profile that changes with the seasons, a nonlinear response to temperature, and a long-term trend. You can inspect each component’s contribution to the load at a particular hour. You can also build forecasts for several horizons using the information available at a forecast origin.

The formulation builds on Bennet Meyers and Stephen Boyd’s [*Signal Decomposition Using Masked Proximal Operators* (2023)](https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html). TSGAM uses the [`signaldecomp` library](https://github.com/cvxgrp/cvx-sd-skill) to assemble component models in CVXPY. TSGAM provides the time indexing, estimator interface, and forecasting methods.

Choose linear or natural cubic spline responses for exogenous variables, truncated Fourier series for periodic patterns, and linear or monotone trends. Cross terms between Fourier series let the shape of one cycle vary with the phase of another. For example, summer and winter can have different hours of peak demand. For monotone trends, use `nonlinear_increasing` or `nonlinear_decreasing` (also accepted as `nonlinear`).

## Installation

Install `tsgam` from Git in a downstream project (Python 3.12 or newer):

```bash
uv add "tsgam @ git+https://github.com/cvxgrp/tsgam.git"
```

From a local checkout:

```bash
uv sync
```

For local development:

```bash
uv sync --group dev
```

## Worked examples

- [What explains hourly load?](examples/example_la_energy_marimo.py) Fit Los Angeles electricity demand from weather and calendar patterns. The notebook holds out the last week of each month and estimates demand for those hours using observed weather.
- [What can I forecast from this hour?](examples/example_simple_dmh_forecast_marimo.py) Follow one synthetic weather observation through a direct multi-horizon model. Each row of predictions contains a nowcast and forecasts for the next three hours.

See the [example guide](examples/README_RUN_SCRIPTS.md) for launch commands and more notebooks.

For a related application of the periodic cross terms, Giray Ogut, Bennet Meyers, and Stephen Boyd used interacting daily, weekly, and annual patterns to model hourly Rhode Island net load. Their study evaluates point forecasts, quantile forecasts, and samples of future load trajectories. [*Interpretable Net Load Forecasting Using Smooth Multiperiodic Features* (2024)](https://web.stanford.edu/~boyd/papers/multiperiodic_forecasting.html)

## Documentation

### Building Documentation Locally

To build the documentation locally:

1. Install documentation dependencies:
   ```bash
   uv sync --group docs
   ```

2. Generate documentation:
   ```bash
   python generate_docs.py
   ```

   Or to open in browser after building:
   ```bash
   python generate_docs.py --open
   ```

3. View the documentation:
   Open `docs/_build/html/index.html` in your browser.

### Alternative: Using Make

You can also use the Makefile in the `docs` directory:

```bash
cd docs
make html
```

## Prediction support

`predict` returns NaN where lagged inputs are unavailable. `estimator.score`
computes R² on supported rows only. For explicit sklearn metrics, use
`make_supported_scorer("neg_root_mean_squared_error")` from `tsgam`
as the `scoring` argument to `GridSearchCV` or `cross_validate`. Ordinary sklearn
scorers do not ignore NaNs. The wrapper predicts on the full validation window
before filtering, preserving lag alignment. Targets must be finite; an entirely
unsupported window raises an error. When tuning lag sets, check that candidates
are compared on equivalent support. For multi-output scoring, a row must be
supported by every output.

Spline configs accept `whiten=True` to delegate exact full-rank numerical
whitening to SignalDecomp. Ridge, lag-smoothing, and horizon-coupling penalties
remain on the original coefficients, as do the compatibility views. Whitening
uses the joint fitting mask and raises on rank deficiency; it never silently
drops basis directions. The existing training-extrema knot policy is unchanged.
Native component metadata retains `support_diagnostics` (knot-interval counts
and base-basis rank/conditioning) and, when enabled, `whitening` (the full lagged
design audit and coordinate transform). Coupled forecasts retain these under
`horizon_component_metadata_` in horizon order.

Duplicate timestamps are rejected before grid expansion; aggregate repeated
observations and their weights explicitly. Spline responses require at least
three knots. For former two-knot linear responses, use `TsgamLinearConfig` with
the same lags and regularization weights.

TSGAM uses SignalDecomp as its decomposition engine, pinned to an immutable Git
revision. SignalDecomp owns generic bases, component penalties, and masked
residual linking; TSGAM owns timestamps, model configuration, forecasting, and AR
policy. Examples also use SignalDecomp's basis helpers; `spcqe` is not a direct
dependency (it remains a transitive dependency of `solar-data-tools`).

After an ordinary fit, `estimator.decomposition_` is the native solved result.
Use `signaldecomp.components_to_frame(result, mask=result["fit_mask"])` for
fitted components and reconstruction. This is distinct from `predict`, which
applies forecasting policy and does not replay fitted outliers.

Coupled forecasts expose numeric `horizon_values_` in `horizons_` order, with
native names such as `exog_0_beta`, `exog_0_coef`, and `periodic_theta`.
For existing consumers, `variables_` preserves the historical keys, coefficient
shapes, and `.value` access using views of native CVXPY expressions. Ordinary
fits also expose `problem_` and `output_` (an alias of `decomposition_`). Coupled
`variables_` retains horizon-column matrices and lists of exogenous coefficient
matrices. These are inspection interfaces, not supported mutation hooks:
prediction continues to use native solved values. No optimization is duplicated.
The historical `_make_H` response-plot helper delegates directly to
SignalDecomp's spline basis builder, so existing response plots can stay unchanged.

New consumers should use `components_to_frame` for fitted time-series components
and reconstruction, reserving coefficient access for response diagnostics.
Native single-offset spline coefficients can be vectors; reshape to
`(basis_width, n_offsets)` with `order="F"` for offset-specific responses.

Exogenous offsets use `x[t + lag]`. Include the required past or future driver
rows in the prediction input, then select the desired output interval. Rows
without the required driver history return `NaN`, including in forecast and
sample outputs. Removing exogenous components removes this requirement.
Residual AR preserves the fitted time grid and excludes lag windows spanning
missing observations.

## Direct Target-History Forecasting

Forecast mode can use values of the target known at each forecast origin as
direct autoregressive features. Unlike `TsgamArConfig`, this is a deterministic
multi-horizon predictor: it fits each future target directly and does not roll
predictions or sampled residuals forward.

```python
import pandas as pd

from tsgam import (
    TsgamForecastArConfig,
    TsgamForecastConfig,
    TsgamForecastEstimator,
)

forecaster = TsgamForecastEstimator(
    TsgamForecastConfig(
        horizon=24,
        base_config=base_config,
        include_nowcast=False,
        forecast_ar_config=TsgamForecastArConfig(
            lags=[0, 1, 2, 24],
            reg_weight=1e-4,
        ),
    )
).fit(X_train, y_train)

predictions = forecaster.predict(
    X_origins,
    y_history=pd.Series(y_observed, index=observed_times),
)
```

Lag `0` is the target observed at the origin, lag `1` is the previous sample,
and so on. Set `include_nowcast=True` (the default) to additionally fit and
return `horizon_0`; that nowcast never uses target history, avoiding the
tautological prediction `y[t] = y[t]`. Fitted coefficients in original target
units are available in `forecast_ar_coefficients_`; the internally standardized
coefficients are in `forecast_ar_standardized_coefficients_`.

## Forecast Visualization

Install the optional Matplotlib support and plot the origin-indexed output from
`TsgamForecastEstimator.predict` directly:

```bash
uv sync --extra viz
```

```python
from tsgam import plot_forecast_horizon, plot_forecast_origin

predictions = forecaster.predict(X_test)

# One path with its horizon-zero nowcast and horizon 1..H forecasts.
plot_forecast_origin(
    predictions,
    actual=y,
    origin=predictions.index[24],
)

# One fixed horizon aligned to target time across all evaluation origins.
plot_forecast_horizon(predictions, actual=y, horizon=6)

# The horizon-zero baseline over time.
plot_forecast_horizon(predictions, actual=y, horizon=0)
```

Both functions also accept a mapping of labels to prediction DataFrames for
side-by-side model comparisons. Use `forecast_to_long_dataframe` when a notebook
needs the aligned origin/target data for Altair, Seaborn, or another plotting
library.

## Development

Development takes place at [cvxgrp/tsgam](https://github.com/cvxgrp/tsgam), where contributors can open issues and pull requests.

### Running Tests

```bash
uv sync --group test
uv run pytest
```

### Running Type Checks

```bash
uv sync --group typecheck
uv run ty check
```

### Running Tests with Coverage

```bash
uv run pytest --cov=tsgam --cov-report=html
```

### Working with Examples and Notebooks

```bash
uv sync --group examples
uv sync --group notebooks
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.

## Contributors

See [CONTRIBUTORS](CONTRIBUTORS) for a list of contributors.
