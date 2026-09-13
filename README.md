# Time-series generalized additive model (tsgam)

Time series generalized additive model (tsgam) is a package for fitting "generalized addive models" (GAMs) augmented with time-dependent features. The idea is to fit a statistical model that estimates a target time-series based on linear or nonlinear responses to exogenous variables, features encoding one or more natural periodicies, and a features encoding long-term trends. For nonlinear exogenous variables, we model the response with natural cubic basis splines. The (multi-)periodic components are modeled with trucated Fourier series, plus cross terms when multiple periods are present (e.g., daily and year periodicities). We currently have long-term trend models for linear trends and monotonic nonlinear trends. Nonlinear trends can now be configured explicitly as `nonlinear_decreasing` or `nonlinear_increasing`, while legacy `nonlinear` remains a backward-compatible alias for the decreasing form.

## Installation

From PyPI:

```bash
uv add tsgam-estimator
```

For local development:

```bash
uv sync --group dev
```

This branch requires Python 3.13 or newer, matching the pinned SignalDecomp
development revision and its dependency minimums; Python 3.12 is not supported.

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
`make_supported_scorer("neg_root_mean_squared_error")` from `tsgam_estimator`
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

from tsgam_estimator import (
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
uv add "tsgam-estimator[viz]"
```

```python
from tsgam_estimator import plot_forecast_horizon, plot_forecast_origin

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
uv run pytest --cov=tsgam_estimator --cov-report=html
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
