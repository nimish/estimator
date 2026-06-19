# Compact Tidal Diagnostics Design

## Summary

`examples/example_tidal_compact.py` currently focuses its post-fit diagnostics on
test-window figures plus a spline-basis inspection view for regressors. The goal
of this change is to make the compact notebook more useful for exploratory tidal
model iteration by adding train-set prediction views, PACT-style regressor
response plots, investigative typical-cycle summaries, safer high-harmonic
exploration, explicit fit verbosity/debug controls, and clearer MAPE semantics.

## Goals

- Add train-set predicted-vs-observed diagnostics alongside the current test-set
  views.
- Add PACT-style regressor response tabs that show training-target scatterplots
  with overlaid fitted spline/linear responses.
- Add investigative `Typical day`, `Typical week`, and `Typical month` plots,
  where `Typical month` means a generic 30-day phase cycle.
- Extend harmonic exploration from `0..8` to `0..32` while automatically
  increasing Fourier regularization when long-period harmonic orders become
  aggressive.
- Add notebook fit controls for solver `verbose` output and estimator `debug`
  mode.
- Make the MAPE reference point explicit in the notebook output, including the
  subset of samples eligible for MAPE.

## Non-Goals

- Do not refactor the estimator API to support per-period Fourier regularization
  weights.
- Do not redesign the compact notebook around a new layout or split it into
  multiple files.
- Do not change the underlying MAPE formula in `examples/tidal_model_shared.py`.
- Do not add broad plot snapshot tests or notebook end-to-end tests.
- Do not change the Shapley workflow or attribution semantics in this change.

## Proposed Changes

### Fit State And Controls

- Keep `run_tidal_model()` as the single entry point for notebook fitting.
- Extend the compact notebook fit state so `FitResult` carries enough
  train-aligned metadata to support:
  - train predicted-vs-observed plots;
  - regressor-response overlays tied to the fitted model;
  - investigative typical day/week/30-day observed-vs-predicted plots;
  - train/test MAPE eligibility counts.
- Add notebook controls for:
  - `verbose` -> threaded to `TsgamSolverConfig.verbose`;
  - `debug` -> threaded to `TsgamEstimatorConfig.debug`.

### Train/Test Prediction Diagnostics

- Keep the existing test-window time-series figure.
- Add a train-set predicted-vs-observed figure alongside the existing test-set
  version.
- Keep the test-set predicted-vs-observed view in the diagnostics tabs and add a
  train-set sibling tab rather than replacing the existing figure.

### Regressor Response Tabs

- Replace or augment the current basis-only regressor inspection section with one
  tab per active fitted regressor.
- Each regressor tab should:
  - show a lag selector populated from that regressor's active fitted lags;
  - default to `lag=0` when available, otherwise the lag closest to `0`;
  - plot the lag-aligned regressor values on the x-axis and the training target
    on the y-axis;
  - color points by time, following the PACT pattern;
  - overlay the fitted spline or linear response for the selected lag, shifted
    vertically onto the target scale so the curve is interpretable on the same
    axes as the scatter.
- The overlay should be derived from the fitted estimator coefficients and knot
  state, not from a rolling smoother.

### Typical Cycle Plots

- Add exploratory observed-vs-predicted plots that collapse timestamps onto
  generic cycle phases:
  - `Typical day`: sample position within `24 h`;
  - `Typical week`: sample position within `7 d`;
  - `Typical month`: sample position within `30 d`.
- Treat these plots as investigative summaries of a typical cycle, not as formal
  error metrics or train/test score panels.
- Keep the aggregation on the model's native sample grid instead of resampling to
  a coarser frequency first.

### Harmonic Exploration And Stabilization

- Raise each harmonic slider cap from `8` to `32`.
- Keep the estimator-facing `TsgamMultiPeriodicConfig.reg_weight` as a single
  scalar, but compute that scalar in notebook code using a conservative schedule
  driven by the maximum active harmonic order among long-period constituents.
- Use the current compact default (`1e-4`) as the base weight and scale upward
  only when long-period harmonic orders exceed the safe baseline explored today.
- Leave short-period-only selections at the current effective behavior unless a
  long-period constituent activates the stronger schedule.

### MAPE Semantics In Notebook Output

- Keep `tidal_metrics()` as the source of truth for metric computation.
- Update the notebook metric presentation so MAPE is labeled explicitly as:
  `MAPE (vs observed; |obs| > 0.01 m)`.
- Add short explanatory text stating that MAPE is computed against observed water
  levels and only on samples with `|obs| > 0.01 m`.
- Surface the number of eligible train/test samples used for MAPE so the metric's
  reference subset is visible.

## Error Handling

- If no regressors are active after fitting, show a clear notebook-visible
  message in the regressor-response section instead of rendering empty tabs.
- If a selected regressor/lag combination has no valid aligned samples after
  masking, show a notebook-visible message inside that regressor tab.
- If a fit falls back to NaN predictions due to solver failure, keep the current
  fallback behavior and let the new plots degrade gracefully rather than adding a
  new failure mode.
- If MAPE has no eligible samples, display `nan` and an eligible count of `0`.

## Testing

- Add only light helper-level regression coverage in
  `test/test_example_tidal_compact.py`.
- Cover the smallest durable seams introduced by this change, such as:
  - default lag selection for regressor tabs;
  - Fourier regularization scheduling for long-period high-order selections;
  - MAPE eligibility metadata/reporting;
  - typical day/week/30-day aggregation helpers.
- Prefer testing data contracts and simple numeric behavior over broad figure
  snapshots.

## Acceptance Criteria

- The compact notebook shows train-set prediction diagnostics in addition to the
  existing test-set ones.
- Active regressors can be inspected in per-regressor tabs with a lag selector
  defaulting to `0` or nearest-to-zero, and each tab shows a PACT-style scatter
  plus fitted response overlay.
- The notebook shows investigative `Typical day`, `Typical week`, and `Typical
  month` plots, where `Typical month` is a generic 30-day phase cycle.
- Harmonic controls allow up to `32` harmonics while long-period high-order fits
  automatically use stronger Fourier regularization than the current fixed
  baseline.
- The notebook exposes `verbose` and `debug` fit toggles.
- The notebook explicitly states the MAPE reference semantics and reports the
  eligible sample counts used for train/test MAPE.
