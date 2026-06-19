# Tidal Multi-Station Report Design

## Summary

Add a new standalone script that runs the same fixed Battery-style tidal model
across the stations already exposed by `examples/tidal_analysis.ipynb` via
`STATION_CATALOG`, then produces a shareable report artifact. The purpose is to
measure how well one common model form and one common parameterization transport
across stations, not to retune each station independently.

The output should be a batch-friendly script that writes:

- a results CSV for included stations;
- an exclusions CSV for skipped stations;
- a Markdown report with full metric tables and embedded charts;
- a log file;
- PNG figures for metric-wide comparisons and per-station diagnostics.

## Goals

- Reuse the notebook's station set by iterating the same `STATION_CATALOG`
  source of truth.
- Apply one fixed, Battery-style model form and one fixed parameterization to
  every station.
- Keep the model focused on the lower-timescale tidal periods plus the weather
  regressors `pressure`, `dp_dt`, and `wind_u`.
- Skip stations that cannot support the required data window or regressors, and
  report the exclusion reason clearly.
- Produce a Markdown report that includes all fit metrics for each included
  station plus embedded charts.
- Produce both across-station comparison charts and per-station diagnostic
  figures.
- Keep the script usable as a normal CLI batch job with durable output files.

## Non-Goals

- Do not run a per-station hyperparameter search in this first report.
- Do not create a notebook-first artifact as the primary deliverable.
- Do not broaden the model form to new regressors beyond `pressure`, `dp_dt`,
  and `wind_u`.
- Do not automatically shorten the analysis window per station; stations that
  cannot support the requested window should be excluded.
- Do not replace the current single-station Battery search script.

## Fixed Model Configuration

The report should use one fixed configuration for every station.

### Data Window

- Train window: `2022-01-01` through `2023-12-31`
- Held-out window: `2024-01-01` through `2024-12-31`
- Data loading window should include the full train + held-out span needed by
  the model and requested regressors.

### Tidal Structure

Use the same lower-timescale structure as the Battery anchor point:

- `M2 = 4`
- `S2 = 1`
- `N2 = 1`
- `K1 = 2`
- `O1 = 1`
- `Mf = 0`
- `Mm = 0`
- `annual = 0`

This keeps the periodic model focused on the shorter-timescale constituents
while intentionally omitting the long-period terms for this transport test.

### Exogenous Regressors

Use the same fixed regressor set at every station:

- `pressure`
- `dp_dt`
- `wind_u`

Use the same lag ranges as the Battery configuration:

- `pressure: (-2, 0)`
- `dp_dt: (-2, 0)`
- `wind_u: (-1, 0)`

Use the same knot preset across those regressors:

- `med`

Use the same Fourier regularization weight as the Battery fixed point:

- `1e-4`

The implementation should keep the solver choice aligned with the current
single-station Battery script path rather than introducing a separate
station-by-station solver decision in this report script.

## Station Selection

- Build the station list from the same `STATION_CATALOG` that powers the
  notebook station dropdown.
- Preserve the station metadata needed for reporting, especially:
  - station id;
  - station name;
  - tidal regime when available.
- Process stations independently so one station failure does not abort the
  entire report run.

## Coverage And Exclusion Policy

- Attempt to load each station with weather enabled so the fixed regressor set
  can be evaluated honestly.
- Require enough data coverage to support the requested train window, held-out
  window, and the derived `dp_dt`/lagged regressor features.
- If a station cannot support the required window or regressors:
  - skip it;
  - record a clear exclusion reason;
  - continue the run.
- Do not silently shorten the station window to rescue incomplete stations in
  this first-pass report.

## Script Shape

Add a new script rather than overloading the existing Battery grid runner.

Recommended target:

- `examples/run_tidal_multi_station_report.py`

The script should:

1. Build the station list from `STATION_CATALOG`.
2. Load each station frame with weather enabled.
3. Validate whether the station can support the fixed report configuration.
4. Run the fixed model fit for included stations.
5. Collect per-station metrics and fit metadata.
6. Write CSV outputs as the run progresses or at safe checkpoints.
7. Produce comparison and per-station figures.
8. Write the Markdown summary with embedded images.

The script should support a normal CLI batch workflow similar to the existing
examples:

- explicit output directory;
- optional no-download mode;
- logging to disk;
- station-by-station progress messages.

## Output Files

### Results CSV

Write a machine-readable included-stations table, e.g.
`multi_station_results.csv`, with one row per included station and at least:

- station id;
- station name;
- tidal regime;
- active regressors;
- train metrics:
  - `rmse`
  - `mae`
  - `mape`
  - `r2`
- held-out metrics:
  - `rmse`
  - `mae`
  - `mape`
  - `r2`
- train and held-out sample counts;
- metric gap columns where useful, such as held-out minus train.

### Exclusions CSV

Write `multi_station_excluded.csv` with:

- station id;
- station name;
- exclusion category;
- clear human-readable reason.

### Markdown Report

Write `multi_station_summary.md` as the shareable report artifact.

It should include:

- a setup section describing the fixed model form and data window;
- an included-stations table with all metrics;
- a skipped-stations table with reasons;
- ranked summaries of best/worst held-out performance;
- embedded metric-wide charts;
- embedded per-station figures.

### Figures

Write charts into a figures subdirectory, for example
`figures/multi_station/`, and reference them from the Markdown using relative
paths.

## Charts

The Markdown report should embed both comparison charts and per-station
diagnostics.

### Across-Station Charts

Include one chart per key metric across stations. At minimum:

- held-out `MAPE` by station;
- held-out `RMSE` by station;
- held-out `MAE` by station;
- held-out `R^2` by station.

Also include train-versus-held-out comparisons so generalization is visible,
either as:

- paired bars per station; or
- separate train/held-out ranked plots; or
- another compact comparison form that stays readable in Markdown.

The charts should prioritize readability over density; station labels may need
rotation or ranking by held-out score.

### Per-Station Figures

Include one compact figure per included station. Each figure should provide
useful context without reproducing the entire notebook diagnostic suite.

Recommended content:

- held-out observed versus predicted time-series overlay;
- a compact title/annotation block with train and held-out metrics;
- optionally a small residual panel if it materially improves interpretability.

## Error Handling

- A failure on one station should produce an exclusion/failure record and allow
  the script to continue with the next station.
- Data-loading failures, weather-coverage failures, and fit failures should be
  distinguished in logs and exclusion output where practical.
- The script should preserve partial outputs when possible so long multi-station
  runs remain diagnosable.

## Testing

Add focused tests around the script's durable logic rather than brittle
end-to-end plotting assertions.

Cover helper seams such as:

- building the station list from `STATION_CATALOG`;
- constructing the fixed fit configuration;
- classifying included versus excluded stations;
- shaping results and exclusion rows;
- generating the Markdown sections/relative figure paths.

Avoid brittle image-content snapshot tests. Where figures are tested, prefer
asserting that the expected files are created and referenced.

## Acceptance Criteria

- Running the new multi-station script iterates the same station set exposed by
  the notebook via `STATION_CATALOG`.
- Every included station is fit with the same fixed model form and parameter
  values.
- Stations lacking required coverage are skipped and written to an exclusions
  CSV with clear reasons.
- The output directory contains:
  - results CSV;
  - exclusions CSV;
  - Markdown summary;
  - log file;
  - embedded chart images.
- The Markdown report contains all train and held-out metrics for each included
  station.
- The Markdown report embeds both:
  - across-station metric comparison charts;
  - per-station diagnostic figures.
- The script can be rerun as a normal batch CLI without requiring notebook
  interaction.
