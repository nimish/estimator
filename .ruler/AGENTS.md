# Project Instructions

* Use uv to run Python scripts and code.
* Use pytest for testing.
* Keep durable agent guidance in `.ruler/`; do not encode preferences in Cursor runtime state files or generated agent outputs.

## Learned User Preferences

* Prefer positive boolean config flags, such as `use_dpp`, over negated names, such as `ignore_dpp`.
* Prefer bash scripts with bashisms over POSIX sh.
* Prefer standalone test functions over test classes when a class is not necessary.
* Work with data at its natural frequency; do not resample to a coarser resolution.
* Type parameters as strongly as reasonable; avoid `Any` when a concrete union or recursive type alias suffices.
* For substantive tidal/met and shareable research reports, prioritize `examples/tidal_analysis.ipynb` as the polished artifact. Keep `examples/example_tidal_marimo.py` out of scope unless Marimo changes are explicitly requested.
* Prefer PACT-style notebook structure from `PACT_data_analysis_v2.py` and `PACT_data_full_analysis.py`: configuration-first cells, concise section headers, analysis-first plots, and factored plotting/helpers over tutorial exposition or long inline blocks.
* Prefer human-readable station selection, such as `example_tidal.find_station` or `load_station` with `STATION_CATALOG`, over raw numeric station IDs when it improves clarity.
* For large notebook refactors, tight context, or resource limits, prefer small focused edits with incremental review. Before large tidal-notebook refactors, snapshot the current state on a dedicated branch or worktree, but do not leave requested work isolated there without clearly bringing it back or committing it as requested.
* For user-authored YAML that builds `Tsgam*` configs, validate at the load boundary with explicit discriminators for union fields such as exogenous spline versus linear. Keep existing runtime dataclasses as the constructed target.
* For tidal or other physics-informed periodic structure, encode known constituent periods and harmonics in multi-periodic config when justified. Define named constituents once and reuse them for diagnostics and models.
* Prefer `mo.ui.date` over `mo.ui.text` for date inputs in marimo notebooks.
* Prefer seaborn `sns.set_theme` over manual matplotlib `rcParams` for analysis notebook plot styling.
* For shareable analysis notebooks and Marimo examples, keep tone and layout engineer-native and concise. Avoid generic or visibly machine-generated structure.
* For synthetic or interactive examples, prioritize a generator-first flow: configure data, inspect generated truth with charts/summaries and clear truth-vs-fit legends, then fit and inspect performance/components with per-component fit quality.
* Keep regressor controls focused on response shape, noise, and distributions rather than periodic harmonic knobs. Reserve harmonic profiles, order, and cross terms for periodic structure.
* Present scalar fit metrics in formatted text or tables with appropriate units, such as percent versus absolute, not bar-chart summaries of single scalars.

## Learned Workspace Facts

* Inferred frequency strings must use a numeric prefix, such as `1h` or `1D`; store with prefix and lowercase.
* Fit allows gaps and masks NaNs; predict requires strictly regular timestamps with no gaps.
* Data-dependent tests use `examples/data/iso` for ISO data files.
* Tests that use `pandas.read_excel` require `openpyxl` in the dev dependency group so CI with `uv sync --group dev` passes.
* `TsgamSplineConfig.knots` accepts numpy arrays or lists; use len-based emptiness checks, not truthiness on arrays.
* `sort_index` defaults to true; when true, sort by index, and when false, require an already sorted index or raise.
* Fit supports optional `sample_weight` for weighted least squares; the AR step is unweighted.
* Example and run scripts use Click and Rich from `dependency-groups.examples` and should produce publication-quality figures as PDF and PNG.
* Exogenous regressors must be standardized before fitting because raw scale mismatch against Fourier features can ill-condition design matrices and solvers.
* In tidal example spline configs, positive lag indices mean the regressor leads the target, so use only lags <= 0 for strictly causal fits.
* Exogenous interactions are 2-way basis/response-matrix tensor products, default off per factor, and use only lag 0 even when main effects use lag windows.
* CO-OPS tide-gauge meteorology is often incomplete; merge NCDC LCD hourly weather-station series when fuller regressors are needed.
* `TsgamSolverConfig` has solver, verbose, warm_start default true, and solver_opts forwarded as kwargs to cvxpy `Problem.solve`.
* CVXPY DPP compliance requires pre-weighting design matrices by `sqrt(w/sum(w))`; products of `cp.Parameter` violate DPP rules.
* SCS is more robust than CLARABEL for large high-resolution tidal problems.
* `TsgamEstimatorConfig.exog_config` is required; pass `None` when no exogenous regressors are used.
* `TsgamMultiPeriodicConfig.periods`, `TsgamTrendConfig.grouping`, and outlier period bucketing use the regular-grid sample index from fit, not raw wall-clock hours.
* Multiply intended hour-based widths by `samples_per_hour` for sub-hourly data.
* Marimo cells cannot use implicit globals; names must be imported or returned from earlier cells.
* Avoid early return in `@app.cell`; prefer a single-exit cell shape.
* Assignments like `fig`, `axes`, and `label` are cross-cell symbols and duplicate across cells; use underscore-prefixed plot locals.
* Read `UIElement.value` in a later cell, not the cell that creates the UI element.
* Prefer Altair via marimo's Altair integration for charts.
* Ensure raw/tidal and weather-regressor overview plots run on initial load, not only after UI changes.
* `marimo run` may work while `marimo check` warns until the file matches canonical marimo layout.
* Large chart or spec objects may exceed marimo's default output byte limit; raise `[tool.marimo.runtime] output_max_bytes` or `MARIMO_OUTPUT_MAX_BYTES` when intentional.
* Altair brush/filter interactivity can mis-compare timezone-aware `datetime64` time fields; use UTC-naive datetimes or consistent time encoding.
