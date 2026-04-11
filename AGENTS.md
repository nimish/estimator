# Project Instructions

* Use uv to run python scripts and code
* Use pytest for testing

## Learned User Preferences

* Prefer positive boolean config flags (e.g. `use_dpp: True`) over negated ones (e.g. `ignore_dpp: False`).
* Prefer bash scripts with bashisms over POSIX sh.
* Prefer standalone test functions over test classes when a class isn't necessary.
* Prefer working with data at its natural frequency; do not resample to a coarser resolution.
* Type parameters as strongly as reasonable; avoid `Any` when a concrete union or recursive type alias suffices.
* For substantive tidal/met and shareable research reports: prioritize the Jupyter notebook (`examples/tidal_analysis.ipynb`) as the polished artifact; keep `examples/example_tidal_marimo.py` out of scope for agent-driven work unless the user explicitly requests Marimo changes (default to Jupyter, not Marimo polish). Prefer PACT-style structure: configuration-first cells, concise section headers, analysis-first plots, and factored plotting/helpers over tutorial-style exposition or long inline blocks; prefer human-readable station selection (catalog or name-based loaders) over raw numeric station IDs when it improves clarity.
* When refactoring large notebooks, context is tight, or iterative runs hit resource limits, prefer small focused edits with incremental review rather than bulk changes; before large tidal-notebook refactors, snapshot the current state on a dedicated git branch or worktree so the prior version stays easy to recover.
* For user-authored YAML that builds `Tsgam*` configs, validate at the load boundary (e.g. Pydantic models) with explicit discriminators for union fields such as exogenous spline vs linear; keep the existing runtime dataclasses as the constructed target.
* For tidal (or other physics-informed) periodic structure, prefer encoding known constituent periods and harmonics in the multi-periodic config when justified, rather than relying only on generic Fourier settings or residual spectrum alone; define named constituents in one place and reuse the same definitions for diagnostics (e.g. periodogram overlays) and the model.
* In marimo notebooks, prefer `mo.ui.date` over `mo.ui.text` for date inputs.
* Prefer seaborn (`sns.set_theme`) over manual `plt.rcParams` for plot styling in analysis notebooks.
* For shareable analysis notebooks and Marimo examples, keep tone and layout engineer-native and concise; avoid generic or visibly machine-generated structure. Present scalar fit metrics in formatted text or tables with appropriate units (e.g. percent vs absolute), not bar-chart summaries of single scalars.

## Learned Workspace Facts

* Inferred frequency strings must use numeric prefix (e.g. '1h', '1D'); store with prefix and lowercase.
* Fit allows gaps and masks NaNs; predict requires strictly regular timestamps (no gaps).
* Data-dependent tests use examples/data/iso for ISO data files; tests that use pd.read_excel require openpyxl in the dev dependency group so CI (uv sync --group dev) passes.
* TsgamSplineConfig.knots accepts numpy array or list; use len-based emptiness checks, not truthiness on arrays.
* sort_index config (default True): when True sort by index; when False, index must already be sorted or error.
* Optional sample_weight in fit for weighted least squares; AR step is unweighted.
* Example/run scripts use Click and Rich (deps in dependency-groups.examples); produce publication-quality figures (data overview, model summary, ablation) as PDF and PNG.
* Exogenous regressors must be standardized (zero-mean, unit-variance) before fitting; raw scale mismatch against [-1,1] Fourier features causes ill-conditioned design matrices and solver failures. In tidal example spline configs, positive lag indices mean the regressor leads the target (uses future samples); for strictly causal exogenous features use lags ≤ 0 only. For exogenous splines, positive lag indices mean the regressor leads the target (uses future samples); use only lags ≤ 0 for strictly causal fits.
* For NOAA tidal + meteorology examples, CO-OPS tide-gauge met is often incomplete; merge NCDC Local Climatological Data (LCD) hourly weather-station series on the time index when fuller regressors (e.g. wind) are needed.
* TsgamSolverConfig has solver, verbose, warm_start (default True), and solver_opts (dict forwarded as **kwargs to cvxpy Problem.solve; SolverOptionValue is the recursive type alias); CVXPY DPP compliance requires pre-weighting design matrices by sqrt(w/sum(w)); products of cp.Parameter violate DPP rules. SCS is more robust than CLARABEL for large (high-resolution, many-sample) tidal problems. `TsgamEstimatorConfig.exog_config` is required (no default); pass `None` when no exogenous regressors are used.
* TsgamMultiPeriodicConfig.periods, TsgamTrendConfig.grouping, and outlier period bucketing use the same regular-grid sample index as fit (not raw wall-clock hours); multiply intended hour-based widths by samples_per_hour for sub-hourly data.
* Marimo (`examples/example_tidal_marimo.py`): cells cannot use implicit globals—names must be imported or returned from earlier cells; avoid early `return` in `@app.cell` (single-exit); assignments like `fig`/`axes`/`label` are cross-cell symbols and duplicate across cells → `multiple-definitions` (use `_`-prefixed plot locals). Prefer Altair via marimo’s Altair integration for charts; ensure raw/tidal and weather-regressor overview plots run on initial load, not only after UI changes. `marimo run` may work while `marimo check` warns until the file matches canonical marimo layout. Large chart or spec objects may exceed Marimo’s default output byte limit—raise `[tool.marimo.runtime] output_max_bytes` or `MARIMO_OUTPUT_MAX_BYTES` when that is intentional. Altair brush/filter interactivity can mis-compare timezone-aware `datetime64[tz]` time fields; use UTC-naive datetimes or a consistent time encoding for interactive Altair charts.
