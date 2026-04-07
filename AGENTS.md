# Project Instructions

* Use uv to run python scripts and code
* Use pytest for testing

## Learned User Preferences

* Prefer positive boolean config flags (e.g. `use_dpp: True`) over negated ones (e.g. `ignore_dpp: False`).
* Prefer bash scripts with bashisms over POSIX sh.
* Prefer standalone test functions over test classes when a class isn't necessary.
* Prefer working with data at its natural frequency; do not resample to a coarser resolution.
* Type parameters as strongly as reasonable; avoid `Any` when a concrete union or recursive type alias suffices.
* For PACT-style analysis notebooks, prefer configuration-first cells, concise section headers, and analysis-first plots over tutorial-style exposition.
* For user-authored YAML that builds `Tsgam*` configs, validate at the load boundary (e.g. Pydantic models) with explicit discriminators for union fields such as exogenous spline vs linear; keep the existing runtime dataclasses as the constructed target.
* For tidal (or other physics-informed) periodic structure, prefer encoding known constituent periods and harmonics in the multi-periodic config when justified, rather than relying only on generic Fourier settings or residual spectrum alone.
* In marimo notebooks, prefer `mo.ui.date` over `mo.ui.text` for date inputs.

## Learned Workspace Facts

* Inferred frequency strings must use numeric prefix (e.g. '1h', '1D'); store with prefix and lowercase.
* Fit allows gaps and masks NaNs; predict requires strictly regular timestamps (no gaps).
* Data-dependent tests use examples/data/iso for ISO data files.
* TsgamSplineConfig.knots accepts numpy array or list; use len-based emptiness checks, not truthiness on arrays.
* sort_index config (default True): when True sort by index; when False, index must already be sorted or error.
* Optional sample_weight in fit for weighted least squares; AR step is unweighted.
* Example/run scripts use Click and Rich (deps in dependency-groups.examples); produce publication-quality figures (data overview, model summary, ablation) as PDF and PNG.
* Exogenous regressors must be standardized (zero-mean, unit-variance) before fitting; raw scale mismatch against [-1,1] Fourier features causes ill-conditioned design matrices and solver failures.
* Tests that use pd.read_excel require openpyxl; include openpyxl in the dev dependency group so CI (uv sync --group dev) passes.
* TsgamSolverConfig has solver, verbose, warm_start (default True), and solver_opts (dict forwarded as **kwargs to cvxpy Problem.solve; SolverOptionValue is the recursive type alias); CVXPY DPP compliance requires pre-weighting design matrices by sqrt(w/sum(w)); products of cp.Parameter violate DPP rules.
* TsgamMultiPeriodicConfig.periods, TsgamTrendConfig.grouping, and outlier period bucketing use the same regular-grid sample index as fit (not raw wall-clock hours); multiply intended hour-based widths by samples_per_hour for sub-hourly data.
* Marimo (`examples/example_tidal_marimo.py`): cells cannot use implicit globals—names must be imported or returned from earlier cells; avoid early `return` in `@app.cell` (single-exit); assignments like `fig`/`axes`/`label` are cross-cell symbols and duplicate across cells → `multiple-definitions` (use `_`-prefixed plot locals). `marimo run` may work while `marimo check` warns until the file matches canonical marimo layout.
