# Project Instructions

* Use uv to run python scripts and code
* Use pytest for testing

## Learned User Preferences

* Prefer positive boolean config flags (e.g. `use_dpp: True`) over negated ones (e.g. `ignore_dpp: False`).
* Prefer bash scripts with bashisms over POSIX sh.

## Learned Workspace Facts

* Inferred frequency strings must use numeric prefix (e.g. '1h', '1D'); store with prefix and lowercase.
* Fit allows gaps and masks NaNs; predict requires strictly regular timestamps (no gaps).
* Data-dependent tests use examples/data/iso for ISO data files.
* TsgamSplineConfig.knots accepts numpy array or list; use len-based emptiness checks, not truthiness on arrays.
* sort_index config (default True): when True sort by index; when False, index must already be sorted or error.
* Optional sample_weight in fit for weighted least squares; AR step is unweighted.
* junit.xml is gitignored.
* Example/run scripts use Click and Rich for CLI and UI; deps are in dependency-groups.examples (uv sync --group examples).
* Example run scripts produce publication-quality figures (data overview, model summary, ablation comparison) as PDF and PNG for journal or presentation use.
* Tests that use pd.read_excel require openpyxl; include openpyxl in the dev dependency group so CI (uv sync --group dev) passes.
* TsgamSolverConfig has warm_start (default True) and use_dpp (default False); call invalidate_compiled_problem() before fitting with different data shapes.
* CVXPY DPP compliance requires pre-weighting design matrices by sqrt(w/sum(w)); products of cp.Parameter violate DPP rules.
