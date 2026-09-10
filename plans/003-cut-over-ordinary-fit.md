# Plan 003: Route ordinary TSGAM fits through signaldecomp

> **Executor instructions**: Change only ordinary `TsgamEstimator.fit`
> composition. Reuse the existing `_TsgamDesign` matrices exactly. Do not adopt
> signaldecomp catalog basis builders in this plan and do not touch coupled
> forecasting.
>
> **Drift check**:
> `git diff --stat 57cebe0..HEAD -- src/tsgam_estimator/_problem.py src/tsgam_estimator/_estimator.py test/test_tsgam_signaldecomp_contract.py`

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/001-compatible-signaldecomp-dependency.md` and
  `plans/002-characterize-decomposition-contract.md`
- **Category**: migration / architecture
- **Planned at**: commit `57cebe0`, 2026-09-01

## Why this matters

TSGAM currently duplicates the generic decomposition engine: residual fidelity,
variables, component penalties, constraints, final CVXPY problem construction,
solve dispatch, and status validation. This plan moves those responsibilities
to signaldecomp without mixing in basis or forecast changes.

## Current state

- `src/tsgam_estimator/_problem.py:38-108` builds the ordinary residual loss,
  standard variables, and regularization.
- `_problem.py:184-207` constructs the fitted ordinary prediction expression.
- `src/tsgam_estimator/_estimator.py:1313-1422` adds grouped trend/outlier terms,
  creates `cvxpy.Problem`, and solves it.
- `_estimator.py:1436-1463` separately reconstructs a baseline for residual AR.
- `_problem.py:242-322` evaluates coefficients on a prediction design and must
  remain for out-of-sample prediction.
- Existing callers inspect `problem_` and scalar/coefficient expressions in
  `variables_`; preserve those views.

Use `signaldecomp.Component`, `make_problem`, and `solve` directly. Do not add
an interface, backend flag, factory hierarchy, or registry class.

## Commands

| Purpose | Command | Expected result |
|---|---|---|
| Contract | `uv run pytest -q test/test_tsgam_signaldecomp_contract.py` | all pass |
| Ordinary fit regression | `uv run pytest -q test/test_tsgam_multi_frequency.py test/test_tsgam_exog_interactions.py test/test_tsgam_trend.py test/test_tsgam_solver_opts.py test/test_tsgam_parametric_refit.py test/test_tsgam_nan_handling.py` | all pass |
| Typecheck | `uv run ty check` | exit 0 |
| Full tests | `uv run pytest -q` | all pass |

## Scope

**In scope**:

- `src/tsgam_estimator/_problem.py`
- `src/tsgam_estimator/_estimator.py`
- `test/test_tsgam_signaldecomp_contract.py`
- additional focused tests only when a discovered compatibility behavior is not
  representable in the contract file

**Out of scope**:

- `_design.py` basis generation;
- `_forecast.py` coupled composition;
- public config fields or offset sign conventions;
- residual AR mathematics or sampling;
- signaldecomp source;
- a second selectable backend.

## Target shape

Add plain helper functions to `_problem.py`:

1. `make_single_output_components(config, design, trend_mapping,
   outlier_mapping)` returns a list of signaldecomp `Component` objects built
   from existing matrices.
2. `solve_single_output_decomposition(...)` calls `make_problem` and `solve`,
   checks the final mask against `design.valid_mask`, and returns the built/solved
   dictionary plus the compatibility `variables_` mapping.

Use role names for fitted signals:

- `constant`, `exog_0`, `exog_1`, ...;
- `periodic`;
- `interaction_0`, ...;
- `trend` and `outlier`.

Expose coefficient expressions under distinct auxiliary names, then map those
back to current `variables_` keys such as `constant`, `exog_coef_0`,
`fourier_coef`, `interaction_coef_0`, `trend`, `trend_slope`, and `outlier`.

## Steps

### Step 1: Translate existing matrices into Components

In `_problem.py`, create each component with a local `build(T)` closure:

- constant: scalar coefficient and length-`T` constant expression;
- exogenous: current `design.exog_Hs` blocks and coefficient matrix;
- periodic: current `design.fourier_basis` and current Fourier regularizer;
- interaction: current `design.interaction_Hs` and coefficient vector;
- trend/outlier: the mappings still constructed by TSGAM from timestamps.

Each closure must reproduce the current penalty and constraints exactly. Attach
component availability masks derived from its finite design rows. Keep trend
linear/monotone constraints unchanged.

**Verify**: contract tests comparing objective terms and coefficient shapes pass.

### Step 2: Supply exact weighted residual fidelity

Create a residual-loss callable closing over full-length sample weights. Set
weights outside the expected fit mask to zero and return:

```python
cp.sum_squares(cp.multiply(np.sqrt(weights), residual)) / weights.sum()
```

Reject a nonpositive fitted weight sum before building the problem.

After `make_problem`, assert `built["fit_mask"]` equals `design.valid_mask`.
This assertion is an integration invariant, not a fallback.

**Verify**: weighted objective and mask tests pass.

### Step 3: Solve through signaldecomp

Call `solve` with TSGAM's solver, `verbose`, `warm_start`, and validated
`solver_opts`. Keep `verify_dcp=True`. Store:

- `self.problem_ = solved["problem"]`;
- a private solved result for diagnostics;
- role-valued structural signals separately from coefficient auxiliaries;
- the existing `self.variables_` compatibility mapping.

Do not copy numeric values into new CVXPY variables; reuse the expressions from
the solved dictionary.

**Verify**: solver option, warm-start, refit, and compatibility tests pass.

### Step 4: Make fitted reconstruction authoritative

Sum structural role values, excluding `residual`, once after solve and store the
full-length fitted deterministic reconstruction. Assert on the final fit mask:

```text
y == residual + deterministic reconstruction
```

within solver tolerance.

Change `_fit_ar_model` to consume this reconstruction instead of rebuilding a
partial baseline from coefficient helpers. This must include configured
outliers; do not alter the AR optimization itself.

**Verify**: decomposition identity and AR tests pass.

### Step 5: Remove ordinary-only assembly calls

Delete `make_single_output_standard_variables` and
`single_output_prediction_expression` after `rg` shows no callers. Keep
`weighted_squared_loss`, `solve_problem`, horizon builders, and out-of-sample
evaluation while `_forecast.py` still uses them.

**Verify**:

```bash
rg -n 'make_single_output_standard_variables|single_output_prediction_expression' src test
```

returns no matches, and the full suite passes.

## Test plan

- Extend the Plan 002 contract tests to compare role-valued reconstruction and
  compatibility coefficient expressions.
- Run existing multi-frequency, interaction, trend, solver, refit, NaN, AR, and
  sklearn compatibility tests.
- Add one test asserting signaldecomp DCP verification rejects a deliberately
  malformed TSGAM component only if no upstream test already covers that gate.

## Done criteria

- [ ] ordinary `fit` uses `signaldecomp.make_problem` and `solve`;
- [ ] final fit mask and weighted objective match the prior contract;
- [ ] structural values are keyed by role and reconstruct the fit;
- [ ] AR residuals use that reconstruction;
- [ ] `problem_` and `variables_` compatibility tests pass;
- [ ] ordinary-only legacy assembly functions have no callers and are deleted;
- [ ] typecheck and full tests pass;
- [ ] `_design.py` and coupled forecast composition are unchanged.

## STOP conditions

- signaldecomp cannot reproduce the exact final mask or weighted objective.
- Compatibility requires copying solved numbers into new fake variables.
- A basis or coefficient shape must change to make the cutover work.
- Coupled forecast code must change for ordinary tests to pass.
- The dirty estimator file has not been stabilized before execution.

## Maintenance notes

Review the distinction between role-valued signals and coefficient auxiliaries.
Future components must provide both when TSGAM needs out-of-sample evaluation.
