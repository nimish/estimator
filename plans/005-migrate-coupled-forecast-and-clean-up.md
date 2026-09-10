# Plan 005: Compose coupled forecasts with signaldecomp and delete legacy assembly

> **Executor instructions**: Independent forecast mode should already inherit
> Plan 003 through child `TsgamEstimator` instances. Change only coupled mode,
> using one stacked signaldecomp problem; do not add a new upstream joint-problem
> API unless the stacked formulation proves impossible.
>
> **Drift check**:
> `git diff --stat 57cebe0..HEAD -- src/tsgam_estimator/_forecast.py src/tsgam_estimator/_problem.py test/test_tsgam_forecast_mode.py`

## Status

- **Priority**: P2
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/004-adopt-signaldecomp-components.md`
- **Category**: migration / cleanup
- **Planned at**: commit `57cebe0`, 2026-09-01

## Why this matters

`_fit_coupled` still duplicates final loss aggregation, CVXPY problem creation,
solver dispatch, and status handling across forecast horizons. A concatenated
signal is already sufficient for `make_problem`: horizon-specific coefficients
and cross-horizon roughness belong inside TSGAM-built Components, while generic
residual/mask/problem/solve behavior remains in signaldecomp.

## Current state

- `_forecast.py:417-437` independent mode fits ordinary child estimators.
- `_forecast.py:493-580` coupled mode creates horizon variables, predictions,
  weighted losses, cross-horizon penalties, the final `cvxpy.Problem`, and
  solves it.
- `_problem.py:111-239` contains horizon variable and prediction helpers.
- `test/test_tsgam_forecast_mode.py` covers zero-coupling parity, coefficient
  smoothing, origin alignment, forecast AR, invalid configs, and outlier
  rejection.
- Coupled mode deliberately rejects `outlier_config`; keep that policy unless a
  separate feature request changes it.

## Commands

| Purpose | Command | Expected result |
|---|---|---|
| Forecast tests | `uv run pytest -q test/test_tsgam_forecast_mode.py test/test_tsgam_forecast_nowcast.py test/test_forecast_real_data_support.py` | all pass |
| Ordinary regression | `uv run pytest -q test/test_tsgam_signaldecomp_contract.py test/test_tsgam_multi_frequency.py` | all pass |
| Typecheck/full suite | `uv run ty check && uv run pytest -q` | both exit 0 |
| Legacy search | `rg -n 'weighted_squared_loss|solve_problem|make_horizon_standard_variables|horizon_prediction_expression' src/tsgam_estimator` | no matches after cleanup |

## Scope

**In scope**:

- `src/tsgam_estimator/_forecast.py`
- `src/tsgam_estimator/_problem.py`
- `test/test_tsgam_forecast_mode.py`
- focused forecast tests only when required

**Out of scope**:

- forecast origin/target alignment;
- nowcast inclusion policy;
- forecast AR feature construction or regularization;
- generative residual AR and sampling;
- support for outlier components in coupled mode;
- a generic upstream multi-output estimator;
- a selectable legacy backend.

## Target formulation

For `H` horizons, concatenate the horizon targets and weights in deterministic
horizon order:

```text
y_stacked = [y_0, y_1, ..., y_H]
```

Each TSGAM Component returns the correspondingly stacked signal. Its auxiliary
coefficient values retain a horizon axis. Its loss contains both:

1. the sum of existing per-horizon coefficient penalties; and
2. the existing cross-horizon roughness penalty.

The residual-loss callable applies the concatenated, per-horizon normalized
weights. `make_problem` and `solve` then create and solve one scalar-vector
decomposition without any new signaldecomp API.

## Steps

### Step 1: Add a stacked coupled parity test

Before changing source, extend `test_tsgam_forecast_mode.py` with a small case
that records:

- predictions for every horizon;
- coefficient arrays and horizon axis order;
- objective value for zero and nonzero coupling;
- per-horizon final masks and weight normalization.

Keep existing zero-coupling parity with independent fits.

**Verify**: the new test passes on the legacy coupled path.

### Step 2: Build stacked horizon Components

In `_forecast.py`, translate the existing list of `_TsgamDesign` objects into
Components for constant, exogenous, periodic, interaction, and configured
forecast-AR terms. Use `cp.hstack`/`cp.concatenate` only to form the stacked
signal; retain coefficient matrices with horizon as the last axis.

Put cross-horizon roughness inside each owning component's loss. Preserve the
current rule that horizon zero is not coupled when that is what the existing
roughness helper implements.

Attach a stacked availability mask built in the same order as `y_stacked`.

**Verify**: coefficient order/shape and mask tests pass.

### Step 3: Supply stacked weighted residual fidelity

Build a full stacked weight vector with zero outside each horizon's final mask.
Preserve current per-horizon normalization: if legacy code sums separately
normalized horizon losses, scale each horizon block so the single residual
callable evaluates the identical sum.

Assert signaldecomp's returned `fit_mask` equals the expected concatenated mask.

**Verify**: zero/nonzero coupling objective parity tests pass.

### Step 4: Solve and map results

Call `make_problem` and `solve` with current solver configuration. Store
`problem_`, compatibility `variables_`, and role values. Split stacked role
signals back into horizons only at the TSGAM forecast boundary.

Do not change prediction indexing, returned DataFrame columns, or target-time
alignment.

**Verify**: forecast mode, nowcast, and real-data support tests pass.

### Step 5: Delete legacy final assembly

After all callers move, delete from `_problem.py`:

- `weighted_squared_loss`;
- `solve_problem`;
- `make_horizon_standard_variables`;
- `horizon_prediction_expression`.

Retain `evaluate_horizon_prediction` only if prediction still uses it; otherwise
delete it too. Keep direct CVXPY use for residual AR and component definitions.

**Verify**: the legacy `rg` command returns no matches and the full suite passes.

### Step 6: Verify installed-package behavior

Build TSGAM, install the wheel in a clean temporary environment, and smoke-test
ordinary fit plus independent and coupled forecast modes. This catches an
undeclared or path-only signaldecomp dependency.

**Verify**: `uv build` succeeds and the installed-package smoke script exits 0.

## Test plan

- Preserve every existing forecast-mode test.
- Add stacked objective/mask/role-split tests.
- Explicitly test one horizon, several horizons, zero coupling, nonzero
  coupling, include/exclude nowcast, exogenous offsets, and forecast AR.
- Run ordinary contract tests to catch shared-helper regressions.

## Done criteria

- [ ] ordinary and coupled deterministic fits use signaldecomp composition;
- [ ] independent forecast remains ordinary child-estimator orchestration;
- [ ] coupled objective, masks, coefficients, and predictions preserve the
  characterized contract;
- [ ] no generic joint API was added upstream unless stacked composition was
  proven impossible and documented;
- [ ] legacy final-assembly helpers have no callers and are deleted;
- [ ] AR, sampling, and forecast policy remain in TSGAM;
- [ ] focused/full tests, typecheck, build, and installed smoke test pass.

## STOP conditions

- Stacking cannot reproduce the sum of separately normalized horizon losses.
- A role cannot retain deterministic horizon/auxiliary ordering.
- The change requires altering origin alignment, nowcast, or forecast AR policy.
- A new upstream joint API appears necessary before a minimal stacked prototype
  has been tested.
- Dirty forecast changes have not been stabilized before execution.

## Maintenance notes

The stacked block order is part of the internal contract; keep one helper and
one test authoritative. Review future horizon-specific components for both
within-horizon penalties and cross-horizon coupling.
