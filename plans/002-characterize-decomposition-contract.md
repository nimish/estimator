# Plan 002: Freeze TSGAM decomposition behavior before cutover

> **Executor instructions**: Add characterization tests only. Do not alter the
> estimator to make a test easier to write. These tests define compatibility
> for Plan 003.
>
> **Drift check**:
> `git diff --stat 57cebe0..HEAD -- src/tsgam_estimator test pyproject.toml`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `57cebe0`, 2026-09-01

## Why this matters

Solver success does not protect offset signs, masks, regularization scaling,
component reconstruction, or time phase. Plan 003 replaces the final convex
assembly, so these semantics need executable tests before the old path is
removed.

## Current state

- `_problem.py:38-207` creates the ordinary weighted loss, variables,
  regularizers, and prediction expression.
- `_estimator.py:1290-1422` adds grouped trend/outlier terms and solves.
- `_problem.py:242-322` separately evaluates coefficients for prediction.
- `_estimator.py:1436-1510` reconstructs a baseline for residual AR.
- Existing structural patterns live in `test/test_tsgam_multi_frequency.py`,
  `test/test_tsgam_exog_interactions.py`, and `test/test_tsgam_trend.py`.
- Focused baseline on this working tree: 124 tests pass across those files plus
  solver, refit, and forecast tests.

## Commands

| Purpose | Command | Expected result |
|---|---|---|
| New contract tests | `uv run pytest -q test/test_tsgam_signaldecomp_contract.py` | all pass |
| Focused regression | `uv run pytest -q test/test_tsgam_multi_frequency.py test/test_tsgam_exog_interactions.py test/test_tsgam_trend.py test/test_tsgam_solver_opts.py test/test_tsgam_parametric_refit.py` | all pass |
| Typecheck | `uv run ty check` | exit 0 |

## Scope

**In scope**:

- create `test/test_tsgam_signaldecomp_contract.py`

**Out of scope**:

- all files under `src/`;
- forecast behavior;
- changing public offset or lag names;
- binary golden files;
- tolerances broad enough to hide sign or phase errors.

## Steps

### Step 1: Add a deterministic all-component fixture

Create hourly data with a fixed RNG seed and enough samples for constant, one
periodic block, spline offsets `[-1, 0, 1]`, a linear term containing offset
`0`, their interaction, grouped trend, grouped outlier, and nonuniform positive
sample weights. Use standalone test functions.

**Verify**: fixture fits with an optimal or optimal-inaccurate status.

### Step 2: Assert the fitted decomposition identity

Using existing fitted variables and design matrices, compute each structural
signal separately. Assert their sum equals `predict(X)` on fitted rows to a
tight tolerance. Include trend and outlier explicitly; coefficient vectors are
not component signals.

**Verify**: the test fails if any contribution is omitted or an offset sign is
reversed, then passes with the complete reconstruction.

### Step 3: Freeze masks, weighting, and scaling

Assert:

- boundary rows excluded by `[-1, 0, 1]` are exactly expected;
- `problem_.value` matches manually evaluated residual plus regularization;
- sample weights are normalized only over fitted rows;
- outlier L1 and Fourier terms use current TSGAM scaling.

**Verify**: the new test module passes.

### Step 4: Freeze phase across timestamp gaps

Fit a periodic signal on a continuous index and after removing interior
timestamps. Assert shared timestamps retain the same Fourier phase rather than
compressing the gap into adjacent samples.

**Verify**: the test fails if Fourier rows use `0..len(X)-1`.

### Step 5: Freeze compatibility attributes

Assert existing `variables_` keys/shapes and `problem_` status for constant,
exogenous, Fourier, interaction, trend, outlier, and trend slope configurations.

**Verify**: contract and parametric-refit tests pass.

## Test plan

Use small deterministic arrays and CLARABEL unless an existing test requires
SCS. Avoid snapshots and external data.

## Done criteria

- [ ] decomposition identity includes every configured component;
- [ ] offset boundaries, gap phase, weighting, and penalty scaling are asserted;
- [ ] compatibility keys and shapes are asserted;
- [ ] new and focused regression tests pass;
- [ ] `uv run ty check` exits 0;
- [ ] no production source changed.

## STOP conditions

- Current code cannot reconstruct its own fitted prediction.
- The objective cannot be reproduced from documented terms.
- Dirty source differs from the current-state locations above.

Report a STOP as a pre-existing decision or bug; do not encode an accidental
result as the new contract.

## Maintenance notes

Delete assertions only after a deliberate public deprecation, not during the
internal cutover.
