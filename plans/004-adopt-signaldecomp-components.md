# Plan 004: Replace duplicated generic components with signaldecomp

> **Executor instructions**: Replace one component family at a time, running its
> gate before continuing. Preserve TSGAM's public configuration and prediction
> behavior. A stricter signaldecomp validation is not permission to silently
> reject previously valid TSGAM models.
>
> **Drift check**:
> `git diff --stat 57cebe0..HEAD -- src/tsgam_estimator/_design.py src/tsgam_estimator/_problem.py src/tsgam_estimator/_estimator.py test examples pyproject.toml`

## Status

- **Priority**: P2
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/003-cut-over-ordinary-fit.md`
- **Category**: migration / tech debt
- **Planned at**: TSGAM `57cebe0`, signaldecomp `4cfcc32`, 2026-09-01

## Why this matters

After Plan 003, signaldecomp composes the problem but TSGAM still duplicates
generic spline, offset, interaction, grouping, and periodic implementations.
The latest signaldecomp contains those facilities, many adapted from TSGAM.
This plan makes signaldecomp the implementation owner while TSGAM remains the
translator from timestamps and configuration.

## Current state

- `_design.py:298-367` implements the same natural-spline basis and ordered
  shifted blocks now available in signaldecomp.
- `_design.py:384-401` implements tensor-product interaction construction and
  evaluation.
- `_design.py:259-295` and `503-541` build periodic basis/regularization at
  explicit TSGAM sample indices.
- `_estimator.py:1027-1192` exposes private wrappers used by tests and examples.
- signaldecomp provides `make_spline_basis`, `make_offset_basis`,
  `exog_linear`, `exog_spline`, `make_interaction_basis`, `exog_interaction`,
  `make_group_basis`, `grouped_trend`, and `grouped_sparse`.
- Live probes show exact equality for the spline basis and for offset blocks
  after sign conversion.

## Required conversions

- TSGAM: `x[t + offset]`, negative means past.
- signaldecomp: `z[t - offset]`, positive means past.
- Always pass `tuple(-offset for offset in tsgam_offsets)` and preserve order.
- `multiperiodic`: pass `sqrt(tsgam_reg_weight)` because the weight is inside a
  regularization matrix whose norm is squared.
- `grouped_sparse`: pass `tsgam_reg_weight * n_groups` because signaldecomp
  divides the supplied weight by `n_groups`.

## Commands

| Purpose | Command | Expected result |
|---|---|---|
| Contract | `uv run pytest -q test/test_tsgam_signaldecomp_contract.py` | all pass |
| Exogenous/interactions | `uv run pytest -q test/test_tsgam_linear_config.py test/test_tsgam_exog_interactions.py test/test_tsgam_multi_frequency.py` | all pass |
| Trend | `uv run pytest -q test/test_tsgam_trend.py` | all pass |
| Examples compatibility | `uv run pytest -q test/test_example_tidal_compact.py test/test_notebook_compatibility.py` | all pass |
| Typecheck/full suite | `uv run ty check && uv run pytest -q` | both exit 0 |

## Scope

**In TSGAM scope**:

- `src/tsgam_estimator/_design.py`
- `src/tsgam_estimator/_problem.py`
- `src/tsgam_estimator/_estimator.py`
- affected tests and in-repo callers of retained private compatibility wrappers

**In signaldecomp scope, only if periodic parity needs it**:

- `src/signaldecomp/periodic.py`
- `tests/` periodic coverage
- public export documentation for the added argument

**Out of scope**:

- timestamp/frequency validation;
- public config or offset convention changes;
- residual AR, sampling, and forecast policy;
- coupled forecast composition;
- wholesale notebook cleanup;
- removing `spcqe` while tests/examples still import it directly.

## Steps

### Step 1: Delegate spline and offset numerics

Replace `_make_spline_H` internals with `signaldecomp.make_spline_basis` and
replace `_make_offset_H` internals with `make_offset_basis`, applying the sign
conversion. Retain the existing private methods as thin deprecated wrappers for
one compatibility cycle because examples and tests call them.

Move fit component construction to `exog_linear` and `exog_spline` with stored
knots and exact coefficient/penalty mapping. Keep TSGAM prediction design
construction using the same signaldecomp basis utilities.

**Verify**: contract, linear, multi-frequency, and notebook compatibility tests
pass without tolerance relaxation.

### Step 2: Delegate grouped components

TSGAM continues to derive group labels/mappings from timestamps. Use
`grouped_trend` for nonlinear monotone grouped trends and `grouped_sparse` for
outliers with the documented weight conversion.

Keep the current linear grouped-trend component local because signaldecomp
`grouped_trend` does not enforce equal first differences. Add generic grouped
affine support upstream only if at least one non-TSGAM use exists; otherwise the
small local component is cheaper.

Map group-value auxiliaries back to current `variables_` keys.

**Verify**: trend, outlier contract, and prediction tests pass.

### Step 3: Adopt interaction construction conditionally

Use `exog_interaction` with the current zero-offset factor bases and fit mask.
Run all existing interaction configurations through signaldecomp's
interaction-only rank validation.

If a previously supported model is rejected, stop and determine whether the
old model was unidentifiable. Do not disable the upstream check merely for
parity. A deliberate behavior change needs its own documented decision and
test; otherwise retain the current TSGAM interaction component while importing
only the generic tensor-product utility.

**Verify**: interaction and contract tests pass; coefficient matrix order still
matches TSGAM's C-order contraction.

### Step 4: Add explicit sample positions to periodic upstream

`signaldecomp.multiperiodic` currently evaluates consecutive positions. Add an
optional immutable one-dimensional `sample_positions` argument to
signaldecomp. Validate length, finiteness, and uniqueness; when absent, retain
`0..T-1`. Build the Fourier rows at those positions without rounding float
periods or compressing gaps. Keep regularization unchanged.

Add upstream tests for continuous, gapped, negative-origin, and float-period
positions. Release or pin the resulting immutable revision, then pass
`design.time_indices` from TSGAM and remove TSGAM's periodic basis/regularizer
implementation.

**Verify**: upstream full suite passes; TSGAM gap-phase and objective-scaling
tests pass.

### Step 5: Reduce compatibility wrappers to delegation

Keep `_make_H`, `_make_offset_H`, `_process_exog_config`, and related estimator
methods only as thin delegating wrappers if current examples still call them.
Emit `FutureWarning` only if removal is intended and migrate all in-repo callers
in the same cycle. Delete duplicated mathematical implementations immediately.

Run `rg` before considering `spcqe` removal; current tests and examples import it
directly, so dependency removal is not part of this plan unless those callers
are explicitly migrated.

**Verify**:

```bash
rg -n 'def _make_spline_H|def _make_regularization_matrix|make_basis_matrix' src/tsgam_estimator
```

returns no duplicated implementation/import, aside from documented delegating
wrappers whose bodies call signaldecomp.

## Test plan

- Preserve all Plan 002 contract cases.
- Add direct equality tests for spline basis, each offset sign/boundary, grouped
  penalty scaling, interaction column order, and gapped periodic phase.
- Test private wrapper warnings and results only if wrappers remain.
- Run signaldecomp and TSGAM full suites.

## Done criteria

- [ ] generic spline, offset, grouping, and periodic math is implemented once in
  signaldecomp;
- [ ] interactions use signaldecomp or have a documented identifiability reason
  to remain local;
- [ ] TSGAM owns timestamp-to-group/sample-position translation only;
- [ ] public offset sign, coefficient shapes, objective scaling, and predictions
  remain compatible;
- [ ] no mutable or local-path dependency is introduced;
- [ ] both full suites and typecheck pass.

## STOP conditions

- Catalog adoption changes the represented function space or coefficient
  penalty.
- Interaction validation rejects an established model without a reviewed
  identifiability decision.
- Periodic explicit positions cannot reproduce TSGAM gap phase exactly.
- Required notebook/example migration grows beyond callers of the delegated
  helpers.

## Maintenance notes

Review every future signaldecomp weight normalization change against TSGAM's
adapter tests. Offset sign conversion must remain at the TSGAM boundary.
