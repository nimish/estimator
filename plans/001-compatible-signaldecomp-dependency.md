# Plan 001: Make signaldecomp consumable by TSGAM

> **Executor instructions**: Follow every step and verification gate. This plan
> changes two repositories. Use a proper development clone of
> `cvxgrp/cvx-sd-skill`, not the installed Codex skill as a long-term dependency.
> Do not change TSGAM's supported Python floor merely to make dependency
> resolution easier.
>
> **Drift check**:
> `git diff --stat 57cebe0..HEAD -- pyproject.toml uv.lock .github/workflows/test.yml`
> and, in signaldecomp,
> `git diff --stat 4cfcc32..HEAD -- pyproject.toml uv.lock .github/workflows/`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: dependency / migration
- **Planned at**: TSGAM `57cebe0`, signaldecomp `4cfcc32`, 2026-09-01

## Why this matters

TSGAM cannot declare the current signaldecomp package: TSGAM supports Python
3.12 with CVXPY 1.7, NumPy 2.3, pandas 2.3, and SciPy 1.16, while signaldecomp
declares Python 3.13 and higher direct dependency floors. The integration must
start with a tested overlapping support window and an immutable package
reference.

## Current state

- `pyproject.toml:9-34` declares TSGAM's Python and runtime dependency ranges.
- signaldecomp `pyproject.toml:1-15` declares Python `>=3.13` and higher direct
  dependency floors.
- TSGAM CI uses `uv run ty check` and `uv run pytest` in
  `.github/workflows/test.yml`.
- signaldecomp at `4cfcc32` passes 203 tests on Python 3.13 and has no checked-in
  GitHub Actions workflow.

## Commands

| Purpose | Command | Expected result |
|---|---|---|
| signaldecomp tests | `uv run --python 3.13 pytest -q` | 203 or more passed |
| Python 3.12 compatibility | `uv run --python 3.12 pytest -q` | all pass |
| signaldecomp build | `uv build` | wheel and sdist created |
| TSGAM dependency resolution | `uv lock --check` | exit 0 |
| TSGAM import smoke test | `uv run python -c "import signaldecomp, tsgam_estimator"` | exit 0 |

## Scope

**In signaldecomp scope**:

- `pyproject.toml`
- `uv.lock`
- create `.github/workflows/test.yml`
- source only if a concrete Python-3.12 incompatibility is found

**In TSGAM scope**:

- `pyproject.toml`
- `uv.lock`

**Out of scope**:

- TSGAM estimator source;
- any API or mathematical change;
- copying signaldecomp source into TSGAM;
- using the installed skill directory as a path dependency;
- raising TSGAM's Python floor without explicit maintainer approval.

## Steps

### Step 1: Test signaldecomp on TSGAM's support floor

In a clean signaldecomp clone at `4cfcc32`, lower only `requires-python` to
`>=3.12`; then test with Python 3.12. Lower direct dependency floors one at a
time to TSGAM's current compatible versions, regenerating the lock and running
the full suite after each change. Do not guess that a lower version works.

**Verify**: `uv run --python 3.12 pytest -q` → all tests pass.

### Step 2: Add an upstream version matrix

Create a small uv-based signaldecomp CI workflow testing Python 3.12 and 3.13.
Do not introduce another environment manager.

**Verify**: run the workflow commands locally for both interpreters → all pass.

### Step 3: Publish or pin an immutable artifact

Preferred: tag and publish the compatible signaldecomp revision. Acceptable
before a registry release: use an immutable Git commit URL. Never use a branch,
local path, or skill path.

Record the selected version or SHA in the TSGAM dependency entry.

**Verify**: `uv build` in signaldecomp → wheel and sdist succeed.

### Step 4: Add the dependency to TSGAM

Add signaldecomp to TSGAM's runtime dependencies and regenerate `uv.lock`.
Retain the existing explicit CVXPY dependency because TSGAM's AR and coupled
forecast code still imports CVXPY directly.

**Verify**:

```bash
uv lock --check
uv run python -c "import signaldecomp, tsgam_estimator"
```

Both commands exit 0.

## Test plan

- signaldecomp full suite on Python 3.12 and 3.13;
- build and install its wheel in a clean environment;
- TSGAM import smoke test after lock regeneration;
- `uv run pytest -q test/test_synthetic_problem.py` in TSGAM.

## Done criteria

- [ ] signaldecomp metadata supports Python 3.12 and tested dependency versions;
- [ ] upstream CI covers Python 3.12 and 3.13;
- [ ] TSGAM depends on an immutable signaldecomp artifact;
- [ ] `uv lock --check` and both import/test smoke checks pass;
- [ ] no source from signaldecomp was copied into TSGAM.

## STOP conditions

- Python 3.12 requires a behavior-changing signaldecomp rewrite.
- A signaldecomp dependency cannot coexist with TSGAM's declared ranges.
- Only a mutable branch or local skill path is available as the dependency.
- The intended TSGAM source changes are still uncommitted in the execution
  checkout.

## Maintenance notes

Keep the TSGAM pin immutable until signaldecomp has a stable release cadence.
Review future dependency-floor raises in both repositories together.
