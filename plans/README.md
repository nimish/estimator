# TSGAM / SignalDecomp refactor — completed

Finalized 2026-09-09 in the working tree. Plans 001–005 are historical proposals,
not instructions to rerun. Their goals are implemented with the boundaries below.

## Reconstructed commit history

On 2026-09-10 the accumulated refactor was split into reviewable checkpoints on
`codex/signaldecomp-refactor`, starting from `57cebe0`. These are reconstructed
logical checkpoints, not recovered chronological edits; overwritten intermediate
implementations were not preserved in Git.

- `0921228`: native engine, dependencies, forecasting, and core regression tests
  (218 core tests and source type checks passed on the exact staged export).
- `aeddcf4`: legacy and research consumer migration (42 focused tests passed;
  changed Python consumers compiled).
- `03b2e0a`: tidal migration and notebook retirement; the complete staged refactor passed
  358 tests, the primary tidal Marimo check, and the lock consistency check.

Unrelated solar-analysis and optional-nowcast work, the forecast decision draft,
and generated Marimo state remain outside the refactor commits.

| Plan | Outcome |
|---|---|
| 001 | TSGAM dependency floors aligned; SignalDecomp pinned to immutable Git commit `4cfcc3271f9a24fb3a442882df45ca1668590e6f`, not PyPI or a local skill checkout. |
| 002 | Reconstruction, masks, weights, sample cadence, offsets, interactions, and AR behavior covered by regression tests. |
| 003 | Ordinary fitting builds and solves native SignalDecomp problems. |
| 004 | Native spline/offset/interaction/Fourier/grouped components replace generic local implementations. |
| 005 | Coupled forecasting combines per-horizon native problems and adds TSGAM cross-horizon penalties; legacy coefficient dictionaries are removed. |

## Final ownership

SignalDecomp owns component construction, generic bases, fitting masks, residual
linking, and native solved values. TSGAM owns sklearn lifecycle, timestamp and
sample-unit policy, component selection, forecasting, grouped-trend extrapolation,
and residual AR fitting/sampling. No upstream SignalDecomp changes are required.

The coupled implementation deliberately combines existing per-horizon CVXPY
objectives and constraints instead of the originally proposed concatenated signal.
This preserves per-horizon loss normalization and native component behavior.
No compatibility backend or result-layout adapter is retained.

Ordinary results live in `decomposition_`; coupled numeric coefficient snapshots
live in `horizon_values_`, using native role names. Prediction excludes fitted
outliers; fitted reconstruction and diagnostics use `components_to_frame`.
The primary tidal consumer is `examples/example_tidal_compact.py`.

## Verification

Run from the repository root:

```bash
uv run pytest -q
uv run ty check
uv run marimo check examples/example_tidal_compact.py
uv lock --check
git diff --check
```

Solver success is not a uniqueness claim: regression checks establish numerical
reconstruction and prediction contracts. No new multi-year NOAA study or
end-to-end uncertainty calibration is claimed by this refactor.
