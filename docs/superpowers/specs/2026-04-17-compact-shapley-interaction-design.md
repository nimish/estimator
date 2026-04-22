# Compact Notebook Shapley Interaction Design

## Summary

`examples/example_tidal_compact.py` currently passes selected exogenous
interaction pairs into ordinary model fits, but the notebook's Shapley analysis
only treats harmonics and main-effect regressors as components. The goal of
this change is to promote selected interaction pairs into first-class Shapley
components while enforcing the agreed semantics that an interaction component is
only valid when both parent regressors are present in the coalition.

## Goals

- Include active interaction pairs as explicit Shapley components in the compact
  notebook.
- Keep the existing interaction-selection UI and estimator behavior unchanged
  outside the Shapley path.
- Enforce the rule that an interaction component cannot be evaluated unless both
  parent regressors are active in the same coalition.
- Avoid redundant model fits by deduplicating invalid raw coalitions that repair
  to the same valid coalition.
- Show interaction terms in the Shapley waterfall output alongside harmonics and
  main-effect regressors.

## Non-Goals

- Do not change estimator-level interaction semantics or validation in
  `tsgam_estimator.py`.
- Do not add new interaction-selection controls to the notebook.
- Do not introduce higher-order interactions or automatic interaction discovery.
- Do not replace the existing Shapley formula with a different attribution
  method.
- Do not redesign the notebook's non-Shapley fit, diagnostics, or regressor UI.

## Proposed Changes

### Shapley Component Set

- Extend the Shapley component list to include:
  - active harmonic terms;
  - active main-effect regressors;
  - active selected interaction pairs whose parents survive preprocessing.
- Keep interaction components distinct from parent regressors in the Shapley
  output so users can see whether a gain comes from the marginal regressor fit
  or from the explicit interaction term.
- Use stable human-readable interaction labels such as
  `Pressure (hPa) x Wind U (m/s)` in the Shapley plot.

### Coalition Semantics

- Treat the Shapley game as operating on a component list that now includes
  interactions.
- When a raw coalition includes an interaction component without both of its
  parent regressors, canonicalize that coalition by dropping the invalid
  interaction component before running the model.
- Leave parent-regressor inclusion unchanged; only the invalid interaction bit
  is repaired.
- Reuse canonicalized coalition metrics for every raw coalition that repairs to
  the same valid coalition so the existing `compute_shapley()` helper can still
  consume a full `bits -> metrics` mapping.

### Model-Run Translation

- Add notebook-level helpers that translate a canonical Shapley coalition into:
  - the ordinary `component_mask` for harmonics and main-effect regressors;
  - the subset of selected `interaction_pairs` whose interaction components are
    active and whose parents are active.
- Keep `run_tidal_model()` as the single entry point for fitting. The Shapley
  layer should decide which parent components and interaction pairs are active
  for a coalition, then pass those through the existing model path.
- Preserve the current behavior that interactions whose parents were dropped by
  preprocessing are excluded cleanly.

### Run Counting and Progress

- Count interactions toward the existing Shapley component cap.
- Build the raw component power set from the full component list, but only run
  unique canonical coalitions.
- Update the notebook progress bar to track the number of unique canonical
  coalitions actually evaluated.
- Update `ShapleyResult["coalitions"]` to mean the number of evaluated canonical
  coalitions, since that value is displayed in the figure title.

### Output and Presentation

- Keep the current Shapley figure structure, but allow interaction labels to
  appear as their own waterfall bars.
- Preserve the current baseline/full-model framing:
  - baseline remains the empty coalition;
  - full model remains the coalition with all active harmonics, regressors, and
    valid interaction components.
- Keep solver-failure handling unchanged; failed canonical coalitions still fall
  back through the current metric sanitization path.

## Error Handling

- If fewer than two Shapley components are active after preprocessing, continue
  to fail closed with the current notebook-visible message.
- If a selected interaction pair does not survive preprocessing because one or
  both parent regressors are inactive or unavailable, omit it from the Shapley
  component list rather than creating a dead component.
- If multiple raw coalitions canonicalize to the same valid coalition, evaluate
  that model once and reuse the result rather than treating the duplication as
  an error.
- If a canonical coalition fit produces non-finite metrics, continue using the
  current baseline fallback behavior inside `compute_shapley()`.

## Testing

- Add focused regression tests for the notebook helpers that build Shapley
  interaction components and canonicalize coalitions.
- Cover at least:
  - promoting active interaction pairs into the Shapley component set;
  - dropping interaction bits when one or both parents are missing from a raw
    coalition;
  - deduplicating repaired coalitions so equivalent invalid raw coalitions do
    not trigger repeated model runs;
  - passing only the valid active interaction subset through the Shapley
    `run_tidal_model()` path;
  - showing interaction labels in the Shapley result/figure path.
- Keep existing Shapley tests passing for the no-interaction case.

## Acceptance Criteria

- Running Shapley analysis in the compact notebook includes selected valid
  interaction pairs as explicit components in the attribution output.
- A coalition that turns on an interaction without both parents is evaluated as
  the corresponding parent-only coalition with that interaction removed.
- Equivalent repaired coalitions share the same fitted metrics rather than
  triggering duplicate model runs.
- The Shapley waterfall can display interaction contributions separately from
  parent regressor contributions.
- The notebook reports the number of canonical coalitions actually evaluated,
  not the raw unrepaired power-set size.
