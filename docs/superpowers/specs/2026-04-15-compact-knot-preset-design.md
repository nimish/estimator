# Compact Notebook Knot Preset Design

## Summary

`examples/example_tidal_compact.py` currently hardcodes spline knot counts for
exogenous regressors inside `build_exog_design_matrices()`. That makes it
awkward to explore model stiffness in a notebook and to prepare a small
human-readable hyperparameter grid. The goal is to expose a simple
`low / med / high` knot-count control per regressor while leaving the rest of
the regressor UX unchanged.

## Goals

- Add a per-regressor spline knot preset control in the compact notebook.
- Map presets uniformly across regressors as:
  - `low` -> `4`
  - `med` -> `8`
  - `high` -> `12`
- Feed the selected preset directly into `TsgamSplineConfig(n_knots=...)`.
- Add a regressor visualization section that shows both the loaded regressor
  values and the spline basis implied by the selected knot preset.
- Keep the change easy to reason about as preparation for later manual or
  programmatic hyperparameter search.

## Non-Goals

- Do not add a linear-vs-spline selector in this change.
- Do not change lag controls, regressor toggles, or harmonic controls.
- Do not add an actual grid-search runner in this notebook.
- Do not add fitted partial-effect interpretation charts in this change.

## Proposed Changes

### Notebook Controls

- Extend each regressor row in the `Configure model` section with a knot preset
  control.
- Use a compact dropdown or radio-style choice labeled `Knots` with values
  `low`, `med`, and `high`.
- Only render the knot preset control for regressors that are available in the
  loaded dataset and already shown in the current regressor section.
- Keep the current regressor toggle and lag range control in place; the new knot
  preset is additive rather than a redesign.

### Parameter Mapping

- Add a small notebook-level mapping such as:
  - `{"low": 4, "med": 8, "high": 12}`
- Keep the mapping global and uniform across regressors.
- Preserve the current behavior conceptually by treating `med = 8` as the
  standard default and removing the special-cased hardcoded `10` knots for
  `pressure`.

### Model Assembly

- Extend the current parameter collection path so regressor knot presets are
  captured alongside lag ranges and toggles.
- Update `build_exog_design_matrices()` to accept the selected preset-derived
  knot counts and use them when constructing each `TsgamSplineConfig`.
- Continue using spline exogenous configs for all regressors in this change.

### Regressor Visualization

- Add a separate notebook section for inspecting one currently available model
  regressor at a time.
- Include a regressor selector tied to the currently available regressor list.
- Show two complementary views for the selected regressor:
  - a data view showing the raw loaded regressor values over the loaded time
    window;
  - a basis view showing spline knot locations and basis-function shapes implied
    by the current `low / med / high` preset for that regressor.
- Keep the visualization descriptive rather than inferential:
  - it should show the input data and basis flexibility;
  - it should not attempt to show a fitted regressor effect or causal
    interpretation in this change.
- Derive the basis visualization from the same knot-count mapping and regressor
  data range used for model assembly so the chart and model stay aligned.

## Error Handling

- If a regressor is disabled, its knot preset selection should have no effect.
- If a regressor is enabled but unavailable after preprocessing, the notebook
  should continue following the current active-regressor behavior without adding
  new failure modes.
- The preset mapping should be total and explicit so invalid preset labels do
  not silently fall back to arbitrary counts.
- If no model regressors are available in the loaded dataset, the visualization
  section should continue to fail closed with a clear notebook-visible message
  rather than rendering an empty chart.

## Testing

- Add focused regression coverage for the preset mapping and exogenous config
  assembly path.
- Cover at least:
  - translating `low / med / high` into `4 / 8 / 12`;
  - building spline exogenous configs with the selected knot counts instead of
    the previous hardcoded defaults;
  - building the basis visualization inputs from the same preset-derived knot
    counts used by the model path.

## Acceptance Criteria

- The compact notebook shows a `low / med / high` knot preset control for each
  available regressor.
- Choosing different presets changes the resulting `TsgamSplineConfig.n_knots`
  values used for fitting.
- The notebook includes a regressor inspection section that shows both the raw
  selected regressor series and the spline basis implied by the current preset.
- The notebook still behaves the same in all other regressor dimensions
  (toggle, lag range, spline-only model family).
- Users can use the notebook to define simple stiffness settings that are easy
  to translate into a later hyperparameter grid, while seeing what those
  settings mean for the selected regressor basis.
