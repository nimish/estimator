# Compact Estimator Config Export Design

## Summary

`examples/example_tidal_compact.py` currently builds a `TsgamEstimatorConfig`
implicitly from notebook controls, but the selected model configuration is not
visible in a shareable form. The goal is to surface the exact estimator config
for the selected model as a YAML document so users can copy it into scripts,
notebooks, or shared notes.

## Goals

- Show the selected model's exact `TsgamEstimatorConfig` in the compact notebook.
- Make the export human-readable and easy to copy/share.
- Guarantee that the displayed config matches the config actually used for the
  fitted model by deriving both from the same helper.
- Keep the export scoped to the estimator config only, not broader notebook run
  context like station choice or train/test window.

## Non-Goals

- Do not export station metadata, selected date window, or other notebook UI
  state that is not part of `TsgamEstimatorConfig`.
- Do not add full run replay or file-save workflow in this change.
- Do not introduce a second parallel config-construction path that could drift
  from the fitted model.

## Proposed Changes

### Config Construction

- Extract the inline `TsgamEstimatorConfig(...)` creation in
  `run_tidal_model()` into a reusable helper that returns either:
  - a real `TsgamEstimatorConfig`, or
  - `None` when the notebook is using the mean-baseline fallback instead of a
    TSGAM model.
- Reuse that helper in both:
  - the model-fit path, and
  - the config-export display path.

This keeps one source of truth for estimator assembly.

### Serialization

- Add `PyYAML` as a direct dependency.
- Convert the selected `TsgamEstimatorConfig` into plain Python structures
  (dicts, lists, scalars, `None`) and serialize it with `yaml.safe_dump(...)`.
- Prefer stable, readable output:
  - preserve key order;
  - use block-style YAML;
  - avoid Python object tags or repr-style formatting.

### Notebook UI

- Add a new section near the fitted-model outputs, labeled for example
  `Estimator config`.
- Display the YAML in a code-friendly block so it is easy to copy.
- Only render the YAML when a real estimator config exists.
- When the selected model collapses to the mean-baseline case, show a short note
  such as `No TSGAM estimator config for mean-baseline model.`

## Data Included

The export should include only estimator config fields, such as:

- `multi_periodic_config`
- `exog_config`
- `solver_config`
- any explicit `None` component fields that are part of the estimator config

The export should not include:

- station selection
- train/test date selections
- Shapley settings
- derived diagnostics or fitted coefficients

## Error Handling

- If YAML serialization fails unexpectedly, show a notebook-visible message
  instead of breaking the rest of the notebook output.
- If no estimator config exists, render the baseline note rather than an empty
  or misleading YAML document.

## Testing

- Add a focused regression test for the serialization/helper path so the
  exported YAML reflects the selected estimator config structure.
- Cover both:
  - a real TSGAM config with periodic and/or exogenous components;
  - the no-components mean-baseline path that should not emit a config.

## Acceptance Criteria

- The compact notebook shows a shareable YAML document for the selected
  `TsgamEstimatorConfig`.
- The YAML is derived from the same config object used to fit the estimator.
- The mean-baseline path is handled explicitly and does not pretend to have a
  TSGAM config.
- Users can copy the displayed config without needing to reconstruct it from the
  notebook UI manually.
