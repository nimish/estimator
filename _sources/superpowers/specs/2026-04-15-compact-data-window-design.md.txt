# Compact Notebook Configurable Data Window Design

## Summary

`examples/example_tidal_compact.py` currently loads one station dataset up
front, then all later datepickers only slice that in-memory frame. This makes
the modeling `test_end` picker look like it should extend the dataset when it
actually cannot fetch more rows. The goal is to make the data load window an
explicit top-level notebook action: users choose a station and data date range,
click `Load data`, and all later exploration/model/Shapley date controls derive
from the loaded dataset.

## Goals

- Add explicit top-level `Data start`, `Data end`, and `Load data` controls.
- Keep the initial data window as a fixed example window, but make it editable
  before loading.
- Reuse existing cached tidal and LCD data when possible, and download the
  requested range only when cache coverage is missing and download is enabled.
- Ensure downstream datepickers operate only within the currently loaded
  dataset, so extending the dataset requires a deliberate top-level reload.
- Make the loaded date span visible so users can tell what data was actually
  loaded.

## Non-Goals

- Do not make downstream model/exploration datepickers auto-download more data.
- Do not add background or implicit reloads when station/date controls change.
- Do not introduce a new cache format or a second storage location.
- Do not redesign the model-fitting UI beyond updating it to reflect the loaded
  dataset window.

## Proposed Changes

### Top-Level Load Controls

- Move data loading into an explicit top-level control group containing:
  - station selector;
  - `Data start` datepicker;
  - `Data end` datepicker;
  - tidal/weather download toggles;
  - weather merge toggle when a mapped LCD station exists;
  - `Load data` run button.
- Seed `Data start` / `Data end` from the current compact-notebook example
  window: `2022-01-01` through `2024-03-31`.
- Do not automatically reload data when station or date controls change. The
  selected dataset updates only after the user clicks `Load data`.
- Validate `Data start <= Data end` before attempting any load or download.

### Data Loading Flow

- Change `load_station_frame(...)` to accept the requested top-level begin/end
  dates in addition to station and weather/download flags.
- For tidal data:
  - first try `resolve_tidal_cache_path(...)` for the requested station/range;
  - if a covering cache file exists, load it and trim the DataFrame to the
    requested date window;
  - if no covering cache exists and `download_tidal` is enabled, download the
    requested range and then load it;
  - if no covering cache exists and `download_tidal` is disabled, fail with a
    notebook-visible message telling the user that the requested range is not
    cached.
- For LCD weather:
  - derive `begin_year` / `end_year` from the requested data window;
  - download missing LCD year files only when `download_weather` is enabled;
  - call `load_lcd_weather(..., begin_date=..., end_date=...)` so weather data
    is trimmed to the same requested window;
  - continue skipping weather controls when the selected tidal station has no
    mapped LCD station.
- Keep the merged dataset on the tidal index as it does today.

### Loaded Window Semantics

- The requested top-level data window defines the dataset exposed to the rest of
  the notebook.
- Even when the notebook reuses a larger covering tidal cache file, trim the
  loaded tidal frame to the requested window before returning `station_data`.
- `station_data["date_min"]` and `station_data["date_max"]` should reflect the
  actual loaded frame after trimming and weather merge, not the full cache file
  span.
- Later datepickers should therefore mean "select a window within the loaded
  dataset", not "fetch more data if needed."

### Downstream Picker Behavior

- Recreate the exploration and model datepickers from the loaded frame's actual
  `date_min` / `date_max`.
- Set `train_start` to the loaded `date_min` and `test_end` to the loaded
  `date_max`.
- Keep the current example-oriented `train_end = 2024-01-01` default when that
  date falls inside the loaded range; otherwise default `train_end` to the
  midpoint date of the loaded range so the model controls still start from a
  sensible split on shorter or shifted windows.
- When a newly loaded dataset changes the available date range, downstream
  pickers should reset to defaults derived from that new loaded frame.
- The model `test_end` picker should no longer be expected to extend the
  dataset; users must return to the top-level data window and click `Load data`
  for that.

### Status And Error Handling

- Expand the existing `status_message` to summarize:
  - whether tidal data came from cache or from a new download;
  - the actual loaded date range;
  - whether LCD weather was merged and from which station.
- Surface invalid top-level date ranges with a notebook-visible message instead
  of silently falling back.
- If tidal/weather data for the requested window is unavailable and downloads
  are disabled, show a clear action-oriented message rather than failing
  opaquely.

## Testing

- Add focused regression coverage for the new data-window logic without relying
  on live network calls.
- Cover at least:
  - loading a requested window from an existing covering tidal cache and
    trimming it to the requested range;
  - requesting a range outside cache coverage with downloads disabled and
    surfacing a clear failure;
  - deriving downstream picker defaults from the loaded frame bounds.

## Acceptance Criteria

- Users can change the notebook's top-level data window and click `Load data` to
  fetch or reuse a dataset for that exact window.
- Reusing a larger covering cache file does not expose extra unrequested history
  in later notebook controls.
- Extending the loaded dataset updates later exploration/model datepickers
  because they derive from the newly loaded frame.
- Changing downstream `test_end` or exploration window controls only slices the
  loaded dataset and does not implicitly download more data.
- The notebook clearly reports what date span was loaded and whether cache or
  download paths were used.
