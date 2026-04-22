# Battery Grid Rich Progress Design

## Summary

`examples/run_tidal_battery_grid.py` currently reports progress through plain
line-oriented logging plus a log file and partial CSV rewrites. That works, but
it is hard to scan during long runs and does not make the current stage, best
candidate, or failure counts obvious at a glance. The goal of this change is to
upgrade the script to a richer terminal experience using the existing `rich`
dependency while keeping the script a normal CLI, not a full-screen app.

## Goals

- Add a richer interactive terminal UI for the Battery grid-search script.
- Show stage progress, elapsed time, counts of completed/failed runs, and the
  current best validation candidate live during execution.
- Make `Ctrl-C` stop the run promptly without requiring manual process killing.
- Keep the existing persistent artifacts:
  - log file;
  - partial CSV;
  - final CSV;
  - final markdown summary.
- Preserve the current parallel execution model and support at least `6`
  workers cleanly.
- Avoid adding a new dependency when the existing `rich` package is sufficient.

## Non-Goals

- Do not convert the script into a full-screen `textual` application.
- Do not add terminal input controls, menus, pausing, filtering, or keyboard
  navigation.
- Do not change the search grid, model defaults, train/validation/test split,
  or promotion logic in this UI-focused change.
- Do not remove the plain-text logging path for non-interactive environments.

## Proposed Changes

### UI Mode Selection

- Keep the script runnable as an ordinary CLI.
- Detect whether live terminal rendering is appropriate using the active Rich
  console / terminal capabilities.
- When the output stream is interactive, enable Rich live rendering.
- When the output stream is non-interactive, piped, or otherwise unsuitable,
  fall back to the current plain logging behavior.

### Live Terminal Layout

- Build the interactive display from Rich primitives already available in the
  repo:
  - `Progress` for the active stage bar and elapsed time;
  - `Live` for periodic screen refresh;
  - `Panel` for summary/status blocks;
  - `Table` for recent completed candidates.
- Show the following sections in the live view:
  - run metadata:
    - station;
    - split summary;
    - worker count;
    - output paths;
  - current stage progress:
    - validation or promoted outer test;
    - completed / total count;
    - elapsed time;
  - stage counters:
    - `ok`;
    - `non-finite`;
    - `exception`;
  - best validation-so-far panel:
    - candidate label;
    - validation MAPE;
    - validation RMSE;
    - validation R^2;
  - recent results table:
    - stage;
    - candidate label;
    - status;
    - key metric values.

### Data Flow And Logging

- Keep file logging independent from the terminal presentation:
  - log file writes should continue for every status update;
  - partial CSV rewrites should continue as results arrive.
- In plain mode:
  - preserve the current line-by-line terminal logging behavior.
- In live mode:
  - update an in-memory render state after each completed fit;
  - refresh the Rich display instead of printing every result as a new terminal
    line;
  - continue appending detailed progress messages to the log file so the run is
    still inspectable after the fact.
- Keep the final summary output after the live view closes so the finished run
  still leaves a readable terminal conclusion.

### Parallel Execution

- Replace the current thread-based execution model with an interruptible worker
  backend that allows the parent process to stop queued and in-flight work more
  reliably on `Ctrl-C`.
- Treat interruptibility as more important than preserving the exact current
  thread-pool implementation.
- Ensure live UI updates happen only on the main thread as futures complete.
- Do not attempt per-worker nested progress bars or per-candidate spinner trees;
  the UI should summarize concurrent work, not try to render each worker
  independently.
- Continue defaulting the script to `--n-jobs 6`.

### Minimal Internal Structure

- Add a small render-state layer rather than a heavy UI framework.
- Prefer a few focused helpers such as:
  - one function to determine whether live mode is enabled;
  - one function to derive status counts from finished rows;
  - one function to build the recent-results table;
  - one function to build the full Rich layout from current state.
- Keep the existing search/evaluation functions intact as much as possible.
- Treat the live UI as a presentation layer around the existing run loop, not a
  rewrite of the script's control flow.

## Error Handling

- If live rendering is unavailable or fails to initialize, fall back to plain
  logging rather than aborting the run.
- On `KeyboardInterrupt`, stop accepting new work, persist the current partial
  CSV/log state, tear down worker execution promptly, and exit with an interrupt
  status instead of hanging.
- Continue counting and surfacing `non-finite` and `exception` outcomes in both
  live and plain modes.
- Keep the persistent log file and partial CSV updates even when live mode is
  active so interrupted runs remain diagnosable.
- Do not suppress underlying fit errors; surface them through the existing row
  status/error fields and include recent failures in the live recent-results
  view.

## Testing

- Add helper-level tests for the new presentation-state functions only.
- Cover durable, non-brittle seams such as:
  - deciding whether live mode is enabled;
  - computing status counters from completed rows;
  - selecting and trimming recent results for display;
  - extracting the best validation row for the live best-so-far panel.
- Do not add ANSI snapshot tests or brittle assertions against exact terminal
  formatting.
- Keep the current script behavior tests passing:
  - candidate-grid tests;
  - promotion tests;
  - evaluation-seam tests.

## Acceptance Criteria

- Running `examples/run_tidal_battery_grid.py` in an interactive terminal shows
  a live Rich-based progress view rather than only plain line-by-line logging.
- The live view makes the current stage, total progress, counts of completed
  outcomes, and best validation candidate visible at a glance.
- The script still writes its log file and partial CSV continuously during the
  run.
- The script still falls back cleanly to plain logging in non-interactive
  contexts.
- Pressing `Ctrl-C` during a live or plain run stops the script without needing
  Activity Monitor or another external kill path.
- No new dependency beyond the existing `rich` package is required for this
  upgrade.
