# TSGAM examples

## Worked notebooks

1. [Los Angeles load estimation](example_la_energy_marimo.py) fits hourly load from weather and calendar patterns. Inspect the components and see how well the model reconstructs the last week of each month when those load observations are withheld.
2. [A simple direct multi-horizon forecast](example_simple_dmh_forecast_marimo.py) shows how a single weather observation at the forecast origin feeds a nowcast and three hourly forecasts.
3. [Forecast-origin explorer](example_forecast_origin_explorer_marimo.py) lets you select an origin and inspect which lagged observations were available and which future values the independent and coupled models predict.

For an application outside electricity demand, the [tidal analysis](example_tidal_compact.py) uses known tidal periods and local weather to model water levels. The [tidal explorer](example_tidal_marimo.py) lets you change stations, weather inputs, and tidal constituents.

Launch a notebook from the repository root:

```bash
uv run --group notebooks --group viz marimo edit examples/example_la_energy_marimo.py
```

## Run scripts and reports

CLI scripts that run each example with ablation studies and write reports (Markdown, CSV, optional PNG). All use **Click** and **Rich** for a consistent, colored terminal UI.

**Install example dependencies:**

```bash
uv sync --group examples
```

**Scripts:**

| Script | Description |
|--------|-------------|
| `run_air_quality.py` | Beijing PM2.5: exogenous variable ablation. Default data: `examples/data/air_quality` (downloads if missing). |
| `run_la_energy.py` | LA load estimation: harmonics → +exog → +AR → +outlier ablation. Default data: `examples/data/energy`. |
| `run_pv.py` | Compare trend models for measured PV power. The example uses [Solar Data Tools](https://solar-data-tools.readthedocs.io/) to prepare the power series. Default data: `examples/data/pv/2107_data_combined.csv`. |
| `run_outlier_detector.py` | Synthetic data: with vs without outlier detector. No data file. |
| `run_tidal.py` | NOAA tidal water level: met-regressor ablation (pressure, water temp, wind, air temp). Combines CO-OPS tide station data with NCEI LCD weather station data. Default: 8518750 (The Battery, NY) + Central Park weather. |

**Common options (where applicable):** `--data-dir`, `--output-dir`, `--train-start`, `--train-end`, `--test-start`, `--test-end`. Reports are written to `--output-dir` (default: `examples/reports`).

**Run from project root:**

```bash
uv run python examples/run_outlier_detector.py
uv run python examples/run_air_quality.py --no-download   # use existing data
uv run python examples/run_la_energy.py
uv run python examples/run_pv.py
uv run python examples/run_tidal.py                        # downloads NOAA data
uv run python examples/run_tidal.py --station 9414290      # San Francisco
```

For air quality, use `--n-jobs 1` if you hit process/semaphore limits (e.g. in some CI or restricted environments).
