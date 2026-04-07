# Example run scripts (ablation + reports)

CLI scripts that run each example with ablation studies and write reports (Markdown, CSV, optional PNG). All use **Click** and **Rich** for a consistent, colored terminal UI.

**Install example dependencies:**

```bash
uv sync --group examples
```

**Scripts:**

| Script | Description |
|--------|-------------|
| `run_air_quality.py` | Beijing PM2.5: exogenous variable ablation. Default data: `examples/data/air_quality` (downloads if missing). |
| `run_la_energy.py` | LA energy demand: harmonics → +exog → +AR → +outlier ablation. Default data: `examples/data/energy`. |
| `run_pv.py` | PV/solar: trend-type ablation (none, linear, nonlinear). Default data: `examples/data/pv/2107_data_combined.csv`. |
| `run_outlier_detector.py` | Synthetic data: with vs without outlier detector. No data file. |
| `run_tidal.py` | NOAA tidal water level: met-regressor ablation (pressure, water temp, wind, air temp). Combines CO-OPS tide station data with NCEI LCD weather station data. Default: 8518750 (The Battery, NY) + Central Park weather. |

**Marimo explorer:**

- `example_tidal_marimo.py`: compact tidal explorer with reactive station/weather summaries, manual weather override, advanced constituent-harmonic controls, and on-demand TSGAM fitting.

**Launch from project root:**

```bash
uv sync --group notebooks --group viz
uv run marimo edit examples/example_tidal_marimo.py
```

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
