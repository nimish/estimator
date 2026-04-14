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
| `run_pv.py` | PV/solar: trend-type ablation (none, linear, nonlinear_decreasing, nonlinear_increasing). Default data: `examples/data/pv/2107_data_combined.csv`. |
| `run_outlier_detector.py` | Synthetic data: with vs without outlier detector. No data file. |

**Common options (where applicable):** `--data-dir`, `--output-dir`, `--train-start`, `--train-end`, `--test-start`, `--test-end`. Reports are written to `--output-dir` (default: `examples/reports`).

**Run from project root:**

```bash
uv run python examples/run_outlier_detector.py
uv run python examples/run_air_quality.py --no-download   # use existing data
uv run python examples/run_la_energy.py
uv run python examples/run_pv.py
```

For air quality, use `--n-jobs 1` if you hit process/semaphore limits (e.g. in some CI or restricted environments).
