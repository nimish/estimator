# Compact Estimator Config Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the compact tidal notebook's currently selected `TsgamEstimatorConfig` as shareable YAML, using the same config assembly path that powers model fitting.

**Architecture:** Refactor `examples/example_tidal_compact.py` so one helper assembles the estimator config, selected harmonic metadata, and exogenous design matrices. Serialize that exact config into plain Python data and YAML once, store it on `FitResult`, and render it with a small markdown helper that also handles the all-components-off mean-baseline case.

**Tech Stack:** Python 3.12, marimo, PyYAML, NumPy, pandas, pytest

---

## File Map

- Modify: `examples/example_tidal_compact.py`
  - Add a shared `build_estimator_config(...)` helper used by `run_tidal_model(...)`
  - Add YAML serialization helpers and a small markdown-rendering helper
  - Extend `FitResult` with `estimator_config_yaml`
  - Render the YAML block or a mean-baseline note in the notebook UI
- Modify: `test/test_example_tidal_compact.py`
  - Add focused regression coverage for YAML serialization and baseline rendering
  - Add a lightweight regression for the mean-baseline `run_tidal_model(...)` path
- Modify: `pyproject.toml`
  - Add `PyYAML` as a runtime dependency because the notebook module imports it directly
- Modify: `uv.lock`
  - Capture the dependency update from `uv add pyyaml`

### Task 1: Shared Config Builder And YAML Helpers

**Files:**
- Modify: `examples/example_tidal_compact.py`
- Modify: `test/test_example_tidal_compact.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing tests**

```python
from example_tidal_compact import (  # noqa: E402
    build_estimator_config,
    build_estimator_config_markdown,
    dump_estimator_config_yaml,
)
import yaml


def test_dump_estimator_config_yaml_round_trips_selected_model():
    index = pd.date_range("2024-01-01", periods=96, freq="1h")
    df = pd.DataFrame(
        {
            "water_level": np.sin(2 * np.pi * np.arange(len(index)) / PERIODS["M2"]),
            "pressure": np.linspace(1010.0, 1012.0, len(index)),
        },
        index=index,
    )
    df_train = df.iloc[:72]
    df_test = df.iloc[72:]
    ok_train = df_train["water_level"].notna()

    picked, active_regs, config, *_ = build_estimator_config(
        {"M2": True, "pressure": True},
        df_train=df_train,
        df_test=df_test,
        ok_train=ok_train,
        harmonic_orders={"M2": 2, "S2": 0},
        lag_ranges={"pressure": (-1, 0)},
        sph=1,
    )

    assert picked == {"M2": (PERIODS["M2"], 2)}
    assert active_regs == ["pressure"]
    yaml_data = yaml.safe_load(dump_estimator_config_yaml(config))
    assert yaml_data["multi_periodic_config"]["periods"] == [PERIODS["M2"]]
    assert yaml_data["multi_periodic_config"]["num_harmonics"] == [2]
    assert yaml_data["exog_config"][0]["lags"] == [-1, 0]
    assert yaml_data["solver_config"]["solver"] == "SCS"
    assert yaml_data["solver_config"]["verbose"] is False


def test_build_estimator_config_markdown_handles_yaml_and_baseline():
    yaml_block = "solver_config:\n  solver: SCS\n"

    baseline_markdown = build_estimator_config_markdown(None)
    config_markdown = build_estimator_config_markdown(yaml_block)

    assert "Mean baseline selected" in baseline_markdown
    assert "TsgamEstimatorConfig" in baseline_markdown
    assert "```yaml" in config_markdown
    assert "solver: SCS" in config_markdown
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest test/test_example_tidal_compact.py -k "config_yaml or config_markdown" -v`

Expected: FAIL with missing imports such as `ImportError: cannot import name 'build_estimator_config'`, `ImportError: cannot import name 'dump_estimator_config_yaml'`, or `ModuleNotFoundError: No module named 'yaml'`.

- [ ] **Step 3: Add the dependency and implement the shared helpers**

Run: `uv add pyyaml`

```python
from dataclasses import asdict
import yaml

class FitResult(TypedDict):
    metrics_train: MetricDict
    metrics_test: MetricDict
    te_index: pd.DatetimeIndex
    te_obs: np.ndarray
    te_pred: np.ndarray
    te_obs_clean: np.ndarray
    te_pred_clean: np.ndarray
    residuals: np.ndarray
    picked: dict[str, tuple[float, int]]
    active_regs: list[str]
    n_train: int
    n_test: int
    sph: int
    estimator_config_yaml: str | None


@app.function
def build_estimator_config(
    component_mask: dict[str, bool],
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ok_train: pd.Series,
    harmonic_orders: dict[str, int],
    lag_ranges: dict[str, tuple[int, int]],
    sph: int,
) -> tuple[
    dict[str, tuple[float, int]],
    list[str],
    TsgamEstimatorConfig | None,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    picked, periodic_config = build_periodic_config(component_mask, harmonic_orders, sph)
    reg_names = [name for name, active in component_mask.items() if active and name not in PERIODS]
    x_train_fit, x_train_pred, x_test_pred, active_regs, exog_config = build_exog_design_matrices(
        df_train,
        df_test,
        ok_train,
        reg_names,
        lag_ranges,
        sph,
    )
    config = None
    if periodic_config is not None or exog_config is not None:
        config = TsgamEstimatorConfig(
            multi_periodic_config=periodic_config,
            exog_config=exog_config,
            solver_config=TsgamSolverConfig(solver="SCS", verbose=False),
        )
    return picked, active_regs, config, x_train_fit, x_train_pred, x_test_pred


@app.function
def estimator_config_to_plain_data(config: TsgamEstimatorConfig) -> dict[str, object]:
    def _plainify(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): _plainify(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plainify(item) for item in value]
        if hasattr(value, "value"):
            return value.value
        return value

    return cast(dict[str, object], _plainify(asdict(config)))


@app.function
def dump_estimator_config_yaml(config: TsgamEstimatorConfig | None) -> str | None:
    if config is None:
        return None
    return yaml.safe_dump(estimator_config_to_plain_data(config), sort_keys=False)


@app.function
def build_estimator_config_markdown(estimator_config_yaml: str | None) -> str:
    if estimator_config_yaml is None:
        return (
            "### Estimator config\n\n"
            "_Mean baseline selected; no `TsgamEstimatorConfig` is created when every component is off._"
        )
    return f"### Estimator config\n\n```yaml\n{estimator_config_yaml}\n```"
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run pytest test/test_example_tidal_compact.py -k "config_yaml or config_markdown" -v`

Expected: PASS for both new regression tests.

- [ ] **Step 5: Commit the helper work**

```bash
git add pyproject.toml uv.lock examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add compact notebook config export helpers"
```

### Task 2: Wire The Fit Path And Notebook Output

**Files:**
- Modify: `examples/example_tidal_compact.py`
- Modify: `test/test_example_tidal_compact.py`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing regression for the mean-baseline fit path**

```python
from example_tidal_compact import run_tidal_model  # noqa: E402


def test_run_tidal_model_baseline_has_no_estimator_yaml():
    index = pd.date_range("2024-01-01", periods=72, freq="1h")
    df = pd.DataFrame(
        {"water_level": np.linspace(-1.0, 1.0, len(index))},
        index=index,
    )

    fit_result = run_tidal_model(
        {"M2": False, "pressure": False},
        df=df,
        sph=1,
        harmonic_orders={"M2": 0},
        lag_ranges={},
        train_start="2024-01-01",
        train_end="2024-01-03",
        test_end="2024-01-04",
    )

    assert fit_result["estimator_config_yaml"] is None
```

- [ ] **Step 2: Run the new regression to verify it fails**

Run: `uv run pytest test/test_example_tidal_compact.py::test_run_tidal_model_baseline_has_no_estimator_yaml -v`

Expected: FAIL with `KeyError: 'estimator_config_yaml'` or equivalent because `FitResult` is not yet populated with the export field on every return path.

- [ ] **Step 3: Refactor the fit path and add the notebook YAML section**

```python
def pack_model_result(
    df_test: pd.DataFrame,
    y_train: np.ndarray,
    ok_train: pd.Series,
    ok_test: pd.Series,
    yhat_train: np.ndarray,
    yhat_test: np.ndarray,
    picked: dict[str, tuple[float, int]],
    active_regs: list[str],
    sph: int,
    estimator_config_yaml: str | None,
) -> FitResult:
    # ... existing metrics assembly ...
    return cast(FitResult, {
        "metrics_train": train_metrics,
        "metrics_test": test_metrics,
        "te_index": te_index,
        "te_obs": test_obs,
        "te_pred": yhat_test,
        "te_obs_clean": test_obs_clean,
        "te_pred_clean": yhat_test[ok_test_np],
        "residuals": test_obs - yhat_test,
        "picked": picked,
        "active_regs": active_regs,
        "n_train": len(y_train),
        "n_test": int(ok_test.sum()),
        "sph": sph,
        "estimator_config_yaml": estimator_config_yaml,
    })


def run_tidal_model(...):
    split_time = pd.Timestamp(train_end)
    window = df[train_start:test_end]
    df_train = window[window.index < split_time]
    df_test = window[window.index >= split_time]
    ok_train = df_train["water_level"].notna()
    ok_test = df_test["water_level"].notna()
    y_train = df_train.loc[ok_train, "water_level"].to_numpy(dtype=float)

    picked, active_regs, estimator_config, x_train_fit, x_train_pred, x_test_pred = build_estimator_config(
        component_mask,
        df_train=df_train,
        df_test=df_test,
        ok_train=ok_train,
        harmonic_orders=harmonic_orders,
        lag_ranges=lag_ranges,
        sph=sph,
    )
    estimator_config_yaml = dump_estimator_config_yaml(estimator_config)

    if estimator_config is None:
        baseline = float(np.nanmean(y_train))
        return pack_model_result(
            df_test,
            y_train,
            ok_train,
            ok_test,
            np.full(len(df_train), baseline),
            np.full(len(df_test), baseline),
            picked,
            active_regs,
            sph,
            estimator_config_yaml,
        )

    model = TsgamEstimator(estimator_config)
    model.fit(x_train_fit, y_train)
    return pack_model_result(
        df_test,
        y_train,
        ok_train,
        ok_test,
        model.predict(x_train_pred),
        model.predict(x_test_pred),
        picked,
        active_regs,
        sph,
        estimator_config_yaml,
    )
```

```python
@app.cell
def _(fit_result):
    mo.md(build_estimator_config_markdown(fit_result["estimator_config_yaml"]))
    return
```

Place that cell immediately after the fit metrics/summary block so the shareable YAML appears near the selected model diagnostics rather than far down in the notebook.

- [ ] **Step 4: Run the focused notebook tests**

Run: `uv run pytest test/test_example_tidal_compact.py -v`

Expected: PASS for the existing periodogram/Shapley tests plus the new config-export regressions.

- [ ] **Step 5: Smoke-test the notebook output manually**

Run: `uv run marimo edit examples/example_tidal_compact.py`

Expected:
- Fit a model with at least one tidal constituent enabled and confirm a new `Estimator config` section appears with YAML output.
- Toggle every tidal constituent and regressor off, rerun, and confirm the section switches to the mean-baseline note instead of a YAML block.
- Verify the YAML reflects the selected harmonics, lag ranges, and SCS solver settings.

- [ ] **Step 6: Commit the notebook integration**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: show estimator config yaml in compact notebook"
```
