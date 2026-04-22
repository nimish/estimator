# Compact Knot Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-regressor `low / med / high` spline knot presets and a regressor/basis inspection section to `examples/example_tidal_compact.py` so users can explore regressor stiffness before setting up a hyperparameter grid.

**Architecture:** Keep the existing compact notebook structure and spline-only exogenous model family. Add one small preset mapping plus regressor-knot widgets to the current parameter-collection path, and factor a pure helper that builds the visualization inputs from the same preset-derived knot counts used for model assembly. Use that helper to drive a new regressor inspection section with one raw-data chart and one spline-basis chart, so the model and visualization stay aligned.

**Tech Stack:** Python 3.12, marimo, pandas, NumPy, plotly, pytest

---

## File Map

- Modify: `examples/example_tidal_compact.py`
  - Add knot preset constants and a preset-to-knot-count helper.
  - Extend the regressor UI rows to include per-regressor knot preset widgets.
  - Extend `ModelKwargs`, `collect_model_params(...)`, and `build_exog_design_matrices(...)` to carry preset-derived knot counts into `TsgamSplineConfig(n_knots=...)`.
  - Add pure helpers for regressor/basis visualization inputs and figures.
  - Add a regressor inspection notebook section driven by the currently available model regressors.
- Modify: `test/test_example_tidal_compact.py`
  - Add focused regression tests for preset mapping, spline-config assembly, and basis-visualization inputs.

### Task 1: Knot Preset Controls And Model Wiring

**Files:**
- Modify: `examples/example_tidal_compact.py`
- Modify: `test/test_example_tidal_compact.py`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing tests**

```python
from example_tidal_compact import (  # noqa: E402
    KNOT_PRESET_TO_COUNT,
    build_knot_count,
    build_exog_design_matrices,
)


def test_build_knot_count_maps_named_presets():
    assert KNOT_PRESET_TO_COUNT == {"low": 4, "med": 8, "high": 12}
    assert build_knot_count("low") == 4
    assert build_knot_count("med") == 8
    assert build_knot_count("high") == 12


def test_build_exog_design_matrices_uses_selected_knot_preset():
    index = pd.date_range("2024-01-01", periods=12, freq="1h")
    df_train = pd.DataFrame(
        {
            "water_level": np.linspace(0.0, 1.0, len(index)),
            "pressure": np.linspace(1010.0, 1012.0, len(index)),
        },
        index=index,
    )
    df_test = df_train.iloc[-4:].copy()
    ok_train = pd.Series(True, index=df_train.index)

    _, _, _, active_regs, exog_config = build_exog_design_matrices(
        df_train,
        df_test,
        ok_train,
        ["pressure"],
        {"pressure": (-1, 0)},
        {"pressure": "high"},
        sph=1,
    )

    assert active_regs == ["pressure"]
    assert exog_config is not None
    assert len(exog_config) == 1
    assert isinstance(exog_config[0], tidal_compact.TsgamSplineConfig)
    assert exog_config[0].n_knots == 12
    assert exog_config[0].lags == [-1, 0]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest test/test_example_tidal_compact.py -k "knot_count or selected_knot_preset" -v`

Expected: FAIL with missing imports such as `ImportError: cannot import name 'build_knot_count'` or a `TypeError` because `build_exog_design_matrices(...)` does not yet accept regressor knot presets.

- [ ] **Step 3: Implement the preset mapping and model path**

```python
KNOT_PRESET_TO_COUNT = {"low": 4, "med": 8, "high": 12}
KNOT_PRESET_OPTIONS = {"Low": "low", "Med": "med", "High": "high"}


@app.function
def build_knot_count(preset: str) -> int:
    try:
        return KNOT_PRESET_TO_COUNT[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown knot preset: {preset}") from exc
```

```python
class ModelKwargs(TypedDict):
    df: pd.DataFrame
    sph: int
    harmonic_orders: dict[str, int]
    lag_ranges: dict[str, tuple[int, int]]
    knot_presets: dict[str, str]
    train_start: str
    train_end: str
    test_end: str
```

```python
@app.function
def collect_model_params(
    harmonic_inputs,
    regressor_lags,
    regressor_toggles,
    regressor_knots,
    *,
    df: pd.DataFrame,
    sph: int,
    train_start,
    train_end,
    test_end,
) -> tuple[dict[str, bool], ModelKwargs]:
    mask = {name: int(widget.value) > 0 for name, widget in harmonic_inputs.items()}
    mask.update({name: widget.value for name, widget in regressor_toggles.items()})
    return mask, {
        "df": df,
        "sph": sph,
        "harmonic_orders": {name: int(widget.value) for name, widget in harmonic_inputs.items()},
        "lag_ranges": {name: widget.value for name, widget in regressor_lags.items()},
        "knot_presets": {name: widget.value for name, widget in regressor_knots.items()},
        "train_start": str(train_start.value),
        "train_end": str(train_end.value),
        "test_end": str(test_end.value),
    }
```

```python
@app.function
def build_exog_design_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ok_train: pd.Series,
    reg_names: list[str],
    lag_ranges: dict[str, tuple[int, int]],
    knot_presets: dict[str, str],
    sph: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[TsgamSplineConfig | TsgamLinearConfig] | None]:
    x_train_fit = pd.DataFrame(index=df_train.index[ok_train])
    x_train_pred = pd.DataFrame(index=df_train.index)
    x_test_pred = pd.DataFrame(index=df_test.index)
    if not reg_names:
        return x_train_fit, x_train_pred, x_test_pred, [], None

    x_train_raw, x_test_raw, active_regs, _ = prepare_split_regressors(df_train, df_test, reg_names)
    if not active_regs:
        return x_train_fit, x_train_pred, x_test_pred, [], None

    exog_config = []
    for column in active_regs:
        lag_start, lag_end = lag_ranges.get(column, (-2, 0))
        knot_count = build_knot_count(knot_presets.get(column, "med"))
        exog_config.append(
            TsgamSplineConfig(
                n_knots=knot_count,
                lags=[hour * sph for hour in range(lag_start, lag_end + 1)],
                reg_weight=1e-5,
                diff_reg_weight=0.3,
            )
        )

    return (
        x_train_raw.loc[ok_train],
        x_train_raw,
        x_test_raw,
        active_regs,
        exog_config,
    )
```

Update the existing caller in `run_tidal_model(...)` to pass `knot_presets`.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run pytest test/test_example_tidal_compact.py -k "knot_count or selected_knot_preset" -v`

Expected: PASS for the new preset-mapping and exog-config assembly regressions.

- [ ] **Step 5: Commit the preset/model wiring**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add knot presets to compact regressors"
```

### Task 2: Regressor Inspection And Basis Visualization

**Files:**
- Modify: `examples/example_tidal_compact.py`
- Modify: `test/test_example_tidal_compact.py`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing visualization tests**

```python
from example_tidal_compact import (  # noqa: E402
    build_regressor_basis_inputs,
    build_regressor_basis_figure,
)


def test_build_regressor_basis_inputs_uses_preset_knot_count():
    df = pd.DataFrame(
        {"pressure": [1008.0, 1010.0, 1012.0, 1014.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
    )

    inputs = build_regressor_basis_inputs(df, "pressure", "low")

    assert inputs["regressor_name"] == "pressure"
    assert inputs["knot_preset"] == "low"
    assert inputs["knot_count"] == 4
    assert len(inputs["knot_locations"]) == 4
    assert inputs["x_grid"][0] == pytest.approx(1008.0)
    assert inputs["x_grid"][-1] == pytest.approx(1014.0)


def test_build_regressor_basis_figure_marks_knots():
    df = pd.DataFrame(
        {"pressure": [1008.0, 1010.0, 1012.0, 1014.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
    )

    inputs = build_regressor_basis_inputs(df, "pressure", "med")
    figure = build_regressor_basis_figure(inputs)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) > 0
    assert any(abs(shape.x0 - inputs["knot_locations"][0]) < 1.0e-9 for shape in figure.layout.shapes)
```

- [ ] **Step 2: Run the targeted visualization tests to verify they fail**

Run: `uv run pytest test/test_example_tidal_compact.py -k "basis_inputs or marks_knots" -v`

Expected: FAIL with `ImportError` because the new basis-visualization helpers do not exist yet.

- [ ] **Step 3: Implement the UI controls and basis visualization**

```python
class RegressorBasisInputs(TypedDict):
    regressor_name: str
    knot_preset: str
    knot_count: int
    observed_values: np.ndarray
    observed_index: pd.DatetimeIndex
    knot_locations: np.ndarray
    x_grid: np.ndarray
    basis_values: np.ndarray
```

```python
@app.function
def build_regressor_basis_inputs(
    df: pd.DataFrame,
    regressor_name: str,
    knot_preset: str,
) -> RegressorBasisInputs:
    series = df[regressor_name].dropna()
    knot_count = build_knot_count(knot_preset)
    knot_locations = np.linspace(series.min(), series.max(), knot_count)
    x_grid = np.linspace(series.min(), series.max(), 200)
    basis_values = TsgamEstimator._make_H(x_grid, knot_locations, include_offset=False)
    return {
        "regressor_name": regressor_name,
        "knot_preset": knot_preset,
        "knot_count": knot_count,
        "observed_values": series.to_numpy(dtype=float),
        "observed_index": pd.DatetimeIndex(series.index),
        "knot_locations": knot_locations,
        "x_grid": x_grid,
        "basis_values": basis_values,
    }
```

```python
@app.function
def build_regressor_basis_figure(inputs: RegressorBasisInputs) -> go.Figure:
    fig = go.Figure()
    basis_values = inputs["basis_values"]
    for idx in range(basis_values.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=inputs["x_grid"],
                y=basis_values[:, idx],
                mode="lines",
                name=f"B{idx + 1}",
                line=dict(width=1.2),
            )
        )
    for knot in inputs["knot_locations"]:
        fig.add_vline(x=float(knot), line_width=0.8, line_color="gray", line_dash="dot")
    fig.update_layout(
        title=f"{inputs['regressor_name']} spline basis ({inputs['knot_preset']} / {inputs['knot_count']} knots)",
        xaxis_title=COLUMN_LABELS.get(inputs["regressor_name"], inputs["regressor_name"]),
        yaxis_title="Basis value",
        height=360,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig
```

```python
@app.function
def build_regressor_series_figure(df: pd.DataFrame, regressor_name: str) -> go.Figure:
    series = df[regressor_name]
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=series.index,
                y=series,
                mode="lines",
                name=regressor_name,
                line=dict(width=0.9),
            )
        ]
    )
    fig.update_layout(
        title=f"{COLUMN_LABELS.get(regressor_name, regressor_name)} over loaded window",
        xaxis_title="Time",
        yaxis_title=COLUMN_LABELS.get(regressor_name, regressor_name),
        height=280,
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig
```

```python
@app.cell
def _(station_data):
    _df = station_data["df"]
    _train_start_default, _train_end_default, _test_end_default = build_model_date_defaults(
        station_data["date_min"],
        station_data["date_max"],
    )
    harmonic_inputs = {
        name: mo.ui.slider(
            start=0,
            stop=8,
            value=order,
            label=f"{name} ({PERIODS.get(name, 8766.0):.1f} h)",
            show_value=True,
            full_width=True,
        )
        for name, order in DEFAULT_HARMONICS.items()
    }

    _available_regressors = available_columns(_df, MODEL_REGRESSOR_CANDIDATES)
    regressor_toggles = {}
    regressor_lags = {}
    regressor_knots = {}
    _regressor_rows = []
    for name in _available_regressors:
        regressor_toggles[name] = mo.ui.switch(value=False, label=name)
        _lag_start, _lag_end = LAG_DEFAULTS.get(name, (-2, 0))
        regressor_lags[name] = mo.ui.range_slider(
            start=-6,
            stop=6,
            value=(_lag_start, _lag_end),
            label="lag (h)",
            show_value=True,
        )
        regressor_knots[name] = mo.ui.dropdown(
            options=KNOT_PRESET_OPTIONS,
            value="med",
            label="knots",
        )
        _regressor_rows.append(
            mo.hstack([regressor_toggles[name], regressor_lags[name], regressor_knots[name]], justify="start")
        )

    train_start = mo.ui.date(value=_train_start_default, label="Train start")
    train_end = mo.ui.date(value=_train_end_default, label="Train end")
    test_end = mo.ui.date(value=_test_end_default, label="Test end")
    run_fit = mo.ui.run_button(label="Fit model")

    mo.vstack(
        [
            mo.md("## Configure model"),
            mo.hstack(
                [
                    mo.vstack([mo.md("**Harmonics** (0 = exclude)")] + list(harmonic_inputs.values())),
                    mo.vstack([mo.md("**Regressors** (toggle + lag range in hours + knots)")] + _regressor_rows)
                    if _available_regressors
                    else mo.md(""),
                    mo.vstack([mo.md("**Date ranges**"), train_start, train_end, test_end]),
                ],
                justify="start",
                gap=2,
            ),
            run_fit,
        ]
    )
    return (
        harmonic_inputs,
        regressor_knots,
        regressor_lags,
        regressor_toggles,
        run_fit,
        test_end,
        train_end,
        train_start,
    )
```

Update the `Fit model` and `Run Shapley` cells to pass `regressor_knots` into `collect_model_params(...)`.

```python
@app.cell
def _(station_data):
    _available_regressors = available_columns(station_data["df"], MODEL_REGRESSOR_CANDIDATES)
    mo.stop(len(_available_regressors) == 0, mo.md("*No model regressors are available in the loaded dataset.*"))
    inspect_regressor = mo.ui.dropdown(
        options={COLUMN_LABELS.get(name, name): name for name in _available_regressors},
        value=_available_regressors[0],
        label="Inspect regressor",
    )
    inspect_regressor
    return (inspect_regressor,)
```

```python
@app.cell
def _(inspect_regressor, regressor_knots, station_data):
    _regressor_name = inspect_regressor.value
    _knot_preset = regressor_knots[_regressor_name].value
    _basis_inputs = build_regressor_basis_inputs(
        station_data["df"],
        _regressor_name,
        _knot_preset,
    )
    mo.vstack(
        [
            mo.md("## Regressor inspection"),
            build_regressor_series_figure(station_data["df"], _regressor_name),
            build_regressor_basis_figure(_basis_inputs),
        ]
    )
    return
```

- [ ] **Step 4: Run the full notebook-adjacent test file**

Run: `uv run pytest test/test_example_tidal_compact.py -v`

Expected: PASS for the existing notebook tests plus the new knot-preset and basis-visualization regressions.

- [ ] **Step 5: Smoke-test the notebook manually**

Run: `uv run marimo edit examples/example_tidal_compact.py`

Expected:
- Each available regressor row now includes a `Knots` preset with default `med`.
- Changing a regressor from `med` to `low` or `high` changes the resulting spline config used by the fit path.
- The new `Regressor inspection` section appears when model regressors are available.
- The inspection section lets you choose a regressor and shows both the raw series and the spline basis with knot markers for the currently selected preset.

- [ ] **Step 6: Commit the visualization work**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add compact regressor knot presets"
```
