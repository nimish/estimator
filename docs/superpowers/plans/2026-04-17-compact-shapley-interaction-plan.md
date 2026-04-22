# Compact Shapley Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the compact tidal notebook's Shapley analysis so selected exogenous interaction pairs appear as first-class Shapley components and are evaluated only through canonical coalitions that include both parent regressors.

**Architecture:** Keep `examples/example_tidal_compact.py` as the only production code surface, but pull the Shapley interaction logic out of the marimo cell into small typed helpers. First add pure helpers for component construction, coalition canonicalization, and model-input translation; then add one helper that evaluates only unique canonical coalitions and expands those metrics back to the raw power set for the existing `compute_shapley(...)` math.

**Tech Stack:** Python 3.13, marimo, NumPy, pandas, plotly, pytest

---

## File Map

- Modify: `examples/example_tidal_compact.py`
  - Add typed helpers for Shapley component construction, coalition canonicalization, and canonical-run evaluation
  - Refactor the Shapley notebook cell to use the new helpers and count canonical coalitions in the progress/title path
- Modify: `test/test_example_tidal_compact.py`
  - Add focused regressions for helper behavior, canonical-coalition deduping, and interaction-aware Shapley result assembly

### Task 1: Extract Shapley Interaction Helpers

**Files:**
- Modify: `examples/example_tidal_compact.py`
- Modify: `test/test_example_tidal_compact.py`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing helper tests**

```python
from example_tidal_compact import (  # noqa: E402
    build_shapley_coalition_plan,
    build_shapley_component_list,
    build_shapley_model_inputs,
)


def test_build_shapley_component_list_appends_surviving_interactions():
    component_mask = {"M2": True, "S2": False, "pressure": True, "wind_u": True, "air_temp": True}

    components, interaction_lookup = build_shapley_component_list(
        component_mask,
        ["pressure", "wind_u"],
        [("pressure", "wind_u"), ("pressure", "air_temp")],
    )

    assert components == ["M2", "pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    assert interaction_lookup == {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}


def test_build_shapley_coalition_plan_drops_invalid_interaction_bits():
    components = ["pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    interaction_lookup = {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}

    raw_to_canonical = build_shapley_coalition_plan(components, interaction_lookup)

    assert raw_to_canonical[0b100] == 0b000
    assert raw_to_canonical[0b101] == 0b001
    assert raw_to_canonical[0b110] == 0b010
    assert raw_to_canonical[0b111] == 0b111
    assert len(set(raw_to_canonical.values())) == 5


def test_build_shapley_model_inputs_activates_only_selected_interactions():
    components = ["M2", "pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    interaction_lookup = {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}

    component_mask, interaction_pairs = build_shapley_model_inputs(
        0b1111,
        components,
        {"M2": True, "S2": False, "pressure": True, "wind_u": True, "air_temp": False},
        interaction_lookup,
    )

    assert component_mask == {"M2": True, "S2": False, "pressure": True, "wind_u": True, "air_temp": False}
    assert interaction_pairs == [("pressure", "wind_u")]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest test/test_example_tidal_compact.py -k "shapley_component_list or shapley_coalition_plan or shapley_model_inputs" -v`

Expected: FAIL with missing imports such as `ImportError: cannot import name 'build_shapley_component_list'`, `ImportError: cannot import name 'build_shapley_coalition_plan'`, or `ImportError: cannot import name 'build_shapley_model_inputs'`.

- [ ] **Step 3: Implement the pure Shapley helpers**

```python
@app.function
def build_shapley_component_list(
    component_mask: dict[str, bool],
    active_regs: list[str],
    interaction_pairs: list[tuple[str, str]],
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    harmonic_components = [name for name, active in component_mask.items() if active and name in PERIODS]
    regressor_components = [name for name in active_regs if component_mask.get(name, False)]
    active_reg_set = set(regressor_components)
    interaction_lookup: dict[str, tuple[str, str]] = {}
    for left_name, right_name in interaction_pairs:
        if left_name not in active_reg_set or right_name not in active_reg_set:
            continue
        interaction_lookup[format_interaction_pair_label(left_name, right_name)] = (left_name, right_name)
    return harmonic_components + regressor_components + list(interaction_lookup), interaction_lookup


@app.function
def canonicalize_shapley_coalition(
    raw_bits: int,
    components: list[str],
    interaction_lookup: dict[str, tuple[str, str]],
) -> int:
    active_components = {
        component
        for idx, component in enumerate(components)
        if raw_bits & (1 << idx)
    }
    canonical_components = set(active_components)
    for label, (left_name, right_name) in interaction_lookup.items():
        if label in canonical_components and (left_name not in canonical_components or right_name not in canonical_components):
            canonical_components.remove(label)
    canonical_bits = 0
    for idx, component in enumerate(components):
        if component in canonical_components:
            canonical_bits |= 1 << idx
    return canonical_bits


@app.function
def build_shapley_coalition_plan(
    components: list[str],
    interaction_lookup: dict[str, tuple[str, str]],
) -> dict[int, int]:
    return {
        raw_bits: canonicalize_shapley_coalition(raw_bits, components, interaction_lookup)
        for raw_bits in range(2 ** len(components))
    }


@app.function
def build_shapley_model_inputs(
    canonical_bits: int,
    components: list[str],
    base_component_mask: dict[str, bool],
    interaction_lookup: dict[str, tuple[str, str]],
) -> tuple[dict[str, bool], list[tuple[str, str]]]:
    active_components = {
        component
        for idx, component in enumerate(components)
        if canonical_bits & (1 << idx)
    }
    coalition_mask = {name: False for name in base_component_mask}
    for component in active_components:
        if component in coalition_mask:
            coalition_mask[component] = True
    coalition_interactions = [
        interaction_lookup[component]
        for component in components
        if component in interaction_lookup and component in active_components
    ]
    return coalition_mask, coalition_interactions
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run pytest test/test_example_tidal_compact.py -k "shapley_component_list or shapley_coalition_plan or shapley_model_inputs" -v`

Expected: PASS for all three new helper regressions.

- [ ] **Step 5: Commit the helper extraction**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add compact shapley interaction helpers"
```

### Task 2: Canonical Shapley Evaluation And Notebook Wiring

**Files:**
- Modify: `examples/example_tidal_compact.py`
- Modify: `test/test_example_tidal_compact.py`
- Test: `test/test_example_tidal_compact.py`

- [ ] **Step 1: Write the failing Shapley-result regression**

```python
from example_tidal_compact import build_shapley_result  # noqa: E402


def test_build_shapley_result_deduplicates_invalid_interaction_runs(monkeypatch):
    calls: list[tuple[dict[str, bool], tuple[tuple[str, str], ...]]] = []
    progress_ticks: list[str] = []

    def fake_run_tidal_model(component_mask, **model_kwargs):
        interaction_pairs = tuple(model_kwargs["interaction_pairs"])
        calls.append((component_mask.copy(), interaction_pairs))
        reg_score = int(component_mask.get("pressure", False)) + int(component_mask.get("wind_u", False))
        interaction_score = len(interaction_pairs)
        return {
            "metrics_test": {"r2": float(reg_score + interaction_score), "rmse": float(10 - reg_score - interaction_score)},
            "picked": {},
            "active_regs": [name for name in ["pressure", "wind_u"] if component_mask.get(name, False)],
            "active_interactions": ["Pressure (hPa) × Wind U (m/s)"] if interaction_pairs else [],
        }

    monkeypatch.setattr(tidal_compact, "run_tidal_model", fake_run_tidal_model)

    components = ["pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    interaction_lookup = {"Pressure (hPa) × Wind U (m/s)": ("pressure", "wind_u")}
    raw_to_canonical = tidal_compact.build_shapley_coalition_plan(components, interaction_lookup)

    shapley_result = build_shapley_result(
        {"pressure": True, "wind_u": True},
        components=components,
        interaction_lookup=interaction_lookup,
        raw_to_canonical=raw_to_canonical,
        df=pd.DataFrame({"water_level": [0.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h")),
        sph=1,
        harmonic_orders={},
        lag_ranges={},
        knot_presets={},
        train_start="2024-01-01",
        train_end="2024-01-01",
        test_end="2024-01-01",
        progress_callback=lambda: progress_ticks.append("tick"),
    )

    assert shapley_result["components"] == ["pressure", "wind_u", "Pressure (hPa) × Wind U (m/s)"]
    assert shapley_result["coalitions"] == 5
    assert len(calls) == 5
    assert len(progress_ticks) == 5
    assert all(
        not interaction_pairs or (component_mask["pressure"] and component_mask["wind_u"])
        for component_mask, interaction_pairs in calls
    )
```

- [ ] **Step 2: Run the targeted regression to verify it fails**

Run: `uv run pytest test/test_example_tidal_compact.py -k "build_shapley_result_deduplicates_invalid_interaction_runs" -v`

Expected: FAIL with `ImportError: cannot import name 'build_shapley_result'` or an equivalent missing-helper error before the notebook Shapley path is refactored.

- [ ] **Step 3: Implement canonical-coalition evaluation and wire the notebook cell**

```python
from collections.abc import Callable, Mapping


@app.function
def build_shapley_result(
    component_mask: dict[str, bool],
    *,
    components: list[str],
    interaction_lookup: dict[str, tuple[str, str]],
    raw_to_canonical: dict[int, int],
    df: pd.DataFrame,
    sph: int,
    harmonic_orders: dict[str, int],
    lag_ranges: dict[str, tuple[int, int]],
    knot_presets: dict[str, str],
    train_start: str,
    train_end: str,
    test_end: str,
    progress_callback: Callable[[], None] | None = None,
) -> ShapleyResult:
    unique_canonical_bits = sorted(set(raw_to_canonical.values()))
    canonical_metrics: dict[int, dict[str, float]] = {}
    failed_runs = 0
    max_workers = min(os.cpu_count() or 4, len(unique_canonical_bits), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for canonical_bits in unique_canonical_bits:
            coalition_mask, coalition_interactions = build_shapley_model_inputs(
                canonical_bits,
                components,
                component_mask,
                interaction_lookup,
            )
            futures[
                pool.submit(
                    run_tidal_model,
                    coalition_mask,
                    df=df,
                    sph=sph,
                    harmonic_orders=harmonic_orders,
                    lag_ranges=lag_ranges,
                    knot_presets=knot_presets,
                    interaction_pairs=coalition_interactions,
                    train_start=train_start,
                    train_end=train_end,
                    test_end=test_end,
                )
            ] = canonical_bits
        for future in as_completed(futures):
            canonical_bits = futures[future]
            fit_result = future.result()
            metrics = fit_result["metrics_test"]
            if np.isnan(metrics.get("r2", 0.0)):
                failed_runs += 1
            canonical_metrics[canonical_bits] = metrics
            if progress_callback is not None:
                progress_callback()

    coalition_metrics = {
        raw_bits: canonical_metrics[canonical_bits]
        for raw_bits, canonical_bits in raw_to_canonical.items()
    }
    baseline_metrics = coalition_metrics[0]
    baseline_r2 = baseline_metrics.get("r2", 0.0)
    if not np.isfinite(baseline_r2):
        baseline_r2 = 0.0
    baseline_rmse = baseline_metrics.get("rmse", np.nan)
    if not np.isfinite(baseline_rmse):
        baseline_rmse = np.nan
    full_bits = 2 ** len(components) - 1
    return {
        "components": components,
        "coalitions": len(unique_canonical_bits),
        "failed": failed_runs,
        "baseline_r2": baseline_r2,
        "baseline_rmse": baseline_rmse,
        "full_r2": coalition_metrics[full_bits].get("r2", 0.0)
        if np.isfinite(coalition_metrics[full_bits].get("r2", 0.0))
        else 0.0,
        "full_rmse": coalition_metrics[full_bits].get("rmse", baseline_rmse),
        "shap_r2": compute_shapley(coalition_metrics, components, "r2", baseline_r2),
        "shap_rmse": compute_shapley(coalition_metrics, components, "rmse", baseline_rmse),
    }


@app.cell
def _(
    harmonic_inputs,
    interaction_pairs_select,
    regressor_knots,
    regressor_lags,
    regressor_toggles,
    run_shapley,
    station_data,
    test_end,
    train_end,
    train_start,
):
    mo.stop(not run_shapley.value, mo.md("*Click above to compute Shapley values for active components.*"))
    _window_message = build_model_window_validation_message(
        station_data["date_min"],
        station_data["date_max"],
        train_start.value,
        train_end.value,
        test_end.value,
    )
    mo.stop(_window_message is not None, mo.md(f"*{_window_message}*"))
    _component_mask, _model_kwargs = collect_model_params(
        harmonic_inputs,
        regressor_lags,
        regressor_toggles,
        regressor_knots,
        interaction_pairs_select,
        df=station_data["df"],
        sph=station_data["sph"],
        train_start=train_start,
        train_end=train_end,
        test_end=test_end,
    )
    _df_train, _df_test, _ok_train, _ok_test = split_model_window(
        station_data["df"],
        _model_kwargs["train_start"],
        _model_kwargs["train_end"],
        _model_kwargs["test_end"],
    )
    _reg_names = [name for name, active in _component_mask.items() if active and name not in PERIODS]
    _x_train_fit, _x_train_pred, _x_test_pred, _active_regs, _exog_config = build_exog_design_matrices(
        _df_train,
        _df_test,
        _ok_train,
        _reg_names,
        _model_kwargs["lag_ranges"],
        _model_kwargs["knot_presets"],
        _model_kwargs["sph"],
    )
    _components, _interaction_lookup = build_shapley_component_list(
        _component_mask,
        _active_regs,
        _model_kwargs["interaction_pairs"],
    )
    _num_components = len(_components)
    mo.stop(_num_components < 2, mo.md("*Need at least 2 active components for Shapley analysis.*"))
    mo.stop(
        _num_components > 12,
        mo.md(f"*{_num_components} components -> {2**_num_components:,} raw coalitions - cap at 12.*"),
    )
    _raw_to_canonical = build_shapley_coalition_plan(_components, _interaction_lookup)
    with mo.status.progress_bar(total=len(set(_raw_to_canonical.values()))) as _progress:
        shapley_result = build_shapley_result(
            _component_mask,
            components=_components,
            interaction_lookup=_interaction_lookup,
            raw_to_canonical=_raw_to_canonical,
            **_model_kwargs,
            progress_callback=_progress.update,
        )
    return (shapley_result,)
```

- [ ] **Step 4: Run the focused regression and then the full compact notebook test file**

Run: `uv run pytest test/test_example_tidal_compact.py -k "build_shapley_result_deduplicates_invalid_interaction_runs" -v`

Expected: PASS for the new interaction-aware Shapley regression.

Run: `uv run pytest test/test_example_tidal_compact.py -q`

Expected: PASS for the new Shapley regressions and the existing compact notebook test suite.

- [ ] **Step 5: Commit the Shapley notebook wiring**

```bash
git add examples/example_tidal_compact.py test/test_example_tidal_compact.py
git commit -m "feat: add compact shapley interaction attribution"
```
