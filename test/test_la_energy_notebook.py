# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Run the LA notebook's fitting, prediction, plots, and ablation cells."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _widget(value):
    return SimpleNamespace(value=value)


def _run_notebook(**overrides):
    path = Path(__file__).parents[1] / "examples" / "example_la_energy_marimo.py"
    spec = importlib.util.spec_from_file_location("la_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controls = {
        "fit_model": _widget(True), "take_log": _widget(True),
        "use_ar": _widget(False), "use_outlier": _widget(False),
        "outlier_reg_weight": _widget(1e-4), "outlier_threshold": _widget(.05),
        "solver_select": _widget("CLARABEL"), "verbose": _widget(False),
    }
    controls.update(overrides)
    try:
        _, definitions = module.app.run(defs=controls)
        return definitions
    finally:
        plt.close("all")


def _assert_holdout_predictions(definitions):
    estimator = definitions["estimator"]
    holdout = definitions["X_test"].index
    train = definitions["X_train"].index
    assert len(holdout.tz_localize(None).to_period("M").unique()) >= 2
    assert not train.intersection(holdout).size
    assert np.isfinite(definitions["y_pred"]).all()
    assert len(definitions["y_pred"]) == len(holdout)
    assert np.isfinite([definitions[k] for k in ("rmse", "mae", "mape", "r2")]).all()
    assert estimator.decomposition_["status"] == "optimal"
    # Held-out targets must remain excluded from the solved training mask.
    holdout_indices = ((holdout - estimator.time_reference_) / pd.Timedelta("1h")).astype(int)
    mask = estimator.decomposition_["fit_mask"]
    assert not mask[holdout_indices[holdout_indices < len(mask)]].any()
    index = definitions["X_predict"].index
    assert ((index[1:] - index[:-1]) == pd.Timedelta("1h")).all()


def test_la_notebook_default_real_data():
    definitions = _run_notebook(ablation_test=_widget(True))
    _assert_holdout_predictions(definitions)
    assert len(definitions["y_pred"]) == 2016
    assert definitions["r2"] > .95
    results = definitions["ablation_results"]
    assert len(results) == 12
    assert all(result["status"] == "optimal" for result in results)
    assert all(np.isfinite(result["rmse"]) for result in results)


@pytest.mark.parametrize("n_weather,take_log", [(1, False), (5, True)])
def test_la_notebook_weather_selection_ar_and_outliers(n_weather, take_log):
    index = pd.date_range("2018-01-01", "2018-02-28 23:00", freq="1h")
    t = np.arange(len(index))
    weather = pd.DataFrame({
        "temperature_degF": 60 + 12 * np.sin(2 * np.pi * t / 24),
        "humidity_pc": 50 + 15 * np.cos(2 * np.pi * t / 37),
        "global_Wpms": np.maximum(0, 600 * np.sin(2 * np.pi * t / 24)),
        "direct_Wpms": np.maximum(0, 400 * np.sin(2 * np.pi * t / 24 + .2)),
        "diffuse_Wpms": 50 + 20 * np.cos(2 * np.pi * t / 29),
    }, index=index)
    energy = pd.DataFrame({
        "elec_total_MW": 300 + 30 * np.cos(2 * np.pi * t / 24)
        + .5 * (weather.temperature_degF - 60)
        + np.random.default_rng(8).normal(0, 1, len(index)),
    }, index=index)
    definitions = _run_notebook(
        df_weather=weather, df_energy=energy,
        exog_vars=_widget(list(weather.columns[:n_weather])),
        take_log=_widget(take_log), use_ar=_widget(True), use_outlier=_widget(True),
    )
    _assert_holdout_predictions(definitions)
    assert definitions["estimator"].ar_coef_ is not None
    assert "outlier" in definitions["estimator"].decomposition_["values"]


def _notebook_cell(*arguments):
    """Execute the notebook's actual cell without running unrelated fitting cells."""
    path = Path(__file__).parents[1] / "examples" / "example_la_energy_marimo.py"
    tree = ast.parse(path.read_text())
    cell = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and {arg.arg for arg in node.args.args} == set(arguments)
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_"]


@pytest.mark.parametrize("take_log", [False, True])
def test_outlier_controls_and_reports_use_target_units(take_log, capsys):
    import marimo as mo

    log_switch = _widget(take_log)
    units, = _notebook_cell("take_log")(take_log=log_switch)
    threshold, = _notebook_cell("mo", "outlier_units")(mo=mo, outlier_units=units)
    assert units == ("log(load + 1)" if take_log else "MW")
    assert f"Significant outlier threshold ({units})" in threshold._repr_html_()
    assert threshold.value == .05

    # Known positive/negative daily corrections exercise significant-outlier,
    # holiday/weekend/weekday reports, and unit-bearing plot labels.
    corrections = np.array([.1, -.2, .01, .01, .01, .1, -.2])
    timestamps = pd.date_range("2018-01-01", periods=24 * len(corrections), freq="1h")
    estimator = SimpleNamespace(
        config=SimpleNamespace(outlier_config=object()),
        decomposition_={"values": {
            "outlier_group_values": corrections,
            "outlier": np.repeat(corrections, 24),
        }},
        time_indices_=np.arange(len(timestamps)), outlier_period_hours_=24,
    )
    shared = dict(
        estimator=estimator, np=np, outlier_threshold=threshold,
        outlier_units=units, take_log=log_switch, timestamps_train_aligned=timestamps,
    )
    try:
        _notebook_cell(*shared, "plt")(**shared, plt=plt)
        axes = plt.gcf().axes
        assert all(units in label for label in (
            axes[0].get_ylabel(), axes[1].get_ylabel(), axes[2].get_xlabel(),
        ))
        _notebook_cell(*shared, "pd")(**shared, pd=pd)
        output = capsys.readouterr().out
        assert f"|outlier| > 0.050 {units}" in output
        assert "Significant outlier periods (|outlier| > 0.050): 4" in output
        if take_log:
            assert "Threshold boundary effects on (load + 1): -4.9% / +5.1%" in output
            assert "Multiplier range on (load + 1): 0.819x to 1.105x" in output
            assert "Average holiday multiplier on (load + 1)" in output
            assert "Average weekend multiplier on (load + 1)" in output
            assert "Average weekday multiplier on (load + 1)" in output
        else:
            assert "Additive correction range: -0.2000 to +0.1000 MW" in output
            assert "Outlier (MW)" in output
            assert "multiplier" not in output.lower()
            assert "log space" not in output
            assert "Threshold boundary effects" not in output
    finally:
        plt.close("all")
