"""Evaluate the actual notebook response expressions without fetching datasets."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("lags", [[-2, -1, 0], [0], [0, -2]])
@pytest.mark.parametrize("notebook", ["example_air_quality_marimo.py", "combined_examples_marimo.py"])
def test_response_plots_select_configured_zero_lag(notebook, lags):
    path = Path(__file__).parents[1] / "examples" / notebook
    tree = ast.parse(path.read_text())
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.BinOp):
            continue
        if not isinstance(node.value.op, ast.MatMult) or not any(
            isinstance(target, ast.Name) and target.id in {
                "_log_response", "_log_response_aq", "_log_response_la", "_log_response_la_hum",
            } for target in node.targets
        ):
            continue
        basis = np.arange(12).reshape(4, 3)
        coefficients = np.arange(3 * len(lags)).reshape(3, len(lags))
        cfg = SimpleNamespace(exog_config=[SimpleNamespace(lags=lags)] * 2)
        env = {"len": len, "_var_idx": 0, "_idx": 0}
        for name in (n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)):
            if name.startswith("_H"):
                env[name] = basis
            elif name.startswith("_exog_coef"):
                env[name] = coefficients[:, 0] if len(lags) == 1 else coefficients
            elif name.startswith("_knots"):
                env[name] = np.arange(4)
            elif name.startswith("estimator"):
                env[name] = SimpleNamespace(config=cfg)
        result = eval(compile(ast.Expression(node.value), str(path), "eval"), env)
        np.testing.assert_array_equal(result, basis @ coefficients[:, lags.index(0)])
        checked += 1
    assert checked == (2 if notebook == "example_air_quality_marimo.py" else 4)
