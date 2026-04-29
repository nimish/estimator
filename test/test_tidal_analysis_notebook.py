import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "examples" / "tidal_analysis.ipynb"


def _notebook_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def test_harmonic_selection_sweep_includes_trend_candidate():
    sweep_source = next(
        source for source in _notebook_sources()
        if "harmonic_sweep_specs = {" in source
    )

    assert "trend_config" in sweep_source
    assert "TsgamTrendConfig(" in sweep_source
    assert "TrendType.LINEAR" in sweep_source
    assert "TrendType.NONLINEAR_DECREASING" in sweep_source
    assert "TrendType.NONLINEAR_INCREASING" in sweep_source


def test_trend_candidate_has_ui_controls_and_drives_sweep():
    notebook_source = "\n".join(_notebook_sources())
    config_source = next(
        source for source in _notebook_sources()
        if "config_box = widgets.VBox" in source
    )
    sweep_source = next(
        source for source in _notebook_sources()
        if "harmonic_sweep_specs = {" in source
    )

    assert "trend_candidate_toggle = widgets.Checkbox(" in config_source
    assert "trend_type_dropdown = widgets.Dropdown(" in config_source
    assert '("Linear", "linear")' in config_source
    assert '("Nonlinear decreasing", "nonlinear_decreasing")' in config_source
    assert '("Nonlinear increasing", "nonlinear_increasing")' in config_source
    assert "trend_candidate_toggle.value" in sweep_source
    assert "trend_type_dropdown.value" in sweep_source
    assert "selected_trend_config" in notebook_source


def test_periodogram_plots_are_log_log():
    notebook_source = "\n".join(_notebook_sources())

    assert notebook_source.count('.set_xscale("log")') > 0
    assert notebook_source.count('.set_yscale("log")') == notebook_source.count(
        '.set_xscale("log")',
    )
