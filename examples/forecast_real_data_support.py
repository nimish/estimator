"""Local, bounded benchmark helpers for the forecast research notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tsgam_estimator import (
    TsgamEstimatorConfig,
    TsgamForecastConfig,
    TsgamForecastCouplingConfig,
    TsgamForecastEstimator,
    TsgamLinearConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    plot_forecast_origin,
)


EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
NOTEBOOK_NAME = "example_forecast_real_data_benchmarks.ipynb"
TIDAL_CONSTITUENTS_HOURS: Mapping[str, float] = {
    "M2": 12.4206,
    "S2": 12.0,
    "K1": 23.9345,
}


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for one bounded native-frequency benchmark."""

    name: str
    display_name: str
    source: str
    target_label: str
    feature_columns: tuple[str, ...]
    feature_labels: tuple[str, ...]
    train_samples: int
    eval_samples: int
    horizon: int
    feature_fill_limit: int
    periods_in_samples: tuple[float, ...]
    num_harmonics: tuple[int, ...]
    coupling_weight: float
    solver: str
    solver_opts: Mapping[str, float] | None
    caveat: str

    @property
    def required_samples(self) -> int:
        return self.train_samples + self.eval_samples + self.horizon


@dataclass
class PreparedDataset:
    """A regular, complete source window with a deterministic split."""

    spec: DatasetSpec
    frame: pd.DataFrame
    step: pd.Timedelta
    source_rows: int
    native_grid_rows: int
    filled_feature_cells: int


@dataclass
class BenchmarkResult:
    """Independent and coupled forecast outputs for a dataset."""

    dataset: PreparedDataset
    actuals: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    roughness: pd.DataFrame
    runtime_seconds: dict[str, float]


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "pv_solar": DatasetSpec(
        name="pv_solar",
        display_name="PV solar AC power",
        source="pv/2107_data_combined.csv",
        target_label="sum of inverter AC power",
        feature_columns=("power_now", "poa_irradiance"),
        feature_labels=("AC power at origin", "POA irradiance at origin"),
        train_samples=1_008,
        eval_samples=288,
        horizon=24,
        feature_fill_limit=12,
        periods_in_samples=(288.0,),
        num_harmonics=(2,),
        coupling_weight=0.2,
        solver="CLARABEL",
        solver_opts={"max_iter": 200},
        caveat=(
            "Ambient and wind are mostly missing in this source; the bounded PV run "
            "uses AC power and POA irradiance only."
        ),
    ),
    "iso_load": DatasetSpec(
        name="iso_load",
        display_name="ISO New England real-time load",
        source="iso/2020_smd_hourly.xlsx through 2022_smd_hourly.xlsx, RI sheet",
        target_label="RI real-time demand",
        feature_columns=("load_now", "day_ahead_demand", "dry_bulb", "dew_point"),
        feature_labels=(
            "real-time load at origin",
            "day-ahead demand at origin",
            "dry-bulb temperature at origin",
            "dew-point temperature at origin",
        ),
        train_samples=720,
        eval_samples=168,
        horizon=24,
        feature_fill_limit=2,
        periods_in_samples=(24.0, 168.0),
        num_harmonics=(2, 1),
        coupling_weight=1.0,
        solver="CLARABEL",
        solver_opts={"max_iter": 200},
        caveat=(
            "The bundled ISO workbook has load, prices, and weather but no metered "
            "solar-generation or net-load column, so this row reports load only."
        ),
    ),
    "tidal_water_level": DatasetSpec(
        name="tidal_water_level",
        display_name="NOAA tidal water level",
        source="tidal/9414290_combined.csv",
        target_label="water level",
        feature_columns=("water_level_now",),
        feature_labels=("water level at origin",),
        train_samples=1_008,
        eval_samples=288,
        horizon=20,
        feature_fill_limit=0,
        periods_in_samples=tuple(
            period * 10.0 for period in TIDAL_CONSTITUENTS_HOURS.values()
        ),
        num_harmonics=(1, 1, 1),
        coupling_weight=2.0,
        solver="SCS",
        solver_opts={"eps": 1.0e-4, "max_iters": 10_000},
        caveat=(
            "Station weather columns are about 90% missing, so this benchmark uses "
            "the observed origin water level and named tidal constituents."
        ),
    ),
}


def _as_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def _load_pv() -> pd.DataFrame:
    ac_columns = [
        "inv_01_ac_power_inv_149583",
        "inv_02_ac_power_inv_149588",
        "inv_03_ac_power_inv_149593",
    ]
    raw = pd.read_csv(
        DATA_DIR / "pv" / "2107_data_combined.csv",
        usecols=["measured_on", "poa_irradiance_o_149574", *ac_columns],
        parse_dates=["measured_on"],
    ).set_index("measured_on")
    raw = _as_numeric(raw)
    power = raw[ac_columns].sum(axis=1, min_count=len(ac_columns))
    return pd.DataFrame(
        {
            "target": power,
            "power_now": power,
            "poa_irradiance": raw["poa_irradiance_o_149574"],
        },
        index=raw.index,
    )


def _load_iso() -> pd.DataFrame:
    yearly_frames = []
    for year in (2020, 2021, 2022):
        raw = pd.read_excel(
            DATA_DIR / "iso" / f"{year}_smd_hourly.xlsx", sheet_name="RI"
        )
        hour = pd.to_numeric(raw["Hr_End"], errors="coerce").astype("Int64") - 1
        raw.index = pd.to_datetime(
            raw["Date"].astype(str) + " " + hour.astype(str) + ":00:00"
        )
        raw.index = pd.DatetimeIndex(raw.index + pd.Timedelta(hours=1))
        yearly_frames.append(
            _as_numeric(raw[["RT_Demand", "DA_Demand", "Dry_Bulb", "Dew_Point"]])
        )
    raw = pd.concat(yearly_frames).sort_index()
    return pd.DataFrame(
        {
            "target": raw["RT_Demand"],
            "load_now": raw["RT_Demand"],
            "day_ahead_demand": raw["DA_Demand"],
            "dry_bulb": raw["Dry_Bulb"],
            "dew_point": raw["Dew_Point"],
        },
        index=raw.index,
    )


def _load_tidal() -> pd.DataFrame:
    raw = pd.read_csv(
        DATA_DIR / "tidal" / "9414290_combined.csv",
        usecols=["datetime", "water_level"],
        parse_dates=["datetime"],
    ).set_index("datetime")
    raw = _as_numeric(raw)
    return pd.DataFrame(
        {"target": raw["water_level"], "water_level_now": raw["water_level"]},
        index=raw.index,
    )


def _load_source(name: str) -> pd.DataFrame:
    loaders = {
        "pv_solar": _load_pv,
        "iso_load": _load_iso,
        "tidal_water_level": _load_tidal,
    }
    try:
        return loaders[name]()
    except KeyError as error:
        raise KeyError(f"Unknown dataset {name!r}.") from error


def _native_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    ordered = pd.DatetimeIndex(index).sort_values().unique()
    deltas = ordered.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError("Need at least two timestamps to infer a native step.")
    step = pd.Timedelta(deltas.value_counts().index[0])
    if step <= pd.Timedelta(0):
        raise ValueError(f"Expected a positive native step, got {step!r}.")
    return step


def _first_valid_run(valid: pd.Series, required: int) -> int:
    run_id = valid.ne(valid.shift()).cumsum()
    for _, values in valid[valid].groupby(run_id[valid]):
        if len(values) >= required:
            return int(valid.index.get_indexer([values.index[0]])[0])
    longest = max(
        (len(values) for _, values in valid[valid].groupby(run_id[valid])), default=0
    )
    raise ValueError(
        f"Need {required} contiguous complete samples; longest available run is {longest}."
    )


def _prepare_window(source: pd.DataFrame, spec: DatasetSpec) -> PreparedDataset:
    source = source.sort_index()
    source = source.loc[~source.index.duplicated(keep="last")]
    step = _native_step(pd.DatetimeIndex(source.index))
    native = source.reindex(
        pd.date_range(source.index.min(), source.index.max(), freq=step)
    )
    original_features = native.loc[:, list(spec.feature_columns)]
    known_features = (
        original_features.ffill(limit=spec.feature_fill_limit)
        if spec.feature_fill_limit > 0
        else original_features.copy()
    )
    native.loc[:, list(spec.feature_columns)] = known_features
    valid = native["target"].notna() & known_features.notna().all(axis=1)
    start = _first_valid_run(valid, spec.required_samples)
    window = native.iloc[start : start + spec.required_samples].copy()
    if len(window) != spec.required_samples or window.isna().any().any():
        raise AssertionError(
            "The selected benchmark window must be complete and bounded."
        )
    if pd.infer_freq(window.index) is None:
        raise AssertionError(
            "The selected benchmark window must remain on its native regular grid."
        )
    filled = int(
        (
            original_features.iloc[start : start + spec.required_samples].isna()
            & known_features.iloc[start : start + spec.required_samples].notna()
        )
        .sum()
        .sum()
    )
    return PreparedDataset(
        spec=spec,
        frame=window,
        step=step,
        source_rows=len(source),
        native_grid_rows=len(native),
        filled_feature_cells=filled,
    )


def load_all_datasets(names: Sequence[str] | None = None) -> dict[str, PreparedDataset]:
    """Load deterministic local-only windows for every requested dataset."""

    selected = tuple(names) if names is not None else tuple(DATASET_SPECS)
    return {
        name: _prepare_window(_load_source(name), DATASET_SPECS[name])
        for name in selected
    }


def _duration_label(value: pd.Timedelta) -> str:
    minutes = value.total_seconds() / 60
    if minutes.is_integer() and minutes % 60 == 0:
        return f"{int(minutes // 60)} h"
    return f"{int(minutes)} min"


def overview_table(datasets: Mapping[str, PreparedDataset]) -> pd.DataFrame:
    """Show source, native grid, known-at-origin features, and data policy."""

    rows = []
    for prepared in datasets.values():
        spec = prepared.spec
        rows.append(
            {
                "dataset": spec.display_name,
                "local source": spec.source,
                "target": spec.target_label,
                "known-at-origin features": ", ".join(spec.feature_labels),
                "native frequency": _duration_label(prepared.step),
                "selected window": f"{prepared.frame.index[0]} to {prepared.frame.index[-1]}",
                "train origins": prepared.spec.train_samples,
                "test origins": prepared.spec.eval_samples,
                "max horizon": _duration_label(prepared.spec.horizon * prepared.step),
                "forward-filled feature cells": prepared.filled_feature_cells,
                "caveat": spec.caveat,
            }
        )
    return pd.DataFrame(rows)


def split_and_alignment_table(datasets: Mapping[str, PreparedDataset]) -> pd.DataFrame:
    """Expose the train/test origin split and matching future target times."""

    rows = []
    for prepared in datasets.values():
        spec = prepared.spec
        frame = prepared.frame
        origin = frame.index[spec.train_samples]
        for horizon in sorted({0, 1, max(1, spec.horizon // 2), spec.horizon}):
            rows.append(
                {
                    "dataset": spec.display_name,
                    "train origins end": frame.index[spec.train_samples - 1],
                    "test origins": (
                        f"{frame.index[spec.train_samples]} to "
                        f"{frame.index[spec.train_samples + spec.eval_samples - 1]}"
                    ),
                    "last held-out target": frame.index[
                        spec.train_samples + spec.eval_samples - 1 + spec.horizon
                    ],
                    "test origin": origin,
                    "known feature time": origin,
                    "horizon steps": horizon,
                    "target time": origin + horizon * prepared.step,
                    "target value": float(
                        frame.loc[origin + horizon * prepared.step, "target"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _standardize(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = train.mean()
    scale = train.std(ddof=0).replace(0.0, 1.0)
    return (train - mean) / scale, (evaluation - mean) / scale


def _actual_targets(
    prepared: PreparedDataset, origins: pd.DatetimeIndex
) -> pd.DataFrame:
    target = prepared.frame["target"]
    values = {
        f"horizon_{horizon}": target.reindex(
            origins + horizon * prepared.step
        ).to_numpy()
        for horizon in range(prepared.spec.horizon + 1)
    }
    actuals = pd.DataFrame(values, index=origins)
    if actuals.isna().any().any():
        raise AssertionError("Held-out target alignment contains missing values.")
    return actuals


def _make_model(
    spec: DatasetSpec, n_features: int, mode: str
) -> TsgamForecastEstimator:
    base_config = TsgamEstimatorConfig(
        multi_periodic_config=TsgamMultiPeriodicConfig(
            num_harmonics=list(spec.num_harmonics),
            periods=list(spec.periods_in_samples),
            reg_weight=1.0e-4,
        ),
        exog_config=[
            TsgamLinearConfig(lags=[0], reg_weight=1.0e-4) for _ in range(n_features)
        ],
        solver_config=TsgamSolverConfig(
            solver=spec.solver,
            verbose=False,
            solver_opts=dict(spec.solver_opts) if spec.solver_opts else None,
        ),
    )
    coupling = (
        TsgamForecastCouplingConfig(
            roughness_weight=spec.coupling_weight,
            roughness_order=1,
        )
        if mode == "coupled"
        else None
    )
    return TsgamForecastEstimator(
        TsgamForecastConfig(
            horizon=spec.horizon,
            base_config=base_config,
            mode=mode,
            coupling_config=coupling,
        )
    )


def _roughness(model: TsgamForecastEstimator) -> float:
    vectors = []
    forecast_horizons = [horizon for horizon in model.horizons_ if horizon > 0]
    if model.config.mode == "independent":
        for horizon in forecast_horizons:
            variables = model.forecast_estimators_[horizon].variables_
            vectors.append(
                np.concatenate(
                    [
                        np.asarray(variable.value, dtype=float).ravel(order="F")
                        for _, variable in sorted(variables.items())
                    ]
                )
            )
    else:
        horizon_count = len(model.horizons_)
        for horizon in forecast_horizons:
            horizon_ix = model.horizons_.index(horizon)
            parts = []
            for _, variable in sorted(model.variables_.items()):
                if isinstance(variable, list):
                    value = variable[horizon_ix].value
                else:
                    value = np.asarray(variable.value)
                    if value.ndim == 1 and value.shape[0] == horizon_count:
                        value = value[horizon_ix]
                    elif value.ndim >= 2 and value.shape[-1] == horizon_count:
                        value = value[..., horizon_ix]
                parts.append(np.asarray(value, dtype=float).ravel(order="F"))
            vectors.append(np.concatenate(parts))
    if len(vectors) < 2:
        return 0.0
    coefficients = np.vstack(vectors)
    return float(np.sqrt(np.mean(np.diff(coefficients, axis=0) ** 2)))


def benchmark_dataset(prepared: PreparedDataset) -> BenchmarkResult:
    """Fit the independent and coupled direct forecast models on one data window."""

    spec = prepared.spec
    train_end = spec.train_samples
    eval_end = train_end + spec.eval_samples
    X = prepared.frame.loc[:, list(spec.feature_columns)]
    X_train, X_eval = _standardize(X.iloc[:train_end], X.iloc[train_end:eval_end])
    y_train = prepared.frame["target"].iloc[:train_end].to_numpy(dtype=float)
    actuals = _actual_targets(prepared, pd.DatetimeIndex(X_eval.index))
    predictions: dict[str, pd.DataFrame] = {}
    runtime_seconds: dict[str, float] = {}
    metric_rows = []
    roughness_rows = []
    for mode in ("independent", "coupled"):
        model = _make_model(spec, X_train.shape[1], mode)
        started = perf_counter()
        model.fit(X_train, y_train)
        runtime_seconds[mode] = perf_counter() - started
        prediction = model.predict(X_eval)
        if not prediction.index.equals(X_eval.index):
            raise AssertionError(
                "Prediction rows must stay indexed by forecast origin."
            )
        predictions[mode] = prediction
        roughness_rows.append(
            {
                "dataset": spec.display_name,
                "model": mode,
                "coefficient RMS first difference": _roughness(model),
            }
        )
        for horizon in range(spec.horizon + 1):
            column = f"horizon_{horizon}"
            error = prediction[column].to_numpy() - actuals[column].to_numpy()
            metric_rows.append(
                {
                    "dataset": spec.display_name,
                    "model": mode,
                    "horizon": horizon,
                    "horizon duration": _duration_label(horizon * prepared.step),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                }
            )
    return BenchmarkResult(
        dataset=prepared,
        actuals=actuals,
        predictions=predictions,
        metrics=pd.DataFrame(metric_rows),
        roughness=pd.DataFrame(roughness_rows),
        runtime_seconds=runtime_seconds,
    )


def run_all_benchmarks(
    datasets: Mapping[str, PreparedDataset],
) -> dict[str, BenchmarkResult]:
    """Run every local benchmark with identical independent/coupled protocol."""

    return {name: benchmark_dataset(dataset) for name, dataset in datasets.items()}


def metrics_table(results: Mapping[str, BenchmarkResult]) -> pd.DataFrame:
    return pd.concat([result.metrics for result in results.values()], ignore_index=True)


def compact_summary(results: Mapping[str, BenchmarkResult]) -> pd.DataFrame:
    """Keep cross-dataset h=0/h=1/h=max metrics, roughness, and runtime compact."""

    rows = []
    for result in results.values():
        spec = result.dataset.spec
        row: dict[str, object] = {
            "dataset": spec.display_name,
            "frequency": _duration_label(result.dataset.step),
            "max horizon": _duration_label(spec.horizon * result.dataset.step),
        }
        for mode in ("independent", "coupled"):
            metrics = result.metrics[result.metrics["model"] == mode].set_index(
                "horizon"
            )
            roughness = result.roughness.loc[
                result.roughness["model"] == mode, "coefficient RMS first difference"
            ].iloc[0]
            row[f"{mode} RMSE h=0"] = float(metrics.loc[0, "rmse"])
            row[f"{mode} MAE h=0"] = float(metrics.loc[0, "mae"])
            row[f"{mode} RMSE h=1"] = float(metrics.loc[1, "rmse"])
            row[f"{mode} MAE h=1"] = float(metrics.loc[1, "mae"])
            row[f"{mode} RMSE h=max"] = float(metrics.loc[spec.horizon, "rmse"])
            row[f"{mode} MAE h=max"] = float(metrics.loc[spec.horizon, "mae"])
            row[f"{mode} roughness"] = float(roughness)
            row[f"{mode} fit seconds"] = result.runtime_seconds[mode]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_horizon_metrics(results: Mapping[str, BenchmarkResult]) -> plt.Figure:
    """Plot RMSE and MAE at every horizon for all three datasets."""

    figure, axes = plt.subplots(
        len(results), 2, figsize=(13, 3.4 * len(results)), squeeze=False
    )
    colors = {"independent": "#1f77b4", "coupled": "#d95f02"}
    for row_ix, result in enumerate(results.values()):
        for col_ix, metric in enumerate(("rmse", "mae")):
            axis = axes[row_ix, col_ix]
            for mode in ("independent", "coupled"):
                values = result.metrics[result.metrics["model"] == mode]
                axis.plot(
                    values["horizon"],
                    values[metric],
                    marker="o",
                    markersize=3,
                    color=colors[mode],
                    label=mode,
                )
            axis.set_title(f"{result.dataset.spec.display_name}: {metric.upper()}")
            axis.set_xlabel("forecast horizon (native steps)")
            axis.set_ylabel(metric.upper())
            axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_forecast_paths(
    results: Mapping[str, BenchmarkResult], origin_fraction: float = 0.5
) -> plt.Figure:
    """Plot one origin-to-target forecast path per dataset."""

    figure, axes = plt.subplots(
        len(results), 1, figsize=(13, 3.6 * len(results)), squeeze=False
    )
    for axis, result in zip(axes[:, 0], results.values(), strict=True):
        prediction = result.predictions["independent"]
        position = min(
            len(prediction) - 1, max(0, int((len(prediction) - 1) * origin_fraction))
        )
        origin = prediction.index[position]
        spec = result.dataset.spec
        plot_forecast_origin(
            {
                "Independent": result.predictions["independent"],
                "Coupled": result.predictions["coupled"],
            },
            actual=result.dataset.frame["target"],
            origin=origin,
            history_steps=min(24, spec.train_samples),
            freq=result.dataset.step,
            ax=axis,
        )
        axis.set_title(f"{spec.display_name}: nowcast and future forecast path")
        axis.set_ylabel(spec.target_label)
    figure.tight_layout()
    return figure


def plot_roughness(results: Mapping[str, BenchmarkResult]) -> plt.Figure:
    """Compare coefficient roughness between independent and coupled fits."""

    roughness = pd.concat(
        [result.roughness for result in results.values()], ignore_index=True
    )
    datasets = list(roughness["dataset"].drop_duplicates())
    positions = np.arange(len(datasets))
    figure, axis = plt.subplots(figsize=(11, 4.2))
    for offset, mode, color in (
        (-0.18, "independent", "#1f77b4"),
        (0.18, "coupled", "#d95f02"),
    ):
        values = [
            roughness.loc[
                (roughness["dataset"] == dataset) & (roughness["model"] == mode),
                "coefficient RMS first difference",
            ].iloc[0]
            for dataset in datasets
        ]
        axis.bar(positions + offset, values, width=0.36, color=color, label=mode)
    axis.set_xticks(positions, datasets)
    axis.set_ylabel("coefficient RMS first difference")
    axis.set_title("Coefficient-path roughness by dataset")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _cell_source(*lines: str) -> str:
    return "\n".join(lines)


def write_notebook(path: Path | None = None) -> Path:
    """Generate the checked-in, configuration-first benchmark notebook."""

    import nbformat as nbf

    output_path = path or EXAMPLES_DIR / NOTEBOOK_NAME
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Real-data direct multi-horizon forecast benchmark\n\n"
            "This local-only research notebook compares independent and coupled "
            "TsgamForecastEstimator fits on bundled PV solar, ISO load, and tidal "
            "water-level data. It performs no downloads."
        ),
        nbf.v4.new_code_cell(
            _cell_source(
                "from pathlib import Path",
                "from time import perf_counter",
                "import sys",
                "",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "from IPython.display import display",
                "",
                "examples_dir = Path.cwd() / 'examples'",
                "if not (examples_dir / 'forecast_real_data_support.py').exists():",
                "    examples_dir = Path.cwd()",
                "if str(examples_dir) not in sys.path:",
                "    sys.path.insert(0, str(examples_dir))",
                "",
                "from forecast_real_data_support import (",
                "    compact_summary,",
                "    load_all_datasets,",
                "    metrics_table,",
                "    overview_table,",
                "    plot_forecast_paths,",
                "    plot_horizon_metrics,",
                "    plot_roughness,",
                "    run_all_benchmarks,",
                "    split_and_alignment_table,",
                ")",
                "",
                "sns.set_theme(style='whitegrid', context='notebook')",
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Configuration\n\n"
            "These are deterministic bounded windows at native source frequency. "
            "The final horizon rows are reserved only for held-out targets."
        ),
        nbf.v4.new_code_cell(
            _cell_source(
                "DATASET_NAMES = ('pv_solar', 'iso_load', 'tidal_water_level')",
                "FORECAST_MODES = ('independent', 'coupled')",
                "FORECAST_PATH_ORIGIN_FRACTION = 0.5",
                "LOCAL_DATA_ONLY = True",
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Local data, known-at-origin information, and missing-data policy\n\n"
            "Every feature is observed at the forecast origin. Targets are never "
            "filled; source features are forward-filled only up to a short "
            "dataset-specific limit, so no future feature value reaches an origin. "
            "The loaders preserve the modal source grid and do not resample."
        ),
        nbf.v4.new_code_cell(
            _cell_source(
                "datasets = load_all_datasets(DATASET_NAMES)",
                "display(overview_table(datasets))",
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Train/test origins and target alignment\n\n"
            "For origin o and horizon h, direct multi-horizon fitting uses "
            "X(o) to y(o + h). The table shows the split plus concrete examples "
            "of the origin, known-feature timestamp, and scored target timestamp, "
            "including the h=0 nowcast."
        ),
        nbf.v4.new_code_cell(
            _cell_source("display(split_and_alignment_table(datasets))")
        ),
        nbf.v4.new_markdown_cell(
            "## Run the benchmark\n\n"
            "Independent and coupled modes use the same origin-time basis. Coupling "
            "adds a first-difference penalty across positive-horizon coefficients; "
            "h=0 remains an uncoupled diagnostic baseline. "
            "Lower coefficient roughness is a smoothness result rather than a "
            "guarantee of lower error."
        ),
        nbf.v4.new_code_cell(
            _cell_source(
                "benchmark_started = perf_counter()",
                "results = run_all_benchmarks(datasets)",
                "total_runtime_seconds = perf_counter() - benchmark_started",
                "print(f'Completed {len(results)} bounded local benchmarks in {total_runtime_seconds:.1f} seconds.')",
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Horizon-wise RMSE and MAE\n\n"
            "Horizon zero is the aligned nowcast baseline; positive horizons are "
            "future forecasts from the same origin rows."
        ),
        nbf.v4.new_code_cell(
            _cell_source(
                "metrics = metrics_table(results)",
                "display(metrics)",
                "metric_figure = plot_horizon_metrics(results)",
                "plt.show()",
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Forecast paths\n\n"
            "Each panel shows observed history, the h=0 nowcast at the forecast "
            "origin, and the future target path. Prediction rows are indexed by "
            "origin and unwrapped onto their target timestamps for this view."
        ),
        nbf.v4.new_code_cell(
            _cell_source(
                "path_figure = plot_forecast_paths(",
                "    results,",
                "    origin_fraction=FORECAST_PATH_ORIGIN_FRACTION,",
                ")",
                "plt.show()",
            )
        ),
        nbf.v4.new_markdown_cell("## Coefficient roughness and compact summary"),
        nbf.v4.new_code_cell(
            _cell_source(
                "roughness_figure = plot_roughness(results)",
                "plt.show()",
                "",
                "summary = compact_summary(results)",
                "display(summary)",
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Caveats\n\n"
            "These are small reproducible research windows, not a production "
            "backtest. Runtime depends on the local CVXPY solver. The ISO workbook "
            "does not include a metered solar or net-load series, so its result is "
            "reported honestly as a real-time load benchmark."
        ),
    ]
    nbf.write(notebook, output_path)
    return output_path
