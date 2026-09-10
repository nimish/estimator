# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import cvxpy
import numpy as np
from numpy import ndarray
from signaldecomp import (
    exog_interaction,
    exog_linear,
    exog_spline,
    grouped_sparse,
    grouped_trend,
    make_problem,
    multiperiodic,
)
from signaldecomp.spline import make_spline_basis

from ._design import (
    _is_spline_config,
    _TsgamDesign,
)

if TYPE_CHECKING:
    from ._estimator import TsgamEstimatorConfig, TsgamSolverConfig


def legacy_variable_views(
    config: TsgamEstimatorConfig,
    variables: Mapping[str, cvxpy.Expression],
) -> dict[str, cvxpy.Expression]:
    """Expose historical coefficient names/shapes without new optimization variables."""
    views = {"constant": variables["intercept_group_values"][0]}
    for ix, cfg in enumerate(config.exog_config or []):
        suffix = "coef" if _is_spline_config(cfg) else "beta"
        coefficient = variables[f"exog_{ix}_{suffix}"]
        views[f"exog_coef_{ix}"] = cvxpy.reshape(
            coefficient, (coefficient.size // len(cfg.lags), len(cfg.lags)), order="F",
        )
    for old, native in (
        ("fourier_coef", "periodic_theta"),
        ("trend", "trend_group_values"),
        ("trend_slope", "trend_slope"),
        ("outlier", "outlier_group_values"),
    ):
        if native in variables:
            views[old] = (
                cvxpy.vec(variables[native], order="F")
                if old == "fourier_coef" else variables[native]
            )
    for role, coefficient in variables.items():
        if role.startswith("interaction_") and role.endswith("_coef"):
            pair = role.removeprefix("interaction_").removesuffix("_coef")
            views[f"interaction_coef_{pair}"] = cvxpy.vec(coefficient, order="C")
    return views


def solve_problem(
    problem: cvxpy.Problem,
    solver_config: TsgamSolverConfig,
    *,
    failure_message: str,
) -> None:
    problem.solve(
        solver=solver_config.solver,
        verbose=solver_config.verbose,
        warm_start=solver_config.warm_start,
        **solver_config._solve_kwargs(),
    )
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"{failure_message} Status: {problem.status}.")


def build_single_output_decomposition(
    config: TsgamEstimatorConfig,
    design: _TsgamDesign,
    *,
    knots_by_exog: list[ndarray | None] | None = None,
    trend_period: float | None = None,
    outlier_period: float | None = None,
) -> dict[str, object]:
    """Build the shared SignalDecomp formulation for one output."""
    assert design.y is not None and design.sample_weight is not None
    observed_indices = design.time_indices.astype(int)
    T = int(observed_indices.max()) + 1
    y = np.full(T, np.nan)
    y[observed_indices] = design.y
    sample_weight = np.zeros(T)
    sample_weight[observed_indices] = design.sample_weight
    drivers = np.full((T, design.X_array.shape[1]), np.nan)
    drivers[observed_indices] = design.X_array
    components = [grouped_trend(groups=np.zeros(T, dtype=int), role="intercept")]
    interaction_bases: dict[int, ndarray] = {}
    interaction_parents = {ix for pair in design.interaction_pairs for ix in pair}

    for ix, exog_cfg in enumerate(config.exog_config or []):
        kwargs = {
            "weight": exog_cfg.reg_weight,
            "role": f"exog_{ix}",
            "offsets": tuple(-offset for offset in exog_cfg.lags),
            "lag_smooth_weight": exog_cfg.diff_reg_weight,
        }
        if _is_spline_config(exog_cfg):
            configured_knots = np.asarray(exog_cfg.knots, dtype=float)
            knots = (
                knots_by_exog[ix] if knots_by_exog is not None
                else configured_knots if configured_knots.size else None
            )
            if knots is None and exog_cfg.n_knots is None:
                raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
            component = exog_spline(
                drivers[:, ix],
                n_knots=exog_cfg.n_knots or 10,
                knots=knots,
                **kwargs,
            )
            components.append(component)
            if ix in interaction_parents:
                interaction_bases[ix] = make_spline_basis(
                    drivers[:, ix], component.metadata["knots"]
                )
        else:
            components.append(exog_linear(drivers[:, ix], **kwargs))
            if ix in interaction_parents:
                interaction_bases[ix] = drivers[:, [ix]]

    if config.multi_periodic_config:
        periodic_config = config.multi_periodic_config
        components.append(
            multiperiodic(
                periodic_config.periods,
                num_harmonics=periodic_config.num_harmonics,
                weight=float(np.sqrt(periodic_config.reg_weight)),
                role="periodic",
            )
        )

    for pair_ix, (left_ix, right_ix) in enumerate(design.interaction_pairs):
        assert config.exog_config is not None
        weight = float(
            np.sqrt(
                config.exog_config[left_ix].reg_weight
                * config.exog_config[right_ix].reg_weight
            )
        )
        components.append(
            exog_interaction(
                interaction_bases[left_ix],
                interaction_bases[right_ix],
                weight=weight,
                role=f"interaction_{pair_ix}",
            )
        )

    if trend_period is not None:
        assert config.trend_config is not None
        trend_type = config.trend_config.trend_type.value
        monotonic = (
            "increasing" if trend_type.endswith("increasing") else "decreasing"
        ) if trend_type.startswith("nonlinear") else None
        trend = grouped_trend(
            groups=(np.arange(T) / trend_period).astype(int),
            weight=config.trend_config.reg_weight,
            monotonic=monotonic,
            baseline=0.0,
            role="trend",
        )
        if trend_type == "linear":
            grouped_build = trend.build

            def build_linear_trend(T: int):
                expression, loss, constraints = grouped_build(T)
                slope = cvxpy.Variable(name="trend_slope")
                trend.aux["trend_slope"] = slope
                constraints.append(
                    cvxpy.diff(trend.aux["trend_group_values"]) == slope
                )
                return expression, loss, constraints

            trend.build = build_linear_trend
        components.append(trend)

    if outlier_period is not None:
        assert config.outlier_config is not None
        outlier_groups = (np.arange(T) / outlier_period).astype(int)
        components.append(
            grouped_sparse(
                groups=outlier_groups,
                weight=config.outlier_config.reg_weight
                * np.unique(outlier_groups).size,
                role="outlier",
            )
        )

    component_mask = np.ones(T, dtype=bool)
    for component in components:
        if component.valid_mask is not None:
            component_mask &= component.valid_mask
    effective_weight = np.where(np.isfinite(y) & component_mask, sample_weight, 0.0)
    if not np.any(effective_weight > 0):
        raise ValueError("sample_weight must be positive on at least one valid row.")
    return make_problem(
        y,
        components,
        residual_loss=lambda residual: cvxpy.sum_squares(
            cvxpy.multiply(np.sqrt(effective_weight / effective_weight.sum()), residual)
        ),
    )


def evaluate_single_output_prediction(
    config: TsgamEstimatorConfig,
    design: _TsgamDesign,
    values: Mapping[str, ndarray | float],
    *,
    remove_periodic: bool = False,
    remove_exogenous: bool = False,
) -> ndarray:
    constant_value = float(cast(ndarray, values["intercept_group_values"])[0])
    if np.isnan(constant_value):
        raise ValueError(f"Constant term is None or NaN: {constant_value}")
    predictions = np.full(len(design.timestamps), constant_value)

    if config.exog_config and not remove_exogenous:
        for ix, Hs in enumerate(design.exog_Hs):
            exog_var = design.X_array[:, ix]
            if np.any(np.isnan(exog_var)):
                raise ValueError(
                    f"Exogenous variable {ix} contains NaN values. "
                    f"NaN count: {np.sum(np.isnan(exog_var))} out of {len(exog_var)}"
                )
            suffix = "coef" if _is_spline_config(config.exog_config[ix]) else "beta"
            exog_coef = np.asarray(values[f"exog_{ix}_{suffix}"]).reshape(
                Hs[0].shape[1], len(Hs), order="F"
            )
            if np.any(np.isnan(exog_coef)):
                raise ValueError(f"Exogenous coefficients for variable {ix} contain NaN.")
            for lag_ix, H in enumerate(Hs):
                predictions += np.nan_to_num(H, nan=0.0) @ exog_coef[:, lag_ix]

    if not remove_exogenous and design.interaction_Hs:
        for pair_ix, interaction_H in enumerate(design.interaction_Hs):
            interaction_coef = np.asarray(values[f"interaction_{pair_ix}_coef"]).reshape(-1)
            predictions += np.nan_to_num(interaction_H, nan=0.0) @ interaction_coef

    if config.multi_periodic_config and not remove_periodic:
        assert design.fourier_basis is not None
        fourier_coef = cast(ndarray, values["periodic_theta"])
        fourier_contrib = design.fourier_basis @ fourier_coef
        if np.any(np.isnan(fourier_contrib)):
            raise ValueError(
                f"Fourier contribution contains NaN. F shape: {design.fourier_basis.shape}, "
                f"fourier_coef shape: {fourier_coef.shape}"
            )
        predictions += fourier_contrib

    if np.any(np.isnan(predictions)):
        nan_indices = np.where(np.isnan(predictions))[0]
        raise ValueError(
            f"Predictions contain {len(nan_indices)} NaN values out of {len(predictions)}. "
            f"First few NaN indices: {nan_indices[:10] if len(nan_indices) > 0 else []}. "
            f"Constant value: {constant_value}, "
            f"Time indices range: [{design.time_indices.min():.1f}, "
            f"{design.time_indices.max():.1f}]"
        )
    if not remove_exogenous:
        predictions[~design.valid_mask] = np.nan
    return predictions
