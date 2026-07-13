# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy
import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix

from ._design import (
    _TsgamDesign,
    _interaction_contribution_from_blocks,
    _make_regularization_matrix,
)

if TYPE_CHECKING:
    from ._estimator import TsgamEstimatorConfig, TsgamSolverConfig


def make_fourier_regularization_matrix(config: TsgamEstimatorConfig) -> spmatrix:
    if config.multi_periodic_config is None:
        raise ValueError("multi_periodic_config is required for Fourier regularization.")
    return _make_regularization_matrix(
        num_harmonics=config.multi_periodic_config.num_harmonics,
        weight=1.0,
        periods=config.multi_periodic_config.periods,
        drop_constant=True,
    )


def weighted_squared_loss(
    y: ndarray,
    model_term: cvxpy.Expression,
    sample_weight: ndarray,
) -> cvxpy.Expression:
    return cvxpy.sum_squares(
        cvxpy.multiply(np.sqrt(sample_weight), y - model_term)
    ) / np.sum(sample_weight)


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


def make_single_output_standard_variables(
    config: TsgamEstimatorConfig,
    design: _TsgamDesign,
) -> tuple[dict[str, cvxpy.Variable], cvxpy.Expression]:
    variables: dict[str, cvxpy.Variable] = {
        "constant": cvxpy.Variable(),
    }
    regularization_term: cvxpy.Expression = cvxpy.Constant(0.0)

    if config.exog_config:
        for ix, exog_cfg in enumerate(config.exog_config):
            basis_dim = design.exog_Hs[ix][0].shape[1]
            num_lags = len(exog_cfg.lags)
            exog_coef = cvxpy.Variable((basis_dim, num_lags))
            variables[f"exog_coef_{ix}"] = exog_coef
            regularization_term += exog_cfg.reg_weight * cvxpy.sum_squares(exog_coef)
            if num_lags > 1:
                regularization_term += (
                    exog_cfg.diff_reg_weight
                    * cvxpy.sum_squares(cvxpy.diff(exog_coef, axis=1))
                )

    if config.multi_periodic_config:
        assert design.fourier_basis is not None
        fourier_coef = cvxpy.Variable(design.fourier_basis.shape[1])
        variables["fourier_coef"] = fourier_coef
        regularization_term += (
            config.multi_periodic_config.reg_weight
            * cvxpy.sum_squares(
                make_fourier_regularization_matrix(config) @ fourier_coef
            )
        )

    if config.exog_config:
        for pair_ix, (left_ix, right_ix) in enumerate(design.interaction_pairs):
            interaction_coef = cvxpy.Variable(design.interaction_Hs[pair_ix].shape[1])
            variables[f"interaction_coef_{pair_ix}"] = interaction_coef
            left_cfg = config.exog_config[left_ix]
            right_cfg = config.exog_config[right_ix]
            interaction_weight = float(np.sqrt(left_cfg.reg_weight * right_cfg.reg_weight))
            regularization_term += interaction_weight * cvxpy.sum_squares(
                interaction_coef
            )

    return variables, regularization_term


def single_output_prediction_expression(
    config: TsgamEstimatorConfig,
    design: _TsgamDesign,
    variables: dict[str, cvxpy.Variable],
    valid_mask: ndarray,
) -> cvxpy.Expression:
    model_term = variables["constant"]
    if config.exog_config:
        for ix, Hs in enumerate(design.exog_Hs):
            exog_coef = variables[f"exog_coef_{ix}"]
            model_term += cvxpy.sum(
                expr=[
                    H[valid_mask] @ exog_coef[:, lag_ix]
                    for lag_ix, H in enumerate(Hs)
                ]
            )
    if config.multi_periodic_config:
        assert design.fourier_basis is not None
        model_term += design.fourier_basis[valid_mask] @ variables["fourier_coef"]
    for pair_ix, interaction_H in enumerate(design.interaction_Hs):
        model_term += (
            interaction_H[valid_mask] @ variables[f"interaction_coef_{pair_ix}"]
        )
    return model_term


def evaluate_single_output_prediction(
    config: TsgamEstimatorConfig,
    design: _TsgamDesign,
    variables: dict[str, cvxpy.Variable],
    *,
    remove_periodic: bool = False,
    remove_exogenous: bool = False,
) -> ndarray:
    constant_value = variables["constant"].value
    if constant_value is None or np.isnan(constant_value):
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
            exog_coef = variables[f"exog_coef_{ix}"].value
            if exog_coef is None:
                raise ValueError(
                    f"Exogenous coefficients for variable {ix} are None. "
                    "Model may not have converged."
                )
            if np.any(np.isnan(exog_coef)):
                raise ValueError(f"Exogenous coefficients for variable {ix} contain NaN.")
            for lag_ix, H in enumerate(Hs):
                predictions += np.nan_to_num(H, nan=0.0) @ exog_coef[:, lag_ix]

    if not remove_exogenous and design.interaction_Hs:
        exog_config = config.exog_config
        assert exog_config is not None
        for pair_ix, interaction_H in enumerate(design.interaction_Hs):
            interaction_coef = variables[f"interaction_coef_{pair_ix}"].value
            if interaction_coef is None:
                raise ValueError(
                    f"Interaction coefficients for pair {pair_ix} are None. "
                    "Model may not have converged."
                )
            if np.any(np.isnan(interaction_coef)):
                raise ValueError(
                    f"Interaction coefficients for pair {pair_ix} contain NaN."
                )
            interaction_pred = _interaction_contribution_from_blocks(
                design.exog_Hs[design.interaction_pairs[pair_ix][0]][
                    exog_config[design.interaction_pairs[pair_ix][0]].lags.index(0)
                ],
                design.exog_Hs[design.interaction_pairs[pair_ix][1]][
                    exog_config[design.interaction_pairs[pair_ix][1]].lags.index(0)
                ],
                interaction_coef,
                nan_to_zero=True,
            )
            predictions += interaction_pred

    if config.multi_periodic_config and not remove_periodic:
        assert design.fourier_basis is not None
        fourier_coef = variables["fourier_coef"].value
        if fourier_coef is None:
            raise ValueError("Fourier coefficients are None. Model may not have converged.")
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
            f"Constant value: {variables['constant'].value}, "
            f"Time indices range: [{design.time_indices.min():.1f}, "
            f"{design.time_indices.max():.1f}]"
        )
    return predictions
