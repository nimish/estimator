# Copyright (c) 2026 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from spcqe import make_regularization_matrix

from tsgam_estimator._design import _make_regularization_matrix


@pytest.mark.parametrize("trend", [False, True])
def test_regularization_wrapper_matches_spcqe(trend: bool):
    expected = make_regularization_matrix(
        num_harmonics=[2, 1],
        weight=0.5,
        periods=[24.0, 168.0],
        trend=trend,
    )

    actual = _make_regularization_matrix(
        num_harmonics=[2, 1],
        weight=0.5,
        periods=[24.0, 168.0],
        trend=trend,
    )

    np.testing.assert_array_equal(actual.toarray(), expected.toarray())


@pytest.mark.parametrize("trend", [False, True])
def test_regularization_wrapper_can_remove_only_the_intercept(trend: bool):
    full_matrix = make_regularization_matrix(
        num_harmonics=[2, 1],
        weight=0.5,
        periods=[24.0, 168.0],
        trend=trend,
    ).toarray()

    actual = _make_regularization_matrix(
        num_harmonics=[2, 1],
        weight=0.5,
        periods=[24.0, 168.0],
        trend=trend,
        drop_constant=True,
    )

    np.testing.assert_array_equal(actual.toarray(), full_matrix[1:, 1:])
