# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for notebook-compatible timestamp phase alignment."""

import numpy as np
import pandas as pd
from signaldecomp.basis import make_basis_matrix

from tsgam_estimator._design import _timestamps_to_indices


def test_timestamp_phase_matches_notebook_pattern():
    reference = pd.Timestamp("2020-01-01")
    train_timestamps = pd.date_range(reference, periods=100, freq="1h")
    pred_timestamps = pd.date_range(reference, periods=150, freq="1h")[100:]

    train_indices = _timestamps_to_indices(train_timestamps, reference, "1h")
    pred_indices = _timestamps_to_indices(pred_timestamps, reference, "1h")
    np.testing.assert_array_equal(train_indices, np.arange(100))
    np.testing.assert_array_equal(pred_indices, np.arange(100, 150))

    basis = make_basis_matrix(num_harmonics=[1], periods=[24], length=150)
    np.testing.assert_allclose(
        basis[pred_indices, 1:],
        basis[np.arange(100, 150), 1:],
    )
