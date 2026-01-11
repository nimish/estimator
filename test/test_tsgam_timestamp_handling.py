# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for timestamp handling in TsgamEstimator.

This test suite verifies that:
1. Timestamps are correctly extracted from DataFrames
2. Frequency validation works correctly
3. Phase alignment is correct between fit and predict
4. Error cases are handled appropriately
"""

import pytest
import numpy as np
import pandas as pd
from tsgam_estimator import (
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiHarmonicConfig,
    TsgamSplineConfig,
    TsgamSolverConfig,
)


@pytest.fixture
def hourly_data():
    """Create hourly data for testing."""
    n_samples = 100
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='h')
    temp = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1, n_samples)
    y = 100 + 2 * temp + 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 2, n_samples)
    return timestamps, temp, y


@pytest.fixture
def basic_config():
    """Create basic config for testing."""
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[2, 1],
        periods=[24, 7 * 24]  # Daily and weekly
    )
    solver_config = TsgamSolverConfig(solver='CLARABEL', verbose=False)

    return TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=None,
        ar_config=None,
        solver_config=solver_config,
        random_state=None,
        debug=False
    )


class TestTimestampExtraction:
    """Test timestamp extraction from different input formats."""

    def test_datetime_index_extraction(self, hourly_data, basic_config):
        """Test extraction from DataFrame with DatetimeIndex."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({'temp': temp}, index=timestamps)

        estimator = TsgamEstimator(config=basic_config)
        extracted_timestamps, X_array = estimator._ensure_timestamp_index(X)

        assert isinstance(extracted_timestamps, pd.DatetimeIndex)
        assert len(extracted_timestamps) == len(timestamps)
        assert extracted_timestamps.equals(timestamps)
        assert X_array.shape == (len(timestamps), 1)

    def test_datetime_column_extraction(self, hourly_data, basic_config):
        """Test extraction from DataFrame with datetime column."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({
            'timestamp': timestamps,
            'temp': temp
        })

        estimator = TsgamEstimator(config=basic_config)
        extracted_timestamps, X_array = estimator._ensure_timestamp_index(X)

        assert isinstance(extracted_timestamps, pd.DatetimeIndex)
        assert len(extracted_timestamps) == len(timestamps)
        assert X_array.shape == (len(timestamps), 1)

    def test_no_timestamps_error(self, hourly_data, basic_config):
        """Test that error is raised when no timestamps found."""
        timestamps, temp, y = hourly_data
        X = np.array(temp).reshape(-1, 1)  # NumPy array without timestamps

        estimator = TsgamEstimator(config=basic_config)
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            estimator._ensure_timestamp_index(X)

    def test_dataframe_no_datetime_error(self, hourly_data, basic_config):
        """Test that error is raised when DataFrame has no datetime."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({'temp': temp})  # No DatetimeIndex or datetime column

        estimator = TsgamEstimator(config=basic_config)
        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            estimator._ensure_timestamp_index(X)


class TestFrequencyValidation:
    """Test frequency validation."""

    def test_correct_frequency(self, hourly_data, basic_config):
        """Test that correct frequency passes validation."""
        timestamps, temp, y = hourly_data

        estimator = TsgamEstimator(config=basic_config)
        # Should not raise
        estimator._validate_frequency(timestamps, 'H')

    def test_wrong_frequency_error(self, hourly_data, basic_config):
        """Test that wrong frequency raises error."""
        timestamps, temp, y = hourly_data

        estimator = TsgamEstimator(config=basic_config)
        with pytest.raises(ValueError, match="frequency"):
            estimator._validate_frequency(timestamps, 'D')  # Daily instead of hourly

    def test_irregular_timestamps_error(self, basic_config):
        """Test that irregular timestamps raise error."""
        # Create irregular timestamps
        timestamps = pd.DatetimeIndex([
            '2020-01-01 00:00:00',
            '2020-01-01 01:00:00',
            '2020-01-01 03:00:00',  # Missing 02:00:00
            '2020-01-01 04:00:00',
        ])

        estimator = TsgamEstimator(config=basic_config)
        with pytest.raises(ValueError, match="frequency"):
            estimator._validate_frequency(timestamps, 'H')

    def test_single_sample_no_error(self, basic_config):
        """Test that single sample doesn't cause error (can't validate)."""
        timestamps = pd.DatetimeIndex(['2020-01-01 00:00:00'])

        estimator = TsgamEstimator(config=basic_config)
        # Should not raise (both 'h' and 'H' should work)
        estimator._validate_frequency(timestamps, 'h')
        estimator._validate_frequency(timestamps, 'H')  # Test backward compatibility


class TestTimestampConversion:
    """Test timestamp to index conversion."""

    def test_timestamps_to_indices(self, hourly_data, basic_config):
        """Test conversion of timestamps to hours since reference."""
        timestamps, temp, y = hourly_data

        estimator = TsgamEstimator(config=basic_config)
        reference = timestamps[0]
        indices = estimator._timestamps_to_indices(timestamps, reference)

        assert len(indices) == len(timestamps)
        assert indices[0] == 0.0  # First timestamp is reference
        assert np.allclose(indices[1], 1.0)  # One hour later
        assert np.allclose(indices[-1], len(timestamps) - 1)  # Last timestamp

    def test_different_reference(self, hourly_data, basic_config):
        """Test conversion with different reference point."""
        timestamps, temp, y = hourly_data

        estimator = TsgamEstimator(config=basic_config)
        reference = timestamps[10]  # Use 10th timestamp as reference
        indices = estimator._timestamps_to_indices(timestamps, reference)

        assert indices[10] == 0.0  # Reference point is zero
        assert indices[9] < 0  # Before reference is negative
        assert indices[11] > 0  # After reference is positive


class TestPhaseCorrectness:
    """Test that phase alignment is correct between fit and predict."""

    def test_continuous_prediction(self, hourly_data, basic_config):
        """Test prediction on continuous period after fit."""
        timestamps, temp, y = hourly_data

        # Split into train and test
        split_idx = 80
        train_timestamps = timestamps[:split_idx]
        test_timestamps = timestamps[split_idx:]

        X_train = pd.DataFrame({'temp': temp[:split_idx]}, index=train_timestamps)
        y_train = y[:split_idx]
        X_test = pd.DataFrame({'temp': temp[split_idx:]}, index=test_timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X_train, y_train)

        # Should not raise
        predictions = estimator.predict(X_test)

        assert len(predictions) == len(test_timestamps)
        assert np.all(np.isfinite(predictions))

    def test_future_prediction(self, hourly_data, basic_config):
        """Test prediction on future period."""
        timestamps, temp, y = hourly_data

        # Train on first 80 samples
        train_timestamps = timestamps[:80]
        X_train = pd.DataFrame({'temp': temp[:80]}, index=train_timestamps)
        y_train = y[:80]

        # Predict on future 20 samples (hours 80-99)
        future_timestamps = pd.date_range(
            start=timestamps[80],
            periods=20,
            freq='h'
        )
        X_test = pd.DataFrame({
            'temp': temp[80:100]  # Use actual temp for testing
        }, index=future_timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X_train, y_train)

        # Should not raise
        predictions = estimator.predict(X_test)

        assert len(predictions) == 20
        assert np.all(np.isfinite(predictions))

    def test_frequency_mismatch_error(self, hourly_data, basic_config):
        """Test that frequency mismatch raises error."""
        timestamps, temp, y = hourly_data

        X_train = pd.DataFrame({'temp': temp[:80]}, index=timestamps[:80])
        y_train = y[:80]

        # Test with daily frequency instead of hourly
        test_timestamps = pd.date_range(
            start=timestamps[80],
            periods=10,
            freq='D'  # Daily instead of hourly
        )
        X_test = pd.DataFrame({'temp': temp[80:90]}, index=test_timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X_train, y_train)

        with pytest.raises(ValueError, match="frequency"):
            estimator.predict(X_test)

    def test_same_period_prediction(self, hourly_data, basic_config):
        """Test prediction on same period as fit (should work)."""
        timestamps, temp, y = hourly_data

        X = pd.DataFrame({'temp': temp}, index=timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X, y)

        # Predict on same data
        predictions = estimator.predict(X)

        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))


class TestBasicFunctionality:
    """Test basic fit and predict functionality with timestamps."""

    def test_fit_with_datetime_index(self, hourly_data, basic_config):
        """Test that fit works with DatetimeIndex."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({'temp': temp}, index=timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X, y)

        assert hasattr(estimator, 'time_reference_')
        assert hasattr(estimator, 'freq_')
        assert hasattr(estimator, 'time_indices_')
        assert estimator.freq_ == 'h'
        assert estimator.time_reference_ == timestamps[0]

    def test_predict_with_datetime_index(self, hourly_data, basic_config):
        """Test that predict works with DatetimeIndex."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({'temp': temp}, index=timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X, y)
        predictions = estimator.predict(X)

        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))

    def test_with_exogenous_variables(self, hourly_data):
        """Test fit and predict with exogenous variables."""
        timestamps, temp, y = hourly_data

        exog_config = [
            TsgamSplineConfig(
                n_knots=5,
                lags=[0],
                reg_weight=1e-4
            )
        ]

        multi_harmonic_config = TsgamMultiHarmonicConfig(
            num_harmonics=[2, 1],
            periods=[24, 7 * 24]
        )

        config = TsgamEstimatorConfig(
            multi_harmonic_config=multi_harmonic_config,
            exog_config=exog_config,
            ar_config=None,
            solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
            random_state=None,
            debug=False
        )

        X = pd.DataFrame({'temp': temp}, index=timestamps)
        estimator = TsgamEstimator(config=config)
        estimator.fit(X, y)

        predictions = estimator.predict(X)
        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))


class TestErrorCases:
    """Test error cases."""

    def test_fit_infers_frequency(self, hourly_data):
        """Test that fit infers frequency from data timestamps."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({'temp': temp}, index=timestamps)

        # Config without freq - frequency is inferred from data
        config = TsgamEstimatorConfig(
            multi_harmonic_config=TsgamMultiHarmonicConfig(
                num_harmonics=[2],
                periods=[24]
            ),
            exog_config=None,
            ar_config=None,
            solver_config=TsgamSolverConfig(solver='CLARABEL', verbose=False),
            random_state=None,
            debug=False
        )

        estimator = TsgamEstimator(config=config)
        # Should work - frequency is inferred from timestamps
        estimator.fit(X, y)
        # Verify frequency was inferred correctly
        assert estimator.freq_ == 'h'

    def test_predict_before_fit_error(self, hourly_data, basic_config):
        """Test that predict raises error if called before fit."""
        timestamps, temp, y = hourly_data
        X = pd.DataFrame({'temp': temp}, index=timestamps)

        estimator = TsgamEstimator(config=basic_config)
        with pytest.raises(Exception):  # Should check for fitted attributes
            estimator.predict(X)

    def test_wrong_freq_in_predict(self, hourly_data, basic_config):
        """Test error when predict frequency doesn't match fit frequency."""
        timestamps, temp, y = hourly_data

        # Use hourly data - frequency will be inferred as 'h'
        X = pd.DataFrame({'temp': temp[:50]}, index=timestamps[:50])

        estimator = TsgamEstimator(config=basic_config)
        # Should work - frequency is inferred from data
        estimator.fit(X, y[:50])

        # Now try to predict with daily frequency data - should fail
        daily_timestamps = pd.date_range(
            start=timestamps[50],
            periods=10,
            freq='D'  # Daily instead of hourly
        )
        X_daily = pd.DataFrame({'temp': temp[50:60]}, index=daily_timestamps)

        with pytest.raises(ValueError, match="frequency"):
            estimator.predict(X_daily)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_sample_prediction(self, hourly_data, basic_config):
        """Test prediction with single sample."""
        timestamps, temp, y = hourly_data

        X_train = pd.DataFrame({'temp': temp[:80]}, index=timestamps[:80])
        y_train = y[:80]
        X_test = pd.DataFrame({'temp': [temp[80]]}, index=[timestamps[80]])

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_test)
        assert len(predictions) == 1
        assert np.isfinite(predictions[0])

    def test_prediction_with_gaps(self, hourly_data, basic_config):
        """Test prediction with gaps in timestamps."""
        timestamps, temp, y = hourly_data

        X_train = pd.DataFrame({'temp': temp[:80]}, index=timestamps[:80])
        y_train = y[:80]

        # Create test data with gaps (every other hour)
        test_timestamps = pd.date_range(
            start=timestamps[80],
            periods=10,
            freq='2h'  # Every 2 hours
        )
        X_test = pd.DataFrame({'temp': temp[80:90]}, index=test_timestamps)

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X_train, y_train)

        # Should raise error due to frequency mismatch
        with pytest.raises(ValueError, match="frequency"):
            estimator.predict(X_test)

    def test_prediction_before_fit_period(self, hourly_data, basic_config):
        """Test prediction on period before fit period."""
        timestamps, temp, y = hourly_data

        # Fit on later period
        X_train = pd.DataFrame({'temp': temp[20:80]}, index=timestamps[20:80])
        y_train = y[20:80]

        # Predict on earlier period
        X_test = pd.DataFrame({'temp': temp[:10]}, index=timestamps[:10])

        estimator = TsgamEstimator(config=basic_config)
        estimator.fit(X_train, y_train)

        # Should work but indices will be negative
        predictions = estimator.predict(X_test)
        assert len(predictions) == 10
        assert np.all(np.isfinite(predictions))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


