"""Tests for NAVIGEN IMU preprocessing utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from backend.data.schemas import IMUSample
from backend.preprocessing import (
    SensorPreprocessingValidationError,
    calculate_sampling_rate_hz,
    calculate_time_differences,
    validate_imu_sequence,
    validate_sampling_rate,
)


class SensorPreprocessingTests(unittest.TestCase):
    def _sample(self, seconds: float) -> IMUSample:
        return IMUSample(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
            accelerometer_x=0.1,
            accelerometer_y=0.2,
            accelerometer_z=9.81,
            gyroscope_x=0.01,
            gyroscope_y=0.02,
            gyroscope_z=0.03,
        )

    def _invalid_sample(self, **attributes: object) -> IMUSample:
        """Construct an invalid sample solely to test defensive preprocessing checks."""

        sample = object.__new__(IMUSample)
        defaults: dict[str, object] = {
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "accelerometer_x": 0.1,
            "accelerometer_y": 0.2,
            "accelerometer_z": 9.81,
            "gyroscope_x": 0.01,
            "gyroscope_y": 0.02,
            "gyroscope_z": 0.03,
        }
        defaults.update(attributes)
        for name, value in defaults.items():
            object.__setattr__(sample, name, value)
        return sample

    def test_accepts_correctly_ordered_timestamps(self) -> None:
        samples = (self._sample(0.0), self._sample(0.1), self._sample(0.2))

        validate_imu_sequence(samples)

    def test_rejects_unsorted_timestamps(self) -> None:
        samples = (self._sample(0.1), self._sample(0.0))

        with self.assertRaisesRegex(SensorPreprocessingValidationError, "earlier"):
            validate_imu_sequence(samples)

    def test_rejects_duplicate_timestamps(self) -> None:
        samples = (self._sample(0.0), self._sample(0.0))

        with self.assertRaisesRegex(SensorPreprocessingValidationError, "duplicate"):
            validate_imu_sequence(samples)

    def test_rejects_invalid_timestamp(self) -> None:
        samples = (self._invalid_sample(timestamp="not-a-timestamp"),)

        with self.assertRaisesRegex(SensorPreprocessingValidationError, "timestamp must be a datetime"):
            validate_imu_sequence(samples)

    def test_calculates_time_differences(self) -> None:
        samples = (self._sample(0.0), self._sample(0.1), self._sample(0.3))

        self.assertEqual(calculate_time_differences(samples), (0.1, 0.2))

    def test_rejects_non_positive_time_differences(self) -> None:
        with self.assertRaisesRegex(SensorPreprocessingValidationError, "finite positive"):
            calculate_sampling_rate_hz((0.1, 0.0))

    def test_validates_sampling_rate_bounds(self) -> None:
        samples = (self._sample(0.0), self._sample(0.1), self._sample(0.2))

        self.assertEqual(validate_sampling_rate(samples, minimum_hz=9.0, maximum_hz=11.0), 10.0)
        with self.assertRaisesRegex(SensorPreprocessingValidationError, "below minimum_hz"):
            validate_sampling_rate(samples, minimum_hz=11.0)

    def test_rejects_non_finite_sensor_values(self) -> None:
        samples = (self._invalid_sample(accelerometer_x=float("nan")),)

        with self.assertRaisesRegex(SensorPreprocessingValidationError, "finite numeric"):
            validate_imu_sequence(samples)


if __name__ == "__main__":
    unittest.main()
