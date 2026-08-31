"""Tests for NAVIGEN gyroscope-only quaternion orientation propagation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import cos, pi, sin, sqrt
import unittest

from backend.data.schemas import IMUSample
from backend.navigation.orientation import (
    OrientationValidationError,
    identity_quaternion,
    integrate_angular_velocity,
    multiply_quaternions,
    normalize_quaternion,
    propagate_orientation,
    propagate_orientation_sequence,
)


class OrientationTests(unittest.TestCase):
    def _sample(self, seconds: float, *, gyro_z: float = 0.0) -> IMUSample:
        return IMUSample(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
            accelerometer_x=0.0,
            accelerometer_y=0.0,
            accelerometer_z=9.81,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=gyro_z,
        )

    def _invalid_sample(self, **attributes: object) -> IMUSample:
        """Construct invalid data only to verify defensive validation behavior."""

        sample = object.__new__(IMUSample)
        defaults: dict[str, object] = {
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "accelerometer_x": 0.0,
            "accelerometer_y": 0.0,
            "accelerometer_z": 9.81,
            "gyroscope_x": 0.0,
            "gyroscope_y": 0.0,
            "gyroscope_z": 0.0,
        }
        defaults.update(attributes)
        for name, value in defaults.items():
            object.__setattr__(sample, name, value)
        return sample

    def assertQuaternionAlmostEqual(self, actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
        for actual_component, expected_component in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_component, expected_component, places=12)

    def test_identity_quaternion(self) -> None:
        self.assertEqual(identity_quaternion(), (1.0, 0.0, 0.0, 0.0))

    def test_normalizes_quaternion(self) -> None:
        self.assertQuaternionAlmostEqual(
            normalize_quaternion((2.0, 0.0, 0.0, 0.0)),
            identity_quaternion(),
        )

    def test_multiplies_quaternions(self) -> None:
        half_turn_x = (0.0, 1.0, 0.0, 0.0)
        half_turn_y = (0.0, 0.0, 1.0, 0.0)

        self.assertQuaternionAlmostEqual(
            multiply_quaternions(half_turn_x, half_turn_y),
            (0.0, 0.0, 0.0, 1.0),
        )

    def test_zero_angular_velocity_preserves_orientation(self) -> None:
        initial_orientation = normalize_quaternion((1.0, 1.0, 0.0, 0.0))

        propagated = propagate_orientation(initial_orientation, self._sample(0.0), 0.25)

        self.assertQuaternionAlmostEqual(propagated, initial_orientation)

    def test_constant_angular_velocity_produces_expected_rotation(self) -> None:
        delta_orientation = integrate_angular_velocity((0.0, 0.0, pi / 2.0), 1.0)
        root_half = sqrt(0.5)

        self.assertQuaternionAlmostEqual(delta_orientation, (root_half, 0.0, 0.0, root_half))

    def test_propagates_multiple_imu_samples(self) -> None:
        samples = (
            self._sample(0.0, gyro_z=1.0),
            self._sample(1.0, gyro_z=1.0),
            self._sample(2.0, gyro_z=1.0),
        )

        orientations = propagate_orientation_sequence(samples)

        self.assertEqual(len(orientations), 3)
        self.assertQuaternionAlmostEqual(orientations[0], identity_quaternion())
        self.assertQuaternionAlmostEqual(
            orientations[1],
            (cos(0.5), 0.0, 0.0, sin(0.5)),
        )
        self.assertQuaternionAlmostEqual(
            orientations[2],
            (cos(1.0), 0.0, 0.0, sin(1.0)),
        )

    def test_rejects_invalid_dt(self) -> None:
        with self.assertRaisesRegex(OrientationValidationError, "finite positive"):
            integrate_angular_velocity((0.0, 0.0, 1.0), 0.0)

    def test_rejects_non_finite_quaternion_and_gyro_values(self) -> None:
        with self.assertRaisesRegex(OrientationValidationError, "finite numeric"):
            normalize_quaternion((float("nan"), 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(OrientationValidationError, "invalid IMU sample"):
            propagate_orientation(identity_quaternion(), self._invalid_sample(gyroscope_z=float("inf")), 0.1)


if __name__ == "__main__":
    unittest.main()
