"""Tests for NAVIGEN's baseline inertial dead-reckoning pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import pi, sqrt
import unittest

from backend.data.schemas import IMUSample
from backend.navigation.dead_reckoning import (
    DeadReckoningValidationError,
    propagate_inertial_trajectory,
)


class DeadReckoningTests(unittest.TestCase):
    gravity = 9.80665

    def _sample(
        self,
        seconds: float,
        *,
        acceleration: tuple[float, float, float] = (0.0, 0.0, 9.80665),
        gyro_z: float = 0.0,
    ) -> IMUSample:
        return IMUSample(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
            accelerometer_x=acceleration[0],
            accelerometer_y=acceleration[1],
            accelerometer_z=acceleration[2],
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=gyro_z,
        )

    def assertVectorAlmostEqual(
        self, actual: tuple[float, float, float], expected: tuple[float, float, float]
    ) -> None:
        for actual_component, expected_component in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_component, expected_component, places=12)

    def test_stationary_case_has_no_motion_after_gravity_compensation(self) -> None:
        states = propagate_inertial_trajectory((self._sample(0.0), self._sample(1.0)))

        self.assertVectorAlmostEqual(states[-1].position_enu_m, (0.0, 0.0, 0.0))
        self.assertVectorAlmostEqual(states[-1].velocity_enu_m_s, (0.0, 0.0, 0.0))

    def test_zero_specific_force_includes_navigation_frame_gravity(self) -> None:
        zero_force = (0.0, 0.0, 0.0)
        states = propagate_inertial_trajectory(
            (self._sample(0.0, acceleration=zero_force), self._sample(1.0, acceleration=zero_force)),
        )

        self.assertVectorAlmostEqual(states[-1].position_enu_m, (0.0, 0.0, -0.5 * self.gravity))
        self.assertVectorAlmostEqual(states[-1].velocity_enu_m_s, (0.0, 0.0, -self.gravity))

    def test_constant_acceleration_integrates_velocity_and_position(self) -> None:
        samples = (
            self._sample(0.0, acceleration=(2.0, 0.0, self.gravity)),
            self._sample(3.0, acceleration=(2.0, 0.0, self.gravity)),
        )

        states = propagate_inertial_trajectory(samples)

        self.assertVectorAlmostEqual(states[-1].velocity_enu_m_s, (6.0, 0.0, 0.0))
        self.assertVectorAlmostEqual(states[-1].position_enu_m, (9.0, 0.0, 0.0))

    def test_gravity_compensation_removes_stationary_specific_force(self) -> None:
        states = propagate_inertial_trajectory((self._sample(0.0), self._sample(2.0)))

        self.assertVectorAlmostEqual(states[-1].velocity_enu_m_s, (0.0, 0.0, 0.0))

    def test_known_orientation_rotates_body_acceleration_into_navigation_frame(self) -> None:
        quarter_turn_z = (sqrt(0.5), 0.0, 0.0, sqrt(0.5))
        samples = (
            self._sample(0.0, acceleration=(1.0, 0.0, self.gravity)),
            self._sample(1.0, acceleration=(1.0, 0.0, self.gravity)),
        )

        states = propagate_inertial_trajectory(
            samples,
            initial_orientation_body_to_navigation=quarter_turn_z,
        )

        self.assertVectorAlmostEqual(states[-1].velocity_enu_m_s, (0.0, 1.0, 0.0))
        self.assertVectorAlmostEqual(states[-1].position_enu_m, (0.0, 0.5, 0.0))

    def test_uses_configured_initial_velocity_and_position(self) -> None:
        states = propagate_inertial_trajectory(
            (self._sample(0.0), self._sample(2.0)),
            initial_position_enu_m=(1.0, 2.0, 3.0),
            initial_velocity_enu_m_s=(4.0, 0.0, -1.0),
        )

        self.assertVectorAlmostEqual(states[-1].position_enu_m, (9.0, 2.0, 1.0))
        self.assertVectorAlmostEqual(states[-1].velocity_enu_m_s, (4.0, 0.0, -1.0))

    def test_rejects_invalid_inputs(self) -> None:
        valid_samples = (self._sample(0.0), self._sample(1.0))

        with self.assertRaisesRegex(DeadReckoningValidationError, "initial_position"):
            propagate_inertial_trajectory(valid_samples, initial_position_enu_m=(0.0, float("nan"), 0.0))
        with self.assertRaisesRegex(DeadReckoningValidationError, "gravity_m_s2"):
            propagate_inertial_trajectory(valid_samples, gravity_m_s2=-1.0)
        with self.assertRaisesRegex(DeadReckoningValidationError, "invalid inertial input"):
            propagate_inertial_trajectory(
                valid_samples,
                initial_orientation_body_to_navigation=(0.0, 0.0, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
