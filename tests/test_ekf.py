"""Tests for NAVIGEN's error-state extended Kalman filter."""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
import unittest

from backend.data.schemas import GNSSSample, IMUSample
from backend.fusion.ekf import EKFValidationError, ErrorStateEKF


class ErrorStateEKFTests(unittest.TestCase):
    def _imu(
        self,
        *,
        acceleration: tuple[float, float, float] = (0.0, 0.0, 9.80665),
        gyro_z: float = 0.0,
    ) -> IMUSample:
        return IMUSample(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            accelerometer_x=acceleration[0],
            accelerometer_y=acceleration[1],
            accelerometer_z=acceleration[2],
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=gyro_z,
        )

    def test_initialization_exposes_nominal_state_and_uncertainty(self) -> None:
        ekf = ErrorStateEKF(initial_position_enu_m=(1.0, 2.0, 3.0))

        self.assertEqual(ekf.state.position_enu_m, (1.0, 2.0, 3.0))
        self.assertEqual(ekf.state.orientation_body_to_navigation, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(len(ekf.state.error_covariance), 15)
        self.assertEqual(len(ekf.state.error_covariance[0]), 15)

    def test_prediction_propagates_nominal_state(self) -> None:
        ekf = ErrorStateEKF()

        state = ekf.predict(self._imu(acceleration=(2.0, 0.0, 9.80665)), 2.0)

        self.assertAlmostEqual(state.velocity_enu_m_s[0], 4.0)
        self.assertAlmostEqual(state.position_enu_m[0], 4.0)

    def test_prediction_propagates_covariance_symmetrically(self) -> None:
        ekf = ErrorStateEKF()
        initial_position_variance = ekf.state.error_covariance[0][0]

        state = ekf.predict(self._imu(), 1.0)

        self.assertGreater(state.error_covariance[0][0], initial_position_variance)
        for row in range(15):
            for column in range(15):
                self.assertAlmostEqual(state.error_covariance[row][column], state.error_covariance[column][row])

    def test_gnss_update_returns_innovation_and_corrects_position(self) -> None:
        ekf = ErrorStateEKF(initial_position_enu_m=(10.0, 0.0, 0.0))
        measurement_covariance = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        update = ekf.update_gnss_position((0.0, 0.0, 0.0), measurement_covariance)

        self.assertEqual(update.innovation_enu_m, (-10.0, 0.0, 0.0))
        self.assertLess(abs(update.state.position_enu_m[0]), 10.0)
        self.assertEqual(len(update.kalman_gain), 15)
        self.assertLess(update.state.error_covariance[0][0], 100.0)

    def test_update_applies_attitude_error_consistently(self) -> None:
        covariance = [[0.0 for _ in range(15)] for _ in range(15)]
        for index in range(15):
            covariance[index][index] = 1.0
        covariance[6][0] = covariance[0][6] = 0.1
        ekf = ErrorStateEKF(initial_covariance=covariance)
        measurement_covariance = ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01))

        update = ekf.update_gnss_position((1.0, 0.0, 0.0), measurement_covariance)

        self.assertNotEqual(update.state.orientation_body_to_navigation, (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(sum(component * component for component in update.state.orientation_body_to_navigation), 1.0)

    def test_update_with_gnss_schema_uses_accuracy_when_covariance_is_omitted(self) -> None:
        ekf = ErrorStateEKF()
        sample = GNSSSample(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            latitude=28.6139,
            longitude=77.2090,
            altitude=216.0,
            accuracy=2.0,
        )

        update = ekf.update_gnss_sample(sample, (1.0, 2.0, 3.0))

        self.assertEqual(update.innovation_enu_m, (1.0, 2.0, 3.0))
        self.assertEqual(update.state.timestamp, sample.timestamp)

    def test_rejects_invalid_measurements_and_covariances(self) -> None:
        ekf = ErrorStateEKF()
        with self.assertRaisesRegex(EKFValidationError, "GNSS position"):
            ekf.update_gnss_position((float("nan"), 0.0, 0.0), ((1.0, 0.0, 0.0),) * 3)
        with self.assertRaisesRegex(EKFValidationError, "symmetric"):
            ekf.update_gnss_position(
                (0.0, 0.0, 0.0),
                ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            )
        with self.assertRaisesRegex(EKFValidationError, "must be finite"):
            ekf.update_gnss_position(
                (0.0, 0.0, 0.0),
                ((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            )
        with self.assertRaisesRegex(EKFValidationError, "dt_seconds"):
            ekf.predict(self._imu(), 0.0)

    def test_rejects_invalid_initial_covariance(self) -> None:
        covariance = [[0.0 for _ in range(15)] for _ in range(15)]
        covariance[0][0] = -1.0

        with self.assertRaisesRegex(EKFValidationError, "diagonal"):
            ErrorStateEKF(initial_covariance=covariance)


if __name__ == "__main__":
    unittest.main()
