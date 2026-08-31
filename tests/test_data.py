"""Tests for NAVIGEN's recorded sensor data layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.data import (
    SensorDataValidationError,
    load_gnss_csv,
    load_imu_csv,
    load_sensor_session,
)


class SensorDataLoaderTests(unittest.TestCase):
    def _write_csv(self, directory: Path, name: str, content: str) -> Path:
        path = directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_valid_imu_and_gnss_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            imu_path = self._write_csv(
                directory,
                "imu.csv",
                "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z\n"
                "2026-01-01T00:00:00Z,0.1,0.2,9.81,0.01,0.02,0.03\n",
            )
            gnss_path = self._write_csv(
                directory,
                "gnss.csv",
                "timestamp,latitude,longitude,altitude,accuracy\n"
                "2026-01-01T00:00:00+00:00,28.6139,77.2090,216.0,4.5\n",
            )

            session = load_sensor_session(imu_path, gnss_path, session_id="recording-001")

            self.assertEqual(session.session_id, "recording-001")
            self.assertEqual(len(session.imu_samples), 1)
            self.assertEqual(len(session.gnss_samples), 1)
            self.assertEqual(session.imu_samples[0].accelerometer_z, 9.81)
            self.assertEqual(session.gnss_samples[0].accuracy, 4.5)

    def test_accepts_gnss_without_optional_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "gnss.csv",
                "timestamp,latitude,longitude,altitude\n"
                "2026-01-01T00:00:00Z,28.6139,77.2090,216.0\n",
            )

            samples = load_gnss_csv(path)

            self.assertEqual(samples[0].accuracy, None)

    def test_rejects_missing_required_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "imu.csv",
                "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y\n"
                "2026-01-01T00:00:00Z,0,0,9.81,0,0\n",
            )

            with self.assertRaisesRegex(SensorDataValidationError, "gyroscope_z"):
                load_imu_csv(path)

    def test_rejects_missing_required_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "imu.csv",
                "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z\n"
                "2026-01-01T00:00:00Z,,0,9.81,0,0,0\n",
            )

            with self.assertRaisesRegex(SensorDataValidationError, "accelerometer_x"):
                load_imu_csv(path)

    def test_rejects_non_numeric_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "imu.csv",
                "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z\n"
                "2026-01-01T00:00:00Z,fast,0,9.81,0,0,0\n",
            )

            with self.assertRaisesRegex(SensorDataValidationError, "must be numeric"):
                load_imu_csv(path)

    def test_rejects_timestamp_without_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "imu.csv",
                "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z\n"
                "2026-01-01T00:00:00,0,0,9.81,0,0,0\n",
            )

            with self.assertRaisesRegex(SensorDataValidationError, "UTC offset"):
                load_imu_csv(path)

    def test_rejects_out_of_range_gnss_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "gnss.csv",
                "timestamp,latitude,longitude,altitude\n"
                "2026-01-01T00:00:00Z,91.0,77.2090,216.0\n",
            )

            with self.assertRaisesRegex(SensorDataValidationError, "latitude must be between"):
                load_gnss_csv(path)

    def test_rejects_non_finite_numeric_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                Path(temporary_directory),
                "imu.csv",
                "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z\n"
                "2026-01-01T00:00:00Z,nan,0,9.81,0,0,0\n",
            )

            with self.assertRaisesRegex(SensorDataValidationError, "finite number"):
                load_imu_csv(path)


if __name__ == "__main__":
    unittest.main()
