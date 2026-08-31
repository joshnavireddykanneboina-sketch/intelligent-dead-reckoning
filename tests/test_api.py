"""HTTP-level tests for NAVIGEN's available navigation API."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.api.app import create_app


class NavigationAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def _valid_payload(self) -> dict[str, object]:
        return {
            "imu_samples": [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "accelerometer_x": 0.0,
                    "accelerometer_y": 0.0,
                    "accelerometer_z": 9.80665,
                    "gyroscope_x": 0.0,
                    "gyroscope_y": 0.0,
                    "gyroscope_z": 0.0,
                },
                {
                    "timestamp": "2026-01-01T00:00:01Z",
                    "accelerometer_x": 0.0,
                    "accelerometer_y": 0.0,
                    "accelerometer_z": 9.80665,
                    "gyroscope_x": 0.0,
                    "gyroscope_y": 0.0,
                    "gyroscope_z": 0.0,
                },
            ]
        }

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "navigen"})

    def test_processes_valid_sensor_input_and_returns_actual_pipeline_structure(self) -> None:
        response = self.client.post("/navigation/process", json=self._valid_payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["ml_correction"], {"available": False, "applied": False})
        self.assertEqual(len(body["baseline_states"]), 2)
        self.assertEqual(len(body["fused_states"]), 2)
        self.assertEqual(body["fused_states"][1]["position_enu_m"], [0.0, 0.0, 0.0])
        self.assertIn("error_covariance", body["fused_states"][0])

    def test_rejects_invalid_sensor_input(self) -> None:
        payload = self._valid_payload()
        payload["imu_samples"][1]["timestamp"] = "2026-01-01T00:00:00Z"

        response = self.client.post("/navigation/process", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("duplicate timestamp", response.json()["detail"])

    def test_returns_request_validation_error_for_missing_sensor_field(self) -> None:
        payload = self._valid_payload()
        del payload["imu_samples"][0]["gyroscope_z"]

        response = self.client.post("/navigation/process", json=payload)

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
