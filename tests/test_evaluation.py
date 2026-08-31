"""Deterministic mathematical tests for ENU trajectory metrics only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import sqrt
import unittest

from backend.evaluation.metrics import (
    EvaluationValidationError,
    TrajectoryPoint,
    evaluate_trajectory,
    horizontal_position_error_m,
    mean_absolute_error,
    position_error_enu_m,
    root_mean_square_error,
    three_dimensional_position_error_m,
    velocity_error_enu_m_s,
)


class EvaluationMetricTests(unittest.TestCase):
    def _point(
        self,
        seconds: float,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float] | None = None,
    ) -> TrajectoryPoint:
        return TrajectoryPoint(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
            position_enu_m=position,
            velocity_enu_m_s=velocity,
        )

    def test_individual_position_and_velocity_errors(self) -> None:
        self.assertEqual(position_error_enu_m((4.0, 6.0, 8.0), (1.0, 2.0, 3.0)), (3.0, 4.0, 5.0))
        self.assertEqual(horizontal_position_error_m((4.0, 6.0, 8.0), (1.0, 2.0, 3.0)), 5.0)
        self.assertAlmostEqual(three_dimensional_position_error_m((4.0, 6.0, 8.0), (1.0, 2.0, 3.0)), sqrt(50.0))
        self.assertEqual(velocity_error_enu_m_s((3.0, 2.0, 1.0), (1.0, 1.0, 1.0)), (2.0, 1.0, 0.0))

    def test_mae_and_rmse(self) -> None:
        self.assertEqual(mean_absolute_error((-1.0, 2.0, -3.0)), 2.0)
        self.assertAlmostEqual(root_mean_square_error((3.0, 4.0)), sqrt(12.5))

    def test_evaluates_timestamp_aligned_trajectory_and_drift(self) -> None:
        estimated = (
            self._point(0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            self._point(1.0, (4.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        )
        reference = (
            self._point(0.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            self._point(1.0, (3.0, 4.0, 0.0), (1.0, 0.0, 0.0)),
        )

        metrics = evaluate_trajectory(estimated, reference)

        self.assertEqual(metrics.position_error_vectors_enu_m, ((0.0, 0.0, 0.0), (1.0, -4.0, 0.0)))
        self.assertEqual(metrics.horizontal_position_errors_m, (0.0, sqrt(17.0)))
        self.assertEqual(metrics.endpoint_drift_m, sqrt(17.0))
        self.assertAlmostEqual(metrics.endpoint_drift_ratio, sqrt(17.0) / 5.0)
        self.assertEqual(metrics.velocity_error_vectors_enu_m_s, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        self.assertEqual(metrics.velocity_error_mae_m_s, 0.5)

    def test_rejects_mismatched_or_invalid_trajectories(self) -> None:
        point = self._point(0.0, (0.0, 0.0, 0.0))
        later_point = self._point(1.0, (0.0, 0.0, 0.0))

        with self.assertRaisesRegex(EvaluationValidationError, "non-empty"):
            evaluate_trajectory((), ())
        with self.assertRaisesRegex(EvaluationValidationError, "matching lengths"):
            evaluate_trajectory((point,), (point, later_point))
        with self.assertRaisesRegex(EvaluationValidationError, "timestamp mismatch"):
            evaluate_trajectory((point,), (later_point,))
        with self.assertRaisesRegex(EvaluationValidationError, "metric values"):
            mean_absolute_error((float("nan"),))


if __name__ == "__main__":
    unittest.main()
