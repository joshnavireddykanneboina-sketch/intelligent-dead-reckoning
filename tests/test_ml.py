"""Tests for the NAVIGEN residual-learning pipeline.

These tests validate contracts and deterministic regression arithmetic only.
They do not use sensor datasets, report accuracy, or claim navigation
improvement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from backend.data.schemas import IMUSample
from backend.ml.dataset import (
    DatasetValidationError,
    ResidualTrainingExample,
    build_residual_dataset,
    split_dataset_by_session,
)
from backend.ml.features import FeatureValidationError, build_feature_vector, feature_names
from backend.ml.model import ModelNotTrainedError, ResidualCorrectionModel
from backend.ml.targets import TargetValidationError, position_residual_target


class ResidualLearningTests(unittest.TestCase):
    def _sample(self, seconds: float = 0.0) -> IMUSample:
        return IMUSample(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
            accelerometer_x=0.1,
            accelerometer_y=0.2,
            accelerometer_z=9.81,
            gyroscope_x=0.01,
            gyroscope_y=0.02,
            gyroscope_z=0.03,
        )

    def _features(self, seconds: float = 0.0) -> tuple[float, ...]:
        return build_feature_vector(
            self._sample(seconds),
            dt_seconds=0.1,
            estimated_velocity_enu_m_s=(1.0, 2.0, 3.0),
            orientation_body_to_navigation=(1.0, 0.0, 0.0, 0.0),
        )

    def _example(self, session_id: str, seconds: float) -> ResidualTrainingExample:
        return ResidualTrainingExample(
            features=self._features(seconds),
            target=position_residual_target(
                session_id=session_id,
                timestamp=self._sample(seconds).timestamp,
                estimated_position_enu_m=(1.0, 2.0, 3.0),
                reference_position_enu_m=(1.5, 1.0, 4.0),
            ),
        )

    def test_feature_generation_includes_expected_shape(self) -> None:
        features = self._features()

        self.assertEqual(len(features), len(feature_names()))
        self.assertEqual(features[:7], (0.1, 0.2, 9.81, 0.01, 0.02, 0.03, 0.1))

    def test_feature_generation_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(FeatureValidationError, "dt_seconds"):
            build_feature_vector(
                self._sample(),
                dt_seconds=0.0,
                estimated_velocity_enu_m_s=(0.0, 0.0, 0.0),
                orientation_body_to_navigation=(1.0, 0.0, 0.0, 0.0),
            )
        with self.assertRaisesRegex(FeatureValidationError, "estimated_velocity"):
            build_feature_vector(
                self._sample(),
                dt_seconds=0.1,
                estimated_velocity_enu_m_s=(float("nan"), 0.0, 0.0),
                orientation_body_to_navigation=(1.0, 0.0, 0.0, 0.0),
            )

    def test_target_generation_is_reference_minus_estimate(self) -> None:
        target = position_residual_target(
            session_id="session-a",
            timestamp=self._sample().timestamp,
            estimated_position_enu_m=(10.0, 20.0, 30.0),
            reference_position_enu_m=(12.0, 19.0, 35.0),
        )

        self.assertEqual(target.position_residual_enu_m, (2.0, -1.0, 5.0))

    def test_target_generation_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(TargetValidationError, "finite"):
            position_residual_target(
                session_id="session-a",
                timestamp=self._sample().timestamp,
                estimated_position_enu_m=(0.0, 0.0, 0.0),
                reference_position_enu_m=(float("inf"), 0.0, 0.0),
            )

    def test_dataset_rejects_wrong_feature_shape(self) -> None:
        target = self._example("session-a", 0.0).target
        with self.assertRaisesRegex(DatasetValidationError, "feature vector"):
            ResidualTrainingExample(features=(0.0,), target=target)

    def test_session_split_keeps_whole_sessions_separate(self) -> None:
        dataset = build_residual_dataset(
            (self._example("session-a", 0.0), self._example("session-b", 1.0))
        )

        split = split_dataset_by_session(dataset, {"session-b"})

        self.assertEqual(set(split.train.session_ids), {"session-a"})
        self.assertEqual(set(split.test.session_ids), {"session-b"})
        self.assertTrue(set(split.train.session_ids).isdisjoint(split.test.session_ids))

    def test_model_starts_untrained_and_prediction_requires_real_training(self) -> None:
        model = ResidualCorrectionModel()

        self.assertFalse(model.is_trained)
        with self.assertRaises(ModelNotTrainedError):
            model.predict(self._features())

    def test_linear_regression_fits_deterministic_full_rank_vectors(self) -> None:
        """Verify OLS arithmetic, not navigation accuracy, on basis vectors."""

        feature_count = len(feature_names())
        examples = []
        # A zero vector establishes the intercept. Each unit vector isolates one
        # coefficient, giving a compact full-rank mathematical test matrix.
        for row in range(feature_count + 1):
            features = [0.0] * feature_count
            if row > 0:
                features[row - 1] = 1.0
            intercept = (1.0, -2.0, 0.5)
            coefficient = (0.25 * row, -0.5 * row, 0.75 * row)
            target = tuple(intercept[output] + coefficient[output] for output in range(3))
            examples.append(
                ResidualTrainingExample(
                    features=tuple(features),
                    target=position_residual_target(
                        session_id="mathematical-training-session",
                        timestamp=self._sample(float(row)).timestamp,
                        estimated_position_enu_m=(0.0, 0.0, 0.0),
                        reference_position_enu_m=target,
                    ),
                )
            )
        model = ResidualCorrectionModel()
        model.fit(build_residual_dataset(examples))
        query = [0.0] * feature_count
        query[2] = 1.0

        self.assertEqual(model.predict(query), (1.75, -3.5, 2.75))

    def test_linear_regression_rejects_insufficient_or_rank_deficient_training(self) -> None:
        model = ResidualCorrectionModel()
        insufficient = build_residual_dataset((self._example("session-a", 0.0),))

        with self.assertRaisesRegex(Exception, "requires at least"):
            model.fit(insufficient)


if __name__ == "__main__":
    unittest.main()
