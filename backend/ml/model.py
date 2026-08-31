"""Explicit standard-library linear residual-correction model.

The model predicts an ENU position residual to be fused later as an uncertain
measurement. It does not output an absolute position and does not train at
module import time.

Mathematical method
-------------------
For feature row ``x`` (with a leading intercept of 1), the three residual
components are fitted independently but solved together as:
``B = (X^T X)^-1 X^T Y``. The normal-equation system is solved with Gaussian
elimination and partial pivoting. This is deliberately small and explainable,
but it requires a full-rank real training matrix and is not a replacement for
careful data collection, normalization, or evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .dataset import ResidualDataset
from .features import feature_names


class ModelNotTrainedError(RuntimeError):
    """Raised when inference is requested before an explicit fit call."""


class ModelValidationError(ValueError):
    """Raised when model features or training data are invalid."""


class ResidualCorrectionModel:
    """Multi-output ordinary least-squares regression for ENU residuals."""

    def __init__(self) -> None:
        self._coefficients: tuple[tuple[float, float, float], ...] | None = None

    @property
    def is_trained(self) -> bool:
        """Whether ``fit`` has completed successfully in this process."""

        return self._coefficients is not None

    def fit(self, training_dataset: ResidualDataset) -> None:
        """Fit only on a caller-provided, session-separated training dataset.

        Evaluation/session splitting is intentionally external and must be done
        with ``split_dataset_by_session`` before fitting.
        """

        if not isinstance(training_dataset, ResidualDataset):
            raise ModelValidationError("training_dataset must be a ResidualDataset.")
        feature_count = len(feature_names())
        minimum_examples = feature_count + 1
        if len(training_dataset.features) < minimum_examples:
            raise ModelValidationError(
                "ordinary least squares requires at least "
                f"{minimum_examples} examples for {feature_count} features plus an intercept."
            )
        design_matrix = tuple((1.0, *features) for features in training_dataset.features)
        normal_matrix = _normal_matrix(design_matrix)
        normal_targets = _normal_targets(design_matrix, training_dataset.targets)
        self._coefficients = _solve_linear_system(normal_matrix, normal_targets)

    def predict(self, features: Sequence[float]) -> tuple[float, float, float]:
        """Predict one ENU residual correction; requires explicit prior training."""

        self._validate_features(features)
        if self._coefficients is None:
            raise ModelNotTrainedError(
                "ResidualCorrectionModel must be trained on real session-separated data before prediction."
            )
        return self._predict_validated(features)

    def predict_many(self, feature_vectors: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
        """Predict residual corrections for a validated batch after training."""

        for features in feature_vectors:
            self._validate_features(features)
        if self._coefficients is None:
            raise ModelNotTrainedError(
                "ResidualCorrectionModel must be trained on real session-separated data before prediction."
            )
        return tuple(self._predict_validated(features) for features in feature_vectors)

    @staticmethod
    def _validate_features(features: Sequence[float]) -> None:
        if len(features) != len(feature_names()):
            raise ModelValidationError(
                f"features must contain {len(feature_names())} values."
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) for value in features):
            raise ModelValidationError("features must contain finite numeric values.")

    def _predict_validated(self, features: Sequence[float]) -> tuple[float, float, float]:
        if self._coefficients is None:
            raise ModelNotTrainedError("ResidualCorrectionModel has not been trained.")
        augmented_features = (1.0, *features)
        prediction = tuple(
            sum(augmented_features[row] * self._coefficients[row][output] for row in range(len(augmented_features)))
            for output in range(3)
        )
        if not all(isfinite(value) for value in prediction):
            raise ModelValidationError("model produced a non-finite residual prediction.")
        return (prediction[0], prediction[1], prediction[2])


def _normal_matrix(design_matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    parameter_count = len(design_matrix[0])
    return [
        [
            sum(row[left] * row[right] for row in design_matrix)
            for right in range(parameter_count)
        ]
        for left in range(parameter_count)
    ]


def _normal_targets(
    design_matrix: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]
) -> list[list[float]]:
    parameter_count = len(design_matrix[0])
    return [
        [sum(row[parameter] * target[output] for row, target in zip(design_matrix, targets, strict=True)) for output in range(3)]
        for parameter in range(parameter_count)
    ]


def _solve_linear_system(
    coefficient_matrix: Sequence[Sequence[float]], right_hand_side: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    """Solve ``A B = C`` using partial-pivot Gaussian elimination."""

    size = len(coefficient_matrix)
    augmented = [
        [*coefficient_matrix[row], *right_hand_side[row]]
        for row in range(size)
    ]
    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row: abs(augmented[row][pivot_column]))
        pivot_value = augmented[pivot_row][pivot_column]
        if not isfinite(pivot_value) or abs(pivot_value) < 1e-12:
            raise ModelValidationError(
                "training features are rank-deficient or numerically singular; "
                "supply sufficiently diverse real training sessions."
            )
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        for row in range(pivot_column + 1, size):
            scale = augmented[row][pivot_column] / pivot_value
            for column in range(pivot_column, size + 3):
                augmented[row][column] -= scale * augmented[pivot_column][column]

    solution = [[0.0, 0.0, 0.0] for _ in range(size)]
    for row in range(size - 1, -1, -1):
        pivot = augmented[row][row]
        for output in range(3):
            solution[row][output] = (
                augmented[row][size + output]
                - sum(augmented[row][column] * solution[column][output] for column in range(row + 1, size))
            ) / pivot
    result = tuple((row[0], row[1], row[2]) for row in solution)
    if not all(isfinite(value) for row in result for value in row):
        raise ModelValidationError("linear regression fitting produced non-finite coefficients.")
    return result
