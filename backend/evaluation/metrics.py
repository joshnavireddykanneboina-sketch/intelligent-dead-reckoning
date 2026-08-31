"""Timestamp-aligned ENU trajectory metrics.

Estimated and reference trajectories must have the same non-zero number of
points and identical timestamps at each index. This module deliberately does
not interpolate, infer, or fabricate reference data. Position and velocity are
local East-North-Up (ENU), in metres and metres per second respectively.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from typing import TypeAlias


Vector3: TypeAlias = tuple[float, float, float]


class EvaluationValidationError(ValueError):
    """Raised when trajectories or metric inputs are invalid."""


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One timestamped ENU position and optional ENU velocity observation."""

    timestamp: datetime
    position_enu_m: Vector3
    velocity_enu_m_s: Vector3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise EvaluationValidationError("timestamp must be a datetime.")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise EvaluationValidationError("timestamp must include a UTC offset.")
        _validate_vector3(self.position_enu_m, "position_enu_m")
        if self.velocity_enu_m_s is not None:
            _validate_vector3(self.velocity_enu_m_s, "velocity_enu_m_s")


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Metrics computed from a timestamp-aligned ENU trajectory comparison."""

    position_error_vectors_enu_m: tuple[Vector3, ...]
    horizontal_position_errors_m: tuple[float, ...]
    three_dimensional_position_errors_m: tuple[float, ...]
    position_mae_m: float
    position_rmse_m: float
    horizontal_position_mae_m: float
    horizontal_position_rmse_m: float
    velocity_error_vectors_enu_m_s: tuple[Vector3, ...] | None
    velocity_error_mae_m_s: float | None
    velocity_error_rmse_m_s: float | None
    endpoint_drift_m: float
    endpoint_drift_ratio: float | None


def position_error_enu_m(estimated_position_enu_m: Sequence[float], reference_position_enu_m: Sequence[float]) -> Vector3:
    """Return the signed ENU error ``estimated - reference`` in metres."""

    estimated = _validate_vector3(estimated_position_enu_m, "estimated_position_enu_m")
    reference = _validate_vector3(reference_position_enu_m, "reference_position_enu_m")
    return (
        estimated[0] - reference[0],
        estimated[1] - reference[1],
        estimated[2] - reference[2],
    )


def horizontal_position_error_m(estimated_position_enu_m: Sequence[float], reference_position_enu_m: Sequence[float]) -> float:
    """Return horizontal ENU error magnitude in metres."""

    error = position_error_enu_m(estimated_position_enu_m, reference_position_enu_m)
    return sqrt(error[0] ** 2 + error[1] ** 2)


def three_dimensional_position_error_m(
    estimated_position_enu_m: Sequence[float], reference_position_enu_m: Sequence[float]
) -> float:
    """Return three-dimensional ENU position error magnitude in metres."""

    return _magnitude(position_error_enu_m(estimated_position_enu_m, reference_position_enu_m))


def velocity_error_enu_m_s(estimated_velocity_enu_m_s: Sequence[float], reference_velocity_enu_m_s: Sequence[float]) -> Vector3:
    """Return the signed ENU velocity error ``estimated - reference`` in m/s."""

    estimated = _validate_vector3(estimated_velocity_enu_m_s, "estimated_velocity_enu_m_s")
    reference = _validate_vector3(reference_velocity_enu_m_s, "reference_velocity_enu_m_s")
    return (
        estimated[0] - reference[0],
        estimated[1] - reference[1],
        estimated[2] - reference[2],
    )


def mean_absolute_error(values: Sequence[float]) -> float:
    """Return the arithmetic mean of absolute finite scalar errors."""

    validated = _validate_metric_values(values)
    return sum(abs(value) for value in validated) / len(validated)


def root_mean_square_error(values: Sequence[float]) -> float:
    """Return root-mean-square error for finite scalar errors."""

    validated = _validate_metric_values(values)
    return sqrt(sum(value * value for value in validated) / len(validated))


def evaluate_trajectory(
    estimated_trajectory: Sequence[TrajectoryPoint], reference_trajectory: Sequence[TrajectoryPoint]
) -> TrajectoryMetrics:
    """Evaluate estimated ENU trajectory points against supplied reference points."""

    _validate_trajectory_pair(estimated_trajectory, reference_trajectory)
    position_vectors = tuple(
        position_error_enu_m(estimated.position_enu_m, reference.position_enu_m)
        for estimated, reference in zip(estimated_trajectory, reference_trajectory, strict=True)
    )
    horizontal_errors = tuple(sqrt(error[0] ** 2 + error[1] ** 2) for error in position_vectors)
    three_dimensional_errors = tuple(_magnitude(error) for error in position_vectors)

    has_velocity = estimated_trajectory[0].velocity_enu_m_s is not None
    velocity_vectors: tuple[Vector3, ...] | None = None
    velocity_mae: float | None = None
    velocity_rmse: float | None = None
    if has_velocity:
        velocity_vectors = tuple(
            velocity_error_enu_m_s(estimated.velocity_enu_m_s, reference.velocity_enu_m_s)
            for estimated, reference in zip(estimated_trajectory, reference_trajectory, strict=True)
            if estimated.velocity_enu_m_s is not None and reference.velocity_enu_m_s is not None
        )
        velocity_magnitudes = tuple(_magnitude(error) for error in velocity_vectors)
        velocity_mae = mean_absolute_error(velocity_magnitudes)
        velocity_rmse = root_mean_square_error(velocity_magnitudes)

    endpoint_drift = three_dimensional_errors[-1]
    reference_distance = _reference_path_length_m(reference_trajectory)
    drift_ratio = endpoint_drift / reference_distance if reference_distance > 0.0 else None
    return TrajectoryMetrics(
        position_error_vectors_enu_m=position_vectors,
        horizontal_position_errors_m=horizontal_errors,
        three_dimensional_position_errors_m=three_dimensional_errors,
        position_mae_m=mean_absolute_error(three_dimensional_errors),
        position_rmse_m=root_mean_square_error(three_dimensional_errors),
        horizontal_position_mae_m=mean_absolute_error(horizontal_errors),
        horizontal_position_rmse_m=root_mean_square_error(horizontal_errors),
        velocity_error_vectors_enu_m_s=velocity_vectors,
        velocity_error_mae_m_s=velocity_mae,
        velocity_error_rmse_m_s=velocity_rmse,
        endpoint_drift_m=endpoint_drift,
        endpoint_drift_ratio=drift_ratio,
    )


def _validate_trajectory_pair(
    estimated_trajectory: Sequence[TrajectoryPoint], reference_trajectory: Sequence[TrajectoryPoint]
) -> None:
    if not estimated_trajectory or not reference_trajectory:
        raise EvaluationValidationError("estimated and reference trajectories must be non-empty.")
    if len(estimated_trajectory) != len(reference_trajectory):
        raise EvaluationValidationError("estimated and reference trajectories must have matching lengths.")
    if any(not isinstance(point, TrajectoryPoint) for point in estimated_trajectory) or any(
        not isinstance(point, TrajectoryPoint) for point in reference_trajectory
    ):
        raise EvaluationValidationError("trajectory entries must be TrajectoryPoint instances.")
    expected_velocity_presence = estimated_trajectory[0].velocity_enu_m_s is not None
    reference_velocity_presence = reference_trajectory[0].velocity_enu_m_s is not None
    if expected_velocity_presence != reference_velocity_presence:
        raise EvaluationValidationError("estimated and reference trajectories must both provide velocity or neither.")
    for index, (estimated, reference) in enumerate(zip(estimated_trajectory, reference_trajectory, strict=True)):
        if estimated.timestamp != reference.timestamp:
            raise EvaluationValidationError(f"trajectory timestamp mismatch at index {index}.")
        if (estimated.velocity_enu_m_s is not None) != expected_velocity_presence or (
            reference.velocity_enu_m_s is not None
        ) != reference_velocity_presence:
            raise EvaluationValidationError("velocity availability must be consistent across each trajectory.")


def _reference_path_length_m(reference_trajectory: Sequence[TrajectoryPoint]) -> float:
    return sum(
        _magnitude(
            position_error_enu_m(
                reference_trajectory[index].position_enu_m,
                reference_trajectory[index - 1].position_enu_m,
            )
        )
        for index in range(1, len(reference_trajectory))
    )


def _validate_vector3(values: Sequence[float], field_name: str) -> Vector3:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise EvaluationValidationError(f"{field_name} must contain exactly three components.")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise EvaluationValidationError(f"{field_name}[{index}] must be a finite numeric value.")
        result.append(float(value))
    return (result[0], result[1], result[2])


def _validate_metric_values(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise EvaluationValidationError("metric values must be non-empty.")
    validated: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise EvaluationValidationError(f"metric values[{index}] must be a finite numeric value.")
        validated.append(float(value))
    return tuple(validated)


def _magnitude(vector: Vector3) -> float:
    return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
