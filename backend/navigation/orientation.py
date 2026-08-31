"""Gyroscope-only quaternion orientation propagation.

Coordinate convention
---------------------
Quaternions are scalar-first ``(w, x, y, z)`` active rotations that map a
vector from the sensor/device body frame (B) into the local navigation frame
(N): ``v_N = q_B_to_N * v_B * conjugate(q_B_to_N)``. Gyroscope values are
assumed to be angular velocity of B relative to N, expressed in B, in rad/s.

For this convention, an angular increment derived from a body-frame gyro value
is post-multiplied: ``q_next = q_current * delta_q_B``. This module only
propagates attitude. It does not estimate translational motion or correct gyro
bias and therefore accumulates drift in real-world use.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, isfinite, sin, sqrt
from typing import TypeAlias

from backend.data.schemas import IMUSample
from backend.preprocessing.sensor import calculate_time_differences, validate_imu_sequence


Quaternion: TypeAlias = tuple[float, float, float, float]


class OrientationValidationError(ValueError):
    """Raised when quaternion, gyroscope, or timing input is invalid."""


def identity_quaternion() -> Quaternion:
    """Return the identity body-to-navigation rotation."""

    return (1.0, 0.0, 0.0, 0.0)


def normalize_quaternion(quaternion: Sequence[float]) -> Quaternion:
    """Return a unit quaternion, rejecting non-finite and zero-norm inputs."""

    values = _validate_quaternion(quaternion)
    norm = sqrt(sum(component * component for component in values))
    if not isfinite(norm) or norm == 0.0:
        raise OrientationValidationError("quaternion must have a finite non-zero norm.")
    return tuple(component / norm for component in values)  # type: ignore[return-value]


def multiply_quaternions(left: Sequence[float], right: Sequence[float]) -> Quaternion:
    """Multiply two scalar-first quaternions: ``left * right``."""

    left_w, left_x, left_y, left_z = _validate_quaternion(left)
    right_w, right_x, right_y, right_z = _validate_quaternion(right)
    return (
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    )


def integrate_angular_velocity(
    angular_velocity_rad_s: Sequence[float], dt_seconds: float
) -> Quaternion:
    """Convert a constant body-frame angular velocity over ``dt_seconds`` to a quaternion.

    This uses the exact axis-angle increment for a constant angular velocity,
    rather than a first-order Euler approximation.
    """

    angular_velocity = _validate_vector3(angular_velocity_rad_s, "angular_velocity_rad_s")
    _validate_dt(dt_seconds)
    angular_speed = sqrt(sum(component * component for component in angular_velocity))
    if angular_speed == 0.0:
        return identity_quaternion()

    rotation_angle = angular_speed * dt_seconds
    half_angle = rotation_angle / 2.0
    scale = sin(half_angle) / angular_speed
    return normalize_quaternion(
        (
            cos(half_angle),
            angular_velocity[0] * scale,
            angular_velocity[1] * scale,
            angular_velocity[2] * scale,
        )
    )


def propagate_orientation(
    orientation_body_to_navigation: Sequence[float], imu_sample: IMUSample, dt_seconds: float
) -> Quaternion:
    """Propagate a body-to-navigation orientation with one gyro measurement.

    ``imu_sample.gyroscope_*`` is the body-frame angular velocity in rad/s and
    is assumed constant across the supplied positive time interval.
    """

    try:
        validate_imu_sequence((imu_sample,))
    except ValueError as error:
        raise OrientationValidationError(f"invalid IMU sample: {error}") from error

    current_orientation = normalize_quaternion(orientation_body_to_navigation)
    delta_orientation = integrate_angular_velocity(
        (
            imu_sample.gyroscope_x,
            imu_sample.gyroscope_y,
            imu_sample.gyroscope_z,
        ),
        dt_seconds,
    )
    return normalize_quaternion(multiply_quaternions(current_orientation, delta_orientation))


def propagate_orientation_sequence(
    samples: Sequence[IMUSample], initial_orientation: Sequence[float] | None = None
) -> tuple[Quaternion, ...]:
    """Propagate orientation over timestamp-ordered IMU samples.

    The returned orientation at index zero is the normalized initial orientation
    at the first sample timestamp. For interval ``i -> i + 1``, this function
    uses the gyroscope reading at sample ``i`` as a left-end, constant-rate
    measurement. The output has the same length as ``samples``.
    """

    current_orientation = normalize_quaternion(
        identity_quaternion() if initial_orientation is None else initial_orientation
    )
    try:
        time_differences = calculate_time_differences(samples)
    except ValueError as error:
        raise OrientationValidationError(f"invalid IMU sequence: {error}") from error

    if not samples:
        return ()

    orientations: list[Quaternion] = [current_orientation]
    for index, dt_seconds in enumerate(time_differences):
        current_orientation = propagate_orientation(current_orientation, samples[index], dt_seconds)
        orientations.append(current_orientation)
    return tuple(orientations)


def _validate_quaternion(quaternion: Sequence[float]) -> Quaternion:
    if isinstance(quaternion, (str, bytes)) or len(quaternion) != 4:
        raise OrientationValidationError("quaternion must contain exactly four components (w, x, y, z).")
    return _validate_numeric_values(quaternion, "quaternion")  # type: ignore[return-value]


def _validate_vector3(vector: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if isinstance(vector, (str, bytes)) or len(vector) != 3:
        raise OrientationValidationError(f"{field_name} must contain exactly three components.")
    return _validate_numeric_values(vector, field_name)  # type: ignore[return-value]


def _validate_numeric_values(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    validated_values: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise OrientationValidationError(
                f"{field_name}[{index}] must be a finite numeric value."
            )
        validated_values.append(float(value))
    return tuple(validated_values)


def _validate_dt(dt_seconds: float) -> None:
    if (
        isinstance(dt_seconds, bool)
        or not isinstance(dt_seconds, (int, float))
        or not isfinite(dt_seconds)
        or dt_seconds <= 0.0
    ):
        raise OrientationValidationError("dt_seconds must be a finite positive number.")
