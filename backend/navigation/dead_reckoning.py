"""Baseline strapdown inertial dead reckoning.

Coordinate and acceleration conventions
---------------------------------------
The navigation frame is local East-North-Up (ENU): x=east, y=north, z=up.
Orientation is the scalar-first active quaternion ``q_B_to_N`` that rotates a
body-frame vector into ENU. Accelerometer values are assumed to be *specific
force* in the body frame, measured in m/s²:

``f_B = R_N_to_B (a_N - g_N)``

where ``g_N = (0, 0, -gravity_m_s2)``. Therefore the navigation-frame linear
acceleration is calculated as:

``a_N = R_B_to_N f_B + g_N``

For a stationary, level device under this convention, the accelerometer reads
approximately ``(0, 0, +gravity_m_s2)`` and the compensated acceleration is
approximately zero. This module is an uncorrected baseline; integration drift
from sensor noise, bias, and attitude error is expected and is not suppressed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import TypeAlias

from backend.data.schemas import IMUSample
from backend.navigation.orientation import (
    OrientationValidationError,
    Quaternion,
    normalize_quaternion,
    propagate_orientation_sequence,
)
from backend.preprocessing.sensor import (
    SensorPreprocessingValidationError,
    calculate_time_differences,
)


Vector3: TypeAlias = tuple[float, float, float]


class DeadReckoningValidationError(ValueError):
    """Raised when baseline dead-reckoning input is invalid."""


@dataclass(frozen=True, slots=True)
class InertialState:
    """Baseline inertial navigation state at one IMU timestamp.

    Position and velocity are expressed in local ENU metres and metres per
    second. Orientation is the body-to-navigation quaternion ``(w, x, y, z)``.
    """

    timestamp: datetime
    position_enu_m: Vector3
    velocity_enu_m_s: Vector3
    orientation_body_to_navigation: Quaternion


def propagate_inertial_trajectory(
    samples: Sequence[IMUSample],
    *,
    initial_position_enu_m: Sequence[float] = (0.0, 0.0, 0.0),
    initial_velocity_enu_m_s: Sequence[float] = (0.0, 0.0, 0.0),
    initial_orientation_body_to_navigation: Sequence[float] | None = None,
    gravity_m_s2: float = 9.80665,
) -> tuple[InertialState, ...]:
    """Propagate an uncorrected ENU trajectory from timestamped IMU samples.

    The initial state corresponds to the first IMU timestamp. For each interval
    ``i -> i + 1``, the function uses sample ``i`` as a left-end, constant
    accelerometer and gyroscope measurement. Velocity and position are updated
    with constant-acceleration kinematics using the measured time interval.
    """

    position = _validate_vector3(initial_position_enu_m, "initial_position_enu_m")
    velocity = _validate_vector3(initial_velocity_enu_m_s, "initial_velocity_enu_m_s")
    _validate_gravity(gravity_m_s2)

    try:
        time_differences = calculate_time_differences(samples)
        orientations = propagate_orientation_sequence(
            samples,
            initial_orientation=initial_orientation_body_to_navigation,
        )
    except (SensorPreprocessingValidationError, OrientationValidationError, ValueError) as error:
        raise DeadReckoningValidationError(f"invalid inertial input: {error}") from error

    if not samples:
        return ()

    states: list[InertialState] = [
        InertialState(
            timestamp=samples[0].timestamp,
            position_enu_m=position,
            velocity_enu_m_s=velocity,
            orientation_body_to_navigation=orientations[0],
        )
    ]

    for index, dt_seconds in enumerate(time_differences):
        navigation_acceleration = _specific_force_to_navigation_acceleration(
            samples[index], orientations[index], gravity_m_s2
        )
        next_position = tuple(
            position[axis]
            + velocity[axis] * dt_seconds
            + 0.5 * navigation_acceleration[axis] * dt_seconds * dt_seconds
            for axis in range(3)
        )
        next_velocity = tuple(
            velocity[axis] + navigation_acceleration[axis] * dt_seconds
            for axis in range(3)
        )
        position = _validate_vector3(next_position, "integrated position")
        velocity = _validate_vector3(next_velocity, "integrated velocity")
        states.append(
            InertialState(
                timestamp=samples[index + 1].timestamp,
                position_enu_m=position,
                velocity_enu_m_s=velocity,
                orientation_body_to_navigation=orientations[index + 1],
            )
        )
    return tuple(states)


def _specific_force_to_navigation_acceleration(
    sample: IMUSample, orientation_body_to_navigation: Sequence[float], gravity_m_s2: float
) -> Vector3:
    specific_force_body = _validate_vector3(
        (sample.accelerometer_x, sample.accelerometer_y, sample.accelerometer_z),
        "accelerometer measurement",
    )
    rotated_specific_force = _rotate_body_to_navigation(
        orientation_body_to_navigation, specific_force_body
    )
    return _validate_vector3(
        (
            rotated_specific_force[0],
            rotated_specific_force[1],
            rotated_specific_force[2] - gravity_m_s2,
        ),
        "gravity-compensated acceleration",
    )


def _rotate_body_to_navigation(
    orientation_body_to_navigation: Sequence[float], vector_body: Vector3
) -> Vector3:
    """Rotate a body-frame vector into ENU using ``q * v * conjugate(q)``."""

    w, x, y, z = normalize_quaternion(orientation_body_to_navigation)
    vector_x, vector_y, vector_z = vector_body

    # Equivalent to q * (0, vector_body) * conjugate(q), expanded directly.
    twice_x = 2.0 * x
    twice_y = 2.0 * y
    twice_z = 2.0 * z
    return (
        (1.0 - twice_y * y - twice_z * z) * vector_x
        + (twice_x * y - twice_z * w) * vector_y
        + (twice_x * z + twice_y * w) * vector_z,
        (twice_x * y + twice_z * w) * vector_x
        + (1.0 - twice_x * x - twice_z * z) * vector_y
        + (twice_y * z - twice_x * w) * vector_z,
        (twice_x * z - twice_y * w) * vector_x
        + (twice_y * z + twice_x * w) * vector_y
        + (1.0 - twice_x * x - twice_y * y) * vector_z,
    )


def _validate_vector3(values: Sequence[float], field_name: str) -> Vector3:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise DeadReckoningValidationError(f"{field_name} must contain exactly three components.")
    validated_values: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise DeadReckoningValidationError(
                f"{field_name}[{index}] must be a finite numeric value."
            )
        validated_values.append(float(value))
    return tuple(validated_values)  # type: ignore[return-value]


def _validate_gravity(gravity_m_s2: float) -> None:
    if (
        isinstance(gravity_m_s2, bool)
        or not isinstance(gravity_m_s2, (int, float))
        or not isfinite(gravity_m_s2)
        or gravity_m_s2 <= 0.0
    ):
        raise DeadReckoningValidationError("gravity_m_s2 must be a finite positive number.")
