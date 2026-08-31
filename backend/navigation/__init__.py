"""NAVIGEN navigation-domain utilities."""

from .orientation import (
    Quaternion,
    OrientationValidationError,
    identity_quaternion,
    integrate_angular_velocity,
    multiply_quaternions,
    normalize_quaternion,
    propagate_orientation,
    propagate_orientation_sequence,
)

__all__ = [
    "Quaternion",
    "OrientationValidationError",
    "identity_quaternion",
    "integrate_angular_velocity",
    "multiply_quaternions",
    "normalize_quaternion",
    "propagate_orientation",
    "propagate_orientation_sequence",
]
