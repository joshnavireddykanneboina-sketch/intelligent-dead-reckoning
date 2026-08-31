"""Feature generation for supervised inertial-navigation residual learning.

Features are observations available at the prediction timestamp. They contain
IMU values, elapsed time, existing navigation estimates, and trailing IMU
statistics; they never include a future reference position or residual target.
This module only transforms caller-supplied recordings; it does not generate
sensor data.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt

from backend.data.schemas import IMUSample
from backend.navigation.orientation import normalize_quaternion
from backend.preprocessing.sensor import validate_imu_sequence


FeatureVector = tuple[float, ...]


class FeatureValidationError(ValueError):
    """Raised when residual-learning feature inputs are invalid."""


_FEATURE_NAMES = (
    "accelerometer_x_m_s2",
    "accelerometer_y_m_s2",
    "accelerometer_z_m_s2",
    "gyroscope_x_rad_s",
    "gyroscope_y_rad_s",
    "gyroscope_z_rad_s",
    "dt_seconds",
    "velocity_east_m_s",
    "velocity_north_m_s",
    "velocity_up_m_s",
    "orientation_w",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "recent_acceleration_magnitude_mean_m_s2",
    "recent_acceleration_magnitude_std_m_s2",
    "recent_angular_rate_magnitude_mean_rad_s",
    "recent_angular_rate_magnitude_std_rad_s",
)


def feature_names() -> tuple[str, ...]:
    """Return the stable ordered names of each residual-model feature."""

    return _FEATURE_NAMES


def build_feature_vector(
    imu_sample: IMUSample,
    *,
    dt_seconds: float,
    estimated_velocity_enu_m_s: Sequence[float],
    orientation_body_to_navigation: Sequence[float],
    recent_imu_samples: Sequence[IMUSample] = (),
) -> FeatureVector:
    """Build one finite, ordered feature vector from a real navigation sample.

    ``recent_imu_samples`` contains only earlier samples from the same session.
    It is used for trailing statistics and must be strictly timestamp-ordered
    before ``imu_sample``. The returned vector contains no target/reference
    information, preventing target leakage at feature construction time.
    """

    _validate_positive_dt(dt_seconds)
    try:
        validate_imu_sequence((imu_sample,))
        if recent_imu_samples:
            validate_imu_sequence(recent_imu_samples)
            if recent_imu_samples[-1].timestamp >= imu_sample.timestamp:
                raise FeatureValidationError(
                    "recent_imu_samples must occur strictly before imu_sample."
                )
    except ValueError as error:
        if isinstance(error, FeatureValidationError):
            raise
        raise FeatureValidationError(f"invalid IMU feature input: {error}") from error

    velocity = _validate_vector3(estimated_velocity_enu_m_s, "estimated_velocity_enu_m_s")
    try:
        orientation = normalize_quaternion(orientation_body_to_navigation)
    except ValueError as error:
        raise FeatureValidationError(f"invalid orientation: {error}") from error

    statistics_samples = tuple(recent_imu_samples) + (imu_sample,)
    acceleration_magnitudes = tuple(_acceleration_magnitude(sample) for sample in statistics_samples)
    angular_rate_magnitudes = tuple(_angular_rate_magnitude(sample) for sample in statistics_samples)
    values = (
        imu_sample.accelerometer_x,
        imu_sample.accelerometer_y,
        imu_sample.accelerometer_z,
        imu_sample.gyroscope_x,
        imu_sample.gyroscope_y,
        imu_sample.gyroscope_z,
        float(dt_seconds),
        *velocity,
        *orientation,
        _mean(acceleration_magnitudes),
        _population_standard_deviation(acceleration_magnitudes),
        _mean(angular_rate_magnitudes),
        _population_standard_deviation(angular_rate_magnitudes),
    )
    if not all(isfinite(value) for value in values):
        raise FeatureValidationError("feature vector contains non-finite values.")
    return tuple(float(value) for value in values)


def _validate_positive_dt(dt_seconds: float) -> None:
    if (
        isinstance(dt_seconds, bool)
        or not isinstance(dt_seconds, (int, float))
        or not isfinite(dt_seconds)
        or dt_seconds <= 0.0
    ):
        raise FeatureValidationError("dt_seconds must be a finite positive number.")


def _validate_vector3(values: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise FeatureValidationError(f"{field_name} must contain exactly three components.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) for value in values):
        raise FeatureValidationError(f"{field_name} must contain finite numeric values.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _acceleration_magnitude(sample: IMUSample) -> float:
    return sqrt(
        sample.accelerometer_x**2 + sample.accelerometer_y**2 + sample.accelerometer_z**2
    )


def _angular_rate_magnitude(sample: IMUSample) -> float:
    return sqrt(sample.gyroscope_x**2 + sample.gyroscope_y**2 + sample.gyroscope_z**2)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_standard_deviation(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))
