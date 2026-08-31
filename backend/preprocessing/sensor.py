"""Reusable validation and timing utilities for IMU sample sequences.

These functions intentionally do not reorder, deduplicate, resample, filter, or
otherwise alter IMU measurements. Invalid recorded data is rejected so that an
upstream source can be corrected deliberately.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import isfinite

from backend.data.schemas import IMUSample


class SensorPreprocessingValidationError(ValueError):
    """Raised when an IMU sequence is unsuitable for preprocessing."""


_MEASUREMENT_FIELDS = (
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)


def validate_imu_sequence(samples: Sequence[IMUSample]) -> None:
    """Validate IMU timestamps and measurements without modifying samples.

    Samples must be in strictly increasing, timezone-aware timestamp order.
    Duplicate timestamps and non-finite sensor values are rejected.
    """

    previous_timestamp: datetime | None = None
    for index, sample in enumerate(samples):
        timestamp = _validate_sample(sample, index)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            if timestamp == previous_timestamp:
                raise SensorPreprocessingValidationError(
                    f"sample {index}: duplicate timestamp '{timestamp.isoformat()}'."
                )
            raise SensorPreprocessingValidationError(
                f"sample {index}: timestamp '{timestamp.isoformat()}' is earlier than "
                f"the previous sample. Samples must be strictly ordered."
            )
        previous_timestamp = timestamp


def calculate_time_differences(samples: Sequence[IMUSample]) -> tuple[float, ...]:
    """Return positive time intervals in seconds between consecutive IMU samples.

    The input is validated first. An empty or one-sample sequence has no
    intervals and therefore returns an empty tuple.
    """

    validate_imu_sequence(samples)
    time_differences: list[float] = []
    for index in range(1, len(samples)):
        interval_seconds = (samples[index].timestamp - samples[index - 1].timestamp).total_seconds()
        if not isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise SensorPreprocessingValidationError(
                f"sample {index}: time interval must be finite and positive; "
                f"received {interval_seconds!r} seconds."
            )
        time_differences.append(interval_seconds)
    return tuple(time_differences)


def calculate_sampling_rate_hz(time_differences: Sequence[float]) -> float:
    """Calculate the mean sampling rate in hertz from positive intervals."""

    if not time_differences:
        raise SensorPreprocessingValidationError(
            "At least one positive time interval is required to calculate sampling rate."
        )
    _validate_time_differences(time_differences)
    mean_interval_seconds = sum(time_differences) / len(time_differences)
    return 1.0 / mean_interval_seconds


def validate_sampling_rate(
    samples: Sequence[IMUSample],
    *,
    minimum_hz: float | None = None,
    maximum_hz: float | None = None,
) -> float:
    """Validate a sequence's mean sampling rate and return it in hertz.

    Bounds are optional. When supplied, they must be finite positive values and
    ``minimum_hz`` must not exceed ``maximum_hz``.
    """

    _validate_rate_bound(minimum_hz, "minimum_hz")
    _validate_rate_bound(maximum_hz, "maximum_hz")
    if minimum_hz is not None and maximum_hz is not None and minimum_hz > maximum_hz:
        raise SensorPreprocessingValidationError(
            "minimum_hz must not be greater than maximum_hz."
        )

    sampling_rate_hz = calculate_sampling_rate_hz(calculate_time_differences(samples))
    if minimum_hz is not None and sampling_rate_hz < minimum_hz:
        raise SensorPreprocessingValidationError(
            f"sampling rate {sampling_rate_hz:.6g} Hz is below minimum_hz {minimum_hz:.6g}."
        )
    if maximum_hz is not None and sampling_rate_hz > maximum_hz:
        raise SensorPreprocessingValidationError(
            f"sampling rate {sampling_rate_hz:.6g} Hz exceeds maximum_hz {maximum_hz:.6g}."
        )
    return sampling_rate_hz


def _validate_sample(sample: IMUSample, index: int) -> datetime:
    timestamp = getattr(sample, "timestamp", None)
    if not isinstance(timestamp, datetime):
        raise SensorPreprocessingValidationError(f"sample {index}: timestamp must be a datetime.")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise SensorPreprocessingValidationError(
            f"sample {index}: timestamp must include a UTC offset."
        )

    for field_name in _MEASUREMENT_FIELDS:
        value = getattr(sample, field_name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise SensorPreprocessingValidationError(
                f"sample {index}: {field_name} must be a finite numeric value."
            )
    return timestamp


def _validate_time_differences(time_differences: Sequence[float]) -> None:
    for index, interval_seconds in enumerate(time_differences):
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not isfinite(interval_seconds)
            or interval_seconds <= 0.0
        ):
            raise SensorPreprocessingValidationError(
                f"time_differences[{index}] must be a finite positive number."
            )


def _validate_rate_bound(bound: float | None, field_name: str) -> None:
    if bound is None:
        return
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not isfinite(bound) or bound <= 0.0:
        raise SensorPreprocessingValidationError(
            f"{field_name} must be a finite positive number when supplied."
        )
