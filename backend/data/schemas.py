"""Typed, validated sensor data structures for recorded NAVIGEN sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite


def _validate_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be a finite number.")


def _validate_timestamp(timestamp: datetime) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset.")


@dataclass(frozen=True, slots=True)
class IMUSample:
    """One timestamped accelerometer and gyroscope observation.

    Accelerometer values are expressed in m/s² and gyroscope values in rad/s.
    Timestamps must be timezone-aware datetimes.
    """

    timestamp: datetime
    accelerometer_x: float
    accelerometer_y: float
    accelerometer_z: float
    gyroscope_x: float
    gyroscope_y: float
    gyroscope_z: float

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        for field_name in (
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ):
            _validate_finite(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class GNSSSample:
    """One timestamped GNSS observation in WGS84 coordinates.

    ``accuracy`` is optional and, when supplied, represents a non-negative
    accuracy estimate in metres as provided by the recording source.
    """

    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float
    accuracy: float | None = None

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        _validate_finite(self.latitude, "latitude")
        _validate_finite(self.longitude, "longitude")
        _validate_finite(self.altitude, "altitude")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90 degrees.")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180 degrees.")
        if self.accuracy is not None:
            _validate_finite(self.accuracy, "accuracy")
            if self.accuracy < 0.0:
                raise ValueError("accuracy must be non-negative.")


@dataclass(frozen=True, slots=True)
class SensorSession:
    """A recorded sensor session containing IMU data and optional GNSS data."""

    imu_samples: tuple[IMUSample, ...]
    gnss_samples: tuple[GNSSSample, ...] = field(default_factory=tuple)
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.imu_samples:
            raise ValueError("SensorSession requires at least one IMU sample.")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("session_id must not be empty when supplied.")
