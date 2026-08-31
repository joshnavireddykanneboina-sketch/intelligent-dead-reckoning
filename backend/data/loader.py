"""CSV loading and validation for recorded NAVIGEN sensor data."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from .schemas import GNSSSample, IMUSample, SensorSession


class SensorDataError(ValueError):
    """Base exception for recorded sensor data loading failures."""


class SensorDataValidationError(SensorDataError):
    """Raised when a CSV file violates the NAVIGEN sensor-data contract."""


_IMU_COLUMNS = (
    "timestamp",
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)
_GNSS_COLUMNS = ("timestamp", "latitude", "longitude", "altitude")

SampleT = TypeVar("SampleT", IMUSample, GNSSSample)


def load_imu_csv(path: str | Path) -> tuple[IMUSample, ...]:
    """Load validated IMU samples from a UTF-8 CSV file.

    Required columns are ``timestamp``, ``accelerometer_x``,
    ``accelerometer_y``, ``accelerometer_z``, ``gyroscope_x``,
    ``gyroscope_y``, and ``gyroscope_z``. Timestamps must be ISO 8601 values
    with a UTC offset (``Z`` is accepted).
    """

    return _load_csv(path, _IMU_COLUMNS, _parse_imu_row)


def load_gnss_csv(path: str | Path) -> tuple[GNSSSample, ...]:
    """Load validated GNSS samples from a UTF-8 CSV file.

    Required columns are ``timestamp``, ``latitude``, ``longitude``, and
    ``altitude``. The optional ``accuracy`` column is interpreted in metres.
    """

    return _load_csv(path, _GNSS_COLUMNS, _parse_gnss_row)


def load_sensor_session(
    imu_path: str | Path,
    gnss_path: str | Path | None = None,
    *,
    session_id: str | None = None,
) -> SensorSession:
    """Load a sensor session from an IMU CSV and an optional GNSS CSV."""

    imu_samples = load_imu_csv(imu_path)
    if not imu_samples:
        raise SensorDataValidationError("IMU CSV must contain at least one data row.")
    gnss_samples = load_gnss_csv(gnss_path) if gnss_path is not None else ()
    return SensorSession(
        imu_samples=imu_samples,
        gnss_samples=gnss_samples,
        session_id=session_id,
    )


def _load_csv(
    path: str | Path,
    required_columns: tuple[str, ...],
    row_parser: Callable[[dict[str, str], int], SampleT],
) -> tuple[SampleT, ...]:
    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            _validate_columns(reader.fieldnames, required_columns, csv_path)
            samples: list[SampleT] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise SensorDataValidationError(
                        f"{csv_path}: row {row_number} contains more values than headers."
                    )
                samples.append(row_parser(row, row_number))
    except OSError as error:
        raise SensorDataError(f"Could not read CSV file '{csv_path}': {error}") from error
    return tuple(samples)


def _validate_columns(
    fieldnames: list[str] | None,
    required_columns: tuple[str, ...],
    csv_path: Path,
) -> None:
    if fieldnames is None:
        raise SensorDataValidationError(f"{csv_path}: CSV file is missing a header row.")
    duplicate_columns = {name for name in fieldnames if fieldnames.count(name) > 1}
    if duplicate_columns:
        raise SensorDataValidationError(
            f"{csv_path}: duplicate column(s): {', '.join(sorted(duplicate_columns))}."
        )
    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise SensorDataValidationError(
            f"{csv_path}: missing required column(s): {', '.join(missing_columns)}."
        )


def _parse_imu_row(row: dict[str, str], row_number: int) -> IMUSample:
    try:
        return IMUSample(
            timestamp=_parse_timestamp(row, "timestamp", row_number),
            accelerometer_x=_parse_float(row, "accelerometer_x", row_number),
            accelerometer_y=_parse_float(row, "accelerometer_y", row_number),
            accelerometer_z=_parse_float(row, "accelerometer_z", row_number),
            gyroscope_x=_parse_float(row, "gyroscope_x", row_number),
            gyroscope_y=_parse_float(row, "gyroscope_y", row_number),
            gyroscope_z=_parse_float(row, "gyroscope_z", row_number),
        )
    except ValueError as error:
        raise SensorDataValidationError(f"row {row_number}: {error}") from error


def _parse_gnss_row(row: dict[str, str], row_number: int) -> GNSSSample:
    try:
        accuracy = _parse_optional_float(row, "accuracy", row_number)
        return GNSSSample(
            timestamp=_parse_timestamp(row, "timestamp", row_number),
            latitude=_parse_float(row, "latitude", row_number),
            longitude=_parse_float(row, "longitude", row_number),
            altitude=_parse_float(row, "altitude", row_number),
            accuracy=accuracy,
        )
    except ValueError as error:
        raise SensorDataValidationError(f"row {row_number}: {error}") from error


def _parse_timestamp(row: dict[str, str], column: str, row_number: int) -> datetime:
    value = _required_value(row, column, row_number)
    normalized_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized_value)
    except ValueError as error:
        raise ValueError(f"{column} must be an ISO 8601 timestamp with a UTC offset.") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{column} must include a UTC offset.")
    return timestamp


def _parse_float(row: dict[str, str], column: str, row_number: int) -> float:
    value = _required_value(row, column, row_number)
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise ValueError(f"{column} must be numeric.") from error
    return parsed_value


def _parse_optional_float(row: dict[str, str], column: str, row_number: int) -> float | None:
    if column not in row or row[column] is None or not row[column].strip():
        return None
    return _parse_float(row, column, row_number)


def _required_value(row: dict[str, str], column: str, row_number: int) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"{column} is required and must not be empty.")
    return value.strip()
