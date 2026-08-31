"""Thin orchestration service for NAVIGEN's available navigation pipeline.

The service runs the existing baseline dead-reckoning and ES-EKF components on
caller-supplied, validated recordings. It does not synthesize positions,
reference trajectories, or ML corrections. Raw GNSS coordinates are retained in
``GNSSSample`` while local ENU positions are supplied explicitly for ES-EKF
updates, pending a future WGS84-to-ENU frame service.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from backend.data.schemas import GNSSSample, IMUSample, SensorSession
from backend.fusion.ekf import EKFNavigationState, ErrorStateEKF
from backend.navigation.dead_reckoning import InertialState, propagate_inertial_trajectory
from backend.preprocessing.sensor import calculate_time_differences, validate_imu_sequence


Vector3 = tuple[float, float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class NavigationServiceValidationError(ValueError):
    """Raised when an API-ready navigation request cannot be processed."""


@dataclass(frozen=True, slots=True)
class GNSSPositionMeasurement:
    """A validated GNSS schema sample paired with its local ENU position."""

    sample: GNSSSample
    position_enu_m: Vector3
    measurement_covariance_enu_m2: Matrix3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample, GNSSSample):
            raise NavigationServiceValidationError("sample must be a GNSSSample.")
        _validate_vector3(self.position_enu_m, "position_enu_m")
        if self.measurement_covariance_enu_m2 is not None:
            _validate_matrix3(self.measurement_covariance_enu_m2, "measurement_covariance_enu_m2")


@dataclass(frozen=True, slots=True)
class NavigationPipelineResult:
    """Real outputs from baseline and ES-EKF propagation for one session."""

    baseline_states: tuple[InertialState, ...]
    fused_states: tuple[EKFNavigationState, ...]
    fused_timestamps: tuple[datetime, ...]
    ml_correction_available: bool = False


class NavigationService:
    """Run existing NAVIGEN navigation components without adding new mathematics."""

    def process_session(
        self,
        session: SensorSession,
        *,
        gnss_measurements: Sequence[GNSSPositionMeasurement] = (),
    ) -> NavigationPipelineResult:
        """Run baseline and ES-EKF propagation on one recorded sensor session.

        GNSS updates must use timestamps present in the IMU session. A missing
        trained residual model is represented explicitly as unavailable; no ML
        prediction is attempted by this service.
        """

        if not isinstance(session, SensorSession):
            raise NavigationServiceValidationError("session must be a SensorSession.")
        try:
            validate_imu_sequence(session.imu_samples)
            time_differences = calculate_time_differences(session.imu_samples)
            baseline_states = propagate_inertial_trajectory(session.imu_samples)
        except ValueError as error:
            raise NavigationServiceValidationError(f"invalid IMU session: {error}") from error

        measurement_by_timestamp = self._index_gnss_measurements(
            gnss_measurements, session.imu_samples
        )
        ekf = ErrorStateEKF()
        fused_states: list[EKFNavigationState] = []
        first_sample = session.imu_samples[0]
        if first_sample.timestamp in measurement_by_timestamp:
            measurement = measurement_by_timestamp[first_sample.timestamp]
            ekf.update_gnss_sample(
                measurement.sample,
                measurement.position_enu_m,
                measurement.measurement_covariance_enu_m2,
            )
        fused_states.append(ekf.state)

        for index, dt_seconds in enumerate(time_differences):
            # The left-end IMU reading is assumed constant to the next recorded
            # timestamp, consistent with the existing orientation/DR modules.
            ekf.predict(session.imu_samples[index], dt_seconds)
            result_timestamp = session.imu_samples[index + 1].timestamp
            measurement = measurement_by_timestamp.get(result_timestamp)
            if measurement is not None:
                ekf.update_gnss_sample(
                    measurement.sample,
                    measurement.position_enu_m,
                    measurement.measurement_covariance_enu_m2,
                )
            fused_states.append(ekf.state)

        return NavigationPipelineResult(
            baseline_states=baseline_states,
            fused_states=tuple(fused_states),
            fused_timestamps=tuple(sample.timestamp for sample in session.imu_samples),
            ml_correction_available=False,
        )

    @staticmethod
    def _index_gnss_measurements(
        measurements: Sequence[GNSSPositionMeasurement], imu_samples: Sequence[IMUSample]
    ) -> dict[datetime, GNSSPositionMeasurement]:
        imu_timestamps = {sample.timestamp for sample in imu_samples}
        indexed: dict[datetime, GNSSPositionMeasurement] = {}
        for measurement in measurements:
            if not isinstance(measurement, GNSSPositionMeasurement):
                raise NavigationServiceValidationError(
                    "gnss_measurements must contain GNSSPositionMeasurement instances."
                )
            timestamp = measurement.sample.timestamp
            if timestamp not in imu_timestamps:
                raise NavigationServiceValidationError(
                    "GNSS measurement timestamp must match an IMU sample timestamp."
                )
            if timestamp in indexed:
                raise NavigationServiceValidationError("only one GNSS measurement is allowed per timestamp.")
            indexed[timestamp] = measurement
        return indexed


def _validate_vector3(values: Sequence[float], field_name: str) -> None:
    if len(values) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
        for value in values
    ):
        raise NavigationServiceValidationError(
            f"{field_name} must contain three finite numeric values."
        )


def _validate_matrix3(matrix: Sequence[Sequence[float]], field_name: str) -> None:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise NavigationServiceValidationError(f"{field_name} must be a 3-by-3 matrix.")
    for row in range(3):
        for column in range(3):
            value = matrix[row][column]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise NavigationServiceValidationError(
                    f"{field_name}[{row}][{column}] must be finite."
                )
            if abs(value - matrix[column][row]) > 1e-9:
                raise NavigationServiceValidationError(f"{field_name} must be symmetric.")
