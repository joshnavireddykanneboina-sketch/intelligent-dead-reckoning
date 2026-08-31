"""Error-State Extended Kalman Filter for baseline inertial navigation.

Frames, nominal state, and error state
--------------------------------------
The navigation frame is local East-North-Up (ENU), in metres. The nominal
state is ``(p_N, v_N, q_B_to_N, b_a_B, b_g_B)`` where position and velocity
are ENU, the scalar-first quaternion rotates body vectors into ENU, and IMU
biases remain in body coordinates.

The 15-element error state is ordered as:
``[delta_p_N(3), delta_v_N(3), delta_theta_N(3), delta_ba_B(3), delta_bg_B(3)]``.
``delta_theta_N`` is a small left-multiplicative navigation-frame attitude
error. Its correction is applied as ``q <- delta_q_N * q``.

Accelerometers are assumed to measure body-frame specific force in m/s². With
ENU gravity ``g_N = (0, 0, -gravity_m_s2)``, nominal linear acceleration is
``a_N = R_B_to_N (f_measured_B - b_a_B) + g_N``. Gyroscopes are in rad/s.

The prediction uses first-order discretisation of the continuous error model.
Accelerometer and gyro white-noise standard deviations are isotropic, while
bias random-walk standard deviations model slow bias changes. GNSS updates use
the local ENU position model ``z = p_N + measurement_noise``. Raw WGS84 GNSS
coordinates must be converted to the same ENU origin by a future frame layer
before calling this filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import TypeAlias

from backend.data.schemas import GNSSSample, IMUSample
from backend.navigation.orientation import (
    Quaternion,
    integrate_angular_velocity,
    multiply_quaternions,
    normalize_quaternion,
)
from backend.preprocessing.sensor import validate_imu_sequence


Vector3: TypeAlias = tuple[float, float, float]
Matrix: TypeAlias = tuple[tuple[float, ...], ...]
_STATE_SIZE = 15
_POSITION = 0
_VELOCITY = 3
_ATTITUDE = 6
_ACCELEROMETER_BIAS = 9
_GYROSCOPE_BIAS = 12


class EKFValidationError(ValueError):
    """Raised when an EKF measurement, state, or covariance is invalid."""


@dataclass(frozen=True, slots=True)
class EKFNavigationState:
    """Nominal navigation state and its 15-by-15 error covariance."""

    timestamp: datetime | None
    position_enu_m: Vector3
    velocity_enu_m_s: Vector3
    orientation_body_to_navigation: Quaternion
    accelerometer_bias_body_m_s2: Vector3
    gyroscope_bias_body_rad_s: Vector3
    error_covariance: Matrix


@dataclass(frozen=True, slots=True)
class GNSSPositionUpdate:
    """Details of one accepted local-ENU GNSS position measurement update."""

    innovation_enu_m: Vector3
    innovation_covariance: Matrix
    kalman_gain: Matrix
    state: EKFNavigationState


class ErrorStateEKF:
    """A 15-state error-state EKF with IMU prediction and GNSS position updates."""

    def __init__(
        self,
        *,
        initial_position_enu_m: Sequence[float] = (0.0, 0.0, 0.0),
        initial_velocity_enu_m_s: Sequence[float] = (0.0, 0.0, 0.0),
        initial_orientation_body_to_navigation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        initial_accelerometer_bias_body_m_s2: Sequence[float] = (0.0, 0.0, 0.0),
        initial_gyroscope_bias_body_rad_s: Sequence[float] = (0.0, 0.0, 0.0),
        initial_covariance: Sequence[Sequence[float]] | None = None,
        gravity_m_s2: float = 9.80665,
        accelerometer_noise_std_m_s2: float = 0.1,
        gyroscope_noise_std_rad_s: float = 0.01,
        accelerometer_bias_random_walk_std_m_s3: float = 0.001,
        gyroscope_bias_random_walk_std_rad_s2: float = 0.0001,
    ) -> None:
        self._gravity_m_s2 = _validate_positive_scalar(gravity_m_s2, "gravity_m_s2")
        self._accelerometer_noise_std = _validate_nonnegative_scalar(
            accelerometer_noise_std_m_s2, "accelerometer_noise_std_m_s2"
        )
        self._gyroscope_noise_std = _validate_nonnegative_scalar(
            gyroscope_noise_std_rad_s, "gyroscope_noise_std_rad_s"
        )
        self._accelerometer_bias_random_walk_std = _validate_nonnegative_scalar(
            accelerometer_bias_random_walk_std_m_s3,
            "accelerometer_bias_random_walk_std_m_s3",
        )
        self._gyroscope_bias_random_walk_std = _validate_nonnegative_scalar(
            gyroscope_bias_random_walk_std_rad_s2,
            "gyroscope_bias_random_walk_std_rad_s2",
        )

        covariance = (
            _default_initial_covariance()
            if initial_covariance is None
            else _validate_covariance(initial_covariance, _STATE_SIZE, "initial_covariance")
        )
        self._state = EKFNavigationState(
            timestamp=None,
            position_enu_m=_validate_vector3(initial_position_enu_m, "initial_position_enu_m"),
            velocity_enu_m_s=_validate_vector3(initial_velocity_enu_m_s, "initial_velocity_enu_m_s"),
            orientation_body_to_navigation=normalize_quaternion(initial_orientation_body_to_navigation),
            accelerometer_bias_body_m_s2=_validate_vector3(
                initial_accelerometer_bias_body_m_s2,
                "initial_accelerometer_bias_body_m_s2",
            ),
            gyroscope_bias_body_rad_s=_validate_vector3(
                initial_gyroscope_bias_body_rad_s,
                "initial_gyroscope_bias_body_rad_s",
            ),
            error_covariance=covariance,
        )

    @property
    def state(self) -> EKFNavigationState:
        """Return the current nominal state and its uncertainty covariance."""

        return self._state

    def predict(self, imu_sample: IMUSample, dt_seconds: float) -> EKFNavigationState:
        """Propagate nominal state and error covariance using one IMU sample."""

        _validate_dt(dt_seconds)
        try:
            validate_imu_sequence((imu_sample,))
        except ValueError as error:
            raise EKFValidationError(f"invalid IMU sample: {error}") from error

        state = self._state
        corrected_specific_force_body = _subtract_vectors(
            (imu_sample.accelerometer_x, imu_sample.accelerometer_y, imu_sample.accelerometer_z),
            state.accelerometer_bias_body_m_s2,
        )
        corrected_angular_velocity_body = _subtract_vectors(
            (imu_sample.gyroscope_x, imu_sample.gyroscope_y, imu_sample.gyroscope_z),
            state.gyroscope_bias_body_rad_s,
        )
        rotation = _rotation_matrix_body_to_navigation(state.orientation_body_to_navigation)
        specific_force_navigation = _matrix_vector_multiply(rotation, corrected_specific_force_body)
        acceleration_navigation = (
            specific_force_navigation[0],
            specific_force_navigation[1],
            specific_force_navigation[2] - self._gravity_m_s2,
        )

        position = tuple(
            state.position_enu_m[axis]
            + state.velocity_enu_m_s[axis] * dt_seconds
            + 0.5 * acceleration_navigation[axis] * dt_seconds * dt_seconds
            for axis in range(3)
        )
        velocity = tuple(
            state.velocity_enu_m_s[axis] + acceleration_navigation[axis] * dt_seconds
            for axis in range(3)
        )
        delta_orientation = integrate_angular_velocity(corrected_angular_velocity_body, dt_seconds)
        orientation = normalize_quaternion(
            multiply_quaternions(state.orientation_body_to_navigation, delta_orientation)
        )
        covariance = self._propagate_covariance(
            state.error_covariance,
            rotation,
            specific_force_navigation,
            dt_seconds,
        )
        self._state = EKFNavigationState(
            timestamp=imu_sample.timestamp,
            position_enu_m=_validate_vector3(position, "predicted position"),
            velocity_enu_m_s=_validate_vector3(velocity, "predicted velocity"),
            orientation_body_to_navigation=orientation,
            accelerometer_bias_body_m_s2=state.accelerometer_bias_body_m_s2,
            gyroscope_bias_body_rad_s=state.gyroscope_bias_body_rad_s,
            error_covariance=covariance,
        )
        return self._state

    def update_gnss_position(
        self,
        position_enu_m: Sequence[float],
        measurement_covariance_enu_m2: Sequence[Sequence[float]],
        *,
        timestamp: datetime | None = None,
    ) -> GNSSPositionUpdate:
        """Update the state from a valid GNSS position expressed in local ENU.

        The caller owns WGS84-to-ENU conversion and GNSS quality gating. The
        measurement covariance is a finite, symmetric positive-semidefinite
        3-by-3 matrix in m².
        """

        measurement = _validate_vector3(position_enu_m, "GNSS position")
        measurement_covariance = _validate_covariance(
            measurement_covariance_enu_m2, 3, "GNSS measurement covariance"
        )
        state = self._state
        innovation = _subtract_vectors(measurement, state.position_enu_m)
        innovation_covariance = _matrix_add(
            _matrix_slice(state.error_covariance, _POSITION, _POSITION, 3, 3),
            measurement_covariance,
        )
        inverse_innovation_covariance = _invert_3x3(innovation_covariance)
        covariance_position_columns = tuple(row[_POSITION : _POSITION + 3] for row in state.error_covariance)
        kalman_gain = _matrix_multiply(covariance_position_columns, inverse_innovation_covariance)
        error_correction = _matrix_vector_multiply(kalman_gain, innovation)
        corrected_state = self._apply_error_correction(state, error_correction, timestamp)

        identity = _identity_matrix(_STATE_SIZE)
        kh = [list(row) for row in identity]
        for row in range(_STATE_SIZE):
            for column in range(3):
                kh[row][_POSITION + column] -= kalman_gain[row][column]
        covariance_left = tuple(tuple(row) for row in kh)
        # Joseph form preserves symmetry and positive semidefiniteness better
        # than the simplified P <- (I - KH)P update.
        corrected_covariance = _matrix_add(
            _matrix_multiply(_matrix_multiply(covariance_left, state.error_covariance), _transpose(covariance_left)),
            _matrix_multiply(_matrix_multiply(kalman_gain, measurement_covariance), _transpose(kalman_gain)),
        )
        corrected_covariance = _validate_covariance(
            _symmetrize(corrected_covariance), _STATE_SIZE, "updated covariance"
        )
        self._state = EKFNavigationState(
            timestamp=corrected_state.timestamp,
            position_enu_m=corrected_state.position_enu_m,
            velocity_enu_m_s=corrected_state.velocity_enu_m_s,
            orientation_body_to_navigation=corrected_state.orientation_body_to_navigation,
            accelerometer_bias_body_m_s2=corrected_state.accelerometer_bias_body_m_s2,
            gyroscope_bias_body_rad_s=corrected_state.gyroscope_bias_body_rad_s,
            error_covariance=corrected_covariance,
        )
        return GNSSPositionUpdate(
            innovation_enu_m=innovation,
            innovation_covariance=innovation_covariance,
            kalman_gain=kalman_gain,
            state=self._state,
        )

    def update_gnss_sample(
        self,
        sample: GNSSSample,
        position_enu_m: Sequence[float],
        measurement_covariance_enu_m2: Sequence[Sequence[float]] | None = None,
    ) -> GNSSPositionUpdate:
        """Update from a GNSS schema sample plus its already-converted ENU position.

        If a covariance is not supplied, the sample must contain a non-negative
        scalar accuracy value, which is used as an isotropic one-sigma standard
        deviation in metres.
        """

        _validate_gnss_sample(sample)
        if measurement_covariance_enu_m2 is None:
            if sample.accuracy is None:
                raise EKFValidationError(
                    "GNSS measurement covariance is required when sample.accuracy is unavailable."
                )
            variance = sample.accuracy * sample.accuracy
            measurement_covariance_enu_m2 = (
                (variance, 0.0, 0.0),
                (0.0, variance, 0.0),
                (0.0, 0.0, variance),
            )
        return self.update_gnss_position(
            position_enu_m,
            measurement_covariance_enu_m2,
            timestamp=sample.timestamp,
        )

    def _propagate_covariance(
        self,
        covariance: Matrix,
        rotation_body_to_navigation: Matrix,
        specific_force_navigation: Vector3,
        dt_seconds: float,
    ) -> Matrix:
        transition = _identity_matrix(_STATE_SIZE)
        for axis in range(3):
            transition[_POSITION + axis][_VELOCITY + axis] = dt_seconds

        negative_force_skew = _scale_matrix(_skew_symmetric(specific_force_navigation), -dt_seconds)
        negative_rotation = _scale_matrix(rotation_body_to_navigation, -dt_seconds)
        for row in range(3):
            for column in range(3):
                transition[_VELOCITY + row][_ATTITUDE + column] = negative_force_skew[row][column]
                transition[_VELOCITY + row][_ACCELEROMETER_BIAS + column] = negative_rotation[row][column]
                transition[_ATTITUDE + row][_GYROSCOPE_BIAS + column] = negative_rotation[row][column]

        propagated = _matrix_multiply(
            _matrix_multiply(transition, covariance), _transpose(transition)
        )
        process_noise = _zero_matrix(_STATE_SIZE, _STATE_SIZE)
        acceleration_variance = self._accelerometer_noise_std * self._accelerometer_noise_std
        gyro_variance = self._gyroscope_noise_std * self._gyroscope_noise_std
        accelerometer_bias_variance = (
            self._accelerometer_bias_random_walk_std * self._accelerometer_bias_random_walk_std
        )
        gyroscope_bias_variance = (
            self._gyroscope_bias_random_walk_std * self._gyroscope_bias_random_walk_std
        )
        for axis in range(3):
            process_noise[_POSITION + axis][_POSITION + axis] = acceleration_variance * dt_seconds**3 / 3.0
            process_noise[_POSITION + axis][_VELOCITY + axis] = acceleration_variance * dt_seconds**2 / 2.0
            process_noise[_VELOCITY + axis][_POSITION + axis] = acceleration_variance * dt_seconds**2 / 2.0
            process_noise[_VELOCITY + axis][_VELOCITY + axis] = acceleration_variance * dt_seconds
            process_noise[_ATTITUDE + axis][_ATTITUDE + axis] = gyro_variance * dt_seconds
            process_noise[_ACCELEROMETER_BIAS + axis][_ACCELEROMETER_BIAS + axis] = (
                accelerometer_bias_variance * dt_seconds
            )
            process_noise[_GYROSCOPE_BIAS + axis][_GYROSCOPE_BIAS + axis] = (
                gyroscope_bias_variance * dt_seconds
            )
        return _validate_covariance(
            _symmetrize(_matrix_add(propagated, tuple(tuple(row) for row in process_noise))),
            _STATE_SIZE,
            "propagated covariance",
        )

    def _apply_error_correction(
        self,
        state: EKFNavigationState,
        correction: Sequence[float],
        timestamp: datetime | None,
    ) -> EKFNavigationState:
        if len(correction) != _STATE_SIZE or not all(isfinite(value) for value in correction):
            raise EKFValidationError("error-state correction must contain 15 finite values.")
        position = _add_vectors(state.position_enu_m, correction[_POSITION : _POSITION + 3])
        velocity = _add_vectors(state.velocity_enu_m_s, correction[_VELOCITY : _VELOCITY + 3])
        attitude_correction = integrate_angular_velocity(correction[_ATTITUDE : _ATTITUDE + 3], 1.0)
        orientation = normalize_quaternion(
            multiply_quaternions(attitude_correction, state.orientation_body_to_navigation)
        )
        accelerometer_bias = _add_vectors(
            state.accelerometer_bias_body_m_s2,
            correction[_ACCELEROMETER_BIAS : _ACCELEROMETER_BIAS + 3],
        )
        gyroscope_bias = _add_vectors(
            state.gyroscope_bias_body_rad_s,
            correction[_GYROSCOPE_BIAS : _GYROSCOPE_BIAS + 3],
        )
        return EKFNavigationState(
            timestamp=timestamp if timestamp is not None else state.timestamp,
            position_enu_m=position,
            velocity_enu_m_s=velocity,
            orientation_body_to_navigation=orientation,
            accelerometer_bias_body_m_s2=accelerometer_bias,
            gyroscope_bias_body_rad_s=gyroscope_bias,
            error_covariance=state.error_covariance,
        )


def _default_initial_covariance() -> Matrix:
    diagonal = (
        100.0,
        100.0,
        100.0,
        4.0,
        4.0,
        4.0,
        0.03,
        0.03,
        0.03,
        0.01,
        0.01,
        0.01,
        0.0001,
        0.0001,
        0.0001,
    )
    return tuple(
        tuple(value if row == column else 0.0 for column, value in enumerate(diagonal))
        for row in range(_STATE_SIZE)
    )


def _validate_gnss_sample(sample: GNSSSample) -> None:
    if not isinstance(sample, GNSSSample):
        raise EKFValidationError("sample must be a GNSSSample.")
    values = (sample.latitude, sample.longitude, sample.altitude)
    if not all(isfinite(value) for value in values):
        raise EKFValidationError("GNSS sample values must be finite.")
    if not -90.0 <= sample.latitude <= 90.0 or not -180.0 <= sample.longitude <= 180.0:
        raise EKFValidationError("GNSS sample latitude/longitude is out of range.")
    if sample.accuracy is not None and (not isfinite(sample.accuracy) or sample.accuracy < 0.0):
        raise EKFValidationError("GNSS sample accuracy must be finite and non-negative.")


def _validate_vector3(values: Sequence[float], field_name: str) -> Vector3:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise EKFValidationError(f"{field_name} must contain exactly three components.")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise EKFValidationError(f"{field_name}[{index}] must be a finite numeric value.")
        result.append(float(value))
    return (result[0], result[1], result[2])


def _validate_positive_scalar(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
        raise EKFValidationError(f"{field_name} must be a finite positive number.")
    return float(value)


def _validate_nonnegative_scalar(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0.0:
        raise EKFValidationError(f"{field_name} must be a finite non-negative number.")
    return float(value)


def _validate_dt(dt_seconds: float) -> None:
    _validate_positive_scalar(dt_seconds, "dt_seconds")


def _validate_covariance(
    covariance: Sequence[Sequence[float]], size: int, field_name: str
) -> Matrix:
    if len(covariance) != size or any(len(row) != size for row in covariance):
        raise EKFValidationError(f"{field_name} must be a {size}-by-{size} matrix.")
    validated_rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(covariance):
        validated_row: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise EKFValidationError(
                    f"{field_name}[{row_index}][{column_index}] must be finite."
                )
            validated_row.append(float(value))
        validated_rows.append(tuple(validated_row))
    validated = tuple(validated_rows)
    for row in range(size):
        for column in range(size):
            value = validated[row][column]
            if abs(value - validated[column][row]) > 1e-9:
                raise EKFValidationError(f"{field_name} must be symmetric within tolerance.")
        if validated[row][row] < -1e-12:
            raise EKFValidationError(f"{field_name} diagonal entries must be non-negative.")
    return _symmetrize(validated)


def _rotation_matrix_body_to_navigation(quaternion: Sequence[float]) -> Matrix:
    w, x, y, z = normalize_quaternion(quaternion)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _skew_symmetric(vector: Vector3) -> Matrix:
    x, y, z = vector
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def _add_vectors(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return _validate_vector3(tuple(left[index] + right[index] for index in range(3)), "vector sum")


def _subtract_vectors(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return _validate_vector3(tuple(left[index] - right[index] for index in range(3)), "vector difference")


def _matrix_vector_multiply(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    if any(len(row) != len(vector) for row in matrix):
        raise EKFValidationError("matrix/vector dimensions do not match.")
    result = tuple(sum(row[column] * vector[column] for column in range(len(vector))) for row in matrix)
    if not all(isfinite(value) for value in result):
        raise EKFValidationError("matrix/vector multiplication produced non-finite values.")
    return result


def _matrix_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    if not left or not right or len(left[0]) != len(right):
        raise EKFValidationError("matrix dimensions do not match for multiplication.")
    if any(len(row) != len(left[0]) for row in left) or any(len(row) != len(right[0]) for row in right):
        raise EKFValidationError("matrix rows must have consistent lengths.")
    return tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(len(right))) for column in range(len(right[0])))
        for row in range(len(left))
    )


def _matrix_add(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    if len(left) != len(right) or any(len(left[row]) != len(right[row]) for row in range(len(left))):
        raise EKFValidationError("matrix dimensions do not match for addition.")
    return tuple(tuple(left[row][column] + right[row][column] for column in range(len(left[row]))) for row in range(len(left)))


def _transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise EKFValidationError("matrix rows must have consistent lengths.")
    return tuple(tuple(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix[0])))


def _identity_matrix(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _zero_matrix(rows: int, columns: int) -> list[list[float]]:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def _scale_matrix(matrix: Sequence[Sequence[float]], scalar: float) -> Matrix:
    return tuple(tuple(value * scalar for value in row) for row in matrix)


def _symmetrize(matrix: Sequence[Sequence[float]]) -> Matrix:
    if len(matrix) != len(matrix[0]) or any(len(row) != len(matrix) for row in matrix):
        raise EKFValidationError("only square matrices can be symmetrized.")
    return tuple(
        tuple((matrix[row][column] + matrix[column][row]) / 2.0 for column in range(len(matrix)))
        for row in range(len(matrix))
    )


def _matrix_slice(matrix: Matrix, row_start: int, column_start: int, rows: int, columns: int) -> Matrix:
    return tuple(
        tuple(matrix[row_start + row][column_start + column] for column in range(columns))
        for row in range(rows)
    )


def _invert_3x3(matrix: Matrix) -> Matrix:
    _validate_covariance(matrix, 3, "innovation covariance")
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not isfinite(determinant) or abs(determinant) < 1e-15:
        raise EKFValidationError("innovation covariance is singular or ill-conditioned.")
    inverse = (
        ((e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant),
        ((f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant),
        ((d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant),
    )
    return _validate_covariance(_symmetrize(inverse), 3, "inverse innovation covariance")
