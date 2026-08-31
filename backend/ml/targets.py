"""Measurable residual targets for NAVIGEN navigation-error learning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite


ResidualVector = tuple[float, float, float]


class TargetValidationError(ValueError):
    """Raised when reference or estimated trajectory values are invalid."""


@dataclass(frozen=True, slots=True)
class ResidualTarget:
    """A measurable ENU position correction for one timestamp and session."""

    session_id: str
    timestamp: datetime
    position_residual_enu_m: ResidualVector

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise TargetValidationError("session_id must not be empty.")
        if not isinstance(self.timestamp, datetime):
            raise TargetValidationError("target timestamp must be a datetime.")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise TargetValidationError("target timestamp must include a UTC offset.")
        _validate_vector3(self.position_residual_enu_m, "position_residual_enu_m")


def position_residual_target(
    *,
    session_id: str,
    timestamp: datetime,
    estimated_position_enu_m: Sequence[float],
    reference_position_enu_m: Sequence[float],
) -> ResidualTarget:
    """Create ``reference position - estimated position`` as an ENU residual.

    The reference must be sourced from an externally validated trajectory, such
    as RTK GNSS or quality-screened GNSS ground truth. This function does not
    create or infer reference data.
    """

    estimated = _validate_vector3(estimated_position_enu_m, "estimated_position_enu_m")
    reference = _validate_vector3(reference_position_enu_m, "reference_position_enu_m")
    residual = tuple(reference[axis] - estimated[axis] for axis in range(3))
    return ResidualTarget(
        session_id=session_id,
        timestamp=timestamp,
        position_residual_enu_m=_validate_vector3(residual, "position residual"),
    )


def _validate_vector3(values: Sequence[float], field_name: str) -> ResidualVector:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise TargetValidationError(f"{field_name} must contain exactly three components.")
    validated: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise TargetValidationError(f"{field_name}[{index}] must be a finite numeric value.")
        validated.append(float(value))
    return (validated[0], validated[1], validated[2])
