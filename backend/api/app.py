"""FastAPI endpoints for the currently available NAVIGEN pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.data.schemas import GNSSSample, IMUSample, SensorSession
from backend.services.navigation_service import (
    GNSSPositionMeasurement,
    NavigationService,
    NavigationServiceValidationError,
)


class IMUSampleRequest(BaseModel):
    timestamp: datetime
    accelerometer_x: float
    accelerometer_y: float
    accelerometer_z: float
    gyroscope_x: float
    gyroscope_y: float
    gyroscope_z: float


class GNSSPositionMeasurementRequest(BaseModel):
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float
    accuracy: float | None = None
    position_enu_m: Annotated[list[float], Field(min_length=3, max_length=3)]
    measurement_covariance_enu_m2: Annotated[list[Annotated[list[float], Field(min_length=3, max_length=3)]], Field(min_length=3, max_length=3)] | None = None


class NavigationRequest(BaseModel):
    imu_samples: Annotated[list[IMUSampleRequest], Field(min_length=1)]
    gnss_measurements: list[GNSSPositionMeasurementRequest] = Field(default_factory=list)
    session_id: str | None = None


def create_app() -> FastAPI:
    """Create the NAVIGEN HTTP application without starting a server."""

    application = FastAPI(title="NAVIGEN API", version="0.1.0")
    service = NavigationService()

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "navigen"}

    @application.post("/navigation/process")
    def process_navigation(request: NavigationRequest) -> dict[str, object]:
        try:
            session = SensorSession(
                imu_samples=tuple(
                    IMUSample(
                        timestamp=item.timestamp,
                        accelerometer_x=item.accelerometer_x,
                        accelerometer_y=item.accelerometer_y,
                        accelerometer_z=item.accelerometer_z,
                        gyroscope_x=item.gyroscope_x,
                        gyroscope_y=item.gyroscope_y,
                        gyroscope_z=item.gyroscope_z,
                    )
                    for item in request.imu_samples
                ),
                session_id=request.session_id,
            )
            measurements = tuple(
                GNSSPositionMeasurement(
                    sample=GNSSSample(
                        timestamp=item.timestamp,
                        latitude=item.latitude,
                        longitude=item.longitude,
                        altitude=item.altitude,
                        accuracy=item.accuracy,
                    ),
                    position_enu_m=tuple(item.position_enu_m),
                    measurement_covariance_enu_m2=(
                        tuple(tuple(row) for row in item.measurement_covariance_enu_m2)
                        if item.measurement_covariance_enu_m2 is not None
                        else None
                    ),
                )
                for item in request.gnss_measurements
            )
            result = service.process_session(session, gnss_measurements=measurements)
        except (ValueError, NavigationServiceValidationError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return {
            "ml_correction": {"available": result.ml_correction_available, "applied": False},
            "baseline_states": [
                {
                    "timestamp": state.timestamp,
                    "position_enu_m": state.position_enu_m,
                    "velocity_enu_m_s": state.velocity_enu_m_s,
                    "orientation_body_to_navigation": state.orientation_body_to_navigation,
                    "uncertainty": None,
                }
                for state in result.baseline_states
            ],
            "fused_states": [
                {
                    "timestamp": timestamp,
                    "position_enu_m": state.position_enu_m,
                    "velocity_enu_m_s": state.velocity_enu_m_s,
                    "orientation_body_to_navigation": state.orientation_body_to_navigation,
                    "error_covariance": state.error_covariance,
                }
                for timestamp, state in zip(result.fused_timestamps, result.fused_states, strict=True)
            ],
        }

    return application


app = create_app()
