# NAVIGEN Architecture

## Current implementation status

NAVIGEN currently contains a tested, offline navigation pipeline and a thin
FastAPI interface. The implementation operates on caller-supplied sensor
samples; it does not include a live device connector, a deployed service, or a
real sensor dataset.

| Component | Status | Scope and boundary |
|---|---|---|
| Sensor data layer | Implemented | Validated IMU/GNSS/session schemas and CSV loaders. |
| Sensor preprocessing | Implemented | IMU timestamp ordering, interval, sampling-rate, and finite-value validation. |
| Quaternion orientation estimation | Implemented | Quaternion normalization/multiplication and gyro-based propagation. |
| Baseline inertial dead reckoning | Implemented | ENU trajectory propagation with body-to-navigation rotation and gravity handling. |
| 15-state ES-EKF | Implemented | Nominal state, covariance prediction, and error-state correction. |
| GNSS measurement update | Implemented | Local-ENU position updates, including covariance or accuracy-derived noise. |
| Uncertainty/covariance output | Implemented | EKF covariance is retained and returned for fused states. |
| ML residual model architecture | Implemented | Feature/target/dataset utilities and a linear residual-regression model exist. |
| Evaluation metrics | Implemented | Position, velocity, MAE, RMSE, and trajectory metrics are available. |
| Navigation service integration | Implemented | Orchestrates preprocessing, baseline propagation, and ES-EKF updates. |
| FastAPI backend | Implemented | `/health` and `/navigation/process` endpoints are implemented and tested. |

## Data-dependent work that remains

The numerical and transport components are implemented, but meaningful system
validation still requires real, synchronized recordings. No IMU recordings,
GNSS recordings, surveyed/RTK/motion-capture reference trajectories, trained
model artifact, or benchmark result is included in this repository.

Real synchronized IMU/GNSS/reference trajectories are required to:

- quantify inertial drift and GNSS-corrected accuracy;
- validate timestamp alignment, coordinate conventions, and sensor behavior;
- create trustworthy residual-training targets and held-out evaluation splits;
- assess GNSS-denied operation, uncertainty calibration, and cross-device
  generalization.

The repository does not yet implement live device ingestion, persistent
sessions, automatic WGS84-to-local-ENU conversion, GNSS quality/availability
state management, ZUPT, vehicle-motion constraints, magnetometer calibration,
or production deployment configuration.

## Implemented data and navigation flow

Callers provide timestamped IMU samples and optional GNSS measurements. The
service validates the IMU sequence, calculates intervals, runs the baseline
inertial propagator, and runs the 15-state ES-EKF. GNSS measurements are
accepted when their timestamps match IMU timestamps and their local ENU
position (and optional covariance) is valid. The API returns baseline states,
fused states, orientations, velocities, and fused error covariance.

Raw GNSS latitude/longitude/altitude are retained in the GNSS schema, while
the current service expects the caller to provide the corresponding local ENU
position for an EKF update. This explicit boundary avoids claiming that a
geodetic frame conversion is already available.

## Machine-learning residual correction

The residual regression implementation exists. It provides deterministic
feature generation, residual-target and dataset structures, session-aware
splitting, and a standard-library linear regression model that must be trained
explicitly before prediction.

No real training dataset is included, no trained model is shipped, and no
navigation improvement is claimed. Real synchronized IMU/GNSS/reference
trajectories are still required for training, validation, and any future
confidence-gated integration with the navigation filter. The current API does
not apply an ML correction and reports it as unavailable.

## API and deployment status

FastAPI endpoints are implemented:

- `GET /health`
- `POST /navigation/process`

The API tests pass as part of the complete unittest suite. Live deployment,
authentication, persistence, streaming ingestion, and operational
configuration are not yet configured.

## Testing and claims

The test suite covers the implemented schemas, loaders, preprocessing,
orientation, dead reckoning, ES-EKF, service/API behavior, ML utilities, and
evaluation metrics. Tests use deterministic in-memory mathematical fixtures;
they are not a substitute for validation on real recordings. No accuracy,
outage-drift, or ML-improvement metric is claimed by this repository.
