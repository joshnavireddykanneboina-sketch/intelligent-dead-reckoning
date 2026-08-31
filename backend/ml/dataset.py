"""Dataset construction and session-level splitting for residual learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite

from .features import FeatureVector, feature_names
from .targets import ResidualTarget


class DatasetValidationError(ValueError):
    """Raised when residual-learning examples or session splits are invalid."""


@dataclass(frozen=True, slots=True)
class ResidualTrainingExample:
    """One feature vector and measurable residual target from one session."""

    features: FeatureVector
    target: ResidualTarget

    def __post_init__(self) -> None:
        _validate_feature_vector(self.features)


@dataclass(frozen=True, slots=True)
class ResidualDataset:
    """Validated tabular residual-learning data with source session labels."""

    features: tuple[FeatureVector, ...]
    targets: tuple[tuple[float, float, float], ...]
    session_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.features:
            raise DatasetValidationError("ResidualDataset must contain at least one example.")
        if len(self.features) != len(self.targets) or len(self.features) != len(self.session_ids):
            raise DatasetValidationError("features, targets, and session_ids must have equal lengths.")
        for features in self.features:
            _validate_feature_vector(features)
        for target in self.targets:
            _validate_target_vector(target)
        if any(not session_id.strip() for session_id in self.session_ids):
            raise DatasetValidationError("session_ids must not contain empty values.")


@dataclass(frozen=True, slots=True)
class SessionDatasetSplit:
    """Train/test datasets guaranteed to have no shared session identifier."""

    train: ResidualDataset
    test: ResidualDataset

    def __post_init__(self) -> None:
        overlap = set(self.train.session_ids).intersection(self.test.session_ids)
        if overlap:
            raise DatasetValidationError(
                f"train/test session leakage detected: {', '.join(sorted(overlap))}."
            )


def build_residual_dataset(examples: Iterable[ResidualTrainingExample]) -> ResidualDataset:
    """Build a validated dataset from examples tied to real recording sessions."""

    materialized_examples = tuple(examples)
    return ResidualDataset(
        features=tuple(example.features for example in materialized_examples),
        targets=tuple(example.target.position_residual_enu_m for example in materialized_examples),
        session_ids=tuple(example.target.session_id for example in materialized_examples),
    )


def split_dataset_by_session(
    dataset: ResidualDataset, test_session_ids: Iterable[str]
) -> SessionDatasetSplit:
    """Split by whole session identifiers; samples are never randomly mixed.

    Every requested test session must exist. Both resulting partitions must be
    non-empty, preventing accidental evaluation on the same trajectory used for
    training.
    """

    test_sessions = set(test_session_ids)
    if not test_sessions or any(
        not isinstance(session_id, str) or not session_id.strip() for session_id in test_sessions
    ):
        raise DatasetValidationError("test_session_ids must contain at least one non-empty session ID.")
    available_sessions = set(dataset.session_ids)
    unknown_sessions = test_sessions.difference(available_sessions)
    if unknown_sessions:
        raise DatasetValidationError(
            f"test_session_ids are not present in the dataset: {', '.join(sorted(unknown_sessions))}."
        )

    train_features: list[FeatureVector] = []
    train_targets: list[tuple[float, float, float]] = []
    train_session_ids: list[str] = []
    test_features: list[FeatureVector] = []
    test_targets: list[tuple[float, float, float]] = []
    test_session_ids: list[str] = []
    for features, target, session_id in zip(dataset.features, dataset.targets, dataset.session_ids, strict=True):
        if session_id in test_sessions:
            test_features.append(features)
            test_targets.append(target)
            test_session_ids.append(session_id)
        else:
            train_features.append(features)
            train_targets.append(target)
            train_session_ids.append(session_id)
    if not train_features or not test_features:
        raise DatasetValidationError("session split must leave at least one example in both train and test sets.")
    return SessionDatasetSplit(
        train=ResidualDataset(tuple(train_features), tuple(train_targets), tuple(train_session_ids)),
        test=ResidualDataset(tuple(test_features), tuple(test_targets), tuple(test_session_ids)),
    )


def _validate_feature_vector(features: Sequence[float]) -> None:
    if len(features) != len(feature_names()):
        raise DatasetValidationError(
            f"feature vector must contain {len(feature_names())} values."
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) for value in features):
        raise DatasetValidationError("feature vector must contain finite numeric values.")


def _validate_target_vector(target: Sequence[float]) -> None:
    if len(target) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
        for value in target
    ):
        raise DatasetValidationError("target vector must contain three finite numeric values.")
