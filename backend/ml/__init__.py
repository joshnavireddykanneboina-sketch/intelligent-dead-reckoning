"""Supervised residual-learning components for NAVIGEN."""

from .dataset import (
    ResidualDataset,
    ResidualTrainingExample,
    SessionDatasetSplit,
    build_residual_dataset,
    split_dataset_by_session,
)
from .features import FeatureValidationError, build_feature_vector, feature_names
from .model import ModelNotTrainedError, ResidualCorrectionModel
from .targets import ResidualTarget, TargetValidationError, position_residual_target

__all__ = [
    "FeatureValidationError",
    "ModelNotTrainedError",
    "ResidualCorrectionModel",
    "ResidualDataset",
    "ResidualTarget",
    "ResidualTrainingExample",
    "SessionDatasetSplit",
    "TargetValidationError",
    "build_feature_vector",
    "build_residual_dataset",
    "feature_names",
    "position_residual_target",
    "split_dataset_by_session",
]
