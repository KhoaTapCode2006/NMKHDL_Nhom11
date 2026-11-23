"""Model and reference matching package for Milestone 3."""

from src.model.auto_labeler import (
    AutoReferenceMatcher,
    run_auto_labeling_pipeline,
    select_auto_label_publications,
    split_publications_train_val_test,
)
from src.model.feature_engineering import FeatureExtractor

__all__ = [
    "AutoReferenceMatcher",
    "run_auto_labeling_pipeline",
    "select_auto_label_publications",
    "split_publications_train_val_test",
    "FeatureExtractor",
]
