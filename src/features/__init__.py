"""Feature builders for content-based and collaborative models."""

from src.features.collaborative_features import (
    IdEncoder,
    build_user_item_matrix,
    sparsity,
    train_test_split_by_user,
)
from src.features.content_features import (
    ContentFeatures,
    build_content_matrix,
    build_tfidf_matrix,
)

__all__ = [
    "ContentFeatures",
    "IdEncoder",
    "build_content_matrix",
    "build_tfidf_matrix",
    "build_user_item_matrix",
    "sparsity",
    "train_test_split_by_user",
]
