"""Recommendation models: content-based, SVD collaborative filtering, Neural CF, baseline."""

from src.models.baseline import popularity_scores
from src.models.content_based import ContentBasedRecommender
from src.models.svd_cf import SVDRecommender

__all__ = ["ContentBasedRecommender", "SVDRecommender", "popularity_scores"]

# NCFRecommender lives in src.models.neural_cf and is imported explicitly so that
# PyTorch stays an optional dependency for the rest of the package.
