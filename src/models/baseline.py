"""Non-personalized popularity baseline used as a reference point during evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RELEVANCE_THRESHOLD
from src.features.collaborative_features import IdEncoder


def popularity_scores(
    train_ratings: pd.DataFrame,
    user_encoder: IdEncoder,
    item_encoder: IdEncoder,
    threshold: float = RELEVANCE_THRESHOLD,
) -> np.ndarray:
    """Give every user the same score per movie: its number of positive training ratings.

    Args:
        train_ratings: Training ratings.
        user_encoder: Encoder defining the row order.
        item_encoder: Encoder defining the column order.
        threshold: A rating ``>= threshold`` counts as positive.

    Returns:
        Dense ``[n_users × n_items]`` score array (identical rows).
    """
    liked = train_ratings[train_ratings["rating"] >= threshold]
    counts = liked["movieId"].value_counts()
    item_scores = counts.reindex(item_encoder.to_id, fill_value=0).to_numpy(dtype=np.float32)
    return np.tile(item_scores, (len(user_encoder), 1))
