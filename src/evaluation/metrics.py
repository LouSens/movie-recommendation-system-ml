"""Rating-prediction and top-K ranking metrics.

Rating prediction:
    RMSE = sqrt( (1/n) Σ (r̂_i − r_i)² )
    MAE  = (1/n) Σ |r̂_i − r_i|

Ranking (binary relevance, per user, then averaged over users):
    Precision@K = |Rel ∩ Rec@K| / K
    Recall@K    = |Rel ∩ Rec@K| / |Rel|
    NDCG@K      = DCG@K / IDCG@K,   DCG@K = Σ_{i=1..K} rel_i / log2(i + 1)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import RELEVANCE_THRESHOLD, TOP_K_VALUES
from src.features.collaborative_features import IdEncoder


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    t, p = np.asarray(list(y_true), dtype=float), np.asarray(list(y_pred), dtype=float)
    if t.shape != p.shape or t.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same length.")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean Absolute Error."""
    t, p = np.asarray(list(y_true), dtype=float), np.asarray(list(y_pred), dtype=float)
    if t.shape != p.shape or t.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same length.")
    return float(np.mean(np.abs(p - t)))


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top-``k`` recommendations that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive.")
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the relevant items that appear in the top-``k`` recommendations."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return float(dcg / idcg)


def relevant_items_by_user(
    test_ratings: pd.DataFrame, threshold: float = RELEVANCE_THRESHOLD
) -> dict[int, set[int]]:
    """Map each user to the set of test movies they rated ``>= threshold``."""
    liked = test_ratings[test_ratings["rating"] >= threshold]
    return {int(u): set(map(int, g)) for u, g in liked.groupby("userId")["movieId"]}


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise indices of the ``k`` largest scores, sorted descending."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def evaluate_ranking(
    scores: np.ndarray,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    user_encoder: IdEncoder,
    item_encoder: IdEncoder,
    k_values: Iterable[int] = TOP_K_VALUES,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict[str, float]:
    """Evaluate a ``[n_users × n_items]`` score matrix with top-K ranking metrics.

    Movies a user rated in the training set are excluded from their ranking.
    Only users with at least one relevant test movie are evaluated.

    Args:
        scores: Model scores aligned with ``user_encoder`` rows and ``item_encoder`` columns.
        train_ratings: Training ratings (items to mask).
        test_ratings: Held-out ratings defining relevance.
        user_encoder: Row encoder.
        item_encoder: Column encoder.
        k_values: Cut-offs to report.
        threshold: Minimum test rating considered relevant.

    Returns:
        Mapping such as ``{"Precision@10": 0.12, "Recall@10": 0.08, "NDCG@10": 0.15, ...}``.
    """
    k_values = sorted(set(k_values))
    masked = scores.astype(np.float64, copy=True)
    known = train_ratings[
        train_ratings["userId"].isin(user_encoder.to_index)
        & train_ratings["movieId"].isin(item_encoder.to_index)
    ]
    masked[
        user_encoder.transform(known["userId"]), item_encoder.transform(known["movieId"])
    ] = -np.inf

    relevant = {
        u: {i for i in items if i in item_encoder}
        for u, items in relevant_items_by_user(test_ratings, threshold).items()
        if u in user_encoder
    }
    relevant = {u: items for u, items in relevant.items() if items}
    users = sorted(relevant)
    top = top_k_indices(masked[user_encoder.transform(users)], max(k_values))

    totals = {f"{name}@{k}": 0.0 for k in k_values for name in ("Precision", "Recall", "NDCG")}
    for row, user in enumerate(users):
        recommended = item_encoder.inverse_transform(top[row]).tolist()
        for k in k_values:
            totals[f"Precision@{k}"] += precision_at_k(recommended, relevant[user], k)
            totals[f"Recall@{k}"] += recall_at_k(recommended, relevant[user], k)
            totals[f"NDCG@{k}"] += ndcg_at_k(recommended, relevant[user], k)

    n_users = max(len(users), 1)
    results = {name: value / n_users for name, value in totals.items()}
    results["evaluated_users"] = float(len(users))
    return results


def catalog_coverage(
    scores: np.ndarray, train_mask: np.ndarray | None = None, k: int = 10
) -> float:
    """Fraction of the catalogue that appears in at least one user's top-``k`` list."""
    masked = scores if train_mask is None else np.where(train_mask, -np.inf, scores)
    top = top_k_indices(masked, k)
    return float(np.unique(top).size / scores.shape[1])
