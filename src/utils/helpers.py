"""General-purpose helpers: seeding, logging, persistence, timing and top-N formatting."""

from __future__ import annotations

import logging
import pickle
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_TOP_N, RANDOM_SEED
from src.features.collaborative_features import IdEncoder

logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python's and NumPy's random number generators."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a concise root logger."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s")


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path``, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Load a pickled object from ``path``. Only use with files you created yourself."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Log how long the wrapped block took."""
    start = time.perf_counter()
    yield
    logger.info("%s finished in %.2fs", label, time.perf_counter() - start)


def top_n_for_user(
    user_id: int,
    scores: np.ndarray,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    item_encoder: IdEncoder,
    top_n: int = DEFAULT_TOP_N,
    score_name: str = "predicted_rating",
) -> pd.DataFrame:
    """Turn a score vector over ``item_encoder`` into a top-N table, hiding seen movies.

    Args:
        user_id: The user receiving recommendations.
        scores: One score per item in ``item_encoder`` order.
        train_ratings: Ratings whose movies are excluded for ``user_id``.
        movies: Movie metadata with ``movieId``, ``title`` and ``genres``.
        item_encoder: Encoder defining the order of ``scores``.
        top_n: Number of recommendations.
        score_name: Name of the score column in the output.

    Returns:
        DataFrame with ``movieId``, ``title``, ``genres`` and the score column.
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    seen = train_ratings.loc[train_ratings["userId"] == user_id, "movieId"]
    seen = [m for m in seen if m in item_encoder]
    if seen:
        scores[item_encoder.transform(seen)] = -np.inf

    top = np.argsort(-scores, kind="stable")[:top_n]
    result = pd.DataFrame({"movieId": item_encoder.inverse_transform(top), score_name: scores[top]})
    result = result.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    result[score_name] = result[score_name].round(4)
    return result[["movieId", "title", "genres", score_name]]
