"""Shared synthetic fixtures so the test-suite runs without downloading MovieLens."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import build_movie_content, clean_movies, clean_tags  # noqa: E402


@pytest.fixture
def raw_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": [
                "Toy Story (1995)",
                "Toy Story 2 (1999)",
                "Heat (1995)",
                "Casino (1995)",
                "Alien (1979)",
                "Untitled Project",
            ],
            "genres": [
                "Adventure|Animation|Children",
                "Adventure|Animation|Children",
                "Action|Crime|Thriller",
                "Crime|Drama",
                "Horror|Sci-Fi",
                "(no genres listed)",
            ],
        }
    )


@pytest.fixture
def raw_tags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 2, 1, 3, 3],
            "movieId": [1, 2, 2, 3, 3],
            "tag": ["Pixar", "pixar", "Sequel", "Al Pacino", "  "],
            "timestamp": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def movies_content(raw_movies: pd.DataFrame, raw_tags: pd.DataFrame) -> pd.DataFrame:
    return build_movie_content(clean_movies(raw_movies), clean_tags(raw_tags))


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    """30 users × 40 movies with two taste clusters, ~50% density."""
    rng = np.random.default_rng(0)
    rows = []
    for user in range(1, 31):
        taste = user % 2
        for movie in range(1, 41):
            if rng.random() < 0.5:
                liked = (movie % 2) == taste
                rating = float(np.clip(rng.normal(4.3 if liked else 2.2, 0.4), 0.5, 5.0))
                rows.append((user, movie, round(rating * 2) / 2, 1_000_000 + len(rows)))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
