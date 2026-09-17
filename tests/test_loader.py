"""Tests for data loading, validation and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_movielens, validate_schema
from src.data.preprocessor import (
    clean_movies,
    clean_ratings,
    clean_tags,
    filter_cold_start,
    genre_to_token,
    normalize_token,
)


def _write_dataset(directory: Path, raw_movies: pd.DataFrame, raw_tags: pd.DataFrame) -> None:
    raw_movies.to_csv(directory / "movies.csv", index=False)
    raw_tags.to_csv(directory / "tags.csv", index=False)
    pd.DataFrame(
        {"userId": [1, 1], "movieId": [1, 2], "rating": [4.0, 3.5], "timestamp": [10, 20]}
    ).to_csv(directory / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [1, 2], "imdbId": [114709, 120363], "tmdbId": [862.0, None]}).to_csv(
        directory / "links.csv", index=False
    )


def test_load_movielens_reads_all_tables(tmp_path, raw_movies, raw_tags):
    _write_dataset(tmp_path, raw_movies, raw_tags)
    data = load_movielens(tmp_path)
    assert len(data.movies) == len(raw_movies)
    assert list(data.summary().index) == ["movies", "ratings", "tags", "links"]


def test_load_movielens_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_movielens(tmp_path)


def test_validate_schema_detects_missing_column(raw_movies):
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(raw_movies.drop(columns="genres"), "movies")


def test_validate_schema_unknown_table(raw_movies):
    with pytest.raises(KeyError):
        validate_schema(raw_movies, "unknown")


def test_normalize_and_genre_tokens():
    assert normalize_token("  Based on a BOOK! ") == "based on a book"
    assert genre_to_token("Sci-Fi") == "scifi"
    assert genre_to_token("Film-Noir") == "filmnoir"


def test_clean_movies_handles_missing_genres_and_year(raw_movies):
    movies = clean_movies(raw_movies)
    untitled = movies.set_index("movieId").loc[6]
    assert untitled["genres"] == "Unknown"
    assert pd.isna(untitled["year"])
    assert movies.set_index("movieId").loc[1, "year"] == 1995
    assert movies.loc[0, "genre_list"] == ["Adventure", "Animation", "Children"]


def test_clean_ratings_removes_duplicates_and_keeps_latest():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [5, 5, 5],
            "rating": [2.0, 4.5, 9.0],
            "timestamp": [1, 2, 3],
        }
    )
    cleaned = clean_ratings(ratings)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "rating"] == 4.5
    assert "datetime" in cleaned


def test_clean_tags_drops_blank_and_duplicates(raw_tags):
    tags = clean_tags(raw_tags)
    assert "" not in set(tags["tag_clean"])
    assert (tags["movieId"] == 3).sum() == 1


def test_filter_cold_start_is_iterative():
    # Movie 3 is rated only by user 2; removing it drops user 2 below the user threshold.
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [1, 2, 1, 3, 1, 2],
            "rating": [4.0] * 6,
        }
    )
    filtered = filter_cold_start(ratings, min_user_ratings=2, min_movie_ratings=2)
    assert set(filtered["userId"]) == {1, 3}
    assert set(filtered["movieId"]) == {1, 2}
