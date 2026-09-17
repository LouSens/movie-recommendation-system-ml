"""Cleaning, feature derivation and cold-start filtering for MovieLens tables."""

from __future__ import annotations

import re

import pandas as pd

from src.config import MIN_MOVIE_RATINGS, MIN_USER_RATINGS

NO_GENRE_LABEL = "(no genres listed)"
UNKNOWN_GENRE = "Unknown"
_YEAR_PATTERN = r"\((\d{4})\)\s*$"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_token(text: str) -> str:
    """Lowercase ``text`` and collapse every non-alphanumeric run into one space.

    Example: ``"Sci-Fi"`` -> ``"sci fi"``, ``"Based on a book"`` -> ``"based on a book"``.
    """
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def genre_to_token(genre: str) -> str:
    """Turn a genre label into a single TF-IDF token, e.g. ``"Film-Noir"`` -> ``"filmnoir"``."""
    return normalize_token(genre).replace(" ", "")


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean the movies table and derive helper columns.

    Steps:
        1. Drop duplicate ``movieId`` rows.
        2. Replace ``(no genres listed)`` / missing genres with ``Unknown``.
        3. Split ``genres`` into a Python list (``genre_list``).
        4. Extract the release ``year`` from the title (nullable integer).

    Args:
        movies: Raw movies table with ``movieId``, ``title`` and ``genres``.

    Returns:
        A new DataFrame with ``genre_list`` and ``year`` columns added.
    """
    df = movies.drop_duplicates(subset="movieId").copy()
    df["title"] = df["title"].str.strip()
    df["genres"] = df["genres"].fillna(UNKNOWN_GENRE).replace(NO_GENRE_LABEL, UNKNOWN_GENRE)
    df["genre_list"] = df["genres"].str.split("|")
    year = df["title"].str.extract(_YEAR_PATTERN)[0]
    df["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid ratings and duplicate user–movie pairs, then parse timestamps.

    Args:
        ratings: Raw ratings table.

    Returns:
        A new DataFrame with a ``datetime`` column.
    """
    df = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    df = df[df["rating"].between(0.5, 5.0)]
    df = df.sort_values("timestamp").drop_duplicates(subset=["userId", "movieId"], keep="last")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df.sort_values(["userId", "movieId"]).reset_index(drop=True)


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Normalize free-text tags and drop empty or duplicate (movie, tag) pairs.

    Args:
        tags: Raw tags table.

    Returns:
        A new DataFrame with a ``tag_clean`` column.
    """
    df = tags.dropna(subset=["tag"]).copy()
    df["tag_clean"] = df["tag"].map(normalize_token)
    df = df[df["tag_clean"] != ""]
    df = df.drop_duplicates(subset=["movieId", "tag_clean"])
    return df.reset_index(drop=True)


def build_movie_content(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Attach genre tokens and aggregated user tags to every movie.

    Args:
        movies: Output of :func:`clean_movies`.
        tags: Output of :func:`clean_tags`.

    Returns:
        ``movies`` with extra columns:

        * ``genre_tokens`` — genres as single tokens, e.g. ``"action scifi"``.
        * ``tags`` — all normalized tags of the movie joined by spaces (``""`` if none).
        * ``content`` — ``genre_tokens`` and ``tags`` combined, for inspection.
    """
    tag_docs = tags.groupby("movieId")["tag_clean"].apply(" ".join).rename("tags")
    df = movies.merge(tag_docs, on="movieId", how="left")
    df["tags"] = df["tags"].fillna("")
    df["genre_tokens"] = df["genre_list"].map(lambda gs: " ".join(genre_to_token(g) for g in gs))
    df["content"] = (df["genre_tokens"] + " " + df["tags"]).str.strip()
    return df


def filter_cold_start(
    ratings: pd.DataFrame,
    min_user_ratings: int = MIN_USER_RATINGS,
    min_movie_ratings: int = MIN_MOVIE_RATINGS,
    max_iter: int = 10,
) -> pd.DataFrame:
    """Iteratively remove users and movies with too few ratings.

    Removing sparse movies can push a user below the threshold (and vice versa),
    so filtering repeats until the table stops changing.

    Args:
        ratings: Ratings table with ``userId`` and ``movieId`` columns.
        min_user_ratings: Minimum number of ratings a user must have.
        min_movie_ratings: Minimum number of ratings a movie must have.
        max_iter: Safety cap on the number of passes.

    Returns:
        The filtered ratings table.
    """
    df = ratings
    for _ in range(max_iter):
        before = len(df)
        movie_counts = df["movieId"].map(df["movieId"].value_counts())
        df = df[movie_counts >= min_movie_ratings]
        user_counts = df["userId"].map(df["userId"].value_counts())
        df = df[user_counts >= min_user_ratings]
        if len(df) == before:
            break
    return df.reset_index(drop=True)
