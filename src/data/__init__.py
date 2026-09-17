"""Dataset acquisition, loading and preprocessing."""

from src.data.downloader import download_movielens
from src.data.loader import MovieLensData, load_movielens
from src.data.preprocessor import (
    build_movie_content,
    clean_movies,
    clean_ratings,
    clean_tags,
    filter_cold_start,
)

__all__ = [
    "MovieLensData",
    "build_movie_content",
    "clean_movies",
    "clean_ratings",
    "clean_tags",
    "download_movielens",
    "filter_cold_start",
    "load_movielens",
]
