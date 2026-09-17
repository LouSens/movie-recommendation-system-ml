"""Load the raw MovieLens CSV files and validate their schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR

EXPECTED_SCHEMAS: dict[str, dict[str, str]] = {
    "movies": {"movieId": "int64", "title": "object", "genres": "object"},
    "ratings": {"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    "tags": {"userId": "int64", "movieId": "int64", "tag": "object", "timestamp": "int64"},
    "links": {"movieId": "int64", "imdbId": "int64", "tmdbId": "float64"},
}


@dataclass
class MovieLensData:
    """Container for the four MovieLens tables."""

    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Return row/column counts, missing values and duplicates for each table."""
        rows = []
        for name in EXPECTED_SCHEMAS:
            df: pd.DataFrame = getattr(self, name)
            rows.append(
                {
                    "table": name,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "missing_values": int(df.isna().sum().sum()),
                    "duplicate_rows": int(df.duplicated().sum()),
                }
            )
        return pd.DataFrame(rows).set_index("table")


def validate_schema(df: pd.DataFrame, name: str) -> None:
    """Check that ``df`` has the expected columns and dtypes for table ``name``.

    Args:
        df: DataFrame to validate.
        name: One of ``movies``, ``ratings``, ``tags`` or ``links``.

    Raises:
        KeyError: If ``name`` is not a known table.
        ValueError: If a column is missing or has an unexpected dtype.
    """
    if name not in EXPECTED_SCHEMAS:
        raise KeyError(f"Unknown table '{name}'.")

    expected = EXPECTED_SCHEMAS[name]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Table '{name}' is missing columns: {sorted(missing)}")

    for column, dtype in expected.items():
        if str(df[column].dtype) != dtype:
            raise ValueError(
                f"Column '{name}.{column}' has dtype {df[column].dtype}, expected {dtype}."
            )


def load_movielens(data_dir: Path = RAW_DATA_DIR, validate: bool = True) -> MovieLensData:
    """Load ``movies``, ``ratings``, ``tags`` and ``links`` from ``data_dir``.

    Args:
        data_dir: Directory containing the raw CSV files.
        validate: Whether to validate the schema of every table.

    Returns:
        A :class:`MovieLensData` instance.

    Raises:
        FileNotFoundError: If a CSV file is missing. Run ``python -m src.data.downloader``.
    """
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name in EXPECTED_SCHEMAS:
        path = data_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.data.downloader` first.")
        tables[name] = pd.read_csv(path)
        if validate:
            validate_schema(tables[name], name)
    return MovieLensData(**tables)
