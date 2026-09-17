"""Id encoding, train/test splitting and user–item matrices for collaborative filtering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from src.config import RANDOM_SEED, TEST_SIZE


@dataclass
class IdEncoder:
    """Bidirectional mapping between raw ids and contiguous indices ``0..n-1``."""

    to_index: dict[int, int] = field(default_factory=dict)
    to_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @classmethod
    def fit(cls, ids: Iterable[int]) -> IdEncoder:
        """Build an encoder from the unique, sorted values of ``ids``."""
        unique = np.unique(np.asarray(list(ids), dtype=np.int64))
        return cls(to_index={int(v): i for i, v in enumerate(unique)}, to_id=unique)

    def __len__(self) -> int:
        return len(self.to_id)

    def __contains__(self, raw_id: object) -> bool:
        return raw_id in self.to_index

    def transform(self, ids: Iterable[int]) -> np.ndarray:
        """Map raw ids to indices. Raises ``KeyError`` for unseen ids."""
        return np.array([self.to_index[int(v)] for v in ids], dtype=np.int64)

    def inverse_transform(self, indices: Iterable[int]) -> np.ndarray:
        """Map indices back to raw ids."""
        return self.to_id[np.asarray(list(indices), dtype=np.int64)]


def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's ratings into train and test sets (stratified by user).

    Every user keeps at least one rating in the training set, so collaborative
    models always have a learned representation for users in the test set.

    Args:
        ratings: Ratings with a ``userId`` column.
        test_size: Fraction of each user's ratings held out for testing.
        seed: Random seed.

    Returns:
        ``(train, test)`` DataFrames.
    """
    rng = np.random.default_rng(seed)
    shuffled = ratings.iloc[rng.permutation(len(ratings))]
    rank = shuffled.groupby("userId").cumcount()
    n_user = shuffled["userId"].map(shuffled["userId"].value_counts())
    n_test = np.floor(n_user * test_size).clip(upper=n_user - 1)
    is_test = rank < n_test
    train = shuffled[~is_test].sort_index().reset_index(drop=True)
    test = shuffled[is_test].sort_index().reset_index(drop=True)
    return train, test


def build_user_item_matrix(
    ratings: pd.DataFrame,
    user_encoder: IdEncoder,
    item_encoder: IdEncoder,
) -> sparse.csr_matrix:
    """Build a sparse ``[n_users × n_items]`` rating matrix.

    Args:
        ratings: Ratings with ``userId``, ``movieId`` and ``rating``; ids must be known
            to the encoders.
        user_encoder: Encoder for ``userId``.
        item_encoder: Encoder for ``movieId``.

    Returns:
        A CSR matrix whose non-zero entries are ratings.
    """
    rows = user_encoder.transform(ratings["userId"])
    cols = item_encoder.transform(ratings["movieId"])
    return sparse.csr_matrix(
        (ratings["rating"].to_numpy(dtype=np.float32), (rows, cols)),
        shape=(len(user_encoder), len(item_encoder)),
    )


def sparsity(matrix: sparse.spmatrix) -> float:
    """Fraction of empty cells in ``matrix``."""
    n_cells = matrix.shape[0] * matrix.shape[1]
    return 1.0 - matrix.nnz / n_cells if n_cells else 0.0
