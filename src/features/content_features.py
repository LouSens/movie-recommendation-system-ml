"""TF-IDF representation of movie content (genres + user tags).

Genres and tags are vectorized separately, L2-normalized, scaled by ``sqrt(weight)``
and stacked horizontally. Because each block has unit norm, the dot product of two
rows is a *weighted sum of cosine similarities*:

    sim(a, b) = w_genre · cos(genre_a, genre_b) + w_tag · cos(tag_a, tag_b)

Keeping the blocks apart stops a movie with many tags from drowning out its genres
(and vice versa), which a single concatenated document suffers from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import TFIDF, TfidfConfig

TOKEN_PATTERN = r"(?u)\b\w+\b"


@dataclass
class ContentFeatures:
    """Fitted vectorizers and the combined ``[n_movies × (n_genres + n_tag_terms)]`` matrix."""

    genre_vectorizer: TfidfVectorizer
    tag_vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix

    @property
    def vocabulary_size(self) -> int:
        return len(self.genre_vectorizer.vocabulary_) + len(self.tag_vectorizer.vocabulary_)


def build_tfidf_matrix(
    documents: pd.Series,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_features: int | None = None,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Fit a TF-IDF vectorizer on ``documents`` and return it with the L2-normalized matrix."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        token_pattern=TOKEN_PATTERN,
    )
    matrix = vectorizer.fit_transform(documents.fillna(""))
    return vectorizer, matrix.tocsr()


def build_content_matrix(movies: pd.DataFrame, config: TfidfConfig = TFIDF) -> ContentFeatures:
    """Build the weighted genre + tag TF-IDF matrix used by the content-based model.

    Args:
        movies: Output of :func:`src.data.preprocessor.build_movie_content`; needs the
            ``genre_tokens`` and ``tags`` columns.
        config: Vectorizer settings and the genre/tag weighting.

    Returns:
        A :class:`ContentFeatures` bundle.
    """
    if not 0.0 <= config.genre_weight <= 1.0:
        raise ValueError("genre_weight must be between 0 and 1.")

    genre_vec, genre_matrix = build_tfidf_matrix(movies["genre_tokens"])
    tag_vec, tag_matrix = build_tfidf_matrix(
        movies["tags"],
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_features=config.max_features,
    )
    combined = sparse.hstack(
        [
            np.sqrt(config.genre_weight) * normalize(genre_matrix),
            np.sqrt(1.0 - config.genre_weight) * normalize(tag_matrix),
        ]
    ).tocsr()
    return ContentFeatures(genre_vec, tag_vec, combined)
