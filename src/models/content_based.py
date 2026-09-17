"""Content-Based Filtering: TF-IDF over genres + tags with (weighted) cosine similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from src.config import DEFAULT_TOP_N, TFIDF, TfidfConfig
from src.features.collaborative_features import IdEncoder, build_user_item_matrix
from src.features.content_features import ContentFeatures, build_content_matrix


class ContentBasedRecommender:
    """Recommend movies whose genre/tag profile is similar to a movie or a user's taste.

    Two modes are supported:

    * **Item-to-item** (:meth:`recommend_similar`): top-N movies most similar to a
      reference title — no rating history needed, so it also works for new users.
    * **User profile** (:meth:`score_users`): a user vector is the rating-weighted
      average of the TF-IDF vectors of the movies they rated; unseen movies are ranked
      by cosine similarity to that vector. This mode is used for offline evaluation.

    Args:
        movies: Output of :func:`src.data.preprocessor.build_movie_content`; must contain
            ``movieId``, ``title``, ``genres``, ``genre_tokens`` and ``tags``.
        config: TF-IDF hyperparameters.
        popularity: Optional ``movieId -> number of ratings``; among equally similar
            movies the more popular one is recommended first.
    """

    def __init__(
        self,
        movies: pd.DataFrame,
        config: TfidfConfig = TFIDF,
        popularity: pd.Series | None = None,
    ) -> None:
        self.movies = movies.reset_index(drop=True)
        # Number of ratings per movie, used only to break similarity ties.
        counts = popularity if popularity is not None else pd.Series(dtype=float)
        self._popularity = self.movies["movieId"].map(counts).fillna(0).to_numpy(dtype=np.float64)
        self.config = config
        self.features: ContentFeatures | None = None
        self._id_to_row = pd.Series(self.movies.index, index=self.movies["movieId"])
        # Duplicate titles exist in MovieLens; keep the first occurrence.
        titles = self.movies["title"].str.lower()
        self._title_to_row = pd.Series(self.movies.index, index=titles)
        self._title_to_row = self._title_to_row[~self._title_to_row.index.duplicated()]

    def fit(self) -> ContentBasedRecommender:
        """Fit the genre and tag TF-IDF vectorizers."""
        self.features = build_content_matrix(self.movies, self.config)
        return self

    @property
    def tfidf_matrix(self) -> sparse.csr_matrix | None:
        """Combined, weighted TF-IDF matrix (rows follow ``self.movies``)."""
        return None if self.features is None else self.features.matrix

    def _check_fitted(self) -> sparse.csr_matrix:
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before requesting recommendations.")
        return self.tfidf_matrix

    def find_title(self, query: str) -> str:
        """Resolve an exact (case-insensitive) or partial title to a catalogue title.

        Raises:
            ValueError: If no title matches ``query``.
        """
        key = query.strip().lower()
        if key in self._title_to_row.index:
            return self.movies.at[self._title_to_row[key], "title"]
        matches = self.movies[self.movies["title"].str.lower().str.contains(key, regex=False)]
        if matches.empty:
            raise ValueError(f"Movie '{query}' not found in the dataset.")
        return matches.iloc[0]["title"]

    def similarity_matrix(self, movie_ids: list[int] | None = None) -> np.ndarray:
        """Dense weighted cosine-similarity matrix for ``movie_ids`` (all movies when ``None``)."""
        matrix = self._check_fitted()
        if movie_ids is not None:
            matrix = matrix[self._id_to_row.loc[movie_ids].to_numpy()]
        return linear_kernel(matrix, dense_output=True).astype(np.float32)

    def recommend_similar(
        self,
        movie_title: str,
        top_n: int = DEFAULT_TOP_N,
        candidate_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """Return the ``top_n`` movies most similar to ``movie_title``.

        Args:
            movie_title: Exact or partial title of the reference movie.
            top_n: Number of recommendations.
            candidate_ids: Optional whitelist of ``movieId`` values to recommend from.

        Returns:
            DataFrame with ``movieId``, ``title``, ``genres`` and ``similarity``.

        Raises:
            ValueError: If the title cannot be found.
        """
        matrix = self._check_fitted()
        title = self.find_title(movie_title)
        row = self._title_to_row[title.lower()]
        scores = linear_kernel(matrix[row], matrix).ravel()
        scores[row] = -np.inf
        if candidate_ids is not None:
            allowed = self.movies["movieId"].isin(candidate_ids).to_numpy()
            scores[~allowed] = -np.inf

        # Sort by similarity (rounded to absorb float noise), then by popularity.
        top = np.lexsort((-self._popularity, -np.round(scores, 6)))[:top_n]
        top = top[np.isfinite(scores[top])]
        result = self.movies.loc[top, ["movieId", "title", "genres"]].copy()
        result["similarity"] = scores[top].round(4)
        return result.reset_index(drop=True)

    def score_users(
        self,
        train_ratings: pd.DataFrame,
        user_encoder: IdEncoder,
        item_encoder: IdEncoder,
    ) -> np.ndarray:
        """Score every (user, item) pair from mean-centred user taste profiles.

        Args:
            train_ratings: Ratings used to build the profiles.
            user_encoder: Encoder defining the row order.
            item_encoder: Encoder defining the column order (the candidate universe).

        Returns:
            Dense ``[n_users × n_items]`` array of cosine similarities.
        """
        matrix = self._check_fitted()
        item_vectors = matrix[self._id_to_row.loc[item_encoder.to_id].to_numpy()]

        centred = train_ratings.copy()
        centred["rating"] = centred["rating"] - centred.groupby("userId")["rating"].transform(
            "mean"
        )
        # A user whose ratings are all identical gets uniform positive weights instead.
        flat_users = centred.groupby("userId")["rating"].transform(lambda r: (r == 0).all())
        centred.loc[flat_users, "rating"] = 1.0

        weights = build_user_item_matrix(centred, user_encoder, item_encoder)
        profiles = normalize(weights @ item_vectors)
        return np.asarray((profiles @ item_vectors.T).todense(), dtype=np.float32)
