"""Collaborative Filtering with matrix factorization (Surprise SVD, Funk-SVD with biases)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV

from src.config import DEFAULT_TOP_N, RANDOM_SEED, RATING_SCALE, SVD_CONFIG, SVDConfig
from src.features.collaborative_features import IdEncoder
from src.utils.helpers import top_n_for_user


def _to_surprise(ratings: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=RATING_SCALE)
    return Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)


class SVDRecommender:
    """Thin wrapper around :class:`surprise.SVD` with grid search and vectorized scoring.

    The predicted rating is ``r̂_ui = μ + b_u + b_i + q_iᵀ p_u``.

    Args:
        config: Grid-search space and number of CV folds.
        seed: Random seed for factor initialization.
    """

    def __init__(self, config: SVDConfig = SVD_CONFIG, seed: int = RANDOM_SEED) -> None:
        self.config = config
        self.seed = seed
        self.best_params: dict[str, Any] = {}
        self.cv_results: pd.DataFrame | None = None
        self.model: SVD | None = None

    def tune(self, train_ratings: pd.DataFrame, n_jobs: int = -1) -> dict[str, Any]:
        """Run a k-fold ``GridSearchCV`` on the training ratings and keep the best RMSE params.

        Args:
            train_ratings: Training ratings (``userId``, ``movieId``, ``rating``).
            n_jobs: Parallel jobs for Surprise's grid search.

        Returns:
            The best hyperparameters by mean RMSE.
        """
        grid = dict(self.config.param_grid, random_state=[self.seed])
        search = GridSearchCV(
            SVD, grid, measures=["rmse", "mae"], cv=self.config.cv_folds, n_jobs=n_jobs
        )
        search.fit(_to_surprise(train_ratings))

        best = search.best_params["rmse"]
        self.best_params = {k: v for k, v in best.items() if k != "random_state"}
        results = pd.DataFrame(search.cv_results)
        keep = ["params", "mean_test_rmse", "std_test_rmse", "mean_test_mae", "rank_test_rmse"]
        self.cv_results = results[keep].sort_values("rank_test_rmse").reset_index(drop=True)
        return self.best_params

    def fit(self, train_ratings: pd.DataFrame, **params: Any) -> SVDRecommender:
        """Train SVD on all training ratings.

        Args:
            train_ratings: Training ratings.
            **params: SVD hyperparameters; defaults to :attr:`best_params` when omitted.
        """
        params = params or self.best_params
        self.model = SVD(random_state=self.seed, **params)
        self.model.fit(_to_surprise(train_ratings).build_full_trainset())
        return self

    def _check_fitted(self) -> SVD:
        if self.model is None:
            raise RuntimeError("Call fit() before predicting.")
        return self.model

    def predict(self, ratings: pd.DataFrame) -> np.ndarray:
        """Predict ratings for the ``(userId, movieId)`` pairs in ``ratings``."""
        model = self._check_fitted()
        return np.array(
            [
                model.predict(u, i).est
                for u, i in zip(ratings["userId"], ratings["movieId"], strict=True)
            ],
            dtype=np.float32,
        )

    def score_users(
        self, user_encoder: IdEncoder, item_encoder: IdEncoder, clip: bool = False
    ) -> np.ndarray:
        """Predict every (user, item) rating at once from the learned factors.

        Users or items unknown to the trainset fall back to the global mean plus
        whichever bias is known, mirroring Surprise's own behaviour.

        Args:
            user_encoder: Row encoder.
            item_encoder: Column encoder.
            clip: Clip to the rating scale. Leave ``False`` for ranking so that many
                predictions above 5.0 do not collapse into ties.

        Returns:
            Dense ``[n_users × n_items]`` array of predicted ratings.
        """
        model = self._check_fitted()
        trainset = model.trainset
        n_factors = model.pu.shape[1]

        def lookup(raw_ids: np.ndarray, to_inner: Any, factors: np.ndarray, bias: np.ndarray):
            vecs = np.zeros((len(raw_ids), n_factors), dtype=np.float64)
            biases = np.zeros(len(raw_ids), dtype=np.float64)
            for row, raw in enumerate(raw_ids):
                try:
                    inner = to_inner(int(raw))
                except ValueError:
                    continue
                vecs[row], biases[row] = factors[inner], bias[inner]
            return vecs, biases

        p_u, b_u = lookup(user_encoder.to_id, trainset.to_inner_uid, model.pu, model.bu)
        q_i, b_i = lookup(item_encoder.to_id, trainset.to_inner_iid, model.qi, model.bi)
        scores = trainset.global_mean + b_u[:, None] + b_i[None, :] + p_u @ q_i.T
        if clip:
            scores = np.clip(scores, *RATING_SCALE)
        return scores.astype(np.float32)

    def recommend(
        self,
        user_id: int,
        train_ratings: pd.DataFrame,
        movies: pd.DataFrame,
        item_encoder: IdEncoder,
        top_n: int = DEFAULT_TOP_N,
    ) -> pd.DataFrame:
        """Top-N unseen movies for ``user_id``, ranked by (unclipped) predicted rating.

        The ``predicted_rating`` column is clipped to the 0.5–5.0 scale for display.
        """
        scores = self.score_users(IdEncoder.fit([user_id]), item_encoder)[0]
        result = top_n_for_user(user_id, scores, train_ratings, movies, item_encoder, top_n)
        result["predicted_rating"] = result["predicted_rating"].clip(*RATING_SCALE)
        return result
