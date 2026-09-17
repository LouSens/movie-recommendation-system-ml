"""Tests for collaborative features, SVD and Neural CF."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import NCFConfig, SVDConfig
from src.evaluation.metrics import evaluate_ranking, rmse
from src.features.collaborative_features import (
    IdEncoder,
    build_user_item_matrix,
    sparsity,
    train_test_split_by_user,
)
from src.models.baseline import popularity_scores
from src.models.svd_cf import SVDRecommender


def test_id_encoder_roundtrip():
    enc = IdEncoder.fit([30, 10, 20, 10])
    assert len(enc) == 3
    assert enc.transform([10, 30]).tolist() == [0, 2]
    assert enc.inverse_transform([2, 1]).tolist() == [30, 20]
    assert 20 in enc and 99 not in enc
    with pytest.raises(KeyError):
        enc.transform([99])


def test_split_is_stratified_and_disjoint(synthetic_ratings):
    train, test = train_test_split_by_user(synthetic_ratings, test_size=0.2, seed=1)
    assert len(train) + len(test) == len(synthetic_ratings)
    assert set(train["userId"]) == set(synthetic_ratings["userId"])
    merged = train.merge(test, on=["userId", "movieId"])
    assert merged.empty
    ratio = test.groupby("userId").size() / synthetic_ratings.groupby("userId").size()
    assert ratio.between(0.1, 0.25).all()


def test_split_keeps_one_training_rating_per_user():
    ratings = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0]})
    train, test = train_test_split_by_user(ratings, test_size=0.9)
    assert len(train) == 1 and test.empty


def test_user_item_matrix_and_sparsity(synthetic_ratings):
    users = IdEncoder.fit(synthetic_ratings["userId"])
    items = IdEncoder.fit(synthetic_ratings["movieId"])
    matrix = build_user_item_matrix(synthetic_ratings, users, items)
    assert matrix.shape == (30, 40)
    assert matrix.nnz == len(synthetic_ratings)
    assert 0.0 < sparsity(matrix) < 1.0


def test_svd_learns_structure(synthetic_ratings):
    train, test = train_test_split_by_user(synthetic_ratings, seed=3)
    svd = SVDRecommender().fit(train, n_factors=5, n_epochs=40, lr_all=0.01, reg_all=0.05)
    preds = svd.predict(test)
    global_mean_rmse = rmse(test["rating"], np.full(len(test), train["rating"].mean()))
    assert rmse(test["rating"], preds) < global_mean_rmse


def test_svd_vectorized_scores_match_surprise(synthetic_ratings):
    svd = SVDRecommender().fit(synthetic_ratings, n_factors=4, n_epochs=5)
    users = IdEncoder.fit(synthetic_ratings["userId"])
    items = IdEncoder.fit(synthetic_ratings["movieId"])
    scores = svd.score_users(users, items, clip=True)
    for user, item in [(1, 1), (7, 13), (30, 40)]:
        expected = svd.model.predict(user, item).est
        assert scores[users.to_index[user], items.to_index[item]] == pytest.approx(
            expected, abs=1e-4
        )


def test_svd_tune_small_grid(synthetic_ratings):
    config = SVDConfig(param_grid={"n_factors": [2, 4], "n_epochs": [5]}, cv_folds=2)
    svd = SVDRecommender(config=config)
    best = svd.tune(synthetic_ratings, n_jobs=1)
    assert set(best) == {"n_factors", "n_epochs"}
    assert len(svd.cv_results) == 2


def test_svd_recommend_excludes_seen(synthetic_ratings, raw_movies):
    movies = pd.DataFrame(
        {"movieId": range(1, 41), "title": [f"M{i}" for i in range(1, 41)], "genres": "Drama"}
    )
    svd = SVDRecommender().fit(synthetic_ratings, n_factors=4, n_epochs=5)
    items = IdEncoder.fit(synthetic_ratings["movieId"])
    recs = svd.recommend(1, synthetic_ratings, movies, items, top_n=5)
    seen = set(synthetic_ratings.loc[synthetic_ratings["userId"] == 1, "movieId"])
    assert len(recs) == 5 and not seen & set(recs["movieId"])
    assert recs["predicted_rating"].between(0.5, 5.0).all()


def test_popularity_scores_identical_rows(synthetic_ratings):
    users = IdEncoder.fit(synthetic_ratings["userId"])
    items = IdEncoder.fit(synthetic_ratings["movieId"])
    scores = popularity_scores(synthetic_ratings, users, items)
    assert scores.shape == (30, 40)
    assert np.all(scores == scores[0])


def _implicit_ratings(n_users: int = 40, n_items: int = 60, seed: int = 0) -> pd.DataFrame:
    """Users mostly watch (and like) movies from their own taste cluster."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(1, n_users + 1):
        for movie in range(1, n_items + 1):
            in_cluster = (movie % 2) == (user % 2)
            if rng.random() < (0.6 if in_cluster else 0.05):
                rows.append((user, movie, 4.5 if in_cluster else 2.0))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def test_neural_cf_trains_and_beats_random():
    torch = pytest.importorskip("torch")
    from src.models.neural_cf import NCFRecommender, NeuMF

    model = NeuMF(n_users=3, n_items=4, embedding_dim=8, hidden_layers=(8, 4))
    logits = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
    assert logits.shape == (2,)

    train, test = train_test_split_by_user(_implicit_ratings(), seed=5)
    users, items = IdEncoder.fit(train["userId"]), IdEncoder.fit(train["movieId"])
    config = NCFConfig(
        embedding_dim=8, hidden_layers=(16, 8), batch_size=256, max_epochs=15, patience=15
    )
    ncf = NCFRecommender(users, items, config=config, device="cpu").fit(train, verbose=False)
    scores = ncf.score_users()
    assert scores.shape == (len(users), len(items))
    assert np.all((scores >= 0) & (scores <= 1))
    assert len(ncf.history.to_frame()) >= 1

    random_scores = np.random.default_rng(0).random(scores.shape)
    ncf_ndcg = evaluate_ranking(scores, train, test, users, items, (10,))["NDCG@10"]
    random_ndcg = evaluate_ranking(random_scores, train, test, users, items, (10,))["NDCG@10"]
    assert ncf_ndcg > random_ndcg
